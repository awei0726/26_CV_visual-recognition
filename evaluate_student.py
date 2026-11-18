"""Evaluate a trained Student checkpoint without loading the Teacher model."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

from student.data import (
    CharacterVocabulary, GridDataset, collate_grid, PREPROCESSING_VERSION,
    NORMALIZATION_MEAN, NORMALIZATION_STD, INPUT_SIZE,
)
from student.evaluation import evaluate
from student.checkpoint import atomic_json, model_state_sha256, run_lock
from student.model import StudentModel


REPOSITORY_ROOT = Path(__file__).resolve().parent


def resolve_path(path, root):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def completed_run_best(checkpoint_path, checkpoint, state_sha256):
    """Check the current selection when the completed run's records accompany the checkpoint."""
    run_id = checkpoint.get("run_id")
    if not run_id:
        return None
    try:
        status = json.loads(checkpoint_path.with_name("status.json").read_text(encoding="utf-8"))
        summary = json.loads(checkpoint_path.with_name("summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(status, dict) or not isinstance(summary, dict):
        return None
    if status.get("run_id") != run_id or summary.get("run_id") != run_id:
        return None
    last_epoch = summary.get("last_epoch")
    best_epoch = summary.get("best_epoch")
    best_sha256 = summary.get("best_model_sha256")
    if (status.get("state") != "completed" or type(last_epoch) is not int
            or last_epoch != status.get("last_completed_epoch")
            or type(best_epoch) is not int or not 1 <= best_epoch <= last_epoch
            or not isinstance(best_sha256, str) or not best_sha256):
        return None
    return checkpoint.get("epoch") == best_epoch and state_sha256 == best_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Explicit evaluation split, for example data/student_grid/val.jsonl")
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--vocab", type=Path,
                        default=Path("benchmarks/GRID/models/GRID_V_WER4.8/model.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New or empty directory; existing results are never overwritten")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.workers < 0 or args.threads < 1:
        parser.error("--batch-size/--threads must be positive and --workers must be nonnegative.")
    torch.set_num_threads(args.threads)

    root = args.root.resolve()
    checkpoint_path = resolve_path(args.checkpoint, root)
    output_dir = resolve_path(args.output_dir, root)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    output_initialized = False
    output_lock = ExitStack()
    try:
        if output_dir.exists() and (not output_dir.is_dir() or
                                    any(path.name != ".run.lock" for path in output_dir.iterdir())):
            raise ValueError(f"Output directory must be new or empty: {output_dir}")
        requested_device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        device = torch.device(requested_device)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Supported devices are cpu and cuda (or cuda:N).")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is not available; use --device cpu.")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device index {device.index} is unavailable.")

        vocabulary_path = resolve_path(args.vocab, root)
        manifest_path = resolve_path(args.manifest, root)
        vocabulary = CharacterVocabulary.from_file(vocabulary_path)
        manifest_bytes = manifest_path.read_bytes()
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        # Git can check out the same JSONL using LF or CRLF on different machines.
        lf_manifest = manifest_bytes.replace(b"\r\n", b"\n")
        manifest_hashes = {manifest_sha256, hashlib.sha256(lf_manifest).hexdigest(),
                           hashlib.sha256(lf_manifest.replace(b"\n", b"\r\n")).hexdigest()}
        # Hash and load the same open artifact. A concurrently replaced best.pt
        # must not produce a fingerprint for different model weights.
        with checkpoint_path.open("rb") as checkpoint_stream:
            checkpoint_bytes = checkpoint_stream.seek(0, 2)
            checkpoint_stream.seek(0)
            checkpoint_sha256 = hashlib.file_digest(checkpoint_stream, "sha256").hexdigest()
            checkpoint_stream.seek(0)
            checkpoint = torch.load(checkpoint_stream, map_location="cpu", weights_only=True)
        required = {"model_state", "model_config", "vocab_tokens"}
        if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
            raise ValueError("Checkpoint must include model_state, model_config and vocab_tokens.")
        if list(checkpoint["vocab_tokens"]) != list(vocabulary.tokens):
            raise ValueError("Checkpoint vocabulary order differs from the selected vocabulary.")
        training_config = checkpoint.get("config", {})
        if not isinstance(training_config, dict):
            raise ValueError("Checkpoint config must be a dictionary when present.")
        preprocessing_version = checkpoint.get("preprocessing_version", training_config.get("preprocessing_version"))
        if preprocessing_version not in (None, PREPROCESSING_VERSION):
            raise ValueError("Checkpoint preprocessing differs from the current data loader.")
        if checkpoint["model_config"].get("vocab_size", vocabulary.size) != vocabulary.size:
            raise ValueError("Checkpoint model class count differs from its vocabulary.")
        state_sha256 = model_state_sha256(checkpoint["model_state"])
        if checkpoint.get("model_sha256", state_sha256) != state_sha256:
            raise ValueError("Checkpoint model fingerprint does not match its stored weights.")
        model = StudentModel(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device)
        dataset = GridDataset(manifest_path, root, vocabulary,
                              load_teacher=False, validate=True)
        evaluation_data_fingerprint = dataset.fingerprint(False)
        trained_val_fingerprint = training_config.get("val_data_fingerprint")
        matches_validation_manifest = training_config.get("val_manifest_sha256") in manifest_hashes
        matches_training_input = (
            training_config.get("train_manifest_sha256") in manifest_hashes
            or evaluation_data_fingerprint == training_config.get("train_roi_fingerprint")
        )
        if (matches_validation_manifest and trained_val_fingerprint is not None
                and trained_val_fingerprint != evaluation_data_fingerprint):
            raise ValueError("Validation data differs from the ROI and labels recorded in the checkpoint. "
                             "Restore the training run's validation data before comparing scores.")
        validation_data_verified = None
        if trained_val_fingerprint is not None:
            # The trainer fingerprints the manifest before applying --limit-val.
            validation_data_verified = (trained_val_fingerprint == evaluation_data_fingerprint
                                        and len(dataset) == training_config.get("val_samples", len(dataset)))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_grid,
                            pin_memory=device.type == "cuda")
        output_lock.enter_context(run_lock(output_dir))
        # Recheck after reserving the directory in case another process finished
        # an evaluation between the initial check and this lock acquisition.
        if any(path.name != ".run.lock" for path in output_dir.iterdir()):
            raise ValueError(f"Output directory must be new or empty: {output_dir}")
        output_initialized = True
        metrics = evaluate(model, loader, device, vocabulary, predictions_path=predictions_path)
        checkpoint_is_best = completed_run_best(checkpoint_path, checkpoint, state_sha256)
        checkpoint_best_scope = None
        if checkpoint_is_best is not None:
            checkpoint_best_scope = "completed_run"
        elif checkpoint.get("best_epoch") is not None and checkpoint.get("best_model_sha256") is not None:
            checkpoint_is_best = (checkpoint.get("epoch") == checkpoint["best_epoch"]
                                  and state_sha256 == checkpoint["best_model_sha256"])
            checkpoint_best_scope = "checkpoint"
        metrics.update({
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_format_version": checkpoint.get("format_version"),
            "model_sha256": state_sha256,
            "training_run_id": checkpoint.get("run_id"),
            "epoch": checkpoint.get("epoch"),
            "checkpoint_is_best": checkpoint_is_best,
            "checkpoint_best_scope": checkpoint_best_scope,
            "manifest": str(manifest_path), "manifest_sha256": manifest_sha256,
            "evaluation_data_fingerprint": evaluation_data_fingerprint,
            "matches_training_input": matches_training_input,
            "validation_data_verified": validation_data_verified,
            "predictions_sha256": sha256_file(predictions_path),
            "vocabulary_sha256": sha256_file(vocabulary_path),
            "smoke_test": training_config.get("smoke_test"),
            "training_mode": training_config.get("mode"),
            "training_seed": training_config.get("seed"),
            "training_samples": training_config.get("train_samples"),
            "training_validation_samples": training_config.get("val_samples"),
            "training_manifest_sha256": training_config.get("train_manifest_sha256"),
            "training_validation_manifest_sha256": training_config.get("val_manifest_sha256"),
            "training_roi_fingerprint": training_config.get("train_roi_fingerprint"),
            "training_validation_roi_fingerprint": trained_val_fingerprint,
            "training_implementation_sha256": training_config.get("implementation_sha256"),
            "training_hyperparameters": {
                key: training_config.get(key) for key in (
                    "batch_size", "epochs", "lr", "weight_decay", "grad_clip", "lr_patience",
                    "lr_monitor", "amp", "amp_init_scale", "deterministic", "temperature", "kd_weight",
                )
            },
            "model_config": model.get_config(),
            "preprocessing": {
                "version": PREPROCESSING_VERSION,
                "checkpoint_metadata_present": preprocessing_version is not None,
                "input_size": list(INPUT_SIZE), "scale": 255.0,
                "mean": NORMALIZATION_MEAN, "std": NORMALIZATION_STD,
            },
            "decoder": "ctc_greedy", "cer_excludes_spaces": True,
            "metric_unit": "ratio", "device": str(device),
            "batch_size": args.batch_size, "cpu_threads": args.threads,
            "torch_version": str(torch.__version__), "python_version": sys.version.split()[0],
            "evaluation_status": "complete",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "model_weight_bytes": sum(value.numel() * value.element_size()
                                      for value in model.state_dict().values()),
            "checkpoint_bytes": checkpoint_bytes,
        })
        atomic_json(metrics, metrics_path)
    except (Exception, KeyboardInterrupt) as error:
        if output_initialized and not metrics_path.exists():
            try:
                predictions_path.unlink(missing_ok=True)
            except OSError:
                # Preserve the original error if cleanup is blocked.
                pass
            try:
                atomic_json({
                    "evaluation_status": "failed", "error_type": type(error).__name__,
                    "error": str(error), "checkpoint": str(checkpoint_path),
                    "manifest": str(resolve_path(args.manifest, root)),
                    "note": "This directory does not contain a completed evaluation. Use a new output directory to retry.",
                }, output_dir / "failed.json")
            except OSError:
                pass
        if isinstance(error, KeyboardInterrupt):
            raise
        parser.error(str(error))
    finally:
        output_lock.close()

    if metrics["matches_training_input"]:
        print("DIAGNOSTIC: the evaluation matches the training data. Use held-out data to measure generalization.")
    if metrics["checkpoint_is_best"] is False:
        print("CHECKPOINT: the evaluated weights differ from the recorded best selection.")
    if metrics["smoke_test"] is True:
        print("PIPELINE CHECK CHECKPOINT: these are not reportable experiment results.")
    elif metrics["smoke_test"] is None:
        print("Checkpoint has no smoke-test metadata; verify its training provenance before reporting results.")
    print(f"Samples={metrics['samples']}  WER={metrics['wer']:.2%}  "
          f"CER={metrics['cer']:.2%}  CTC loss={metrics['loss']:.6f}")
    print(f"Results: {output_dir}")
    return metrics


if __name__ == "__main__":
    main()
