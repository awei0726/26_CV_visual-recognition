"""Train the GRID Student with CTC, standard KD or adaptive temporal KD."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import pickle
import platform
import random
import sys
import time
import traceback
import uuid
import warnings
from pathlib import Path

# CuBLAS reads this when its first handle is created, so setting it inside
# main() is too late when multiple experiments run in one Python process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from student.data import CharacterVocabulary, GridDataset, collate_grid, PREPROCESSING_VERSION
from student.checkpoint import atomic_json, atomic_save, atomic_text, model_state_sha256, run_lock
from student.evaluation import evaluate
from student.losses import adaptive_kd_loss, ctc_loss, standard_kd_loss
from student.model import StudentModel
from student.splits import manifest_digests

ROOT = Path(__file__).resolve().parent
# Exact implementation fingerprint of the compatible midterm release (ec3836d).
MIDTERM_IMPLEMENTATION_SHA256 = "2f46c006fc6b99538ab299da576520929422a0fb44464059e2343396c018cc37"
# Prior adaptive release used the same training math (aa50ac0 and 6545be1).
PRE_PORTABLE_IMPLEMENTATION_SHA256 = "8cf5485372ea9dcc7b30cf68c55929544ccddabc64977b7e7a9b68d323716c3c"
ADAPTIVE_MODES = ("kd-entropy", "kd-blank")
FIELDS = ["epoch", "train_loss", "train_ctc", "train_kd", "val_loss", "val_wer",
          "val_cer", "lr", "next_lr", "seconds", "peak_cuda_mib", "train_samples", "val_samples",
          "optimizer_steps", "amp_skipped_steps"]


def resolve(path, root):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def select_training_indices(records, size, seed):
    """Rank IDs deterministically for nested subsets, then retain manifest order."""
    if size is None or size == len(records):
        return None
    if not 0 < size < len(records):
        raise ValueError(f"--train-size must be in [1, {len(records)}] for this manifest")

    def rank(index):
        sample_id = records[index]["id"]
        return hashlib.sha256(f"{seed}\n{sample_id}".encode("utf-8")).digest(), sample_id

    return sorted(sorted(range(len(records)), key=rank)[:size])


def save_atomic(value, path):
    atomic_save(value, path)


def write_history(history, path):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(history)
    atomic_text(stream.getvalue(), path)


def capture_rng(device):
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def train_epoch(model, loader, optimizer, scaler, device, vocabulary, args):
    model.train()
    distill = args.mode != "ctc"
    totals = dict(ctc=0.0, kd=0.0, samples=0, valid_frames=0, optimizer_steps=0, amp_skipped_steps=0)
    for step, batch in enumerate(loader, 1):
        frames = batch["frames"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        lengths, target_lengths = batch["input_lengths"], batch["target_lengths"]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            logits = model(frames, lengths)
        # CTC and KL must be computed in FP32, outside mixed-precision autocast.
        ctc = ctc_loss(logits, targets, lengths, target_lengths, blank_id=vocabulary.blank_id)
        kd = logits.new_zeros((), dtype=torch.float32)
        if distill:
            teacher = batch["teacher"].to(device, non_blocking=True)
            if args.mode == "kd":
                kd = standard_kd_loss(logits, teacher, lengths, args.temperature)
            else:
                blank_id = vocabulary.blank_id if args.mode == "kd-blank" else None
                kd = adaptive_kd_loss(logits, teacher, lengths, args.temperature,
                                      epsilon=args.kd_epsilon, blank_id=blank_id)
        loss = ctc + (args.kd_weight * kd if distill else 0)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at batch {step}: {batch['ids']}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # GradScaler may initially overflow and then lower its scale. In AMP,
        # let it skip that optimizer step rather than aborting the whole run.
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip, error_if_nonfinite=not scaler.is_enabled())
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.is_enabled() and scaler.get_scale() < previous_scale
        totals["amp_skipped_steps" if skipped else "optimizer_steps"] += 1
        count = len(batch["ids"])
        valid_frames = int(lengths.sum())
        totals["ctc"] += float(ctc.detach()) * count
        totals["kd"] += float(kd.detach()) * valid_frames
        totals["samples"] += count
        totals["valid_frames"] += valid_frames
        if step == 1 or step % args.log_every == 0 or step == len(loader):
            print(f"  batch {step}/{len(loader)} loss={float(loss.detach()):.5f} "
                  f"ctc={float(ctc.detach()):.5f} kd={float(kd.detach()):.5f} "
                  f"grad_norm={float(norm):.3f}", flush=True)
    if totals["samples"] == 0:
        raise ValueError("Training loader is empty")
    if totals["optimizer_steps"] == 0:
        raise FloatingPointError("No optimizer update in this epoch. AMP overflowed every batch; "
                                 "lower --amp-init-scale or retry without --amp.")
    totals["ctc"] /= totals["samples"]
    totals["kd"] /= totals["valid_frames"]
    totals["loss"] = totals["ctc"] + (args.kd_weight * totals["kd"] if distill else 0)
    return totals


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--mode", choices=("ctc", "kd", *ADAPTIVE_MODES), default="ctc",
                        help="CTC only, standard KD, entropy-aware KD, or entropy+blank-aware KD")
    parser.add_argument("--train-manifest", type=Path, default=Path("data/student_grid/train.jsonl"))
    parser.add_argument("--val-manifest", type=Path, default=Path("data/student_grid/val.jsonl"))
    parser.add_argument("--vocab", type=Path, default=Path("benchmarks/GRID/models/GRID_V_WER4.8/model.json"))
    parser.add_argument("--model-config", type=Path, help="JSON object of StudentModel constructor options")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kd-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kd-epsilon", type=float,
                        help="Adaptive KD weight floor in (0, 1] (adaptive default: 0.1)")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-monitor", choices=["val_loss", "val_wer"], default="val_loss")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--workers", type=int, default=0, help="0 is portable on Windows")
    parser.add_argument("--threads", type=int, default=4, help="CPU operator threads")
    parser.add_argument("--amp", action="store_true", help="FP16 model forward on CUDA; losses remain FP32")
    parser.add_argument("--amp-init-scale", type=float, default=1024.0)
    parser.add_argument("--deterministic", action="store_true", help="Request deterministic PyTorch algorithms")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--resume", type=Path, help="Epoch-boundary checkpoint; repeat original training options")
    parser.add_argument("--smoke-test", action="store_true", help="Label a pipeline-only run, including custom manifests")
    train_selection = parser.add_mutually_exclusive_group()
    train_selection.add_argument("--limit-train", type=int,
                                 help="First N training samples, for pipeline checks only")
    train_selection.add_argument("--train-size", type=int,
                                 help="Deterministic nested training subset for a formal data-scale ablation")
    parser.add_argument("--subset-seed", type=int,
                        help="Subset selection seed for --train-size (default: 42)")
    parser.add_argument("--limit-val", type=int, help="First N validation samples, for pipeline checks only")
    args = parser.parse_args(argv)
    for key in ["epochs", "batch_size", "threads", "save_every", "log_every"]:
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    for key in ["lr", "temperature", "grad_clip", "amp_init_scale"]:
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    for key in ["weight_decay", "kd_weight"]:
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and nonnegative")
    if args.workers < 0 or args.lr_patience < 0:
        parser.error("workers and lr-patience must be nonnegative")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if args.mode != "ctc" and args.kd_weight == 0:
        parser.error("KD mode requires --kd-weight > 0; use --mode ctc for the baseline")
    if args.mode in ADAPTIVE_MODES:
        args.kd_epsilon = 0.1 if args.kd_epsilon is None else args.kd_epsilon
        if not math.isfinite(args.kd_epsilon) or not 0 < args.kd_epsilon <= 1:
            parser.error("--kd-epsilon must be finite and in (0, 1]")
    elif args.kd_epsilon is not None:
        parser.error("--kd-epsilon requires --mode kd-entropy or kd-blank")
    for key in ["limit_train", "limit_val", "train_size"]:
        if getattr(args, key) is not None and getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.train_size is not None:
        args.subset_seed = 42 if args.subset_seed is None else args.subset_seed
        if not 0 <= args.subset_seed < 2**32:
            parser.error("--subset-seed must be in [0, 2**32)")
    elif args.subset_seed is not None:
        parser.error("--subset-seed requires --train-size")
    return args


def main(argv=None):
    args = parse_args(argv)
    root = args.root.resolve()
    output = resolve(args.output_dir, root)
    if args.resume is not None:
        resume_path = resolve(args.resume, root)
        if resume_path.parent != output or resume_path.name != "last.pt":
            raise ValueError("Resume from last.pt in the same --output-dir; copy the whole run directory if relocating")
    with run_lock(output):
        return run(args, root, output)


def run(args, root, output):
    if any(path.name != ".run.lock" for path in output.iterdir()) and args.resume is None:
        raise ValueError(f"Output directory is not empty: {output}. Use a new directory or --resume.")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA devices are supported")
    if device.type == "cuda" and (not torch.cuda.is_available() or
                                  (device.index is not None and device.index >= torch.cuda.device_count())):
        raise ValueError("Requested CUDA device is unavailable")
    if args.amp and device.type != "cuda":
        raise ValueError("--amp currently requires CUDA; omit it for CPU")
    if args.deterministic and device.type == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            raise ValueError("Strict CUDA determinism requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 before Python starts")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = args.deterministic
    torch.use_deterministic_algorithms(args.deterministic)
    if args.deterministic and device.type == "cuda":
        print("Strict deterministic mode: CTC loss/backward uses CPU with GPU autograd transfer.", flush=True)

    vocabulary = CharacterVocabulary.from_file(resolve(args.vocab, root))
    model_config = {}
    if args.model_config is not None:
        model_config = json.loads(resolve(args.model_config, root).read_text(encoding="utf-8"))
    if model_config.get("vocab_size", vocabulary.size) != vocabulary.size:
        raise ValueError("Model vocabulary size must match the teacher vocabulary")
    model_config["vocab_size"] = vocabulary.size
    model = StudentModel(**model_config).to(device)
    model_config = model.get_config()

    train_path, val_path = resolve(args.train_manifest, root), resolve(args.val_manifest, root)
    manifest_paths = {"train_manifest_sha256": train_path, "val_manifest_sha256": val_path}
    manifest_bytes = {key: path.read_bytes() for key, path in manifest_paths.items()}
    train_set = GridDataset(train_path, root, vocabulary, load_teacher=args.mode != "ctc")
    val_set = GridDataset(val_path, root, vocabulary, load_teacher=False)
    train_ids = {record["id"] for record in train_set.records}
    val_ids = {record["id"] for record in val_set.records}
    if train_ids & val_ids:
        raise ValueError("Training and validation sample IDs overlap")
    train_speakers = {key.split("/")[0] for key in train_ids}
    val_speakers = {key.split("/")[0] for key in val_ids}
    if train_speakers & val_speakers:
        raise ValueError("Speaker-independent training requires disjoint train/validation speakers")
    train_files = {train_set.resolve(record["roi"]).resolve() for record in train_set.records}
    val_files = {val_set.resolve(record["roi"]).resolve() for record in val_set.records}
    if train_files & val_files:
        raise ValueError("Training and validation refer to the same ROI files")
    train_indices = select_training_indices(train_set.records, args.train_size, args.subset_seed)
    print("Fingerprinting training/validation data contents...", flush=True)
    train_fingerprint = train_set.fingerprint(indices=train_indices)
    train_roi_fingerprint = (train_fingerprint if args.mode == "ctc"
                             else train_set.fingerprint(include_teacher=False, indices=train_indices))
    val_fingerprint = val_set.fingerprint()
    for key, path in manifest_paths.items():
        if path.read_bytes() != manifest_bytes[key]:
            raise ValueError(f"Manifest changed while checking training data: {path}")
    if train_indices is not None:
        train_set = Subset(train_set, train_indices)
    elif args.limit_train is not None:
        train_set = Subset(train_set, range(min(args.limit_train, len(train_set))))
    if args.limit_val is not None:
        val_set = Subset(val_set, range(min(args.limit_val, len(val_set))))
    generator = torch.Generator().manual_seed(args.seed)
    # A separate validation generator prevents validation workers from consuming
    # the global dropout RNG. No stochastic data augmentation is used.
    val_generator = torch.Generator().manual_seed(args.seed + 1)
    common = dict(batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate_grid,
                  pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_set, shuffle=False, generator=val_generator, **common)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                       patience=args.lr_patience, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp, init_scale=args.amp_init_scale)
    implementation = hashlib.sha256()
    for relative in ["train_student.py", "student/data.py", "student/model.py", "student/losses.py", "student/evaluation.py"]:
        implementation.update(relative.encode())
        implementation.update((ROOT / relative).read_bytes().replace(b"\r\n", b"\n"))
    runtime_info = dict(python=sys.version.split()[0], torch=str(torch.__version__), numpy=str(np.__version__),
                        device_type=device.type, gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        threads=args.threads, cuda=torch.version.cuda if device.type == "cuda" else None,
                        cudnn=torch.backends.cudnn.version() if device.type == "cuda" else None,
                        platform=platform.platform())
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({key: hashlib.sha256(data).hexdigest() for key, data in manifest_bytes.items()})
    config.update(root=str(root), output_dir=str(output), device=str(device),
                  model_config=model_config, vocab_tokens=vocabulary.tokens,
                  train_samples=len(train_set), val_samples=len(val_set),
                  parameters=model.count_parameters(), torch_version=str(torch.__version__),
                  numpy_version=str(np.__version__), decoder="ctc_greedy", metric_unit="ratio",
                  preprocessing_version=PREPROCESSING_VERSION, train_data_fingerprint=train_fingerprint,
                  train_roi_fingerprint=train_roi_fingerprint, val_data_fingerprint=val_fingerprint,
                  implementation_sha256=implementation.hexdigest(),
                  runtime_info=runtime_info,
                  smoke_test=args.smoke_test or args.limit_train is not None or args.limit_val is not None)
    start_epoch, best_score, history = 1, (float("inf"),)*3, []
    best_epoch, best_digest, run_id = None, None, uuid.uuid4().hex
    if args.resume is not None:
        resume_path = resolve(args.resume, root)
        if resume_path.parent != output or resume_path.name != "last.pt":
            raise ValueError("Resume from last.pt in the same --output-dir; copy the whole run directory if relocating")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 2:
            raise ValueError("This trainer requires a format-version 2 checkpoint; earlier smoke checkpoints are incompatible")
        existing_config = output / "config.json"
        if existing_config.is_file():
            saved_config = json.loads(existing_config.read_text(encoding="utf-8"))
            if saved_config.get("run_id") != checkpoint["run_id"]:
                raise ValueError("Checkpoint run ID differs from the output directory's existing experiment")
        immutable = ["mode", "batch_size", "lr", "weight_decay", "kd_weight", "temperature", "kd_epsilon",
                     "grad_clip", "lr_patience", "lr_monitor", "seed", "amp", "amp_init_scale", "deterministic", "workers",
                     "limit_train", "limit_val", "train_size", "subset_seed", "model_config", "vocab_tokens",
                     "train_data_fingerprint", "val_data_fingerprint",
                     "implementation_sha256", "preprocessing_version", "smoke_test", "train_roi_fingerprint"]
        previous_config = checkpoint["config"]
        previous_implementation = previous_config.get("implementation_sha256")
        if (previous_implementation == PRE_PORTABLE_IMPLEMENTATION_SHA256
                or (args.mode in ("ctc", "kd") and previous_implementation == MIDTERM_IMPLEMENTATION_SHA256)):
            immutable.remove("implementation_sha256")
        changed = [key for key in immutable if previous_config.get(key) != config.get(key)]
        changed.extend(key for key, data in manifest_bytes.items()
                       if previous_config.get(key) not in manifest_digests(data))
        if changed:
            raise ValueError(f"Resume options differ for {changed}; repeat original options")
        if model_state_sha256(checkpoint["model_state"]) != checkpoint["model_sha256"]:
            raise ValueError("Resume checkpoint model fingerprint mismatch")
        previous_runtime = checkpoint["config"].get("runtime_info", {})
        runtime_changes = [key for key in runtime_info if runtime_info[key] != previous_runtime.get(key)]
        config["resume_runtime_changes"] = runtime_changes
        if runtime_changes:
            warnings.warn(f"Resume runtime changed in {runtime_changes}; bitwise trajectory equivalence is not guaranteed")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        generator.set_state(checkpoint["loader_generator_state"])
        val_generator.set_state(checkpoint["val_generator_state"])
        restore_rng(checkpoint["rng_state"], device)
        start_epoch = int(checkpoint["epoch"]) + 1
        if args.epochs < start_epoch - 1:
            raise ValueError(f"Checkpoint already reached epoch {start_epoch - 1}; --epochs cannot be lower")
        best_score = tuple(checkpoint["best_score"])
        best_epoch = checkpoint["best_epoch"]
        best_digest = checkpoint["best_model_sha256"]
        run_id = checkpoint["run_id"]
        history = checkpoint["history"]
        if not history or int(history[-1]["epoch"]) != start_epoch - 1:
            raise ValueError("Checkpoint history does not match its epoch")
        best_path = output / "best.pt"
        try:
            stored_best = torch.load(best_path, map_location="cpu", weights_only=True) if best_path.is_file() else None
        except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError):
            stored_best = None
        try:
            valid_best = (isinstance(stored_best, dict) and stored_best.get("run_id") == run_id
                          and stored_best.get("epoch") == best_epoch
                          and stored_best.get("model_sha256") == best_digest
                          and model_state_sha256(stored_best["model_state"]) == best_digest)
        except (KeyError, TypeError, ValueError, RuntimeError):
            valid_best = False
        if not valid_best:
            if checkpoint["epoch"] != best_epoch or checkpoint["model_sha256"] != best_digest:
                raise ValueError("Historical best.pt is missing or inconsistent; restore the original best model before resuming")
            save_atomic(checkpoint, best_path)
    history_path = output / "history.csv"
    summary_path = output / "summary.json"
    config["run_id"] = run_id
    epoch = start_epoch - 1
    atomic_json(dict(state="running", run_id=run_id, pid=os.getpid(), next_epoch=start_epoch), output / "status.json")
    try:
        if args.resume is not None:
            # last.pt is authoritative. Recover outputs that may have been missed
            # if the process stopped after saving last.pt.
            summary_path.unlink(missing_ok=True)
            write_history(history, history_path)
            snapshot = output / f"epoch_{epoch:03d}.pt"
            if epoch % checkpoint["config"]["save_every"] == 0 and not snapshot.exists():
                save_atomic(checkpoint, snapshot)
        atomic_json(config, output / "config.json")
        print(f"Mode={args.mode} device={device} parameters={model.count_parameters():,} "
              f"train={len(train_set)} val={len(val_set)} AMP={args.amp}", flush=True)
        if config["smoke_test"]:
            print("PIPELINE CHECK ONLY: limited data; these are not reportable experiment results.", flush=True)

        for epoch in range(start_epoch, args.epochs + 1):
            print(f"Epoch {epoch}/{args.epochs}", flush=True)
            started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            learning_rate = optimizer.param_groups[0]["lr"]
            train = train_epoch(model, train_loader, optimizer, scaler, device, vocabulary, args)
            metrics = evaluate(model, val_loader, device, vocabulary)
            scheduler.step(metrics["loss"] if args.lr_monitor == "val_loss" else metrics["wer"])
            score = (metrics["wer"], metrics["cer"], metrics["loss"])
            improved = score < best_score
            state_digest = model_state_sha256(model.state_dict())
            if improved:
                best_score, best_epoch, best_digest = score, epoch, state_digest
            record = dict(epoch=epoch, train_loss=train["loss"], train_ctc=train["ctc"], train_kd=train["kd"],
                          val_loss=metrics["loss"], val_wer=metrics["wer"], val_cer=metrics["cer"],
                          lr=learning_rate, next_lr=optimizer.param_groups[0]["lr"],
                          seconds=time.perf_counter() - started,
                          peak_cuda_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0,
                          train_samples=train["samples"], val_samples=metrics["samples"],
                          optimizer_steps=train["optimizer_steps"], amp_skipped_steps=train["amp_skipped_steps"])
            history.append(record)
            checkpoint = dict(format_version=2, epoch=epoch, run_id=run_id,
                              model_state=model.state_dict(), model_sha256=state_digest,
                              model_config=model_config, vocab_tokens=vocabulary.tokens,
                              preprocessing_version=PREPROCESSING_VERSION,
                              optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
                              scaler_state=scaler.state_dict(), config=config, best_val_wer=best_score[0],
                              best_score=list(best_score), best_epoch=best_epoch, best_model_sha256=best_digest,
                              validation=metrics, history=history, rng_state=capture_rng(device),
                              loader_generator_state=generator.get_state(), val_generator_state=val_generator.get_state())
            save_atomic(checkpoint, output / "last.pt")
            if improved:
                save_atomic(checkpoint, output / "best.pt")
            if epoch % args.save_every == 0:
                save_atomic(checkpoint, output / f"epoch_{epoch:03d}.pt")
            write_history(history, history_path)
            atomic_json(dict(state="running", run_id=run_id, pid=os.getpid(), last_completed_epoch=epoch), output / "status.json")
            print(f"  train={train['loss']:.5f} val_ctc={metrics['loss']:.5f} "
                  f"WER={metrics['wer']:.2%} CER={metrics['cer']:.2%} "
                  f"time={record['seconds']:.1f}s peak={record['peak_cuda_mib']:.0f}MiB "
                  f"best={improved}", flush=True)
        summary = dict(last_epoch=args.epochs, best_val_wer=best_score[0], best_val_cer=best_score[1],
                       best_val_loss=best_score[2], best_epoch=best_epoch, run_id=run_id,
                       best_model_sha256=best_digest, best_checkpoint=str(output / "best.pt"),
                       smoke_test=config["smoke_test"])
        atomic_json(summary, summary_path)
        atomic_json(dict(state="completed", run_id=run_id, last_completed_epoch=args.epochs), output / "status.json")
    except BaseException as error:
        try:
            atomic_json(dict(state="failed", run_id=run_id, epoch=epoch,
                             error_type=type(error).__name__, error=str(error),
                             traceback=traceback.format_exc()), output / "status.json")
        except OSError:
            # A full disk can also prevent the failure report; retain the
            # original training/save exception rather than hiding its cause.
            pass
        raise
    return summary


if __name__ == "__main__":
    main()
