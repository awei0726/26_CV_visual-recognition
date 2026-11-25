"""Compare FP32 and INT8 ONNX Students on the same GRID data and CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from time import perf_counter_ns

import numpy as np
import torch

from student.checkpoint import atomic_json, atomic_text, run_lock
from student.data import CharacterVocabulary, GridDataset, PREPROCESSING_VERSION
from student.evaluation import edit_distance
from student.onnx_runtime import OnnxStudent
from student.splits import benchmark_split, GRID_TRAIN_MANIFEST


REPOSITORY_ROOT = Path(__file__).resolve().parent
VOCABULARY_PATH = Path("benchmarks/GRID/models/GRID_V_WER4.8/model.json")


def resolve(path: Path, root: Path) -> Path:
    return (path if path.is_absolute() else root / path).resolve()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def peak_rss(process) -> tuple[int, str]:
    if sys.platform == "win32":
        return process.memory_info().peak_wset, "windows_peak_working_set"
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        import resource

        raw_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(raw_peak) * (1 if sys.platform == "darwin" else 1024), "ru_maxrss"
    raise ValueError("Peak RSS measurement supports Windows, Linux/WSL and macOS.")


def latency_summary(samples: list[float]) -> dict:
    return {
        "samples": len(samples),
        "mean": float(np.mean(samples)),
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
        "p99": float(np.percentile(samples, 99)),
    }


def session_inputs(model: OnnxStudent, sample: dict) -> dict[str, np.ndarray]:
    frames = sample["frames"].unsqueeze(0).numpy()
    lengths = np.array([sample["length"]], dtype=np.int64)
    return model.prepare_inputs(frames, lengths)


def prediction_row(sample: dict, prediction: str) -> dict:
    reference = sample["text"]
    reference_words = reference.split()
    reference_chars = reference.replace(" ", "")
    return {
        "id": sample["id"], "reference": reference, "prediction": prediction,
        "word_errors": edit_distance(prediction.split(), reference_words),
        "word_count": len(reference_words),
        "char_errors": edit_distance(prediction.replace(" ", ""), reference_chars),
        "char_count": len(reference_chars),
    }


def evaluate_variant(plan: dict, precision: str) -> dict:
    """Run one model in a fresh subprocess using only the CPU provider."""
    import psutil

    torch.set_num_threads(1)
    vocabulary = CharacterVocabulary.from_file(plan["vocabulary"])
    manifest_path = Path(plan["manifest"])
    manifest_sha = sha256_file(manifest_path)
    dataset = GridDataset(manifest_path, plan["root"], vocabulary, load_teacher=False)
    evaluation_split = benchmark_split(dataset)
    sample_count = len(dataset) if plan["limit"] is None else plan["limit"]
    if sample_count > len(dataset):
        raise ValueError(f"Requested {sample_count} samples but the manifest has {len(dataset)}.")
    selected = range(sample_count)
    measured = np.linspace(0, sample_count - 1, min(plan["latency_samples"], sample_count),
                           dtype=np.int64).tolist()
    timed = set(measured)
    fingerprint = dataset.fingerprint(include_teacher=False, indices=selected)
    path = Path(plan[precision])
    model_hash = sha256_file(path)
    process = psutil.Process()
    rss_before_session = process.memory_info().rss
    peak_before_session, memory_source = peak_rss(process)
    model = OnnxStudent(path, threads=plan["threads"], vocabulary=vocabulary)
    rss_after_session = process.memory_info().rss
    if model.metadata["precision"] != precision:
        raise ValueError(f"Expected {precision} metadata in {path}.")
    if precision == "int8":
        try:
            calibration_ids = json.loads(model.metadata["calibration_sample_ids"])
        except (KeyError, ValueError) as error:
            raise ValueError("INT8 artifact needs calibration_sample_ids metadata.") from error
        if (not isinstance(calibration_ids, list) or not calibration_ids
                or any(not isinstance(value, str) or not value for value in calibration_ids)
                or len(set(calibration_ids)) != len(calibration_ids)):
            raise ValueError("Invalid INT8 calibration sample IDs.")
        overlap = set(calibration_ids).intersection(dataset.records[index]["id"] for index in selected)
        if overlap:
            raise ValueError(f"INT8 calibration overlaps evaluation: {len(overlap)} sample ID(s).")
    for offset in range(plan["warmup"]):
        sample = dataset[measured[offset % len(measured)]]
        inputs = session_inputs(model, sample)
        logits = model.session.run(["logits"], inputs)[0]
        model.check_output(logits, inputs)

    results, times = [], []
    for index in selected:
        sample = dataset[index]
        inputs = session_inputs(model, sample)
        time_this_sample = index in timed
        if time_this_sample:
            start = perf_counter_ns()
        outputs = model.session.run(["logits"], inputs)
        if time_this_sample:
            times.append((perf_counter_ns() - start) / 1_000_000)
        logits = outputs[0]
        model.check_output(logits, inputs)
        results.append(prediction_row(sample, model.decode(logits, inputs["lengths"])[0]))
    peak, _ = peak_rss(process)
    if sha256_file(path) != model_hash:
        raise ValueError(f"ONNX model changed during benchmarking: {path}")
    if (sha256_file(manifest_path) != manifest_sha
            or dataset.fingerprint(include_teacher=False, indices=selected) != fingerprint):
        raise ValueError("Evaluation manifest or ROI files changed during benchmarking.")

    counts = {key: sum(row[key] for row in results)
              for key in ("word_errors", "word_count", "char_errors", "char_count")}
    if not counts["word_count"] or not counts["char_count"]:
        raise ValueError("Evaluation references must contain words and non-space characters.")
    return {
        "precision": precision,
        "model": str(path),
        "onnx_sha256": model_hash,
        "size_bytes": path.stat().st_size,
        "model_metadata": model.metadata,
        "dataset_fingerprint": fingerprint,
        "manifest_sha256": manifest_sha,
        "evaluation_split": evaluation_split,
        "samples": sample_count,
        **counts,
        "wer": counts["word_errors"] / counts["word_count"],
        "cer": counts["char_errors"] / counts["char_count"],
        "latency_ms": latency_summary(times),
        "latency_measurements": [
            {"id": results[index]["id"], "ms": measured_ms}
            for index, measured_ms in zip(measured, times, strict=True)
        ],
        "memory": {
            "source": memory_source,
            "rss_before_session_bytes": rss_before_session,
            "peak_rss_before_session_bytes": peak_before_session,
            "rss_after_session_bytes": rss_after_session,
            "peak_rss_bytes": peak,
        },
        "predictions": results,
    }


def run_workers(plan: dict, scratch: Path) -> tuple[dict, dict]:
    plan_path = scratch / "plan.json"
    atomic_json(plan, plan_path)
    reports = []
    for precision in ("fp32", "int8"):
        result_path = scratch / f"{precision}.json"
        worker = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "_worker", str(plan_path), precision, str(result_path)],
            capture_output=True, encoding="utf-8", errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        if worker.returncode:
            detail = worker.stderr.strip() or worker.stdout.strip()
            raise RuntimeError(f"{precision} benchmark worker exited with status {worker.returncode}.\n{detail}")
        reports.append(json.loads(result_path.read_text(encoding="utf-8")))
    return reports[0], reports[1]


def pair_predictions(fp32: dict, int8: dict) -> list[dict]:
    if any(fp32[key] != int8[key] for key in (
        "dataset_fingerprint", "manifest_sha256", "evaluation_split", "samples", "word_count", "char_count",
    )):
        raise ValueError("The two benchmarks did not evaluate the same GRID data.")
    for key in ("model_sha256", "checkpoint_sha256", "vocab_tokens", "preprocessing_version",
                "model_config", "training_smoke_test", "parameters"):
        if fp32["model_metadata"].get(key) != int8["model_metadata"].get(key):
            raise ValueError(f"FP32 and INT8 artifacts disagree on {key}. Export them as one pair.")
    if [row["id"] for row in fp32["latency_measurements"]] != [
        row["id"] for row in int8["latency_measurements"]
    ]:
        raise ValueError("FP32 and INT8 timed different examples.")
    paired = []
    for reference, quantized in zip(fp32["predictions"], int8["predictions"], strict=True):
        if any(reference[key] != quantized[key] for key in ("id", "reference", "word_count", "char_count")):
            raise ValueError("FP32 and INT8 predictions are not aligned by sample.")
        paired.append({
            **{key: reference[key] for key in ("id", "reference", "word_count", "char_count")},
            "fp32": {key: reference[key] for key in ("prediction", "word_errors", "char_errors")},
            "int8": {key: quantized[key] for key in ("prediction", "word_errors", "char_errors")},
        })
    return paired


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--int8", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="Use a shared val or test JSONL.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty directory.")
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--vocab", type=Path, default=VOCABULARY_PATH)
    parser.add_argument("--threads", type=int, default=1, help="ONNX Runtime intra-op threads.")
    parser.add_argument("--warmup", type=int, default=10, help="Untimed model calls per variant.")
    parser.add_argument("--latency-samples", type=int, default=100,
                        help="Number of evenly spaced, timed samples.")
    parser.add_argument("--limit", type=int, help="Evaluate the first N entries as a smoke test.")
    args = parser.parse_args(argv)
    if args.threads < 1 or args.latency_samples < 1 or args.warmup < 0:
        parser.error("threads and latency-samples must be positive; warmup cannot be negative.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    root = args.root.resolve()
    manifest = resolve(args.manifest, root)
    if manifest == (root / GRID_TRAIN_MANIFEST).resolve():
        parser.error("Use a held-out validation or test manifest for the benchmark.")
    plan = {
        "root": str(root), "manifest": str(manifest), "vocabulary": str(resolve(args.vocab, root)),
        "fp32": str(resolve(args.fp32, root)), "int8": str(resolve(args.int8, root)),
        "threads": args.threads, "warmup": args.warmup,
        "latency_samples": args.latency_samples, "limit": args.limit,
    }
    if plan["fp32"] == plan["int8"]:
        parser.error("The FP32 and INT8 ONNX files must be different.")
    import onnxruntime as ort

    output = resolve(args.output_dir, root)
    predictions_path = output / "predictions.jsonl"
    benchmark_path = output / "metrics.json"
    with run_lock(output):
        if any(path.name != ".run.lock" for path in output.iterdir()):
            parser.error(f"Output directory must be new or empty: {output}")
        with tempfile.TemporaryDirectory(prefix=".benchmark-", dir=output) as temporary:
            fp32, int8 = run_workers(plan, Path(temporary))
            predictions = pair_predictions(fp32, int8)
            del fp32["predictions"], int8["predictions"]
            checkpoint_smoke_test = json.loads(fp32["model_metadata"].get("training_smoke_test", "null"))
            if checkpoint_smoke_test is not None and not isinstance(checkpoint_smoke_test, bool):
                raise ValueError("Unexpected checkpoint smoke-test metadata.")
            parameters = int(fp32["model_metadata"]["parameters"])
            if parameters < 1:
                raise ValueError("Expected a positive parameter count in ONNX metadata.")
            evaluation_split = fp32["evaluation_split"]
            report = {
                "schema_version": 1,
                "evaluation_status": "complete",
                "manifest": str(manifest), "manifest_sha256": fp32["manifest_sha256"],
                "dataset_fingerprint": fp32["dataset_fingerprint"],
                "samples": fp32["samples"],
                "evaluation_split": evaluation_split,
                "smoke_test": (args.limit is not None or evaluation_split == "custom"
                               or checkpoint_smoke_test is not False),
                "checkpoint_smoke_test": checkpoint_smoke_test,
                "parameters": parameters,
                "preprocessing_version": PREPROCESSING_VERSION,
                "decoder": "ctc_greedy",
                "metric_unit": "ratio",
                "cer_excludes_spaces": True,
                "protocol": {
                    "provider": "CPUExecutionProvider",
                    "inference_batch_size": 1,
                    "intra_op_threads": args.threads,
                    "inter_op_threads": 1,
                    "execution_mode": "sequential",
                    "ort_graph_optimization": "all",
                    "warmup_calls_per_model": args.warmup,
                    "evaluation_order": ["fp32", "int8"],
                    "latency_scope": "One ONNX Runtime session.run per timed video, excluding file I/O, ROI normalization and decoding.",
                    "peak_rss_scope": "Lifetime RSS high-water mark of each isolated benchmark worker, including imports, model load and inference. The pre-session high-water mark is recorded separately.",
                    "python_version": platform.python_version(),
                    "onnxruntime_version": ort.__version__,
                    "platform": platform.platform(),
                    "processor": platform.processor() or platform.machine(),
                    "logical_cpus": os.cpu_count(),
                },
                "models": {"fp32": fp32, "int8": int8},
                "comparison": {
                    "int8_minus_fp32_wer": int8["wer"] - fp32["wer"],
                    "int8_minus_fp32_cer": int8["cer"] - fp32["cer"],
                    "int8_over_fp32_size_ratio": int8["size_bytes"] / fp32["size_bytes"],
                    "fp32_over_int8_p50_latency_ratio": fp32["latency_ms"]["p50"] / int8["latency_ms"]["p50"],
                    "fp32_over_int8_p95_latency_ratio": fp32["latency_ms"]["p95"] / int8["latency_ms"]["p95"],
                    "int8_minus_fp32_peak_rss_bytes": (
                        int8["memory"]["peak_rss_bytes"] - fp32["memory"]["peak_rss_bytes"]
                    ),
                },
            }
            atomic_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
                        predictions_path)
            try:
                report["predictions_sha256"] = sha256_file(predictions_path)
                atomic_json(report, benchmark_path)
            except (Exception, KeyboardInterrupt):
                try:
                    predictions_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise

    for result in (fp32, int8):
        print(f"{result['precision'].upper()}  WER={result['wer']:.2%}  CER={result['cer']:.2%}  "
              f"P50={result['latency_ms']['p50']:.2f} ms  P95={result['latency_ms']['p95']:.2f} ms  "
              f"Size={result['size_bytes'] / 2**20:.2f} MiB  "
              f"Peak RSS={result['memory']['peak_rss_bytes'] / 2**20:.1f} MiB")
    print(f"Results: {benchmark_path}")
    return report


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "_worker":
        worker_plan = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        atomic_json(evaluate_variant(worker_plan, sys.argv[3]), Path(sys.argv[4]))
    else:
        main()
