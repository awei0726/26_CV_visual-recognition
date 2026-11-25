"""Export the visual Student and calibrate an INT8 ONNX Runtime model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory

import numpy as np
import torch

from student.checkpoint import atomic_json, model_state_sha256, run_lock
from student.data import (
    CharacterVocabulary,
    GridDataset,
    INPUT_SIZE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    PREPROCESSING_VERSION,
)
from student.model import StudentModel
from student.splits import GRID_TRAIN_MANIFEST


TRAIN_MANIFEST = GRID_TRAIN_MANIFEST
ONNX_OPSET = 17
QUANTIZED_OPERATORS = ("Conv", "Gemm", "MatMul")
CPU_PROVIDER = "CPUExecutionProvider"


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_student(path: Path):
    with path.open("rb") as stream:
        checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    required = {"model_state", "model_config", "vocab_tokens"}
    if not isinstance(checkpoint, dict) or not required <= checkpoint.keys():
        raise ValueError(f"Checkpoint must contain {sorted(required)}")
    vocabulary = CharacterVocabulary(checkpoint["vocab_tokens"])
    training = checkpoint.get("config", {})
    if not isinstance(training, dict):
        raise ValueError("Checkpoint config must be a dictionary")
    preprocessing = checkpoint.get("preprocessing_version", training.get("preprocessing_version"))
    if preprocessing not in (None, PREPROCESSING_VERSION):
        raise ValueError("Checkpoint preprocessing differs from the deployment data loader")
    model = StudentModel(**checkpoint["model_config"]).float()
    if model.get_config()["vocab_size"] != vocabulary.size:
        raise ValueError("Checkpoint model output width does not match its vocabulary")
    state_sha = model_state_sha256(checkpoint["model_state"])
    if checkpoint.get("model_sha256", state_sha) != state_sha:
        raise ValueError("Checkpoint model fingerprint does not match its stored weights")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    provenance = {
        "path": str(path), "sha256": checkpoint_sha, "model_sha256": state_sha,
        "epoch": checkpoint.get("epoch"), "run_id": checkpoint.get("run_id"),
        "mode": training.get("mode"), "smoke_test": training.get("smoke_test"),
        "train_manifest_sha256": training.get("train_manifest_sha256"),
    }
    return model.eval(), vocabulary, provenance


def _calibration_dataset(root: Path, manifest: Path, vocabulary: CharacterVocabulary,
                         count: int, seed: int):
    manifest = (root / manifest).resolve()
    train = (root / TRAIN_MANIFEST).resolve()
    manifest_sha = _sha256_file(manifest)
    dataset = GridDataset(manifest, root, vocabulary, load_teacher=False, validate=False)
    if manifest != train:
        canonical = GridDataset(train, root, vocabulary, load_teacher=False, validate=False)
        by_id = {row["id"]: row for row in canonical.records}
        for row in dataset.records:
            original = by_id.get(row["id"])
            if (original is None or (row["text"], row["length"]) != (original["text"], original["length"])
                    or dataset.resolve(row["roi"]).resolve() != canonical.resolve(original["roi"]).resolve()):
                raise ValueError(f"Calibration entry {row['id']!r} is not from {TRAIN_MANIFEST}; use training data")
    if count > len(dataset):
        raise ValueError(f"Requested {count} calibration samples, but {manifest} has {len(dataset)}")
    indices = random.Random(seed).sample(range(len(dataset)), count)
    provenance = {
        "manifest": str(manifest), "manifest_sha256": manifest_sha,
        "seed": seed, "samples": count, "method": "MinMax", "format": "QDQ",
        "activation_dtype": "int8", "weight_dtype": "int8", "per_channel_weights": True,
        "sample_ids": [dataset.records[i]["id"] for i in indices],
        "data_sha256": dataset.fingerprint(include_teacher=False, indices=indices),
    }
    return dataset, indices, provenance


def _probe_batches():
    generator = torch.Generator().manual_seed(0)
    for lengths in ((17, 8), (1,), (75, 59, 2)):
        frames = torch.rand((len(lengths), max(lengths), 1, *INPUT_SIZE),
                            generator=generator, dtype=torch.float32)
        frames.sub_(NORMALIZATION_MEAN).div_(NORMALIZATION_STD)
        yield frames, torch.tensor(lengths, dtype=torch.int64)


def _verify_outputs(path, onnx, threads, probes, references, vocabulary, *, fp32):
    from student.onnx_runtime import OnnxStudent

    onnx.checker.check_model(str(path), full_check=True)
    runtime = OnnxStudent(path, threads, vocabulary)
    differences = []
    for (frames, lengths), reference in zip(probes, references, strict=True):
        outputs = runtime.run(frames.numpy(), lengths.numpy())
        valid = np.arange(reference.shape[1])[None, :] < lengths.numpy()[:, None]
        if np.any(outputs[~valid] != 0):
            raise ValueError(f"{path.name} produced nonzero logits in padded frames")
        if fp32 and not np.allclose(outputs, reference, rtol=1e-4, atol=2e-4):
            difference = float(np.abs(outputs - reference).max())
            raise ValueError(f"FP32 ONNX and PyTorch outputs differ beyond tolerance (max {difference:.6g})")
        differences.append(float(np.abs(outputs[valid] - reference[valid]).max()))
    return {"batches": len(differences), "max_abs_logit_difference_vs_pytorch": max(differences)}


def _annotate(path, onnx, metadata, precision):
    graph = onnx.load(str(path), load_external_data=False)
    properties = {entry.key: entry.value for entry in graph.metadata_props}
    onnx.helper.set_model_props(graph, {**properties, **metadata, "precision": precision})
    onnx.save_model(graph, str(path))
    return graph


def _int8_operator_counts(graph, onnx):
    tensors = {item.name: item for item in graph.graph.initializer}
    unpacked = {node.output[0]: node.input[0] for node in graph.graph.node if node.op_type == "DequantizeLinear"}
    counts = dict.fromkeys(QUANTIZED_OPERATORS, 0)
    for node in graph.graph.node:
        if node.op_type in counts and len(node.input) >= 2:
            weights = tensors.get(unpacked.get(node.input[1]))
            if weights is not None and weights.data_type == onnx.TensorProto.INT8 and len(weights.dims) >= 2:
                counts[node.op_type] += 1
    if counts["Conv"] == 0:
        raise ValueError("The INT8 export does not contain quantized convolution weights")
    return counts


def _quantize(fp32_path, quantized_path, dataset, indices):
    from onnxruntime.quantization import (
        CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.rewind()

        def get_next(self):
            index = next(self.pending, None)
            if index is None:
                return None
            sample = dataset[index]
            return {"frames": sample["frames"].numpy()[None, ...],
                    "lengths": np.array([sample["length"]], dtype=np.int64)}

        def rewind(self):
            self.pending = iter(indices)

    prepared = fp32_path.with_name("quantization_input.onnx")
    # Keep portable ONNX Conv/MatMul nodes for the quantizer; let the runtime
    # optimize the final QDQ graph on its target CPU when creating a session.
    quant_pre_process(fp32_path, prepared, skip_optimization=True, skip_symbolic_shape=True)
    quantize_static(prepared, quantized_path, Reader(), quant_format=QuantFormat.QDQ,
                    op_types_to_quantize=list(QUANTIZED_OPERATORS), per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax)


def export_student(checkpoint: Path, output_dir: Path, root: Path, *,
                   fp32_only: bool = False, calibration_manifest: Path = TRAIN_MANIFEST,
                   calibration_samples: int = 128, calibration_seed: int = 42, threads: int = 4) -> dict:
    """Publish checked FP32 and optional QDQ INT8 artifacts, committing the report last."""
    import onnx
    import onnxruntime as ort

    if threads < 1 or calibration_samples < 1:
        raise ValueError("Thread count and calibration sample count must be positive")
    torch.set_num_threads(threads)
    root = root.resolve()
    checkpoint = (root / checkpoint).resolve()
    output_dir = (root / output_dir).resolve()
    with run_lock(output_dir):
        if any(path.name != ".run.lock" for path in output_dir.iterdir()):
            raise ValueError(f"Output directory must be new or empty: {output_dir}")
        model, vocabulary, provenance = _load_student(checkpoint)
        calibration = None
        if not fp32_only:
            dataset, indices, calibration = _calibration_dataset(
                root, calibration_manifest, vocabulary, calibration_samples, calibration_seed,
            )
        metadata = {
            "vocab_tokens": json.dumps(vocabulary.tokens, ensure_ascii=False),
            "preprocessing_version": PREPROCESSING_VERSION,
            "model_sha256": provenance["model_sha256"], "checkpoint_sha256": provenance["sha256"],
            "model_config": json.dumps(model.get_config(), sort_keys=True),
            "parameters": str(model.count_parameters()),
            "training_smoke_test": json.dumps(provenance["smoke_test"]),
        }
        if isinstance(provenance["train_manifest_sha256"], str):
            metadata["training_manifest_sha256"] = provenance["train_manifest_sha256"]
        report = {
            "schema_version": 1, "status": "complete", "checkpoint": provenance,
            "opset": ONNX_OPSET, "model_config": model.get_config(), "parameters": model.count_parameters(),
            "vocab_tokens": vocabulary.tokens, "preprocessing_version": PREPROCESSING_VERSION,
            "input_size": list(INPUT_SIZE), "normalization_mean": NORMALIZATION_MEAN,
            "normalization_std": NORMALIZATION_STD, "calibration": calibration,
            "execution_provider": CPU_PROVIDER, "threads": threads,
            "versions": {"torch": str(torch.__version__), "onnx": onnx.__version__,
                         "onnxruntime": ort.__version__},
            "artifacts": {},
        }
        with TemporaryDirectory(prefix=".export-", dir=output_dir) as work_dir:
            work = Path(work_dir)
            fp32_path = work / "student_fp32.onnx"
            probes = list(_probe_batches())
            if calibration is not None:
                sample = dataset[indices[0]]
                probes.append((sample["frames"].unsqueeze(0), torch.tensor([sample["length"]], dtype=torch.int64)))
            with torch.no_grad():
                references = [model(*batch).numpy() for batch in probes]
                # Legacy tracing supports the valid-frame gather/scatter in
                # StudentModel. Runtime assertions and dynamic-shape checks run below.
                torch.onnx.export(model, probes[0], str(fp32_path), opset_version=ONNX_OPSET,
                                  input_names=["frames", "lengths"], output_names=["logits"], dynamo=False,
                                  dynamic_axes={"frames": {0: "batch", 1: "time"}, "lengths": {0: "batch"},
                                                "logits": {0: "batch", 1: "time"}})
            _annotate(fp32_path, onnx, metadata, "fp32")
            checks = _verify_outputs(fp32_path, onnx, threads, probes, references, vocabulary, fp32=True)
            artifacts = [("fp32", fp32_path, checks)]
            if calibration is not None:
                int8_path = work / "student_int8.onnx"
                _quantize(fp32_path, int8_path, dataset, indices)
                if (_sha256_file(dataset.manifest_path) != calibration["manifest_sha256"]
                        or dataset.fingerprint(include_teacher=False, indices=indices) != calibration["data_sha256"]):
                    raise ValueError("Calibration data changed during export; retry with a stable training split")
                quantized_metadata = {
                    **metadata, "calibration_sample_ids": json.dumps(calibration["sample_ids"], ensure_ascii=False),
                    "calibration_manifest_sha256": calibration["manifest_sha256"],
                }
                graph = _annotate(int8_path, onnx, quantized_metadata, "int8")
                quantized_weights = _int8_operator_counts(graph, onnx)
                checks = _verify_outputs(int8_path, onnx, threads, probes, references, vocabulary, fp32=False)
                checks["int8_weighted_operators"] = quantized_weights
                artifacts.append(("int8", int8_path, checks))
            for precision, path, checks in artifacts:
                report["artifacts"][precision] = {
                    "file": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path), **checks,
                }
            published = []
            try:
                for _, path, _ in artifacts:
                    published.append(path.replace(output_dir / path.name))
                atomic_json(report, output_dir / "export_report.json")
            except BaseException:
                for path in published:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass  # The missing report still marks this export as incomplete.
                raise
    return report
