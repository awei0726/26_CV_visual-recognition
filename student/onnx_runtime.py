"""Run the exported GRID Student on the ONNX Runtime CPU provider."""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import torch

from .data import CharacterVocabulary, INPUT_SIZE, PREPROCESSING_VERSION
from .evaluation import greedy_decode


class OnnxStudent:
    """Validate the exported model's input, output, and vocabulary contract."""

    def __init__(
        self,
        model_path: str | Path,
        threads: int = 1,
        vocabulary: CharacterVocabulary | None = None,
    ) -> None:
        import onnxruntime as ort

        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("CPU inference requires a positive thread count.")
        self.path = Path(model_path).resolve()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.path), options, providers=["CPUExecutionProvider"],
        )
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        self.vocabulary = self._read_metadata(vocabulary)
        self._validate_signature()

    def _read_metadata(self, expected: CharacterVocabulary | None) -> CharacterVocabulary:
        if self.metadata.get("precision") not in {"fp32", "int8"}:
            raise ValueError("ONNX Student needs fp32 or int8 precision metadata.")
        if self.metadata.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ValueError("ONNX Student preprocessing differs from the GRID data loader.")
        if not re.fullmatch("[0-9a-f]{64}", self.metadata.get("model_sha256", "")):
            raise ValueError("ONNX Student needs the checkpoint model_sha256 metadata.")
        try:
            tokens = json.loads(self.metadata["vocab_tokens"])
            if not isinstance(tokens, list):
                raise ValueError("Vocabulary metadata must be an ordered list.")
            vocabulary = CharacterVocabulary(tokens)
        except (KeyError, ValueError) as error:
            raise ValueError("Missing or invalid ONNX Student vocabulary metadata.") from error
        if expected is not None and vocabulary.tokens != expected.tokens:
            raise ValueError("ONNX Student vocabulary order differs from the selected vocabulary.")
        return vocabulary

    def _validate_signature(self) -> None:
        inputs = {value.name: value for value in self.session.get_inputs()}
        outputs = {value.name: value for value in self.session.get_outputs()}
        if inputs.keys() != {"frames", "lengths"} or outputs.keys() != {"logits"}:
            raise ValueError("Expected ONNX graph inputs frames/lengths and output logits.")
        frames, lengths, logits = inputs["frames"], inputs["lengths"], outputs["logits"]
        if (frames.type != "tensor(float)" or lengths.type != "tensor(int64)"
                or logits.type != "tensor(float)"):
            raise ValueError("ONNX inputs/outputs must be float32 frames, int64 lengths, float32 logits.")
        for value, trailing_dims in (
            (frames, (1, *INPUT_SIZE)), (logits, (self.vocabulary.size,)),
        ):
            if len(value.shape) != 2 + len(trailing_dims) or any(
                isinstance(dimension, int) for dimension in value.shape[:2]
            ):
                raise ValueError(f"ONNX {value.name} needs dynamic batch and time axes.")
            for dimension, expected in zip(value.shape[2:], trailing_dims):
                if isinstance(dimension, int) and dimension != expected:
                    raise ValueError(f"ONNX {value.name} has an unexpected fixed dimension.")
        if len(lengths.shape) != 1 or isinstance(lengths.shape[0], int):
            raise ValueError("ONNX lengths needs one dynamic batch axis.")

    @staticmethod
    def prepare_inputs(frames: np.ndarray, lengths: np.ndarray) -> dict[str, np.ndarray]:
        """Prepare normalized frame and valid-length arrays for the runtime."""
        frames, lengths = np.asarray(frames), np.asarray(lengths)
        if (frames.dtype != np.float32 or frames.ndim != 5
                or frames.shape[2:] != (1, *INPUT_SIZE) or min(frames.shape[:2]) < 1):
            raise ValueError("Expected normalized float32 frames [B, T, 1, 64, 64].")
        if lengths.dtype != np.int64 or lengths.shape != (frames.shape[0],):
            raise ValueError("Expected int64 lengths [B].")
        if np.any(lengths < 1) or np.any(lengths > frames.shape[1]):
            raise ValueError("Every input length must be between 1 and padded T.")
        valid_frames = frames
        if np.any(lengths != frames.shape[1]):
            valid = np.arange(frames.shape[1]) < lengths[:, None]
            valid_frames = frames[valid]
        if not np.isfinite(valid_frames).all():
            raise ValueError("Valid input frames contain NaN or Inf.")
        return {"frames": np.ascontiguousarray(frames), "lengths": np.ascontiguousarray(lengths)}

    def check_output(self, logits: np.ndarray, inputs: dict[str, np.ndarray]) -> None:
        shape = (*inputs["frames"].shape[:2], self.vocabulary.size)
        if not isinstance(logits, np.ndarray) or logits.shape != shape or logits.dtype != np.float32:
            raise ValueError(f"Expected float32 ONNX logits with shape {shape}.")
        valid = np.arange(shape[1]) < inputs["lengths"][:, None]
        if not np.isfinite(logits[valid]).all():
            raise ValueError("Valid ONNX Student logits contain NaN or Inf.")

    def run(self, frames: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Return frame logits, including positions padded to the input length."""
        inputs = self.prepare_inputs(frames, lengths)
        logits = self.session.run(["logits"], inputs)[0]
        self.check_output(logits, inputs)
        return logits

    def decode(self, logits: np.ndarray, lengths: np.ndarray) -> list[str]:
        return greedy_decode(torch.from_numpy(logits), lengths, self.vocabulary)

    def predict(self, frames: np.ndarray, lengths: np.ndarray) -> list[str]:
        return self.decode(self.run(frames, lengths), lengths)
