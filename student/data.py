"""Load the prepared GRID ROI sequences and paired teacher CTC logits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

INPUT_SIZE = (64, 64)
NORMALIZATION_MEAN = 0.421
NORMALIZATION_STD = 0.165
PREPROCESSING_VERSION = "grid_gray64_div255_mean0.421_std0.165_v1"


def normalize_roi(roi: np.ndarray, *, expected_length: int | None = None) -> torch.Tensor:
    """Normalize uint8 [T,64,64] mouth frames exactly as in Student training."""
    if (not isinstance(roi, np.ndarray) or roi.dtype != np.uint8 or roi.ndim != 3
            or roi.shape[1:] != INPUT_SIZE or not len(roi)
            or (expected_length is not None and len(roi) != expected_length)):
        frames = expected_length if expected_length is not None else "T > 0"
        raise ValueError(f"expected uint8 ROI [{frames}, {INPUT_SIZE[0]}, {INPUT_SIZE[1]}]")
    return (torch.from_numpy(np.ascontiguousarray(roi)).unsqueeze(1).float()
            .div_(255.0).sub_(NORMALIZATION_MEAN).div_(NORMALIZATION_STD))


class CharacterVocabulary:
    """Use the teacher's token ordering without adding CTC target symbols."""

    def __init__(self, tokens: Sequence[str]):
        if isinstance(tokens, (str, bytes)):
            raise ValueError("Vocabulary must be a sequence of token strings")
        self.tokens = list(tokens)
        if (not self.tokens or any(not isinstance(token, str) or not token for token in self.tokens)
                or len(set(self.tokens)) != len(self.tokens)):
            raise ValueError("Vocabulary must contain distinct nonempty token strings")
        self._ids = {token: i for i, token in enumerate(self.tokens)}
        if "<blank>" not in self._ids or "<space>" not in self._ids:
            raise ValueError("Vocabulary requires <blank> and <space>")
        self.blank_id = self._ids["<blank>"]
        self.space_id = self._ids["<space>"]
        self.size = len(self.tokens)

    @classmethod
    def from_file(cls, path: str | Path) -> CharacterVocabulary:
        with Path(path).open(encoding="utf-8-sig") as stream:
            config = json.load(stream)
        args = config if isinstance(config, dict) else config[2]
        return cls(args["char_list"])

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Transcript must be a nonempty string")
        result = []
        for char in text:
            token = "<space>" if char == " " else char
            if token not in self._ids:
                raise ValueError(f"Transcript character is absent from vocabulary: {char!r}")
            result.append(self._ids[token])
        return result

    def decode(self, ids: Sequence[int]) -> str:
        """Convert collapsed CTC token IDs to text; do not collapse repeats here."""
        result = []
        for value in ids:
            index = int(value)
            if not 0 <= index < self.size:
                raise ValueError(f"Token ID outside vocabulary: {index}")
            token = self.tokens[index]
            if token == "<blank>":
                continue
            if token == "<space>":
                result.append(" ")
            else:
                # Keep unexpected <unk>/<eos> predictions visible as errors.
                result.append(token)
        return "".join(result).strip()


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return value


class GridDataset(Dataset):
    """Read a JSONL manifest; large arrays stay on disk until a sample is used.

    ``validate`` checks referenced file existence at construction time. Shape,
    dtype, length, and finite-value checks always run when reading a sample.
    No temporal augmentation is applied because cached teacher frames must align.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        root: str | Path,
        vocabulary: CharacterVocabulary,
        load_teacher: bool = False,
        validate: bool = True,
    ):
        self.root = Path(root).resolve()
        manifest_path = Path(manifest_path)
        self.manifest_path = manifest_path if manifest_path.is_absolute() else self.root / manifest_path
        self.vocabulary = vocabulary
        self.load_teacher = load_teacher
        self.records = []
        self._targets = []
        seen = set()
        with self.manifest_path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    required = {"id", "roi", "text", "length"}
                    if self.load_teacher:
                        required.add("teacher")
                    if not isinstance(record, dict) or not required <= record.keys():
                        raise ValueError(f"Manifest entry requires fields {sorted(required)}")
                    sample_id = record["id"]
                    if not isinstance(sample_id, str) or not sample_id.strip():
                        raise ValueError("Sample ID must be a nonempty string")
                    if sample_id in seen:
                        raise ValueError(f"Duplicate sample ID: {sample_id}")
                    length = _positive_integer(record["length"], "length")
                    targets = self.vocabulary.encode(record["text"])
                    forbidden = {self.vocabulary.blank_id}
                    if "<eos>" in self.vocabulary.tokens:
                        forbidden.add(self.vocabulary.tokens.index("<eos>"))
                    if any(value in forbidden for value in targets):
                        raise ValueError("CTC targets must not contain blank or <eos>")
                    min_length = len(targets) + sum(a == b for a, b in zip(targets, targets[1:]))
                    if min_length > length:
                        raise ValueError(f"CTC needs at least {min_length} frames, manifest has {length}")
                    for key in ["roi"] + (["teacher"] if self.load_teacher else []):
                        if not isinstance(record[key], str) or not record[key]:
                            raise ValueError(f"{key} must be a nonempty path string")
                        expected = Path(*sample_id.split("/"))
                        path = Path(record[key]).with_suffix("")
                        if Path(*path.parts[-len(expected.parts):]) != expected:
                            raise ValueError(f"{key} path does not match sample ID {sample_id!r}")
                        if validate and not self.resolve(record[key]).is_file():
                            raise FileNotFoundError(f"Missing {key} file: {self.resolve(record[key])}")
                    seen.add(sample_id)
                    self.records.append(record)
                    self._targets.append(targets)
                except (ValueError, TypeError, KeyError, FileNotFoundError) as error:
                    message = f"{self.manifest_path}:{line_number}: {error}"
                    if isinstance(error, FileNotFoundError):
                        raise FileNotFoundError(message) from error
                    raise ValueError(message) from error
        if not self.records:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

    def resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __len__(self) -> int:
        return len(self.records)

    def fingerprint(self, include_teacher: bool | None = None,
                    indices: Sequence[int] | None = None) -> str:
        """Hash ordered labels and actual input bytes, independent of root location.

        Recompute on each call: replacing a cache while leaving its manifest
        untouched must invalidate resume. A baseline hashes only its ROI files;
        KD additionally hashes teacher files. The preprocessing contract and
        exact vocabulary ordering are included to detect changed input semantics.
        With indices, hash only the selected records in the supplied order.
        """
        include_teacher = self.load_teacher if include_teacher is None else include_teacher
        if not isinstance(include_teacher, bool):
            raise ValueError("include_teacher must be a bool or None")
        digest = hashlib.sha256()

        def update(value):
            digest.update(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")

        update({"schema": "grid_dataset_fingerprint_v1", "vocab": self.vocabulary.tokens,
                "preprocessing": PREPROCESSING_VERSION, "input_size": INPUT_SIZE,
                "mean": NORMALIZATION_MEAN, "std": NORMALIZATION_STD,
                "teacher_included": include_teacher})
        records = self.records if indices is None else (self.records[index] for index in indices)
        for record in records:
            value = {key: record[key] for key in ("id", "text", "length")}
            for field in ["roi"] + (["teacher"] if include_teacher else []):
                if field not in record:
                    raise ValueError(f"{record['id']}: no {field} path for fingerprint")
                with self.resolve(record[field]).open("rb") as stream:
                    value[field + "_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            update(value)
        return digest.hexdigest()

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        length = record["length"]
        roi_path = self.resolve(record["roi"])
        roi = np.load(roi_path, allow_pickle=False)
        try:
            frames = normalize_roi(roi, expected_length=length)
        except ValueError as error:
            raise ValueError(f"{record['id']}: {error}") from error
        item = {
            "frames": frames,
            "targets": torch.tensor(self._targets[index], dtype=torch.long),
            "length": length,
            "id": record["id"],
            "text": record["text"],
        }
        if self.load_teacher:
            cache_path = self.resolve(record["teacher"])
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            if not isinstance(cache, dict) or not {"logits", "length", "blank_id"} <= cache.keys():
                raise ValueError(f"{cache_path}: teacher cache requires logits, length, blank_id")
            logits = cache["logits"]
            cache_length = _positive_integer(cache["length"], "teacher length")
            cache_blank = cache["blank_id"]
            if (
                not isinstance(logits, torch.Tensor)
                or not logits.is_floating_point()
                or logits.shape != (length, self.vocabulary.size)
                or cache_length != length
                or isinstance(cache_blank, bool)
                or not isinstance(cache_blank, int)
                or cache_blank != self.vocabulary.blank_id
            ):
                raise ValueError(f"{record['id']}: teacher shape, length, blank_id or dtype mismatch")
            if not torch.isfinite(logits).all():
                raise ValueError(f"{record['id']}: teacher logits contain NaN or Inf")
            item["teacher"] = logits.float()
        return item


def collate_grid(samples: list[dict]) -> dict:
    """Pad time only and retain lengths for CTC, KD masking, and decoding."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    input_lengths = torch.tensor([sample["length"] for sample in samples], dtype=torch.long)
    target_lengths = torch.tensor([sample["targets"].numel() for sample in samples], dtype=torch.long)
    batch_size, max_length = len(samples), int(input_lengths.max())
    frames = samples[0]["frames"].new_zeros((batch_size, max_length, 1, *INPUT_SIZE))
    teacher_flags = ["teacher" in sample for sample in samples]
    if any(teacher_flags) != all(teacher_flags):
        raise ValueError("A batch cannot mix samples with and without teacher caches")
    teacher = None
    if all(teacher_flags):
        teacher = samples[0]["teacher"].new_zeros((batch_size, max_length, samples[0]["teacher"].shape[-1]))
    for i, sample in enumerate(samples):
        length = sample["length"]
        if sample["frames"].shape != (length, 1, *INPUT_SIZE):
            raise ValueError(f"{sample['id']}: frame shape does not match its length")
        frames[i, :length] = sample["frames"]
        if teacher is not None:
            if sample["teacher"].shape != (length, teacher.shape[-1]):
                raise ValueError(f"{sample['id']}: teacher shape does not match its length")
            teacher[i, :length] = sample["teacher"]
    result = {
        "frames": frames,
        "targets": torch.cat([sample["targets"] for sample in samples]),
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "ids": [sample["id"] for sample in samples],
        "texts": [sample["text"] for sample in samples],
    }
    if teacher is not None:
        result["teacher"] = teacher
    return result
