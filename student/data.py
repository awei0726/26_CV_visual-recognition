"""Torch data adapters for the GRID Student and paired teacher CTC logits."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .grid import (
    CharacterVocabulary,
    GridManifest,
    INPUT_SIZE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    PREPROCESSING_VERSION,
    _positive_integer,
    validate_roi,
)


def normalize_roi(roi: np.ndarray, *, expected_length: int | None = None) -> torch.Tensor:
    """Normalize uint8 [T,64,64] mouth frames exactly as in Student training."""
    validate_roi(roi, expected_length=expected_length)
    return (torch.from_numpy(np.ascontiguousarray(roi)).unsqueeze(1).float()
            .div_(255.0).sub_(NORMALIZATION_MEAN).div_(NORMALIZATION_STD))


class GridDataset(GridManifest, Dataset):
    """Prepare Torch inputs and optionally load matching cached teacher logits."""

    def __getitem__(self, index: int) -> dict:
        item = self._sample(index, normalize_roi, torch.tensor(self._targets[index], dtype=torch.long))
        if self.load_teacher:
            record = self.records[index]
            length = item["length"]
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
            logits = logits.float()
            if not torch.isfinite(logits).all():
                raise ValueError(f"{record['id']}: teacher logits contain NaN or Inf")
            item["teacher"] = logits
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
