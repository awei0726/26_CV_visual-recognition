"""Verify the provenance of GRID manifests used for evaluation."""

import hashlib
from pathlib import Path

from .data import GridDataset


GRID_MANIFEST_DIR = Path("data/student_grid")
GRID_TRAIN_MANIFEST = GRID_MANIFEST_DIR / "train.jsonl"
GRID_HELD_OUT_SPLITS = {
    "val": (2000, {"s9", "s10"}),
    "test": (4000, {"s11", "s12", "s13", "s14"}),
}


def manifest_digests(data: bytes) -> set[str]:
    """Allow a Git checkout to translate LF to CRLF between environments."""
    lf = data.replace(b"\r\n", b"\n")
    return {hashlib.sha256(raw).hexdigest()
            for raw in (data, lf, lf.replace(b"\n", b"\r\n"))}


def _reference(dataset: GridDataset, path: Path) -> GridDataset:
    return GridDataset(path, dataset.root, dataset.vocabulary, validate=False)


def _overlap(dataset: GridDataset, reference: GridDataset) -> int:
    reference_ids = {row["id"] for row in reference.records}
    reference_rois = {reference.resolve(row["roi"]).resolve() for row in reference.records}
    return sum(row["id"] in reference_ids or dataset.resolve(row["roi"]).resolve() in reference_rois
               for row in dataset.records)


def training_manifest_overlap(dataset: GridDataset, training_config: dict) -> int | None:
    """Count overlaps with the checkpoint's original training pool when available."""
    digest = training_config.get("train_manifest_sha256")
    if not isinstance(digest, str):
        return None
    candidates = [dataset.root / GRID_TRAIN_MANIFEST]
    recorded = training_config.get("train_manifest")
    if isinstance(recorded, str) and recorded:
        candidates.insert(0, dataset.resolve(recorded))
    for candidate in dict.fromkeys(path.resolve() for path in candidates):
        if candidate.is_file() and digest in manifest_digests(candidate.read_bytes()):
            return _overlap(dataset, _reference(dataset, candidate))
    return None


def _samples(dataset: GridDataset) -> set[tuple]:
    return {(row["id"], row["text"], row["length"], dataset.resolve(row["roi"]).resolve())
            for row in dataset.records}


def benchmark_split(dataset: GridDataset) -> str:
    """Identify a complete GRID held-out split and reject training-pool leakage."""
    train = dataset.root / GRID_TRAIN_MANIFEST
    if not train.is_file():
        return "custom"
    overlap = _overlap(dataset, _reference(dataset, train))
    if overlap:
        raise ValueError(f"Benchmark manifest overlaps the GRID training split: {overlap} sample(s).")

    speakers = {row["id"].split("/", 1)[0] for row in dataset.records}
    for split, (expected_count, expected_speakers) in GRID_HELD_OUT_SPLITS.items():
        path = dataset.root / GRID_MANIFEST_DIR / f"{split}.jsonl"
        if len(dataset) == expected_count and speakers == expected_speakers and path.is_file():
            reference = _reference(dataset, path)
            if _samples(dataset) == _samples(reference):
                return split
    return "custom"
