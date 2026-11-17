import argparse
import json
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import functional as TF

from pipelines.data.data_module import AVSRDataLoader


# 1. 路径与实验划分
ROOT = Path(__file__).resolve().parent
INPUT_ROOT = ROOT / "data" / "GRID"
TEACHER_ROOT = ROOT / "output"
LABEL_ROOT = ROOT / "benchmarks" / "GRID" / "labels"
OUTPUT_ROOT = ROOT / "data" / "student_grid"

SPLITS = {"train": range(1, 9), "val": range(9, 11), "test": range(11, 15)}
VIDEO_EXTENSIONS = (".mpg", ".mpeg", ".mp4")
ROI_SIZE = (64, 64)


def read_labels():
    labels = {}
    for filename in ("unseen_train.ref", "unseen_test.ref"):
        with (LABEL_ROOT / filename).open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                sample_id, text = line.split(maxsplit=1)
                speaker, stem = sample_id.split("/", maxsplit=1)
                labels.setdefault(speaker, {})[stem] = text.strip()
    return labels


@contextmanager
def atomic_output(path, mode, **options):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode, dir=path.parent, delete=False, **options
    ) as stream:
        temporary_path = Path(stream.name)
        try:
            yield stream
            stream.close()
            temporary_path.replace(path)
        finally:
            stream.close()
            temporary_path.unlink(missing_ok=True)


def manifest_path(path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


# 2. 逐视频裁剪，与已有 Teacher 缓存校验时间步数
def prepare_sample(speaker, stem, text, output_root, detector, loader):
    input_dir = INPUT_ROOT / speaker
    candidates = (input_dir / f"{stem}{ext}" for ext in VIDEO_EXTENSIONS)
    video_path = next((path for path in candidates if path.is_file()), None)
    if video_path is None:
        raise FileNotFoundError(f"找不到源视频：{input_dir / stem}")

    teacher_path = TEACHER_ROOT / speaker / f"{stem}.pt"
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=True)
    length = int(teacher["length"])
    logits = teacher["logits"]
    if length <= 0 or logits.shape != (length, 41) or int(teacher["blank_id"]) != 0:
        raise ValueError(f"Teacher 缓存形状、长度或 blank_id 无效：{teacher_path}")
    if not torch.isfinite(logits).all():
        raise ValueError(f"Teacher logits 含 NaN 或 Inf：{teacher_path}")

    roi_path = output_root / "roi" / speaker / f"{stem}.npy"
    expected_shape = (length, *ROI_SIZE)
    skipped = False
    if roi_path.is_file():
        try:
            cached_roi = np.load(roi_path, mmap_mode="r", allow_pickle=False)
            skipped = cached_roi.shape == expected_shape and cached_roi.dtype == np.uint8
            del cached_roi
        except (EOFError, OSError, ValueError):
            pass

    if not skipped:
        landmarks = detector(str(video_path))
        roi = loader.load_data(str(video_path), landmarks)
        if roi.ndim != 3 or roi.dtype != torch.uint8 or roi.shape[0] != length:
            raise ValueError(
                f"灰度 ROI 形状为 {tuple(roi.shape)}，类型为 {roi.dtype}，"
                f"Teacher 时间步数为 {length}"
            )
        roi = TF.center_crop(roi, [88, 88])
        roi = TF.resize(roi.unsqueeze(1), list(ROI_SIZE), antialias=True)
        with atomic_output(roi_path, "wb") as stream:
            np.save(stream, roi.squeeze(1).numpy(), allow_pickle=False)

    record = {
        "id": f"{speaker}/{stem}",
        "roi": manifest_path(roi_path),
        "text": text,
        "teacher": manifest_path(teacher_path),
        "length": length,
    }
    return record, skipped


# 3. 按说话人遍历，打印逐样本进度和汇总
def main():
    parser = argparse.ArgumentParser(description="批量提取 GRID Student 的灰度嘴部视频")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--sample", metavar="SPEAKER/STEM", help="仅生成指定样本的 ROI，用于验证预处理"
    )
    args = parser.parse_args()
    output_root = args.output_dir.resolve()
    labels = read_labels()

    if args.sample:
        speaker, _, stem = args.sample.partition("/")
        if stem not in labels.get(speaker, {}):
            parser.error(f"找不到样本标注：{args.sample}")
        jobs = {None: {speaker: {stem: labels[speaker][stem]}}}
    else:
        jobs = {
            split: {f"s{i}": labels[f"s{i}"] for i in speakers}
            for split, speakers in SPLITS.items()
        }

    from pipelines.detectors.mediapipe.detector import LandmarksDetector

    detector = LandmarksDetector()
    loader = AVSRDataLoader(
        modality="video", speed_rate=1, transform=False, detector="mediapipe", convert_gray=True
    )

    totals = Counter()
    for split, speakers in jobs.items():
        records = []
        for speaker, samples in speakers.items():
            counts = Counter()
            print(f"\n===== {speaker}: {len(samples)} videos =====")
            for index, (stem, text) in enumerate(sorted(samples.items()), start=1):
                start = time.perf_counter()
                try:
                    record, skipped = prepare_sample(speaker, stem, text, output_root, detector, loader)
                except Exception as error:
                    counts["failed"] += 1
                    print(f"[ERROR] {speaker}/{stem}: {error}", flush=True)
                    continue

                records.append(record)
                if skipped:
                    counts["skipped"] += 1
                else:
                    counts["success"] += 1
                    print(
                        f"[{speaker}] {index}/{len(samples)} {stem} "
                        f"-> {(record['length'], *ROI_SIZE)} ({time.perf_counter() - start:.2f}s)",
                        flush=True,
                    )
            totals.update(counts)
            print(f"{speaker} finished | Success={counts['success']}, "
                  f"Skipped={counts['skipped']}, Failed={counts['failed']}")

        if split is not None:
            path = output_root / f"{split}.jsonl"
            with atomic_output(path, "w", encoding="utf-8", newline="\n") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Saved {path}: {len(records)} samples")

    print("\n========== ALL FINISHED ==========")
    print(f"Success={totals['success']}, Skipped={totals['skipped']}, Failed={totals['failed']}")
    print(f"Output path: {output_root}")
    return int(totals["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
