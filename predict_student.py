"""Transcribe a GRID clip using an exported ONNX Student."""

import argparse
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="FP32 or INT8 ONNX Student")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--roi", type=Path, help="Prepared uint8 GRID mouth ROI .npy")
    source.add_argument("--video", type=Path, help="One short video of a single speaker")
    parser.add_argument("--threads", type=int, default=1, help="ONNX Runtime CPU threads")
    parser.add_argument("--max-frames", type=int, default=300, help="Raw video frame limit")
    args = parser.parse_args(argv)

    if args.threads <= 0 or args.max_frames <= 0:
        parser.error("--threads and --max-frames must be positive")
    if args.roi is not None:
        with args.roi.open("rb") as stream:
            roi = np.load(stream, allow_pickle=False)
            if not isinstance(roi, np.ndarray):
                roi.close()
                parser.error("--roi requires a single .npy array")
    else:
        from student.video import extract_video_roi

        roi = extract_video_roi(args.video, max_frames=args.max_frames)
    from student.grid import normalize_roi_numpy
    from student.onnx_runtime import OnnxStudent

    frames = normalize_roi_numpy(roi)[None, ...]
    lengths = np.array([frames.shape[1]], dtype=np.int64)
    student = OnnxStudent(args.model, threads=args.threads)
    print(student.predict(frames, lengths)[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
