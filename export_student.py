"""Export a Student checkpoint as a validated FP32 ONNX model and optionally INT8."""

import argparse
from pathlib import Path

from student.deployment import TRAIN_MANIFEST, export_student


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--fp32-only", action="store_true", help="Export without loading calibration data")
    parser.add_argument("--calibration-manifest", type=Path, default=TRAIN_MANIFEST,
                        help=f"Training manifest or a subset of {TRAIN_MANIFEST.as_posix()}")
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)

    try:
        report = export_student(
            args.checkpoint, args.output_dir, args.root, fp32_only=args.fp32_only,
            calibration_manifest=args.calibration_manifest, calibration_samples=args.calibration_samples,
            calibration_seed=args.calibration_seed, threads=args.threads,
        )
    except ModuleNotFoundError as error:
        parser.error(f"Missing deployment dependency: {error.name}. Install requirements-deployment.txt.")
    except Exception as error:
        parser.error(str(error))

    print(f"Exported {', '.join(report['artifacts'])} to {(args.root / args.output_dir).resolve()}")
    return report


if __name__ == "__main__":
    main()
