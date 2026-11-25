"""Apply the GRID Student mouth crop to one raw video."""

from pathlib import Path

import numpy as np


def resize_student_roi(roi) -> np.ndarray:
    """Center crop the aligned grayscale sequence and resize as in training."""
    import torch
    from torchvision.transforms import functional as TF

    from .data import INPUT_SIZE

    sequence = torch.as_tensor(roi)
    if sequence.ndim != 3 or not sequence.numel() or sequence.dtype != torch.uint8:
        raise ValueError("Expected a nonempty uint8 grayscale sequence [T, H, W].")
    cropped = TF.center_crop(sequence, [88, 88])
    resized = TF.resize(cropped.unsqueeze(1), list(INPUT_SIZE), antialias=True)
    return resized.squeeze(1).numpy()


def extract_video_roi(path: Path, *, max_frames: int = 300) -> np.ndarray:
    """Decode RGB in presentation order and run the existing GRID face alignment."""
    if max_frames <= 0:
        raise ValueError("max_frames must be positive.")
    try:
        import av
        from pipelines.detectors.mediapipe.detector import LandmarksDetector
        from pipelines.detectors.mediapipe.video_process import VideoProcess
    except ImportError as error:
        raise RuntimeError(
            f"Unable to load raw-video dependencies ({error}). See requirements-video.txt."
        ) from error

    frames = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("Input has no video stream.")
        for frame in container.decode(video=0):
            if len(frames) == max_frames:
                raise ValueError(f"Video exceeds {max_frames} frames. Select a short utterance.")
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError("Input contains no decodable video frames.")

    detector = LandmarksDetector()
    try:
        landmarks = detector.detect(frames, detector.full_range_detector)
        if not any(point is not None for point in landmarks):
            landmarks = detector.detect(frames, detector.short_range_detector)
        if not any(point is not None for point in landmarks):
            raise ValueError("No face detected. Use a clip with one visible, front-facing speaker.")
    finally:
        detector.full_range_detector.close()
        detector.short_range_detector.close()
    aligned = VideoProcess(convert_gray=True)(frames, landmarks)
    return resize_student_roi(aligned)
