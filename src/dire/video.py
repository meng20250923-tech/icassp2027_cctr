"""Video loading utilities for multi-frame relation evidence aggregation.

The video extension deliberately uses uniform frame sampling and mean pooling.
It does not model temporal order, so claims should be limited to multi-frame
video--text relation evidence aggregation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset


def uniform_frame_indices(frame_count: int, frames_per_video: int) -> list[int]:
    """Return deterministic, uniformly spaced frame indices.

    Repeated indices are intentional for videos shorter than the requested
    number of frames: every example then has the same tensor shape.
    """
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if frames_per_video <= 0:
        raise ValueError("frames_per_video must be positive")
    if frames_per_video == 1:
        # A single-frame control uses the temporal midpoint, not the first frame.
        return [(frame_count - 1) // 2]
    return torch.linspace(0, frame_count - 1, frames_per_video).round().long().tolist()


def _opencv():
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Video evaluation requires opencv-python-headless. "
            "Install it with: python -m pip install 'opencv-python-headless>=4.10,<5'"
        ) from error
    return cv2


def load_uniform_video_frames(
    video_path: str | Path,
    preprocess: Callable[[Image.Image], torch.Tensor],
    frames_per_video: int,
) -> torch.Tensor:
    """Decode uniformly sampled RGB frames and apply an OpenCLIP transform."""
    cv2 = _opencv()
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = uniform_frame_indices(frame_count, frames_per_video)
        frames: list[torch.Tensor] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {index} from: {path}")
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(preprocess(image))
        return torch.stack(frames)
    finally:
        capture.release()


class VideoCaptionPairDataset(Dataset):
    """Video--caption contrast pairs with a fixed number of decoded frames."""

    def __init__(
        self,
        rows: Sequence[dict],
        video_dir: str | Path,
        preprocess: Callable[[Image.Image], torch.Tensor],
        frames_per_video: int = 8,
        filename_key: str = "filename",
    ) -> None:
        self.rows = list(rows)
        self.video_dir = Path(video_dir)
        self.preprocess = preprocess
        self.frames_per_video = frames_per_video
        self.filename_key = filename_key

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        return index, load_uniform_video_frames(
            self.video_dir / row[self.filename_key],
            self.preprocess,
            self.frames_per_video,
        )
