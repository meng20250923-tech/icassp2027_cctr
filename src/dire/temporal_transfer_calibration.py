"""Unlabelled calibration utilities for safe temporal-residual transfer.

The selectors in this module deliberately consume only the video--caption
similarity matrix. They never receive the benchmark caption-to-video mapping:
that mapping is reserved for reporting held-out retrieval performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class CycleAgreement:
    """Bidirectional reciprocal-neighbour statistics without pair labels."""

    video_to_text: float
    text_to_video: float
    balanced: float
    topk: int


@dataclass(frozen=True)
class UnlabelledConfidence:
    """Simple label-free score-only selector baselines."""

    mean_top1_similarity: float
    mean_top1_margin: float


def blend_temporal_features(
    mean_features: torch.Tensor,
    temporal_features: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Interpolate mean and temporal embeddings, then L2-normalize."""
    if mean_features.ndim != 2 or temporal_features.shape != mean_features.shape:
        raise ValueError("mean_features and temporal_features must share shape [videos, dim]")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return F.normalize(mean_features + alpha * (temporal_features - mean_features), dim=-1)


def reciprocal_cycle_agreement(scores: torch.Tensor, topk: int = 1) -> CycleAgreement:
    """Measure reciprocal retrieval agreement using no ground-truth pairs."""
    if scores.ndim != 2 or min(scores.shape) < 1:
        raise ValueError("scores must have non-empty shape [videos, captions]")
    videos, captions = scores.shape
    if not 1 <= topk <= min(videos, captions):
        raise ValueError("topk must be between 1 and min(videos, captions)")

    top_captions = scores.topk(topk, dim=1).indices
    top_videos = scores.topk(topk, dim=0).indices.T
    video_ids = torch.arange(videos, device=scores.device).view(videos, 1, 1)
    caption_ids = torch.arange(captions, device=scores.device).view(captions, 1, 1)

    reciprocal_video = (top_videos[top_captions] == video_ids).any(dim=(1, 2))
    reciprocal_caption = (top_captions[top_videos] == caption_ids).any(dim=(1, 2))
    video_to_text = float(reciprocal_video.float().mean())
    text_to_video = float(reciprocal_caption.float().mean())
    return CycleAgreement(
        video_to_text=video_to_text,
        text_to_video=text_to_video,
        balanced=(video_to_text + text_to_video) / 2.0,
        topk=topk,
    )


def unlabelled_retrieval_confidence(scores: torch.Tensor) -> UnlabelledConfidence:
    """Return natural score-only confidence baselines without pair labels."""
    if scores.ndim != 2 or min(scores.shape) < 2:
        raise ValueError("scores must have both dimensions at least two")
    top_text = scores.topk(2, dim=1).values
    top_video = scores.topk(2, dim=0).values
    mean_top1_similarity = (top_text[:, 0].mean() + top_video[0].mean()) / 2.0
    mean_top1_margin = (
        (top_text[:, 0] - top_text[:, 1]).mean()
        + (top_video[0] - top_video[1]).mean()
    ) / 2.0
    return UnlabelledConfidence(
        mean_top1_similarity=float(mean_top1_similarity),
        mean_top1_margin=float(mean_top1_margin),
    )


def parse_alpha_grid(values: Iterable[float]) -> list[float]:
    """Validate a unique, sorted residual-strength grid containing zero."""
    grid = sorted({float(value) for value in values})
    if not grid or grid[0] != 0.0:
        raise ValueError("alpha grid must contain 0.0 for safe abstention")
    if any(value < 0.0 or value > 1.0 for value in grid):
        raise ValueError("all alpha values must be in [0, 1]")
    return grid


def choose_alpha_by_metric(rows: list[dict[str, float]], metric: str) -> float:
    """Maximize a label-free metric, breaking ties toward abstention."""
    if not rows:
        raise ValueError("rows must be non-empty")
    required = {"alpha", metric}
    if any(required - row.keys() for row in rows):
        raise ValueError(f"each row needs alpha and {metric}")
    best = max(rows, key=lambda row: (float(row[metric]), -float(row["alpha"])))
    return float(best["alpha"])


def choose_alpha_by_cycle(rows: list[dict[str, float]]) -> float:
    """Choose the strongest reciprocal-cycle score, breaking ties safely."""
    return choose_alpha_by_metric(rows, "cycle_balanced")
