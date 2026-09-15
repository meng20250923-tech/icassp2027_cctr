"""Metrics for retrieval with multiple captions associated with each video."""

from __future__ import annotations

from typing import Iterable

import torch


def validate_caption_video_indices(
    caption_video_indices: torch.Tensor,
    videos: int,
    captions: int,
) -> torch.Tensor:
    """Validate and canonicalize a one-dimensional caption-to-video mapping."""
    indices = torch.as_tensor(caption_video_indices, dtype=torch.long)
    if indices.ndim != 1 or len(indices) != captions:
        raise ValueError("caption_video_indices must have one entry per caption")
    if videos < 1 or captions < 1:
        raise ValueError("videos and captions must be positive")
    if indices.min().item() < 0 or indices.max().item() >= videos:
        raise ValueError("caption_video_indices contain an invalid video index")
    counts = torch.bincount(indices, minlength=videos)
    if (counts == 0).any():
        raise ValueError("Every video must have at least one positive caption")
    return indices


def multicaption_ranks(scores: torch.Tensor, caption_video_indices: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute ranks for V2T and T2V under explicit multi-positive labels.

    ``scores`` is ``[videos, captions]``.  V2T rank uses the best-scoring
    caption belonging to each query video; T2V uses each caption's associated
    video as its sole positive.  Ties favour a positive candidate in both
    directions, consistently with the fixed-pair metric implementation.
    """
    if scores.ndim != 2:
        raise ValueError("scores must have shape [videos, captions]")
    videos, captions = scores.shape
    mapping = validate_caption_video_indices(caption_video_indices, videos, captions).to(scores.device)
    video_indices = torch.arange(videos, device=scores.device)
    positive_mask = mapping.unsqueeze(0) == video_indices.unsqueeze(1)
    best_positive = scores.masked_fill(~positive_mask, float("-inf")).max(dim=1).values
    video_to_text = (scores > best_positive.unsqueeze(1)).sum(dim=1) + 1
    correct_video_scores = scores[mapping, torch.arange(captions, device=scores.device)]
    text_to_video = (scores.T > correct_video_scores.unsqueeze(1)).sum(dim=1) + 1
    return {"video_to_text": video_to_text, "text_to_video": text_to_video}


def recall_from_ranks(ranks: torch.Tensor, cutoffs: Iterable[int] = (1, 5, 10)) -> dict[str, float]:
    """Report recall and rank statistics from one-based ranks."""
    if ranks.ndim != 1 or len(ranks) == 0:
        raise ValueError("ranks must be a non-empty vector")
    report = {f"r_at_{cutoff}": float((ranks <= cutoff).float().mean()) for cutoff in cutoffs}
    report["median_rank"] = float(ranks.float().median())
    report["mean_rank"] = float(ranks.float().mean())
    return report


def multicaption_retrieval_report(scores: torch.Tensor, caption_video_indices: torch.Tensor) -> dict[str, dict[str, float]]:
    """Report V2T and T2V retrieval using a caption-to-video association map."""
    ranks = multicaption_ranks(scores, caption_video_indices)
    return {direction: recall_from_ranks(values) for direction, values in ranks.items()}
