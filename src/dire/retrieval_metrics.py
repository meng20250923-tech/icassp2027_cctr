"""Small, framework-independent metrics for fixed-pair retrieval."""

from __future__ import annotations

from typing import Iterable

import torch


def positive_ranks(scores: torch.Tensor) -> torch.Tensor:
    """Return one-based ranks of the diagonal positives in a square score matrix.

    Rows are queries and columns are candidates.  The fixed-pair protocols use
    the matching index as the sole positive for each query.  Ties are resolved
    deterministically in favour of the positive item, which is conservative
    with respect to neither system because both use this same convention.
    """
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be a square [queries, candidates] matrix")
    positives = scores.diagonal().unsqueeze(1)
    return (scores > positives).sum(dim=1) + 1


def retrieval_recall(scores: torch.Tensor, cutoffs: Iterable[int] = (1, 5, 10)) -> dict[str, float]:
    """Compute R@K and median rank for a fixed-pair retrieval matrix."""
    ranks = positive_ranks(scores)
    report = {f"r_at_{cutoff}": float((ranks <= cutoff).float().mean()) for cutoff in cutoffs}
    report["median_rank"] = float(ranks.float().median())
    report["mean_rank"] = float(ranks.float().mean())
    return report


def symmetric_retrieval_report(scores: torch.Tensor, cutoffs: Iterable[int] = (1, 5, 10)) -> dict[str, dict[str, float]]:
    """Report video-to-text and text-to-video recalls from one square matrix."""
    return {
        "video_to_text": retrieval_recall(scores, cutoffs),
        "text_to_video": retrieval_recall(scores.T, cutoffs),
    }


def symmetric_topk_mask(scores: torch.Tensor, topk: int) -> torch.Tensor:
    """Select pairs retrieved in the top-K in either retrieval direction."""
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be square")
    if not 1 <= topk <= scores.shape[0]:
        raise ValueError("topk must be between 1 and the number of pairs")
    mask = torch.zeros_like(scores, dtype=torch.bool)
    rows = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1)
    columns = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0)
    mask[rows, scores.topk(topk, dim=1).indices] = True
    mask[scores.topk(topk, dim=0).indices, columns] = True
    return mask


def candidate_coverage(scores: torch.Tensor, topk: int) -> dict[str, float]:
    """Fraction of positives eligible for reranking in each direction."""
    return symmetric_retrieval_report(scores, (topk,))
