import pytest
import torch

from dire.retrieval_metrics import (
    candidate_coverage,
    positive_ranks,
    retrieval_recall,
    symmetric_topk_mask,
)


def test_positive_ranks_and_recalls_are_diagonal_fixed_pair_metrics():
    scores = torch.tensor([
        [3.0, 4.0, 2.0],
        [1.0, 5.0, 2.0],
        [4.0, 3.0, 3.5],
    ])
    assert positive_ranks(scores).tolist() == [2, 1, 2]
    report = retrieval_recall(scores)
    assert report["r_at_1"] == pytest.approx(1 / 3)
    assert report["r_at_5"] == 1.0
    assert report["median_rank"] == 2.0


def test_symmetric_topk_mask_contains_candidates_from_both_directions():
    scores = torch.tensor([
        [4.0, 3.0, 1.0],
        [2.0, 5.0, 0.0],
        [6.0, 1.0, 2.0],
    ])
    mask = symmetric_topk_mask(scores, topk=1)
    assert mask[0, 0]  # row-0 top candidate
    assert mask[2, 0]  # column-0 top candidate
    assert mask[1, 1]
    coverage = candidate_coverage(scores, topk=1)
    assert coverage["video_to_text"]["r_at_1"] == pytest.approx(2 / 3)
    assert coverage["text_to_video"]["r_at_1"] == pytest.approx(2 / 3)
