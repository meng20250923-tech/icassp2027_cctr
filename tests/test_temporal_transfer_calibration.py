import pytest
import torch

from dire.temporal_transfer_calibration import (
    blend_temporal_features,
    choose_alpha_by_cycle,
    choose_alpha_by_metric,
    parse_alpha_grid,
    reciprocal_cycle_agreement,
    unlabelled_retrieval_confidence,
)


def test_blending_preserves_mean_and_temporal_endpoints():
    mean = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    temporal = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert torch.allclose(blend_temporal_features(mean, temporal, 0.0), mean)
    assert torch.allclose(blend_temporal_features(mean, temporal, 1.0), temporal)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        blend_temporal_features(mean, temporal, 1.1)


def test_reciprocal_cycle_agreement_needs_no_labels_and_detects_mutual_edges():
    scores = torch.tensor([[4.0, 1.0, 0.0], [0.0, 3.0, 2.0]])
    agreement = reciprocal_cycle_agreement(scores, topk=1)
    assert agreement.video_to_text == 1.0
    assert agreement.text_to_video == pytest.approx(2.0 / 3.0)
    assert agreement.balanced == pytest.approx(5.0 / 6.0)


def test_grid_requires_abstention_and_selector_breaks_ties_toward_zero():
    assert parse_alpha_grid([1.0, 0.0, 0.5, 0.5]) == [0.0, 0.5, 1.0]
    with pytest.raises(ValueError, match="contain 0.0"):
        parse_alpha_grid([0.1, 1.0])
    assert choose_alpha_by_cycle([
        {"alpha": 0.5, "cycle_balanced": 0.8},
        {"alpha": 0.0, "cycle_balanced": 0.8},
    ]) == 0.0


def test_score_only_selector_baselines_need_no_pair_labels():
    scores = torch.tensor([[4.0, 1.0, 0.0], [0.0, 3.0, 2.0]])
    confidence = unlabelled_retrieval_confidence(scores)
    assert confidence.mean_top1_similarity == pytest.approx(3.25)
    assert confidence.mean_top1_margin == pytest.approx(7.0 / 3.0)
    assert choose_alpha_by_metric([
        {"alpha": 0.7, "mean_top1_margin": 0.5},
        {"alpha": 0.2, "mean_top1_margin": 0.5},
    ], "mean_top1_margin") == 0.2
