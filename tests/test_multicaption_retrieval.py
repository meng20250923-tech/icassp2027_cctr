import pytest
import torch

from dire.multicaption_retrieval import multicaption_ranks, multicaption_retrieval_report


def test_multicaption_ranks_use_best_video_caption_and_correct_text_video_pairing():
    scores = torch.tensor([
        [0.8, 0.9, 0.1, 0.2],
        [0.4, 0.3, 0.7, 0.9],
    ])
    mapping = torch.tensor([0, 0, 1, 1])
    ranks = multicaption_ranks(scores, mapping)
    assert ranks["video_to_text"].tolist() == [1, 1]
    assert ranks["text_to_video"].tolist() == [1, 1, 1, 1]
    report = multicaption_retrieval_report(scores, mapping)
    assert report["video_to_text"]["r_at_1"] == 1.0
    assert report["text_to_video"]["r_at_1"] == 1.0


def test_multicaption_metrics_reject_missing_positive_video():
    with pytest.raises(ValueError, match="Every video"):
        multicaption_ranks(torch.ones(2, 2), torch.tensor([0, 0]))
