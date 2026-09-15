import torch

from scripts.data.split_multicaption_features import split_payload


def test_video_level_split_keeps_caption_groups_and_remaps_indices():
    payload = {
        "frame_features": torch.randn(4, 2, 3),
        "text_features": torch.randn(7, 3),
        "caption_video_indices": torch.tensor([0, 0, 1, 2, 2, 2, 3]),
        "frames_per_video": 2,
        "video_ids": ["v0", "v1", "v2", "v3"],
        "captions": ["a", "b", "c", "d", "e", "f", "g"],
    }
    calibration, evaluation = split_payload(payload, calibration_videos=2, split_seed=9)
    calibration_source = set(calibration["source_video_indices"].tolist())
    evaluation_source = set(evaluation["source_video_indices"].tolist())
    assert calibration_source.isdisjoint(evaluation_source)
    assert calibration_source | evaluation_source == {0, 1, 2, 3}
    for result in (calibration, evaluation):
        assert result["caption_video_indices"].min().item() == 0
        assert result["caption_video_indices"].max().item() == len(result["frame_features"]) - 1
        assert len(result["caption_video_indices"]) >= len(result["frame_features"])
