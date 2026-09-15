import json

import pytest
import torch

from dire.video_pipeline import feature_tensors, load_feature_payload, load_multicaption_manifest


def write_manifest(path, videos, captions):
    path.write_text(json.dumps({"videos": videos, "captions": captions}), encoding="utf-8")


def test_load_multicaption_manifest_validates_canonical_contract(tmp_path):
    path = tmp_path / "manifest.json"
    write_manifest(
        path,
        [
            {"index": 1, "video_id": "v1", "filename": "v1.mp4"},
            {"index": 0, "video_id": "v0", "filename": "v0.mp4"},
        ],
        [
            {"index": 1, "video_index": 1, "caption": "second"},
            {"index": 0, "video_index": 0, "caption": "first"},
        ],
    )
    videos, captions = load_multicaption_manifest(path)
    assert [row["video_id"] for row in videos] == ["v0", "v1"]
    assert [row["caption"] for row in captions] == ["first", "second"]


def test_load_multicaption_manifest_rejects_invalid_caption_mapping(tmp_path):
    path = tmp_path / "manifest.json"
    write_manifest(
        path,
        [{"index": 0, "video_id": "v0", "filename": "v0.mp4"}],
        [{"index": 0, "video_index": 2, "caption": "invalid"}],
    )
    with pytest.raises(ValueError, match="invalid"):
        load_multicaption_manifest(path)


def test_feature_payload_and_device_tensors_validate_caption_coverage(tmp_path):
    path = tmp_path / "features.pt"
    torch.save(
        {
            "frame_features": torch.ones(2, 2, 3),
            "text_features": torch.ones(3, 3),
            "caption_video_indices": torch.tensor([0, 1, 1]),
            "frames_per_video": 2,
        },
        path,
    )
    payload = load_feature_payload(path)
    tensors = feature_tensors(payload, torch.device("cpu"))
    assert tensors["frames"].shape == (2, 2, 3)
    assert [len(group) for group in tensors["groups"]] == [1, 2]


def test_feature_payload_rejects_video_without_caption(tmp_path):
    path = tmp_path / "features.pt"
    torch.save(
        {
            "frame_features": torch.ones(2, 2, 3),
            "text_features": torch.ones(1, 3),
            "caption_video_indices": torch.tensor([0]),
            "frames_per_video": 2,
        },
        path,
    )
    with pytest.raises(ValueError, match="Every video"):
        load_feature_payload(path)
