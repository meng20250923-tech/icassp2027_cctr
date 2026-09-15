"""Create deterministic disjoint calibration and evaluation feature files.

The split is made at video level. Every caption of a selected video remains
with that video, and caption-to-video indices are remapped independently in
both outputs. This permits unlabelled target calibration and held-out labelled
reporting without shared video or caption candidates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dire.video_pipeline import load_feature_payload


def select_videos(payload: dict, selected: torch.Tensor) -> dict:
    """Select videos and all associated captions, remapping video indices."""
    source_videos = len(payload["frame_features"])
    selected = torch.as_tensor(selected, dtype=torch.long).sort().values
    if selected.ndim != 1 or len(selected) < 1:
        raise ValueError("selected must contain at least one video index")
    if selected.min().item() < 0 or selected.max().item() >= source_videos:
        raise ValueError("selected video indices are out of range")
    if len(selected.unique()) != len(selected):
        raise ValueError("selected video indices must be unique")
    mapping = payload["caption_video_indices"].long()
    remap = torch.full((source_videos,), -1, dtype=torch.long)
    remap[selected] = torch.arange(len(selected), dtype=torch.long)
    caption_indices = torch.where(remap[mapping] >= 0)[0]

    result = dict(payload)
    result["frame_features"] = payload["frame_features"].index_select(0, selected)
    result["text_features"] = payload["text_features"].index_select(0, caption_indices)
    result["caption_video_indices"] = remap[mapping.index_select(0, caption_indices)]
    if "video_ids" in payload:
        result["video_ids"] = [payload["video_ids"][index] for index in selected.tolist()]
    if "captions" in payload:
        result["captions"] = [payload["captions"][index] for index in caption_indices.tolist()]
    result["source_video_indices"] = selected
    return result


def split_payload(payload: dict, calibration_videos: int, split_seed: int) -> tuple[dict, dict]:
    """Split a payload into non-empty video-disjoint calibration and test sets."""
    total = len(payload["frame_features"])
    if not 1 <= calibration_videos < total:
        raise ValueError(f"calibration_videos must be in [1, {total - 1}]")
    generator = torch.Generator(device="cpu").manual_seed(split_seed)
    order = torch.randperm(total, generator=generator)
    calibration = select_videos(payload, order[:calibration_videos])
    evaluation = select_videos(payload, order[calibration_videos:])
    for result, role in ((calibration, "calibration"), (evaluation, "evaluation")):
        result["split_seed"] = split_seed
        result["split_role"] = role
        result["source_videos"] = total
        result["calibration_videos"] = calibration_videos
    return calibration, evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--calibration-output", type=Path, required=True)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--calibration-videos", type=int, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    args = parser.parse_args()

    payload = load_feature_payload(args.features)
    calibration, evaluation = split_payload(payload, args.calibration_videos, args.split_seed)
    args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
    args.evaluation_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(calibration, args.calibration_output)
    torch.save(evaluation, args.evaluation_output)
    metadata = {
        "source_features": str(args.features),
        "split_seed": args.split_seed,
        "source_videos": len(payload["frame_features"]),
        "source_captions": len(payload["text_features"]),
        "calibration_videos": len(calibration["frame_features"]),
        "calibration_captions": len(calibration["text_features"]),
        "evaluation_videos": len(evaluation["frame_features"]),
        "evaluation_captions": len(evaluation["text_features"]),
        "calibration_source_video_indices": calibration["source_video_indices"].tolist(),
        "evaluation_source_video_indices": evaluation["source_video_indices"].tolist(),
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
