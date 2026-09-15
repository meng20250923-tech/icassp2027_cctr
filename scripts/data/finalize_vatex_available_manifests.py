"""Freeze final VATEX manifests from a deterministic decoded candidate pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_vatex_english_manifests import build_manifest


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("videos"), list):
        raise ValueError(f"Invalid candidate manifest: {path}")
    return payload


def rows_from_available(payload: dict[str, Any], available_names: set[str]) -> list[dict[str, Any]]:
    captions_by_index: dict[int, list[str]] = {}
    for caption in payload.get("captions", []):
        captions_by_index.setdefault(int(caption["video_index"]), []).append(str(caption["caption"]))
    rows = []
    for video in payload["videos"]:
        index = int(video["index"])
        if video["filename"] not in available_names:
            continue
        captions = captions_by_index.get(index, [])
        if not captions:
            raise ValueError(f"Available candidate has no captions: {video['video_id']}")
        rows.append(
            {
                "video_id": video["video_id"],
                "youtube_id": video["youtube_id"],
                "clip_start_seconds": int(video["clip_start_seconds"]),
                "clip_end_seconds": int(video["clip_end_seconds"]),
                "captions": captions,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-candidates", type=Path, required=True)
    parser.add_argument("--target-candidates", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-videos", type=int, default=2000)
    parser.add_argument("--dev-videos", type=int, default=500)
    parser.add_argument("--test-videos", type=int, default=1000)
    parser.add_argument("--video-suffix", default=".mp4")
    args = parser.parse_args()
    if not args.video_suffix.startswith("."):
        raise ValueError("video-suffix must start with a dot")

    source_payload = load_manifest(args.source_candidates)
    target_payload = load_manifest(args.target_candidates)
    available_names = {path.name for path in args.video_dir.glob(f"*{args.video_suffix}") if path.stat().st_size > 0}
    available_source = rows_from_available(source_payload, available_names)
    available_target = rows_from_available(target_payload, available_names)
    required_source = args.train_videos + args.dev_videos
    if len(available_source) < required_source or len(available_target) < args.test_videos:
        raise RuntimeError(
            "Insufficient decoded candidates: "
            f"source={len(available_source)}/{required_source}, "
            f"target={len(available_target)}/{args.test_videos}"
        )

    manifests = {
        "train": build_manifest(available_source[: args.train_videos], "train", args.video_suffix),
        "dev": build_manifest(available_source[args.train_videos : required_source], "dev", args.video_suffix),
        "test": build_manifest(available_target[: args.test_videos], "test", args.video_suffix),
    }
    ids = {name: {row["video_id"] for row in manifest["videos"]} for name, manifest in manifests.items()}
    overlap = {
        "train_dev": sorted(ids["train"] & ids["dev"]),
        "train_test": sorted(ids["train"] & ids["test"]),
        "dev_test": sorted(ids["dev"] & ids["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Final VATEX split overlap detected: {overlap}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, manifest in manifests.items():
        (args.output_dir / f"{name}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    audit = {
        "benchmark": "VATEX English multi-caption retrieval",
        "selection_stage": "post_download_availability_filtered_final_protocol",
        "source_candidate_videos": len(source_payload["videos"]),
        "target_candidate_videos": len(target_payload["videos"]),
        "decoded_source_candidates": len(available_source),
        "decoded_target_candidates": len(available_target),
        "final_splits": {
            name: {"videos": len(manifest["videos"]), "captions": len(manifest["captions"])}
            for name, manifest in manifests.items()
        },
        "overlap": overlap,
        "passed": True,
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
