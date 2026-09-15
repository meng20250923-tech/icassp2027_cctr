"""Build deterministic English-only VATEX manifests for frozen retrieval."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


CLIP_ID = re.compile(r"^(?P<youtube_id>.+)_(?P<start>\d{6})_(?P<end>\d{6})$")


def parse_clip_id(video_id: str) -> tuple[str, int, int]:
    """Validate a VATEX clip ID and return its YouTube ID and boundaries."""
    match = CLIP_ID.fullmatch(video_id)
    if match is None:
        raise ValueError(f"Invalid VATEX videoID: {video_id!r}")
    start = int(match.group("start"))
    end = int(match.group("end"))
    if not match.group("youtube_id") or end <= start:
        raise ValueError(f"Invalid VATEX clip boundaries: {video_id!r}")
    return match.group("youtube_id"), start, end


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load and normalize public VATEX English annotation rows."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"VATEX annotations must be a list: {path}")
    normalized = []
    seen = set()
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError(f"VATEX annotation row is not a dictionary: {row!r}")
        video_id = row.get("videoID")
        captions = row.get("enCap")
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError(f"VATEX annotation has invalid videoID: {row!r}")
        if video_id in seen:
            raise ValueError(f"Duplicate VATEX videoID: {video_id}")
        if not isinstance(captions, list):
            raise ValueError(f"VATEX annotation has invalid enCap: {video_id}")
        cleaned = [str(caption).strip() for caption in captions if str(caption).strip()]
        if not cleaned:
            raise ValueError(f"VATEX annotation has no English captions: {video_id}")
        youtube_id, start, end = parse_clip_id(video_id)
        normalized.append(
            {
                "video_id": video_id,
                "youtube_id": youtube_id,
                "clip_start_seconds": start,
                "clip_end_seconds": end,
                "captions": cleaned,
            }
        )
        seen.add(video_id)
    if not normalized:
        raise ValueError(f"VATEX annotations are empty: {path}")
    return normalized


def choose(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Choose rows reproducibly without relying on annotation-file order."""
    if count < 1 or count > len(rows):
        raise ValueError(f"Requested {count} rows but only {len(rows)} are available")
    ordered = sorted(rows, key=lambda row: row["video_id"])
    random.Random(seed).shuffle(ordered)
    return ordered[:count]


def build_manifest(rows: list[dict[str, Any]], split: str, suffix: str) -> dict[str, Any]:
    """Convert normalized rows to the repository's generic retrieval contract."""
    videos = []
    captions = []
    for video_index, row in enumerate(rows):
        video_id = row["video_id"]
        videos.append(
            {
                "index": video_index,
                "video_id": video_id,
                "filename": f"{video_id}{suffix}",
                "youtube_id": row["youtube_id"],
                "clip_start_seconds": row["clip_start_seconds"],
                "clip_end_seconds": row["clip_end_seconds"],
            }
        )
        for caption in row["captions"]:
            captions.append(
                {
                    "index": len(captions),
                    "video_index": video_index,
                    "video_id": video_id,
                    "caption": caption,
                }
            )
    return {
        "benchmark": "VATEX English multi-caption retrieval",
        "split": split,
        "protocol_note": (
            "Deterministic English-only VATEX slice. Train and development videos are "
            "drawn without overlap from the official training annotations; evaluation "
            "videos are drawn from the disjoint official validation annotations."
        ),
        "videos": videos,
        "captions": captions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-annotations", type=Path, required=True)
    parser.add_argument("--validation-annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-videos", type=int, default=2000)
    parser.add_argument("--dev-videos", type=int, default=500)
    parser.add_argument("--test-videos", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--video-suffix", default=".mp4")
    args = parser.parse_args()
    if not args.video_suffix.startswith("."):
        raise ValueError("video-suffix must start with a dot")

    training = load_rows(args.training_annotations)
    validation = load_rows(args.validation_annotations)
    selected_source = choose(training, args.train_videos + args.dev_videos, args.seed)
    selected_test = choose(validation, args.test_videos, args.seed + 1)
    manifests = {
        "train": build_manifest(selected_source[: args.train_videos], "train", args.video_suffix),
        "dev": build_manifest(selected_source[args.train_videos :], "dev", args.video_suffix),
        "test": build_manifest(selected_test, "test", args.video_suffix),
    }
    ids = {name: {row["video_id"] for row in payload["videos"]} for name, payload in manifests.items()}
    overlap = {
        "train_dev": sorted(ids["train"] & ids["dev"]),
        "train_test": sorted(ids["train"] & ids["test"]),
        "dev_test": sorted(ids["dev"] & ids["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"VATEX split overlap detected: {overlap}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in manifests.items():
        (args.output_dir / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    audit = {
        "benchmark": "VATEX English multi-caption retrieval",
        "seed": args.seed,
        "official_sources": {"train": str(args.training_annotations), "test": str(args.validation_annotations)},
        "splits": {name: {"videos": len(payload["videos"]), "captions": len(payload["captions"])} for name, payload in manifests.items()},
        "overlap": overlap,
        "passed": True,
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
