"""Build a disjoint multi-caption MSR-VTT test-1K retrieval manifest.

The public fixed test-1K annotation supplies the ordered 1,000 video IDs.
Captions are read from the public MSRVTT_data.json annotation, retaining every
available test caption and its video association.  This is intentionally a
separate protocol from the existing one-caption fixed-pair diagnostic.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def caption_groups(payload: dict) -> dict[str, list[tuple[int, str]]]:
    """Return non-empty captions grouped and deterministically ordered by video."""
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for order, row in enumerate(payload.get("sentences", [])):
        video_id = str(row.get("video_id", "")).strip()
        caption = str(row.get("caption", row.get("sentence", ""))).strip()
        if video_id and caption:
            grouped[video_id].append((int(row.get("sen_id", order)), caption))
    return {
        video_id: sorted(rows, key=lambda item: (item[0], item[1]))
        for video_id, rows in grouped.items()
    }


def build_manifest(test_rows: list[dict], raw_payload: dict) -> dict:
    """Create video and caption tables with explicit positive associations."""
    if len(test_rows) != 1000:
        raise ValueError(f"Expected exactly 1,000 test videos, received {len(test_rows)}")
    grouped = caption_groups(raw_payload)
    seen: set[str] = set()
    videos: list[dict] = []
    captions: list[dict] = []
    for video_index, row in enumerate(test_rows):
        video_id = str(row.get("video_id", "")).strip()
        filename = str(row.get("video", "")).strip()
        if not video_id or not filename.lower().endswith(".mp4"):
            raise ValueError(f"Invalid test row {video_index}: {row}")
        if video_id in seen:
            raise ValueError(f"Duplicate video_id: {video_id}")
        if video_id not in grouped:
            raise ValueError(f"Missing raw captions for test video: {video_id}")
        seen.add(video_id)
        videos.append({"index": video_index, "video_id": video_id, "filename": filename})
        for sentence_id, caption in grouped[video_id]:
            captions.append(
                {
                    "index": len(captions),
                    "video_index": video_index,
                    "video_id": video_id,
                    "sentence_id": sentence_id,
                    "caption": caption,
                }
            )
    per_video = [0] * len(videos)
    for row in captions:
        per_video[row["video_index"]] += 1
    return {
        "benchmark": "MSR-VTT test-1K multi-caption retrieval",
        "protocol_note": "Test video IDs are fixed by msrvtt_test_1k.json; every available caption for those IDs is read from MSRVTT_data.json.",
        "videos": videos,
        "captions": captions,
        "audit": {
            "videos": len(videos),
            "captions": len(captions),
            "minimum_captions_per_video": min(per_video),
            "maximum_captions_per_video": max(per_video),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-annotations", type=Path, required=True)
    parser.add_argument("--raw-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(
        json.loads(args.test_annotations.read_text(encoding="utf-8")),
        json.loads(args.raw_annotations.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["audit"], indent=2))
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
