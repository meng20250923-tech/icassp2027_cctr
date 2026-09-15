"""Validate a dataset-agnostic multi-caption video-retrieval manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dire.video_pipeline import load_multicaption_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    videos, captions = load_multicaption_manifest(args.manifest)
    counts = Counter(row["video_index"] for row in captions)
    report = {
        "manifest": str(args.manifest),
        "videos": len(videos),
        "captions": len(captions),
        "minimum_captions_per_video": min(counts.values()),
        "maximum_captions_per_video": max(counts.values()),
        "duplicate_video_ids": len({row["video_id"] for row in videos}) != len(videos),
        "passed": True,
    }
    if report["duplicate_video_ids"]:
        raise ValueError("Manifest contains duplicate video IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
