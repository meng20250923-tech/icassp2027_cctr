"""Audit train, development, and test multi-caption manifests for leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dire.video_pipeline import load_multicaption_manifest


def read_split(name: str, path: Path) -> dict:
    videos, captions = load_multicaption_manifest(path)
    return {
        "name": name,
        "path": str(path),
        "videos": len(videos),
        "captions": len(captions),
        "video_ids": {row["video_id"] for row in videos},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    splits = {name: read_split(name, getattr(args, name)) for name in ("train", "dev", "test")}
    overlap = {
        "train_dev": sorted(splits["train"]["video_ids"] & splits["dev"]["video_ids"]),
        "train_test": sorted(splits["train"]["video_ids"] & splits["test"]["video_ids"]),
        "dev_test": sorted(splits["dev"]["video_ids"] & splits["test"]["video_ids"]),
    }
    report = {
        "splits": {name: {key: value for key, value in split.items() if key != "video_ids"} for name, split in splits.items()},
        "overlap": overlap,
        "passed": not any(overlap.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Split audit failed")


if __name__ == "__main__":
    main()
