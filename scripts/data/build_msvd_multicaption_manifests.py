"""Build canonical multi-caption train/dev/test manifests from MSVD splits."""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def load_split(path: Path) -> list[str]:
    """Load an MSVD split list represented as JSON list or newline text."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Split file is empty: {path}")
    if path.suffix.lower() == ".json":
        values = json.loads(text)
    else:
        values = [line.strip() for line in text.splitlines() if line.strip()]
    if not isinstance(values, list) or not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError(f"Split must contain non-empty strings: {path}")
    return [value.strip() for value in values]


def load_captions(path: Path) -> dict[str, list[str]]:
    """Load the public CLIP4Clip-style ``raw-captions.pkl`` mapping."""
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Caption payload must be a dictionary")
    result: dict[str, list[str]] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, str):
            entries: Iterable[Any] = [value]
        elif isinstance(value, (list, tuple)):
            entries = value
        else:
            continue
        captions = [str(entry).strip() for entry in entries if str(entry).strip()]
        if captions:
            result[key] = captions
    if not result:
        raise ValueError("Caption payload has no usable captions")
    return result


def filename_and_id(value: str, suffix: str) -> tuple[str, str]:
    """Return a stable video ID and filename from a public MSVD split value."""
    path = Path(value)
    video_id = path.stem if path.suffix else path.name
    filename = path.name if path.suffix else f"{video_id}{suffix}"
    return video_id, filename


def captions_for(video_id: str, filename: str, caption_map: dict[str, list[str]]) -> list[str]:
    """Resolve common caption-map key variants without changing captions."""
    candidates = (video_id, filename, Path(filename).stem)
    for candidate in candidates:
        if candidate in caption_map:
            return caption_map[candidate]
    raise ValueError(f"No captions found for video: {video_id}")


def build_manifest(split: str, values: list[str], caption_map: dict[str, list[str]], suffix: str) -> dict[str, Any]:
    """Build a consecutive-index manifest from a public split list."""
    videos = []
    captions = []
    seen = set()
    for value in values:
        video_id, filename = filename_and_id(value, suffix)
        if video_id in seen:
            raise ValueError(f"Duplicate video ID in {split} split: {video_id}")
        seen.add(video_id)
        index = len(videos)
        videos.append({"index": index, "video_id": video_id, "filename": filename})
        for caption in captions_for(video_id, filename, caption_map):
            captions.append({"index": len(captions), "video_index": index, "video_id": video_id, "caption": caption})
    if not videos or not captions:
        raise ValueError(f"Split must contain videos and captions: {split}")
    return {
        "benchmark": "MSVD multi-caption retrieval",
        "split": split,
        "protocol_note": "Public MSVD split lists and raw captions; train, development, and test video IDs are disjoint.",
        "videos": videos,
        "captions": captions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--dev-list", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--raw-captions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-suffix", default=".avi")
    args = parser.parse_args()
    if not args.video_suffix.startswith("."):
        raise ValueError("video-suffix must start with a dot")
    caption_map = load_captions(args.raw_captions)
    manifests = {
        "train": build_manifest("train", load_split(args.train_list), caption_map, args.video_suffix),
        "dev": build_manifest("dev", load_split(args.dev_list), caption_map, args.video_suffix),
        "test": build_manifest("test", load_split(args.test_list), caption_map, args.video_suffix),
    }
    ids = {name: {row["video_id"] for row in manifest["videos"]} for name, manifest in manifests.items()}
    overlaps = {
        "train_dev": sorted(ids["train"] & ids["dev"]),
        "train_test": sorted(ids["train"] & ids["test"]),
        "dev_test": sorted(ids["dev"] & ids["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"MSVD split overlap detected: {overlaps}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, manifest in manifests.items():
        (args.output_dir / f"{name}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    audit = {
        "benchmark": "MSVD multi-caption retrieval",
        "video_suffix": args.video_suffix,
        "splits": {name: {"videos": len(manifest["videos"]), "captions": len(manifest["captions"])} for name, manifest in manifests.items()},
        "overlap": overlaps,
        "passed": True,
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
