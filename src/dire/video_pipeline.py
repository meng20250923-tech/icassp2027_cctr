"""Shared contracts for frozen-feature multi-caption video retrieval.

This module is intentionally dataset-agnostic.  Dataset-specific builders only
need to emit the manifest and feature contracts validated below.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


FEATURE_KEYS = {
    "frame_features",
    "text_features",
    "caption_video_indices",
    "frames_per_video",
}


def load_multicaption_manifest(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load a canonical multi-caption manifest and validate its indices."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    videos = sorted(payload.get("videos", []), key=lambda row: row["index"])
    captions = sorted(payload.get("captions", []), key=lambda row: row["index"])
    if not videos or not captions:
        raise ValueError("Manifest must contain non-empty videos and captions")
    if [row["index"] for row in videos] != list(range(len(videos))):
        raise ValueError("Video indices must be consecutive")
    if [row["index"] for row in captions] != list(range(len(captions))):
        raise ValueError("Caption indices must be consecutive")
    if any(not str(row.get("video_id", "")).strip() or not str(row.get("filename", "")).strip() for row in videos):
        raise ValueError("Every video needs non-empty video_id and filename")
    mapping = [row.get("video_index") for row in captions]
    if any(not isinstance(index, int) for index in mapping):
        raise ValueError("Every caption needs an integer video_index")
    if min(mapping) < 0 or max(mapping) >= len(videos):
        raise ValueError("Caption video indices are invalid")
    if any(not str(row.get("caption", "")).strip() for row in captions):
        raise ValueError("Every caption must be non-empty")
    return videos, captions


def load_feature_payload(path: Path) -> dict[str, Any]:
    """Load and validate a frozen video/text feature file on CPU."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing = FEATURE_KEYS - payload.keys()
    if missing:
        raise ValueError(f"Feature file is missing keys: {sorted(missing)}")
    frames = payload["frame_features"]
    text = payload["text_features"]
    mapping = payload["caption_video_indices"]
    if frames.ndim != 3 or text.ndim != 2 or mapping.ndim != 1:
        raise ValueError("Feature tensors have incompatible ranks")
    if frames.shape[-1] != text.shape[-1] or len(mapping) != len(text):
        raise ValueError("Frame, text, and caption mapping dimensions disagree")
    if frames.shape[1] != int(payload["frames_per_video"]):
        raise ValueError("frames_per_video does not match frame feature shape")
    if mapping.numel() == 0 or mapping.min().item() < 0 or mapping.max().item() >= len(frames):
        raise ValueError("Caption video indices are invalid")
    if (torch.bincount(mapping.long(), minlength=len(frames)) == 0).any():
        raise ValueError("Every video needs at least one caption")
    return payload


def feature_tensors(payload: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move validated features to ``device`` and derive caption groups."""
    frames = payload["frame_features"].float().to(device)
    text = F.normalize(payload["text_features"].float().to(device), dim=-1)
    mapping = payload["caption_video_indices"].long().to(device)
    groups = [torch.where(mapping == index)[0] for index in range(len(frames))]
    return {"frames": frames, "text": text, "mapping": mapping, "groups": groups}


def attention_entropy(attention: torch.Tensor) -> float:
    """Return mean entropy of normalized frame-attention weights."""
    return float((-(attention * attention.clamp_min(1e-8).log()).sum(dim=1)).mean())


def rank_lists(ranks: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    """Convert device tensors to JSON-compatible one-based rank lists."""
    return {key: value.cpu().tolist() for key, value in ranks.items()}
