"""Evaluate a configurable temporal-adapter ablation on frozen features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from dire.configurable_temporal_adapter import build_configurable_adapter
from dire.multicaption_retrieval import multicaption_ranks, multicaption_retrieval_report
from dire.video_pipeline import attention_entropy, feature_tensors, load_feature_payload, rank_lists


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--protocol-note", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    payload = load_feature_payload(args.features)
    features = feature_tensors(payload, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_configurable_adapter(checkpoint["dim"], checkpoint["hidden_dim"], checkpoint["max_frames"], checkpoint["architecture"]).to(device).eval()
    model.load_state_dict(checkpoint["state"])
    with torch.no_grad():
        mean_features = F.normalize(features["frames"].mean(dim=1), dim=-1)
        mean_scores = model.logit_scale.exp().clamp(max=100.0) * mean_features @ features["text"].T
        output = model(features["frames"])
        temporal_scores = model.scaled_similarity(output.features, features["text"])
        mean_ranks = multicaption_ranks(mean_scores, features["mapping"])
        temporal_ranks = multicaption_ranks(temporal_scores, features["mapping"])
    report = {
        "benchmark": args.benchmark,
        "protocol_note": args.protocol_note,
        "architecture": checkpoint["architecture"],
        "videos": len(features["frames"]),
        "captions": len(features["text"]),
        "frames_per_video": int(payload["frames_per_video"]),
        "mean_frame_adapter": multicaption_retrieval_report(mean_scores, features["mapping"]),
        "learned_temporal_adapter": multicaption_retrieval_report(temporal_scores, features["mapping"]),
        "mean_attention_entropy": attention_entropy(output.attention),
        "mean_frame_adapter_ranks": rank_lists(mean_ranks),
        "learned_temporal_adapter_ranks": rank_lists(temporal_ranks),
        "features": str(args.features),
        "checkpoint": str(args.checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if "ranks" not in key}, indent=2))


if __name__ == "__main__":
    main()
