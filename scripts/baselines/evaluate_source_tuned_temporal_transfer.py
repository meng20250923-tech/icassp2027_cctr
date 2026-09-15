"""Evaluate a conventional source-validation-tuned fixed temporal residual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from dire.multicaption_retrieval import multicaption_retrieval_report
from dire.temporal_transfer_calibration import blend_temporal_features, parse_alpha_grid
from dire.temporal_video_adapter import TemporalVideoAdapter
from dire.video_pipeline import attention_entropy, feature_tensors, load_feature_payload


def alpha_values(value: str) -> list[float]:
    try:
        return parse_alpha_grid(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def encode(model: TemporalVideoAdapter, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, float]:
    mean = F.normalize(features["frames"].mean(dim=1), dim=-1)
    output = model(features["frames"])
    return mean, output.features, attention_entropy(output.attention)


def balanced_r_at_1(report: dict[str, dict[str, float]]) -> float:
    return (report["video_to_text"]["r_at_1"] + report["text_to_video"]["r_at_1"]) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-validation-features", type=Path, required=True)
    parser.add_argument("--target-features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--protocol-note", required=True)
    parser.add_argument("--alphas", type=alpha_values, default=alpha_values("0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    source_payload = load_feature_payload(args.source_validation_features)
    target_payload = load_feature_payload(args.target_features)
    source = feature_tensors(source_payload, device)
    target = feature_tensors(target_payload, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TemporalVideoAdapter(checkpoint["dim"], checkpoint["hidden_dim"], checkpoint["max_frames"]).to(device).eval()
    model.load_state_dict(checkpoint["state"])

    with torch.no_grad():
        source_mean, source_temporal, source_entropy = encode(model, source)
        source_rows: list[dict[str, object]] = []
        for alpha in args.alphas:
            videos = blend_temporal_features(source_mean, source_temporal, alpha)
            report = multicaption_retrieval_report(model.scaled_similarity(videos, source["text"]), source["mapping"])
            source_rows.append({
                "alpha": alpha,
                "source_validation_report": report,
                "source_validation_balanced_r_at_1": balanced_r_at_1(report),
            })
        selected = max(source_rows, key=lambda row: (float(row["source_validation_balanced_r_at_1"]), -float(row["alpha"])))
        selected_alpha = float(selected["alpha"])

        target_mean, target_temporal, target_entropy = encode(model, target)
        target_rows: list[dict[str, object]] = []
        for alpha in args.alphas:
            videos = blend_temporal_features(target_mean, target_temporal, alpha)
            report = multicaption_retrieval_report(model.scaled_similarity(videos, target["text"]), target["mapping"])
            target_rows.append({
                "alpha": alpha,
                "held_out_report": report,
                "held_out_balanced_r_at_1": balanced_r_at_1(report),
            })

    target_selected = next(row for row in target_rows if row["alpha"] == selected_alpha)
    report = {
        "benchmark": args.benchmark,
        "protocol_note": args.protocol_note,
        "selection_protocol": {
            "source_validation_pair_labels_used_for_selection": True,
            "target_pair_labels_used_for_selection": False,
            "selector": "source_validation_balanced_r_at_1",
            "alpha_grid": args.alphas,
        },
        "source_validation_videos": len(source["frames"]),
        "source_validation_captions": len(source["text"]),
        "target_videos": len(target["frames"]),
        "target_captions": len(target["text"]),
        "frames_per_video": int(target_payload["frames_per_video"]),
        "source_validation_attention_entropy": source_entropy,
        "target_attention_entropy": target_entropy,
        "source_validation_alpha_sweep": source_rows,
        "target_alpha_sweep_for_analysis_only": target_rows,
        "source_validation_selected_alpha": selected_alpha,
        "source_validation_selected_report": target_selected["held_out_report"],
        "source_validation_selected_balanced_r_at_1": target_selected["held_out_balanced_r_at_1"],
        "mean_pool_report": target_rows[0]["held_out_report"],
        "full_temporal_report": target_rows[-1]["held_out_report"],
        "source_validation_features": str(args.source_validation_features),
        "target_features": str(args.target_features),
        "checkpoint": str(args.checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if "sweep" not in key and "report" not in key}, indent=2))


if __name__ == "__main__":
    main()
