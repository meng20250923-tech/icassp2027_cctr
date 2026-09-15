"""Evaluate unlabelled cycle-calibrated temporal-residual transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from dire.multicaption_retrieval import multicaption_retrieval_report
from dire.temporal_transfer_calibration import (
    blend_temporal_features,
    choose_alpha_by_cycle,
    parse_alpha_grid,
    reciprocal_cycle_agreement,
)
from dire.temporal_video_adapter import TemporalVideoAdapter
from dire.video_pipeline import attention_entropy, feature_tensors, load_feature_payload


def alpha_values(value: str) -> list[float]:
    try:
        return parse_alpha_grid(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def balanced_r_at_1(report: dict[str, dict[str, float]]) -> float:
    return (report["video_to_text"]["r_at_1"] + report["text_to_video"]["r_at_1"]) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--protocol-note", required=True)
    parser.add_argument("--alphas", type=alpha_values, default=alpha_values("0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"))
    parser.add_argument("--cycle-topk", type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    payload = load_feature_payload(args.features)
    features = feature_tensors(payload, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TemporalVideoAdapter(checkpoint["dim"], checkpoint["hidden_dim"], checkpoint["max_frames"]).to(device).eval()
    model.load_state_dict(checkpoint["state"])

    with torch.no_grad():
        mean_features = F.normalize(features["frames"].mean(dim=1), dim=-1)
        temporal_output = model(features["frames"])
        rows: list[dict[str, object]] = []
        for alpha in args.alphas:
            video_features = blend_temporal_features(mean_features, temporal_output.features, alpha)
            scores = model.scaled_similarity(video_features, features["text"])
            cycle = reciprocal_cycle_agreement(scores, args.cycle_topk)
            report = multicaption_retrieval_report(scores, features["mapping"])
            rows.append(
                {
                    "alpha": alpha,
                    "cycle_video_to_text": cycle.video_to_text,
                    "cycle_text_to_video": cycle.text_to_video,
                    "cycle_balanced": cycle.balanced,
                    "held_out_report": report,
                    "held_out_balanced_r_at_1": balanced_r_at_1(report),
                }
            )

    selected_alpha = choose_alpha_by_cycle(rows)
    selected = next(row for row in rows if row["alpha"] == selected_alpha)
    # Ground truth is deliberately used only here, after the unlabelled choice.
    oracle = max(rows, key=lambda row: (float(row["held_out_balanced_r_at_1"]), -float(row["alpha"])))
    report = {
        "benchmark": args.benchmark,
        "protocol_note": args.protocol_note,
        "calibration_protocol": {
            "target_pair_labels_used_for_selection": False,
            "selector": "bidirectional_reciprocal_cycle_agreement",
            "cycle_topk": args.cycle_topk,
            "alpha_grid": args.alphas,
            "safe_abstention_alpha": 0.0,
            "oracle_note": "For analysis only; it uses held-out pair labels and is not a deployable selection rule.",
        },
        "videos": len(features["frames"]),
        "captions": len(features["text"]),
        "frames_per_video": int(payload["frames_per_video"]),
        "mean_attention_entropy": attention_entropy(temporal_output.attention),
        "alpha_sweep": rows,
        "cycle_selected_alpha": selected_alpha,
        "cycle_selected_report": selected["held_out_report"],
        "cycle_selected_balanced_r_at_1": selected["held_out_balanced_r_at_1"],
        "mean_pool_report": rows[0]["held_out_report"],
        "full_temporal_report": rows[-1]["held_out_report"],
        "oracle_alpha_for_analysis_only": oracle["alpha"],
        "oracle_report_for_analysis_only": oracle["held_out_report"],
        "features": str(args.features),
        "checkpoint": str(args.checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"alpha_sweep", "mean_pool_report", "full_temporal_report", "oracle_report_for_analysis_only"}}, indent=2))


if __name__ == "__main__":
    main()
