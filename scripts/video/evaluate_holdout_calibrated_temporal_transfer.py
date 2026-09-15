"""Select a temporal residual scale on unlabelled target data, then test it."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from dire.multicaption_retrieval import multicaption_retrieval_report
from dire.temporal_transfer_calibration import (
    blend_temporal_features,
    choose_alpha_by_metric,
    parse_alpha_grid,
    reciprocal_cycle_agreement,
    unlabelled_retrieval_confidence,
)
from dire.temporal_video_adapter import TemporalVideoAdapter
from dire.video_pipeline import attention_entropy, feature_tensors, load_feature_payload


SELECTOR_METRICS = {
    "cycle": "cycle_balanced",
    "max_similarity": "mean_top1_similarity",
    "max_margin": "mean_top1_margin",
}


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
    parser.add_argument("--calibration-features", type=Path, required=True)
    parser.add_argument("--evaluation-features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--protocol-note", required=True)
    parser.add_argument("--alphas", type=alpha_values, default=alpha_values("0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"))
    parser.add_argument("--cycle-topk", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    calibration_payload = load_feature_payload(args.calibration_features)
    evaluation_payload = load_feature_payload(args.evaluation_features)
    calibration = feature_tensors(calibration_payload, device)
    evaluation = feature_tensors(evaluation_payload, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TemporalVideoAdapter(checkpoint["dim"], checkpoint["hidden_dim"], checkpoint["max_frames"]).to(device).eval()
    model.load_state_dict(checkpoint["state"])

    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats(device)
        calibration_started = time.perf_counter()
        calibration_mean, calibration_temporal, calibration_entropy = encode(model, calibration)
        selection_rows: list[dict[str, float]] = []
        for alpha in args.alphas:
            videos = blend_temporal_features(calibration_mean, calibration_temporal, alpha)
            scores = model.scaled_similarity(videos, calibration["text"])
            cycle = reciprocal_cycle_agreement(scores, args.cycle_topk)
            confidence = unlabelled_retrieval_confidence(scores)
            selection_rows.append({
                "alpha": alpha,
                "cycle_video_to_text": cycle.video_to_text,
                "cycle_text_to_video": cycle.text_to_video,
                "cycle_balanced": cycle.balanced,
                "mean_top1_similarity": confidence.mean_top1_similarity,
                "mean_top1_margin": confidence.mean_top1_margin,
            })
        selected_alphas = {
            name: choose_alpha_by_metric(selection_rows, metric)
            for name, metric in SELECTOR_METRICS.items()
        }
        calibration_seconds = time.perf_counter() - calibration_started
        calibration_peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))

        torch.cuda.reset_peak_memory_stats(device)
        evaluation_started = time.perf_counter()
        evaluation_mean, evaluation_temporal, evaluation_entropy = encode(model, evaluation)
        evaluation_rows: list[dict[str, object]] = []
        for alpha in args.alphas:
            videos = blend_temporal_features(evaluation_mean, evaluation_temporal, alpha)
            scores = model.scaled_similarity(videos, evaluation["text"])
            held_out = multicaption_retrieval_report(scores, evaluation["mapping"])
            evaluation_rows.append({
                "alpha": alpha,
                "held_out_report": held_out,
                "held_out_balanced_r_at_1": balanced_r_at_1(held_out),
            })
        evaluation_sweep_seconds = time.perf_counter() - evaluation_started
        evaluation_peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))

    selected = {
        name: next(row for row in evaluation_rows if row["alpha"] == alpha)
        for name, alpha in selected_alphas.items()
    }
    oracle = max(evaluation_rows, key=lambda row: (float(row["held_out_balanced_r_at_1"]), -float(row["alpha"])))
    report = {
        "benchmark": args.benchmark,
        "protocol_note": args.protocol_note,
        "calibration_protocol": {
            "calibration_pair_labels_used_for_selection": False,
            "evaluation_pair_labels_used_for_selection": False,
            "calibration_and_evaluation_candidate_sets_disjoint": True,
            "selector": "bidirectional_reciprocal_cycle_agreement",
            "selector_baselines": {
                "max_similarity": "Maximize mean bidirectional top-1 similarity without pair labels.",
                "max_margin": "Maximize mean bidirectional top-1 minus top-2 score margin without pair labels.",
            },
            "cycle_topk": args.cycle_topk,
            "alpha_grid": args.alphas,
            "safe_abstention_alpha": 0.0,
            "oracle_note": "For analysis only; it uses evaluation pair labels and is not a deployable selection rule.",
        },
        "calibration_videos": len(calibration["frames"]),
        "calibration_captions": len(calibration["text"]),
        "evaluation_videos": len(evaluation["frames"]),
        "evaluation_captions": len(evaluation["text"]),
        "frames_per_video": int(evaluation_payload["frames_per_video"]),
        "runtime": {
            "calibration_selection_wall_seconds": calibration_seconds,
            "calibration_peak_memory_bytes": calibration_peak_memory_bytes,
            "evaluation_alpha_sweep_wall_seconds_for_analysis_only": evaluation_sweep_seconds,
            "evaluation_peak_memory_bytes": evaluation_peak_memory_bytes,
            "target_parameter_updates": 0,
            "alpha_grid_size": len(args.alphas),
        },
        "calibration_attention_entropy": calibration_entropy,
        "evaluation_attention_entropy": evaluation_entropy,
        "calibration_alpha_sweep": selection_rows,
        "evaluation_alpha_sweep_for_analysis_only": evaluation_rows,
        "cycle_selected_alpha": selected_alphas["cycle"],
        "cycle_selected_report": selected["cycle"]["held_out_report"],
        "cycle_selected_balanced_r_at_1": selected["cycle"]["held_out_balanced_r_at_1"],
        "max_similarity_selected_alpha": selected_alphas["max_similarity"],
        "max_similarity_selected_report": selected["max_similarity"]["held_out_report"],
        "max_similarity_selected_balanced_r_at_1": selected["max_similarity"]["held_out_balanced_r_at_1"],
        "max_margin_selected_alpha": selected_alphas["max_margin"],
        "max_margin_selected_report": selected["max_margin"]["held_out_report"],
        "max_margin_selected_balanced_r_at_1": selected["max_margin"]["held_out_balanced_r_at_1"],
        "mean_pool_report": evaluation_rows[0]["held_out_report"],
        "full_temporal_report": evaluation_rows[-1]["held_out_report"],
        "oracle_alpha_for_analysis_only": oracle["alpha"],
        "oracle_report_for_analysis_only": oracle["held_out_report"],
        "calibration_features": str(args.calibration_features),
        "evaluation_features": str(args.evaluation_features),
        "checkpoint": str(args.checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if "sweep" not in key and "report" not in key}, indent=2))


if __name__ == "__main__":
    main()
