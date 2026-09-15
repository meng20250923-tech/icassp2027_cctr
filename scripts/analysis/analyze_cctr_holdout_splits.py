"""Aggregate CCTR selector baselines over target calibration/test splits."""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path


SYSTEMS = {
    "mean_pool": "mean_pool_report",
    "full_temporal": "full_temporal_report",
    "cycle": "cycle_selected_report",
    "max_similarity": "max_similarity_selected_report",
    "max_margin": "max_margin_selected_report",
}
CALIBRATION_METRICS = {
    "cycle": "cycle_balanced",
    "max_similarity": "mean_top1_similarity",
    "max_margin": "mean_top1_margin",
}


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": stats.mean(values),
        "sample_std": stats.stdev(values) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def average_ranks(values: list[float]) -> list[float]:
    """Return average ranks for ties, using one-based rank values."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def spearman_correlation(first: list[float], second: list[float]) -> float:
    """Compute Spearman correlation without an optional SciPy dependency."""
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("inputs must have the same length of at least two")
    first_rank = average_ranks(first)
    second_rank = average_ranks(second)
    first_mean = stats.mean(first_rank)
    second_mean = stats.mean(second_rank)
    numerator = sum((x - first_mean) * (y - second_mean) for x, y in zip(first_rank, second_rank))
    first_scale = sum((x - first_mean) ** 2 for x in first_rank) ** 0.5
    second_scale = sum((y - second_mean) ** 2 for y in second_rank) ** 0.5
    return numerator / (first_scale * second_scale) if first_scale and second_scale else 0.0


def matching_held_out_scores(report: dict, metric: str) -> tuple[list[float], list[float]]:
    calibration = {float(row["alpha"]): float(row[metric]) for row in report["calibration_alpha_sweep"]}
    evaluation = {
        float(row["alpha"]): float(row["held_out_balanced_r_at_1"])
        for row in report["evaluation_alpha_sweep_for_analysis_only"]
    }
    alphas = sorted(calibration.keys() & evaluation.keys())
    return [calibration[alpha] for alpha in alphas], [evaluation[alpha] for alpha in alphas]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_ints, default=parse_ints("2027,2028,2029"))
    parser.add_argument("--split-seeds", type=parse_ints, required=True)
    parser.add_argument("--transfers", default="msrvtt_to_msvd,msvd_to_msrvtt")
    args = parser.parse_args()
    transfers = [item.strip() for item in args.transfers.split(",") if item.strip()]
    summary: dict[str, object] = {
        "seeds": args.seeds,
        "split_seeds": args.split_seeds,
        "aggregation_unit": "all source-adapter seeds and all disjoint target split seeds",
        "transfers": {},
    }
    for transfer in transfers:
        reports = []
        for split_seed in args.split_seeds:
            for seed in args.seeds:
                path = args.metrics_dir / f"{transfer}_split{split_seed}_seed{seed}.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
                reports.append(json.loads(path.read_text(encoding="utf-8")))

        item: dict[str, object] = {
            "reports": len(reports),
            "systems": {},
            "selected_alpha": {},
            "calibration_to_held_out_spearman": {},
        }
        for name, report_key in SYSTEMS.items():
            metrics: dict[str, object] = {}
            for direction in ("video_to_text", "text_to_video"):
                scores = [float(report[report_key][direction]["r_at_1"]) for report in reports]
                base = [float(report["mean_pool_report"][direction]["r_at_1"]) for report in reports]
                metrics[direction] = {
                    "r_at_1": mean_std(scores),
                    "delta_vs_mean_r_at_1": mean_std([score - value for score, value in zip(scores, base)]),
                }
            item["systems"][name] = metrics
        for name, metric in CALIBRATION_METRICS.items():
            item["selected_alpha"][name] = mean_std([
                float(report[f"{name}_selected_alpha"]) for report in reports
            ])
            correlations = [
                spearman_correlation(*matching_held_out_scores(report, metric))
                for report in reports
            ]
            item["calibration_to_held_out_spearman"][name] = mean_std(correlations)
        summary["transfers"][transfer] = item
        print(f"\n=== {transfer} ({len(reports)} reports) ===")
        for name in ("full_temporal", "cycle", "max_similarity", "max_margin"):
            system = item["systems"][name]
            v2t = system["video_to_text"]["delta_vs_mean_r_at_1"]
            t2v = system["text_to_video"]["delta_vs_mean_r_at_1"]
            print(
                f"{name}: V2T={100*v2t['mean']:+.3f}% +/- {100*v2t['sample_std']:.3f}% "
                f"T2V={100*t2v['mean']:+.3f}% +/- {100*t2v['sample_std']:.3f}%"
            )
        for name, values in item["calibration_to_held_out_spearman"].items():
            print(f"{name}_spearman={values['mean']:.3f} +/- {values['sample_std']:.3f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
