"""Plot target-only CCTR selector scores against held-out retrieval quality."""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

import matplotlib.pyplot as plt


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def mean_std(values: list[float]) -> tuple[float, float]:
    return stats.mean(values), stats.stdev(values) if len(values) > 1 else 0.0


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        value = (start + 1 + end) / 2.0
        for index in order[start:end]:
            result[index] = value
        start = end
    return result


def spearman(first: list[float], second: list[float]) -> float:
    first_rank, second_rank = rank(first), rank(second)
    first_center = stats.mean(first_rank)
    second_center = stats.mean(second_rank)
    numerator = sum((x - first_center) * (y - second_center) for x, y in zip(first_rank, second_rank))
    first_scale = sum((x - first_center) ** 2 for x in first_rank) ** 0.5
    second_scale = sum((y - second_center) ** 2 for y in second_rank) ** 0.5
    return numerator / (first_scale * second_scale) if first_scale and second_scale else 0.0


def normalize(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high == low:
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--transfers", required=True)
    parser.add_argument("--seeds", type=parse_ints, default=parse_ints("2027,2028,2029"))
    parser.add_argument("--split-seeds", type=parse_ints, default=parse_ints("4101,4102,4103"))
    args = parser.parse_args()

    transfers = [item.strip() for item in args.transfers.split(",") if item.strip()]
    columns = 3
    rows = (len(transfers) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(7.35, 4.7), squeeze=False)
    result: dict[str, object] = {"selector": "max_similarity", "transfers": {}}

    for index, transfer in enumerate(transfers):
        reports = []
        for split_seed in args.split_seeds:
            for seed in args.seeds:
                path = args.metrics_dir / f"{transfer}_split{split_seed}_seed{seed}.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
                reports.append(json.loads(path.read_text(encoding="utf-8")))

        alpha_grid = [float(row["alpha"]) for row in reports[0]["calibration_alpha_sweep"]]
        calibration_by_alpha = [[] for _ in alpha_grid]
        heldout_by_alpha = [[] for _ in alpha_grid]
        correlations = []
        for report in reports:
            calibration = [float(row["mean_top1_similarity"]) for row in report["calibration_alpha_sweep"]]
            heldout = [float(row["held_out_balanced_r_at_1"]) for row in report["evaluation_alpha_sweep_for_analysis_only"]]
            correlations.append(spearman(calibration, heldout))
            for position, value in enumerate(normalize(calibration)):
                calibration_by_alpha[position].append(value)
            for position, value in enumerate(heldout):
                heldout_by_alpha[position].append(value)

        calibration_mean = [mean_std(values)[0] for values in calibration_by_alpha]
        heldout_mean = [100.0 * mean_std(values)[0] for values in heldout_by_alpha]
        heldout_std = [100.0 * mean_std(values)[1] for values in heldout_by_alpha]
        corr_mean, corr_std = mean_std(correlations)
        result["transfers"][transfer] = {
            "reports": len(reports),
            "spearman_calibration_similarity_to_held_out_balanced_r_at_1": {
                "mean": corr_mean,
                "sample_std": corr_std,
            },
            "alpha_grid": alpha_grid,
            "normalized_calibration_similarity_mean": calibration_mean,
            "held_out_balanced_r_at_1_percent_mean": heldout_mean,
            "held_out_balanced_r_at_1_percent_std": heldout_std,
        }

        axis = axes[index // columns][index % columns]
        twin = axis.twinx()
        line1 = axis.plot(alpha_grid, calibration_mean, color="#1f77b4", marker="o", label="Calibration similarity")
        line2 = twin.errorbar(alpha_grid, heldout_mean, yerr=heldout_std, color="#d62728", marker="s", capsize=2, label="Held-out balanced R@1")
        axis.set_title(f"{transfer.replace('_to_', ' -> ')}\nSpearman={corr_mean:.2f} +/- {corr_std:.2f}", fontsize=6.5)
        axis.set_xlabel("Residual strength lambda", fontsize=6.5)
        axis.set_ylabel("Normalized calibration score", color="#1f77b4", fontsize=7)
        twin.set_ylabel("Held-out balanced R@1 (%)", color="#d62728", fontsize=8)
        axis.tick_params(labelsize=7)
        twin.tick_params(labelsize=7)
        if index == 0:
            axis.legend(line1 + [line2], ["Calibration similarity", "Held-out balanced R@1"], fontsize=7, loc="best")

    for index in range(len(transfers), rows * columns):
        axes[index // columns][index % columns].axis("off")

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Plot: {args.output}")
    print(f"Summary: {args.summary_output}")


if __name__ == "__main__":
    main()
