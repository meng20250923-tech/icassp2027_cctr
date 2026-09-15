"""Plot selector reliability across all directed CCTR transfers."""
from __future__ import annotations
import argparse
import json
import statistics as stats
from pathlib import Path
import matplotlib.pyplot as plt

def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]

def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i]); result = [0.0] * len(values); start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]: end += 1
        value = (start + 1 + end) / 2.0
        for i in order[start:end]: result[i] = value
        start = end
    return result

def spearman(first: list[float], second: list[float]) -> float:
    x, y = rank(first), rank(second); xbar, ybar = stats.mean(x), stats.mean(y)
    numerator = sum((a - xbar) * (b - ybar) for a, b in zip(x, y))
    denominator = (sum((a - xbar) ** 2 for a in x) * sum((b - ybar) ** 2 for b in y)) ** 0.5
    return numerator / denominator if denominator else 0.0

def display_transfer(name: str) -> str:
    return name.replace("msrvtt", "MSR-VTT").replace("msvd", "MSVD").replace("vatex", "VATEX").replace("_to_", " -> ")

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
    means, stds, details = [], [], {}
    for transfer in transfers:
        correlations = []
        for split_seed in args.split_seeds:
            for seed in args.seeds:
                report = json.loads((args.metrics_dir / f"{transfer}_split{split_seed}_seed{seed}.json").read_text(encoding="utf-8"))
                calibration = [float(row["mean_top1_similarity"]) for row in report["calibration_alpha_sweep"]]
                heldout = [float(row["held_out_balanced_r_at_1"]) for row in report["evaluation_alpha_sweep_for_analysis_only"]]
                correlations.append(spearman(calibration, heldout))
        means.append(stats.mean(correlations)); stds.append(stats.stdev(correlations))
        details[transfer] = {"reports": len(correlations), "mean_spearman": means[-1], "sample_std": stds[-1]}
    order = list(range(len(transfers)))[::-1]; labels = [display_transfer(transfers[i]) for i in order]
    figure, axis = plt.subplots(figsize=(6.7, 3.35))
    axis.axvline(0, color="#9aa5ad", linewidth=0.8, zorder=0)
    axis.errorbar([means[i] for i in order], range(len(order)), xerr=[stds[i] for i in order], fmt="o", markersize=6.2, capsize=3.2, linewidth=1.25, color="#2E5F88", ecolor="#7D9AAF", markerfacecolor="#2E5F88", markeredgecolor="white", markeredgewidth=0.7, zorder=3)
    axis.set_yticks(range(len(order)), labels)
    axis.set_xlabel("Spearman correlation: calibration similarity vs. held-out balanced R@1")
    axis.set_xlim(-0.08, 1.05); axis.set_ylim(-0.7, len(order) - 0.3)
    axis.grid(axis="x", color="#D9E1E7", linewidth=0.7); axis.tick_params(axis="both", labelsize=9); axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(pad=0.4); args.output.parent.mkdir(parents=True, exist_ok=True); figure.savefig(args.output, dpi=350, bbox_inches="tight")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True); args.summary_output.write_text(json.dumps(details, indent=2) + "\\n", encoding="utf-8")
if __name__ == "__main__": main()
