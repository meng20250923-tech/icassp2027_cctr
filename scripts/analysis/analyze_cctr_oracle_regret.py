"""Summarize CCTR's analysis-only oracle gap from saved held-out sweeps.

The oracle uses held-out video--caption labels solely as a diagnostic upper
bound. It is never a deployment-time selector and this utility never reruns
retrieval or changes any saved report.
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

DEFAULT_TRANSFERS = (
    "msrvtt_to_msvd,msrvtt_to_vatex,msvd_to_msrvtt,"
    "msvd_to_vatex,vatex_to_msrvtt,vatex_to_msvd"
)


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def mean_std(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": stats.mean(values),
        "sample_std": stats.stdev(values) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def fraction_at_most(values: list[float], threshold: float) -> dict[str, float | int]:
    count = sum(value <= threshold + 1e-12 for value in values)
    return {"count": count, "total": len(values), "fraction": count / len(values)}


def selected_score_from_sweep(report: dict[str, object], alpha: float) -> float:
    for row in report["evaluation_alpha_sweep_for_analysis_only"]:
        if abs(float(row["alpha"]) - alpha) < 1e-9:
            return float(row["held_out_balanced_r_at_1"])
    raise ValueError(f"selected alpha={alpha} is absent from the saved sweep")


def run_diagnostic(report: dict[str, object]) -> dict[str, float]:
    cctr_alpha = float(report["max_similarity_selected_alpha"])
    cctr_score = selected_score_from_sweep(report, cctr_alpha)
    oracle_alpha = float(report["oracle_alpha_for_analysis_only"])
    oracle_score = selected_score_from_sweep(report, oracle_alpha)
    regret = 100.0 * (oracle_score - cctr_score)
    if regret < -1e-7:
        raise ValueError("oracle score is lower than the CCTR-selected score")
    return {
        "cctr_alpha": cctr_alpha,
        "oracle_alpha": oracle_alpha,
        "absolute_alpha_difference": abs(cctr_alpha - oracle_alpha),
        "cctr_balanced_r_at_1": cctr_score,
        "oracle_balanced_r_at_1": oracle_score,
        "oracle_regret_points": max(0.0, regret),
    }


def summarize(rows: list[dict[str, float]], thresholds: list[float]) -> dict[str, object]:
    regrets = [row["oracle_regret_points"] for row in rows]
    distances = [row["absolute_alpha_difference"] for row in rows]
    return {
        "oracle_regret_points": mean_std(regrets),
        "absolute_alpha_difference": mean_std(distances),
        "within_oracle_points": {
            str(threshold): fraction_at_most(regrets, threshold)
            for threshold in thresholds
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_ints, default=parse_ints("2027,2028,2029"))
    parser.add_argument("--split-seeds", type=parse_ints, default=parse_ints("4101,4102,4103"))
    parser.add_argument("--transfers", default=DEFAULT_TRANSFERS)
    parser.add_argument(
        "--thresholds-points",
        type=parse_floats,
        default=parse_floats("0.25,0.5,1.0"),
        help="Oracle-regret thresholds in balanced R@1 percentage points.",
    )
    args = parser.parse_args()

    transfers = [item.strip() for item in args.transfers.split(",") if item.strip()]
    per_run: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "protocol": (
            "Analysis-only oracle diagnostic. Oracle coefficients use held-out "
            "video--caption labels and are not deployable selectors."
        ),
        "metrics_dir": str(args.metrics_dir),
        "seeds": args.seeds,
        "split_seeds": args.split_seeds,
        "thresholds_points": args.thresholds_points,
        "transfers": {},
    }

    all_rows: list[dict[str, float]] = []
    for transfer in transfers:
        transfer_rows: list[dict[str, float]] = []
        for split_seed in args.split_seeds:
            for seed in args.seeds:
                report_path = args.metrics_dir / f"{transfer}_split{split_seed}_seed{seed}.json"
                if not report_path.is_file():
                    raise FileNotFoundError(report_path)
                diagnostic = run_diagnostic(json.loads(report_path.read_text(encoding="utf-8")))
                transfer_rows.append(diagnostic)
                all_rows.append(diagnostic)
                per_run.append({
                    "transfer": transfer,
                    "split_seed": split_seed,
                    "source_seed": seed,
                    **diagnostic,
                })
        summary["transfers"][transfer] = summarize(transfer_rows, args.thresholds_points)

    summary["macro"] = summarize(all_rows, args.thresholds_points)
    summary["per_run"] = per_run
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("=== CCTR oracle-regret diagnostic ===")
    for transfer in transfers:
        item = summary["transfers"][transfer]
        regret = item["oracle_regret_points"]
        distance = item["absolute_alpha_difference"]
        print(
            f"{transfer}: regret={regret['mean']:.3f} +/- {regret['sample_std']:.3f} points, "
            f"|lambda_CCTR-lambda_oracle|={distance['mean']:.3f} +/- {distance['sample_std']:.3f}"
        )

    macro = summary["macro"]
    regret = macro["oracle_regret_points"]
    distance = macro["absolute_alpha_difference"]
    print("\n=== Macro average over all matched reports ===")
    print(
        f"CCTR oracle regret: {regret['mean']:.3f} +/- {regret['sample_std']:.3f} points\n"
        f"Mean |lambda_CCTR-lambda_oracle|: {distance['mean']:.3f} +/- {distance['sample_std']:.3f}"
    )
    for threshold in args.thresholds_points:
        item = macro["within_oracle_points"][str(threshold)]
        print(
            f"Within {threshold:.2f} oracle points: "
            f"{item['count']}/{item['total']} ({100.0 * item['fraction']:.1f}%)"
        )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
