"""Aggregate source-validation fixed-lambda CCTR transfer baselines."""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": stats.mean(values),
        "sample_std": stats.stdev(values) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument(
        "--cctr-metrics-dir",
        type=Path,
        help="Optional matching CCTR reports for paired comparisons.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transfers", required=True)
    parser.add_argument("--seeds", type=parse_ints, default=parse_ints("2027,2028,2029"))
    parser.add_argument("--split-seeds", type=parse_ints, default=parse_ints("4101,4102,4103"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing summary: {args.output}. "
            "Choose a new output path for an independent run."
        )

    result: dict[str, object] = {
        "seeds": args.seeds,
        "split_seeds": args.split_seeds,
        "transfers": {},
        "aggregation_unit": "all source-adapter seeds and all disjoint target split seeds",
        "cctr_metrics_dir": str(args.cctr_metrics_dir) if args.cctr_metrics_dir else None,
    }
    for transfer in [item.strip() for item in args.transfers.split(",") if item.strip()]:
        reports = []
        for split_seed in args.split_seeds:
            for seed in args.seeds:
                path = args.metrics_dir / f"{transfer}_split{split_seed}_seed{seed}.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
                reports.append(json.loads(path.read_text(encoding="utf-8")))

        item: dict[str, object] = {
            "reports": len(reports),
            "source_selected_alpha": summarize([
                float(report["source_validation_selected_alpha"]) for report in reports
            ]),
            "systems": {},
        }
        systems = {
            "mean_pool": "mean_pool_report",
            "full_temporal": "full_temporal_report",
            "source_fixed": "source_validation_selected_report",
        }
        for name, key in systems.items():
            row: dict[str, object] = {}
            for direction in ("video_to_text", "text_to_video"):
                values = [float(report[key][direction]["r_at_1"]) for report in reports]
                base = [float(report["mean_pool_report"][direction]["r_at_1"]) for report in reports]
                row[direction] = {
                    "r_at_1": summarize(values),
                    "delta_vs_mean_r_at_1": summarize([
                        value - baseline for value, baseline in zip(values, base)
                    ]),
                }
            item["systems"][name] = row

        if args.cctr_metrics_dir:
            cctr_reports = []
            for split_seed in args.split_seeds:
                for seed in args.seeds:
                    path = args.cctr_metrics_dir / f"{transfer}_split{split_seed}_seed{seed}.json"
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    cctr_reports.append(json.loads(path.read_text(encoding="utf-8")))
            cctr_row: dict[str, object] = {}
            cctr_balanced: list[list[float]] = []
            source_balanced: list[list[float]] = []
            for direction in ("video_to_text", "text_to_video"):
                cctr_values = [
                    float(report["max_similarity_selected_report"][direction]["r_at_1"])
                    for report in cctr_reports
                ]
                source_values = [
                    float(report["source_validation_selected_report"][direction]["r_at_1"])
                    for report in reports
                ]
                cctr_row[direction] = {
                    "r_at_1": summarize(cctr_values),
                    "cctr_minus_source_fixed_r_at_1": summarize([
                        value - baseline for value, baseline in zip(cctr_values, source_values)
                    ]),
                }
                cctr_balanced.append(cctr_values)
                source_balanced.append(source_values)
            item["systems"]["cctr"] = cctr_row
            cctr_balanced_values = [sum(values) / 2.0 for values in zip(*cctr_balanced)]
            source_balanced_values = [sum(values) / 2.0 for values in zip(*source_balanced)]
            item["cctr_minus_source_fixed_balanced_r_at_1"] = summarize([
                value - baseline for value, baseline in zip(cctr_balanced_values, source_balanced_values)
            ])
        result["transfers"][transfer] = item

        fixed = item["systems"]["source_fixed"]
        v2t = fixed["video_to_text"]["delta_vs_mean_r_at_1"]
        t2v = fixed["text_to_video"]["delta_vs_mean_r_at_1"]
        alpha = item["source_selected_alpha"]
        print(f"\n=== {transfer} ===")
        print(f"source_alpha={alpha['mean']:.3f} +/- {alpha['sample_std']:.3f}")
        print(
            f"source_fixed: V2T={100*v2t['mean']:+.3f}% +/- {100*v2t['sample_std']:.3f}% "
            f"T2V={100*t2v['mean']:+.3f}% +/- {100*t2v['sample_std']:.3f}%"
        )
        if args.cctr_metrics_dir:
            paired = item["cctr_minus_source_fixed_balanced_r_at_1"]
            print(
                f"CCTR-source_fixed balanced R@1={100*paired['mean']:+.3f}% "
                f"+/- {100*paired['sample_std']:.3f}%"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
