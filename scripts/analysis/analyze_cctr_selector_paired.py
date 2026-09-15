"""Report paired bootstrap intervals among CCTR label-free selectors."""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

import numpy as np


DIRECTIONS = ("video_to_text", "text_to_video")
CUTOFFS = (1, 5, 10)


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def ranks(report: dict, key: str, direction: str) -> np.ndarray:
    values = np.asarray(report[key][direction], dtype=np.int64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError(f"{key}.{direction} must be a non-empty rank vector")
    return values


def bootstrap_ci(deltas: list[np.ndarray], repetitions: int, seed: int) -> list[float]:
    generator = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        estimates[index] = float(np.mean([
            values[generator.integers(0, len(values), size=len(values))].mean()
            for values in deltas
        ]))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def comparison(reports: list[dict], reference: str, candidate: str, repetitions: int, seed: int) -> dict:
    result: dict[str, object] = {}
    for direction in DIRECTIONS:
        result[direction] = {}
        for cutoff in CUTOFFS:
            deltas = [
                (ranks(report, candidate, direction) <= cutoff).astype(np.float64)
                - (ranks(report, reference, direction) <= cutoff).astype(np.float64)
                for report in reports
            ]
            means = [float(values.mean()) for values in deltas]
            result[direction][f"r_at_{cutoff}"] = {
                "delta_mean": stats.mean(means),
                "delta_sample_std": stats.stdev(means) if len(means) > 1 else 0.0,
                "paired_bootstrap_95_ci": bootstrap_ci(deltas, repetitions, seed),
                "bootstrap_unit": "examples resampled within each source-adapter and target-split report",
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_ints, default=parse_ints("2027,2028,2029"))
    parser.add_argument("--split-seeds", type=parse_ints, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    args = parser.parse_args()
    result: dict[str, object] = {
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.bootstrap_seed,
        "transfers": {},
    }
    comparisons = {
        "cycle_minus_mean_pool": ("mean_pool_ranks", "cycle_ranks"),
        "max_similarity_minus_mean_pool": ("mean_pool_ranks", "max_similarity_ranks"),
        "max_margin_minus_mean_pool": ("mean_pool_ranks", "max_margin_ranks"),
        "max_similarity_minus_cycle": ("cycle_ranks", "max_similarity_ranks"),
        "max_margin_minus_cycle": ("cycle_ranks", "max_margin_ranks"),
    }
    for transfer in ("msrvtt_to_msvd", "msvd_to_msrvtt"):
        reports = []
        for split_seed in args.split_seeds:
            for seed in args.seeds:
                path = args.ranks_dir / f"{transfer}_split{split_seed}_seed{seed}.json"
                reports.append(json.loads(path.read_text(encoding="utf-8")))
        result["transfers"][transfer] = {
            name: comparison(reports, reference, candidate, args.bootstrap_repetitions, args.bootstrap_seed)
            for name, (reference, candidate) in comparisons.items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for transfer, rows in result["transfers"].items():
        print(f"\n=== {transfer} ===")
        for name, values in rows.items():
            print(name)
            for direction in DIRECTIONS:
                row = values[direction]["r_at_1"]
                lower, upper = row["paired_bootstrap_95_ci"]
                print(f"{direction} R@1: {100*row['delta_mean']:+.3f}% [{100*lower:+.3f}%, {100*upper:+.3f}%]")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
