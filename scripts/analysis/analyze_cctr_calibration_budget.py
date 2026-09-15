"""Summarize CCTR performance as a function of target calibration budget."""

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transfers", required=True)
    parser.add_argument("--ratios", type=parse_ints, default=parse_ints("10,25,50"))
    parser.add_argument("--seeds", type=parse_ints, default=parse_ints("2027,2028,2029"))
    parser.add_argument("--split-seeds", type=parse_ints, default=parse_ints("4101,4102,4103"))
    args = parser.parse_args()

    result: dict[str, object] = {
        "ratios_percent": args.ratios,
        "seeds": args.seeds,
        "split_seeds": args.split_seeds,
        "selector": "max_similarity",
        "transfers": {},
    }
    transfers = [item.strip() for item in args.transfers.split(",") if item.strip()]
    for transfer in transfers:
        transfer_result: dict[str, object] = {}
        print(f"\n=== {transfer} ===")
        for ratio in args.ratios:
            reports = []
            for split_seed in args.split_seeds:
                for seed in args.seeds:
                    path = args.metrics_dir / f"ratio{ratio}" / f"{transfer}_split{split_seed}_seed{seed}.json"
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    reports.append(json.loads(path.read_text(encoding="utf-8")))

            item: dict[str, object] = {
                "reports": len(reports),
                "calibration_videos": summarize([float(report["calibration_videos"]) for report in reports]),
                "selected_alpha": summarize([float(report["max_similarity_selected_alpha"]) for report in reports]),
                "directions": {},
            }
            for direction in ("video_to_text", "text_to_video"):
                values = [float(report["max_similarity_selected_report"][direction]["r_at_1"]) for report in reports]
                base = [float(report["mean_pool_report"][direction]["r_at_1"]) for report in reports]
                item["directions"][direction] = {
                    "r_at_1": summarize(values),
                    "delta_vs_mean_r_at_1": summarize([
                        value - baseline for value, baseline in zip(values, base)
                    ]),
                }
            transfer_result[str(ratio)] = item
            v2t = item["directions"]["video_to_text"]["delta_vs_mean_r_at_1"]
            t2v = item["directions"]["text_to_video"]["delta_vs_mean_r_at_1"]
            print(
                f"ratio={ratio}% videos={item['calibration_videos']['mean']:.0f} "
                f"lambda={item['selected_alpha']['mean']:.3f} "
                f"V2T={100*v2t['mean']:+.3f}% T2V={100*t2v['mean']:+.3f}%"
            )
        result["transfers"][transfer] = transfer_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
