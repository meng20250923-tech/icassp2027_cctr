"""Train one configurable temporal-adapter component ablation."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch.nn import functional as F

from dire.configurable_temporal_adapter import ARCHITECTURES, ConfigurableTemporalAdapter, build_configurable_adapter
from dire.multicaption_retrieval import multicaption_retrieval_report
from dire.video_pipeline import attention_entropy, feature_tensors, load_feature_payload


def evaluate(model: ConfigurableTemporalAdapter, features: dict) -> dict:
    model.eval()
    with torch.no_grad():
        output = model(features["frames"])
        report = multicaption_retrieval_report(model.scaled_similarity(output.features, features["text"]), features["mapping"])
        report["mean_attention_entropy"] = attention_entropy(output.attention)
        report["selection_score"] = (report["video_to_text"]["r_at_1"] + report["text_to_video"]["r_at_1"]) / 2
        return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--dev-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden_dim) < 1:
        raise ValueError("epochs, batch-size, and hidden-dim must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    train = feature_tensors(load_feature_payload(args.train_features), device)
    dev = feature_tensors(load_feature_payload(args.dev_features), device)
    model = build_configurable_adapter(train["frames"].shape[-1], args.hidden_dim, train["frames"].shape[1], args.architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train["frames"]), generator=generator)
        losses = []
        for start in range(0, len(order), args.batch_size):
            video_indices = order[start : start + args.batch_size].to(device)
            caption_indices = torch.stack([
                train["groups"][index][torch.randint(len(train["groups"][index]), (1,), generator=generator).item()]
                for index in video_indices.cpu().tolist()
            ]).to(device)
            output = model(train["frames"].index_select(0, video_indices))
            logits = model.scaled_similarity(output.features, train["text"].index_select(0, caption_indices))
            targets = torch.arange(len(video_indices), device=device)
            loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        dev_report = evaluate(model, dev)
        if best is None or dev_report["selection_score"] > best["dev_report"]["selection_score"]:
            best = {"epoch": epoch, "state": copy.deepcopy(model.state_dict()), "dev_report": dev_report}
        print(f"architecture={args.architecture} epoch={epoch:03d} train_loss={sum(losses) / len(losses):.5f} dev={dev_report}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": best["state"], "dim": model.dim, "hidden_dim": model.hidden_dim, "max_frames": model.max_frames,
            "architecture": args.architecture, "seed": args.seed, "best_epoch": best["epoch"],
            "dev_report": best["dev_report"], "train_features": str(args.train_features), "dev_features": str(args.dev_features),
        },
        args.output,
    )
    print(f"checkpoint={args.output} architecture={args.architecture} best_epoch={best['epoch']} dev={best['dev_report']}")


if __name__ == "__main__":
    main()
