"""Extract frozen OpenCLIP frame and caption features from a generic manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dire.backbone import CLIPPatchBackbone
from dire.video import VideoCaptionPairDataset
from dire.video_pipeline import load_multicaption_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--frames-per-video", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.batch_size, args.text_batch_size, args.frames_per_video) < 1:
        raise ValueError("Batch sizes and frames-per-video must be positive")
    device = torch.device("cuda")
    videos, captions = load_multicaption_manifest(args.manifest)
    backbone = CLIPPatchBackbone(freeze_backbone=True).to(device).eval()
    loader = DataLoader(
        VideoCaptionPairDataset(videos, args.video_dir, backbone.preprocess, args.frames_per_video),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    frame_chunks = []
    with torch.no_grad():
        for number, (indices, frames) in enumerate(loader, start=1):
            expected = torch.arange(indices[0], indices[0] + len(indices))
            if not torch.equal(indices.cpu(), expected):
                raise RuntimeError("Video DataLoader changed the declared order")
            batch, count = frames.shape[:2]
            features = F.normalize(backbone.model.encode_image(frames.flatten(0, 1).to(device, non_blocking=True)), dim=-1)
            frame_chunks.append(features.view(batch, count, -1).cpu())
            if number % 25 == 0 or number == len(loader):
                print(f"encoded_videos={number}/{len(loader)}", flush=True)
        text_chunks = []
        caption_text = [row["caption"] for row in captions]
        for start in range(0, len(caption_text), args.text_batch_size):
            batch = caption_text[start : start + args.text_batch_size]
            text_chunks.append(F.normalize(backbone._encode_text(batch, device), dim=-1).cpu())
            completed = min(start + len(batch), len(caption_text))
            if completed % (args.text_batch_size * 25) == 0 or completed == len(caption_text):
                print(f"encoded_captions={completed}/{len(caption_text)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "frame_features": torch.cat(frame_chunks).half(),
            "text_features": torch.cat(text_chunks).half(),
            "caption_video_indices": torch.tensor([row["video_index"] for row in captions], dtype=torch.long),
            "video_ids": [row["video_id"] for row in videos],
            "captions": caption_text,
            "frames_per_video": args.frames_per_video,
            "representation": "frozen_openclip_vit_b_32",
            "manifest": str(args.manifest),
        },
        args.output,
    )
    print(json.dumps({"videos": len(videos), "captions": len(captions), "frames_per_video": args.frames_per_video, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
