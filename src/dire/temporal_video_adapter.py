"""A lightweight order-aware adapter for frozen video frame embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TemporalVideoOutput:
    """Temporally pooled video embeddings and normalized frame weights."""

    features: torch.Tensor
    attention: torch.Tensor


class TemporalVideoAdapter(nn.Module):
    """Learn temporal attention and a residual projection over frame features.

    The visual backbone is kept frozen.  A depth-wise temporal convolution and
    learned absolute frame positions make the pooled representation sensitive
    to order, while the residual projection preserves the original embedding
    space used by the frozen text encoder.
    """

    def __init__(self, dim: int = 512, hidden_dim: int = 256, max_frames: int = 8) -> None:
        super().__init__()
        if min(dim, hidden_dim, max_frames) < 1:
            raise ValueError("dim, hidden_dim, and max_frames must be positive")
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.max_frames = max_frames
        self.position = nn.Parameter(torch.zeros(max_frames, dim))
        self.temporal_convolution = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.attention = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / 0.07).log())
        nn.init.normal_(self.position, std=0.01)

    def forward(self, frame_features: torch.Tensor) -> TemporalVideoOutput:
        if frame_features.ndim != 3:
            raise ValueError("frame_features must have shape [videos, frames, dim]")
        batch, frames, dim = frame_features.shape
        if dim != self.dim:
            raise ValueError(f"Expected feature dimension {self.dim}, received {dim}")
        if not 1 <= frames <= self.max_frames:
            raise ValueError(f"frames must be between 1 and {self.max_frames}")
        positioned = frame_features + self.position[:frames].unsqueeze(0)
        temporal = positioned + self.temporal_convolution(positioned.transpose(1, 2)).transpose(1, 2)
        weights = self.attention(temporal).squeeze(-1).softmax(dim=1)
        pooled = (weights.unsqueeze(-1) * temporal).sum(dim=1)
        features = F.normalize(pooled + self.residual_scale * self.projection(pooled), dim=-1)
        return TemporalVideoOutput(features=features, attention=weights)

    def scaled_similarity(self, video_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """Return CLIP-style scaled cosine similarities for normalized features."""
        return self.logit_scale.exp().clamp(max=100.0) * video_features @ text_features.T
