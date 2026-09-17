"""Configurable lightweight temporal adapter for component ablations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURES = {
    "full": (True, True, True),
    "attention_only": (False, False, True),
    "no_position": (False, True, True),
    "no_temporal_convolution": (True, False, True),
    "no_residual_projection": (True, True, False),
}


@dataclass
class ConfigurableTemporalOutput:
    """Temporally pooled video embeddings and normalized frame weights."""

    features: torch.Tensor
    attention: torch.Tensor


class ConfigurableTemporalAdapter(nn.Module):
    """Temporal adapter whose position and convolution components can be removed."""

    def __init__(
        self,
        dim: int = 512,
        hidden_dim: int = 256,
        max_frames: int = 8,
        use_position: bool = True,
        use_temporal_convolution: bool = True,
        use_residual_projection: bool = True,
    ) -> None:
        super().__init__()
        if min(dim, hidden_dim, max_frames) < 1:
            raise ValueError("dim, hidden_dim, and max_frames must be positive")
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.max_frames = max_frames
        self.use_position = use_position
        self.use_temporal_convolution = use_temporal_convolution
        self.use_residual_projection = use_residual_projection
        if use_position:
            self.position = nn.Parameter(torch.zeros(max_frames, dim))
            nn.init.normal_(self.position, std=0.01)
        else:
            self.register_buffer("position", torch.zeros(max_frames, dim), persistent=False)
        self.temporal_convolution: nn.Module
        if use_temporal_convolution:
            self.temporal_convolution = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        else:
            self.temporal_convolution = nn.Identity()
        self.attention = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        if use_residual_projection:
            self.projection: nn.Module | None = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)
            )
            self.residual_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.projection = None
            self.register_buffer("residual_scale", torch.tensor(0.0), persistent=False)
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / 0.07).log())

    def forward(self, frame_features: torch.Tensor) -> ConfigurableTemporalOutput:
        if frame_features.ndim != 3:
            raise ValueError("frame_features must have shape [videos, frames, dim]")
        _, frames, dim = frame_features.shape
        if dim != self.dim:
            raise ValueError(f"Expected feature dimension {self.dim}, received {dim}")
        if not 1 <= frames <= self.max_frames:
            raise ValueError(f"frames must be between 1 and {self.max_frames}")
        positioned = frame_features + self.position[:frames].unsqueeze(0)
        temporal = positioned + self.temporal_convolution(positioned.transpose(1, 2)).transpose(1, 2)
        weights = self.attention(temporal).squeeze(-1).softmax(dim=1)
        pooled = (weights.unsqueeze(-1) * temporal).sum(dim=1)
        if self.projection is None:
            features = F.normalize(pooled, dim=-1)
        else:
            features = F.normalize(pooled + self.residual_scale * self.projection(pooled), dim=-1)
        return ConfigurableTemporalOutput(features=features, attention=weights)

    def scaled_similarity(self, video_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0) * video_features @ text_features.T


def architecture_flags(architecture: str) -> tuple[bool, bool]:
    """Return ``(use_position, use_temporal_convolution)`` for an ablation name."""
    try:
        return ARCHITECTURES[architecture]
    except KeyError as error:
        raise ValueError(f"Unknown architecture: {architecture}; choices={sorted(ARCHITECTURES)}") from error


def build_configurable_adapter(dim: int, hidden_dim: int, max_frames: int, architecture: str) -> ConfigurableTemporalAdapter:
    """Construct an adapter from checkpoint-compatible metadata."""
    use_position, use_temporal_convolution, use_residual_projection = architecture_flags(architecture)
    return ConfigurableTemporalAdapter(
        dim,
        hidden_dim,
        max_frames,
        use_position,
        use_temporal_convolution,
        use_residual_projection,
    )
