from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import open_clip
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CLIPDIREFeatures:
    global_image_features: torch.Tensor
    patch_features: torch.Tensor
    subject_features: torch.Tensor
    object_features: torch.Tensor


class CLIPPatchBackbone(nn.Module):
    """
    CLIP encoder that exposes:
      global image feature: [B, 512]
      patch features:       [B, 49, 512]
      subject/object text:  [B, 512]
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def project_patch_tokens(
        patch_tokens: torch.Tensor,
        projection: torch.Tensor,
    ) -> torch.Tensor:
        if patch_tokens.ndim != 3:
            raise ValueError("patch_tokens must have shape [B, P, D]")

        if projection.ndim != 2:
            raise ValueError("projection must have shape [D, output_dim]")

        if patch_tokens.shape[-1] != projection.shape[0]:
            raise ValueError("patch dimension and projection dimension differ")

        return patch_tokens @ projection

    def _encode_text(
        self,
        texts: Sequence[str],
        device: torch.device,
    ) -> torch.Tensor:
        tokens = self.tokenizer(list(texts)).to(device)
        return F.normalize(self.model.encode_text(tokens), dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        subjects: Sequence[str],
        objects: Sequence[str],
    ) -> CLIPDIREFeatures:
        if len(subjects) != images.shape[0] or len(objects) != images.shape[0]:
            raise ValueError("one subject and one object are required per image")

        grad_enabled = not self.freeze_backbone

        with torch.set_grad_enabled(grad_enabled):
            visual_output = self.model.visual.forward_intermediates(
                images,
                indices=1,
                output_fmt="NLC",
            )

            patch_tokens = visual_output["image_intermediates"][-1]
            patch_features = self.project_patch_tokens(
                patch_tokens,
                self.model.visual.proj,
            )

            global_image_features = F.normalize(
                visual_output["image_features"],
                dim=-1,
            )
            patch_features = F.normalize(patch_features, dim=-1)

            subject_features = self._encode_text(subjects, images.device)
            object_features = self._encode_text(objects, images.device)

        return CLIPDIREFeatures(
            global_image_features=global_image_features,
            patch_features=patch_features,
            subject_features=subject_features,
            object_features=object_features,
        )
