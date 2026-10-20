# value_model_codebase/models/dino_encoder.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FrozenDINOEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        out_dim: int = 384,
        image_size: int = 224,
        pretrained: bool = True,
        freeze: bool = True,
    ):
        super().__init__()

        self.image_size = image_size

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=image_size,
        )

        feat_dim = self.backbone.num_features
        self.proj = nn.Linear(feat_dim, out_dim)

        if freeze:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, N, 3, H, W], usually N=3

        Returns:
            tokens: [B, N, out_dim]
        """
        if images.ndim != 5:
            raise ValueError(f"Expected images shape [B, N, 3, H, W], got {tuple(images.shape)}")

        b, n, c, h, w = images.shape
        if c != 3:
            raise ValueError(f"Expected 3 image channels, got {c}")

        flat = images.reshape(b * n, c, h, w)
        if h != self.image_size or w != self.image_size:
            flat = F.interpolate(
                flat,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        with torch.no_grad():
            feats = self.backbone(flat)

        feats = self.proj(feats)
        feats = feats.reshape(b, n, -1)
        return feats