from __future__ import annotations

import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    """Fuses image tokens and the action token into a condition vector."""

    def __init__(self, image_dim: int, action_dim: int, cond_dim: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, cond_dim)
        self.action_proj = nn.Linear(action_dim, cond_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, cond_dim))
        self.type_embed = nn.Parameter(torch.zeros(1, 8, cond_dim))
        block = nn.TransformerEncoderLayer(
            d_model=cond_dim,
            nhead=heads,
            dim_feedforward=cond_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(cond_dim)

    def forward(self, image_tokens: torch.Tensor, action_token: torch.Tensor) -> torch.Tensor:
        b, cams, _ = image_tokens.shape
        img = self.image_proj(image_tokens)
        act = self.action_proj(action_token).unsqueeze(1)
        cls = self.cls.expand(b, -1, -1)
        tokens = torch.cat([cls, img, act], dim=1)
        tokens = tokens + self.type_embed[:, : tokens.shape[1]]
        tokens = self.encoder(tokens)
        return self.norm(tokens[:, 0])
