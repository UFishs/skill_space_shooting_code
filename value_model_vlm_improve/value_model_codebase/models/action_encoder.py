from __future__ import annotations

import torch
import torch.nn as nn


class ActionChunkEncoder(nn.Module):
    """Encodes a temporal action chunk [B, H, action_dim] into one token."""

    def __init__(self, action_dim: int, token_dim: int, chunk_size: int, layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, token_dim)
        self.pos = nn.Parameter(torch.zeros(1, chunk_size, token_dim))
        block = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        x = self.action_proj(actions) + self.pos[:, : actions.shape[1]]
        x = self.encoder(x)
        return self.norm(x.mean(dim=1))
