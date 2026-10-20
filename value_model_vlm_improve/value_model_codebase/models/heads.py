from __future__ import annotations

import torch
import torch.nn as nn


def mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2, dropout: float = 0.1) -> nn.Sequential:
    layers = []
    d = in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden_dim),  nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),]

        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class PredictionHeads(nn.Module):
    def __init__(self, cond_dim: int, state_latent_dim: int, value_latent_dim: int, target_state_dim: int, target_value_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.state_decoder = mlp(cond_dim + state_latent_dim, hidden_dim, target_state_dim, depth=2, dropout=dropout)
        self.value_head = nn.Sequential(
            nn.LayerNorm(cond_dim + value_latent_dim),
            mlp(
                cond_dim + value_latent_dim,
                hidden_dim,
                target_value_dim,
                depth=2,
                dropout=dropout,
            ),
        )
        self.state_encoder = mlp(target_state_dim, hidden_dim, state_latent_dim, depth=2, dropout=dropout)
        self.value_encoder = mlp(target_value_dim, hidden_dim, value_latent_dim, depth=2, dropout=dropout)

    def encode_state_target(self, target_state: torch.Tensor) -> torch.Tensor:
        return self.state_encoder(target_state)

    def encode_value_target(self, value: torch.Tensor) -> torch.Tensor:
        return self.value_encoder(value.float())

    def decode_state(self, cond: torch.Tensor, z_state: torch.Tensor) -> torch.Tensor:
        return self.state_decoder(torch.cat([cond, z_state], dim=-1))

    def predict_value(self, cond: torch.Tensor, z_value: torch.Tensor) -> torch.Tensor:
        return self.value_head(torch.cat([cond, z_value], dim=-1))
