from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device) * (-math.log(10000.0) / max(half - 1, 1)))
        emb = t * freqs.view(1, -1)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class FlowMLP(nn.Module):
    """Conditional velocity field v_theta(x_t, t, condition)."""

    def __init__(self, latent_dim: int, cond_dim: int, hidden_dim: int = 512, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(hidden_dim)
        in_dim = latent_dim + cond_dim + hidden_dim
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        te = self.time(t)
        return self.net(torch.cat([x_t, te, cond], dim=-1))


def flow_matching_loss(flow: FlowMLP, x1: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], 1, device=x1.device, dtype=x1.dtype)
    x_t = (1.0 - t) * x0 + t * x1
    target_v = x1 - x0
    pred_v = flow(x_t, t, cond)
    return torch.mean((pred_v - target_v) ** 2)


@torch.no_grad()
def euler_integrate(flow: FlowMLP, cond: torch.Tensor, latent_dim: int, steps: int = 16) -> torch.Tensor:
    z = torch.randn(cond.shape[0], latent_dim, device=cond.device, dtype=cond.dtype)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((cond.shape[0], 1), i / steps, device=cond.device, dtype=cond.dtype)
        z = z + dt * flow(z, t, cond)
    return z
