from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class RunningStats:
    mean: torch.Tensor
    std: torch.Tensor


class Normalizer:
    """Simple feature normalizer for actions and target states."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.stats: Dict[str, RunningStats] = {}

    def fit_tensor(self, name: str, x: torch.Tensor) -> None:
        x = x.float()
        dims = tuple(range(x.ndim - 1)) if x.ndim > 1 else (0,)
        mean = x.mean(dim=dims)
        std = x.std(dim=dims).clamp_min(self.eps)
        self.stats[name] = RunningStats(mean=mean, std=std)

    def normalize(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name not in self.stats:
            return x
        s = self.stats[name]
        return (x - s.mean.to(x.device)) / s.std.to(x.device)

    def denormalize(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name not in self.stats:
            return x
        s = self.stats[name]
        return x * s.std.to(x.device) + s.mean.to(x.device)

    def state_dict(self) -> dict:
        return {k: {"mean": v.mean, "std": v.std} for k, v in self.stats.items()}

    def load_state_dict(self, state: dict) -> None:
        self.stats = {k: RunningStats(mean=v["mean"], std=v["std"]) for k, v in state.items()}
