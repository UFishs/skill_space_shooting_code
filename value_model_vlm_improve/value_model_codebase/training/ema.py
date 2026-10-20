from __future__ import annotations

import copy

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if k in msd and torch.is_floating_point(v):
                v.copy_(v * self.decay + msd[k] * (1.0 - self.decay))
            elif k in msd:
                v.copy_(msd[k])
