from __future__ import annotations

import torch
import torch.nn.functional as F


def value_loss_fn(pred: torch.Tensor, target: torch.Tensor, kind: str = "bce") -> torch.Tensor:
    pred = pred.squeeze(-1)
    target = target.float().view_as(pred)
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(pred, target)
    if kind == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(f"Unknown value loss: {kind}")
