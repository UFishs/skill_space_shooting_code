from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from value_model_codebase.models.action_encoder import ActionChunkEncoder
from value_model_codebase.models.condition_encoder import ConditionEncoder
from value_model_codebase.models.dino_encoder import FrozenDINOEncoder
from value_model_codebase.models.flow_model import FlowMLP, euler_integrate, flow_matching_loss
from value_model_codebase.models.heads import PredictionHeads
from value_model_codebase.training.losses import value_loss_fn


@dataclass
class ValueModelConfig:
    image_size: int
    dino_model: str
    dino_dim: int
    freeze_dino: bool
    action_dim: int
    chunk_size: int
    action_token_dim: int
    cond_dim: int
    target_state_dim: int
    target_value_dim: int
    state_latent_dim: int
    value_latent_dim: int
    cond_layers: int = 4
    cond_heads: int = 8
    flow_hidden_dim: int = 512
    flow_depth: int = 4
    dropout: float = 0.1


class ValueFlowModel(nn.Module):
    """No state_t input. Condition = frozen DINO image tokens + action chunk token."""

    def __init__(self, cfg: ValueModelConfig):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = FrozenDINOEncoder(cfg.dino_model, cfg.dino_dim, freeze=cfg.freeze_dino)
        self.action_encoder = ActionChunkEncoder(
            action_dim=cfg.action_dim,
            token_dim=cfg.action_token_dim,
            chunk_size=cfg.chunk_size,
            dropout=cfg.dropout,
        )
        self.condition_encoder = ConditionEncoder(
            image_dim=cfg.dino_dim,
            action_dim=cfg.action_token_dim,
            cond_dim=cfg.cond_dim,
            layers=cfg.cond_layers,
            heads=cfg.cond_heads,
            dropout=cfg.dropout,
        )
        self.state_flow = FlowMLP(cfg.state_latent_dim, cfg.cond_dim, cfg.flow_hidden_dim, cfg.flow_depth, cfg.dropout)
        self.value_flow = FlowMLP(cfg.value_latent_dim, cfg.cond_dim, cfg.flow_hidden_dim, cfg.flow_depth, cfg.dropout)
        self.heads = PredictionHeads(
            cond_dim=cfg.cond_dim,
            state_latent_dim=cfg.state_latent_dim,
            value_latent_dim=cfg.value_latent_dim,
            target_state_dim=cfg.target_state_dim,
            target_value_dim=cfg.target_value_dim,
            hidden_dim=cfg.flow_hidden_dim,
            dropout=cfg.dropout,
        )

    def encode_condition(self, images: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        image_tokens = self.image_encoder(images)
        action_token = self.action_encoder(actions)
        return self.condition_encoder(image_tokens, action_token)

    def forward_train(
        self,
        images: torch.Tensor,
        actions: torch.Tensor,
        target_state: torch.Tensor,
        value: torch.Tensor,
        lambda_state: float = 1.0,
        lambda_flow_state: float = 1.0,
        lambda_flow_value: float = 0.25,
        lambda_value: float = 1.0,
        value_loss: str = "bce",
    ) -> Dict[str, torch.Tensor]:
        # import ipdb; ipdb.set_trace()
        cond = self.encode_condition(images, actions)
        z_state_target = self.heads.encode_state_target(target_state)
        z_value_target = self.heads.encode_value_target(value)

        loss_flow_state = flow_matching_loss(self.state_flow, z_state_target, cond)
        loss_flow_value = flow_matching_loss(self.value_flow, z_value_target, cond)

        # Supervised shortcut heads stabilize early training.
        state_pred = self.heads.decode_state(cond, z_state_target.detach())
        value_pred = self.heads.predict_value(cond, z_value_target.detach())
        loss_state = F.mse_loss(state_pred, target_state)
        loss_value = value_loss_fn(value_pred, value, value_loss)
        loss = (
            lambda_flow_state * loss_flow_state
            + lambda_flow_value * loss_flow_value
            + lambda_state * loss_state
            + lambda_value * loss_value
        )
        return {
            "loss": loss,
            "loss_flow_state": loss_flow_state.detach(),
            "loss_flow_value": loss_flow_value.detach(),
            "loss_state": loss_state.detach(),
            "loss_value": loss_value.detach(),
        }

    @torch.no_grad()
    def predict(self, images: torch.Tensor, actions: torch.Tensor, ode_steps: int = 16) -> Dict[str, torch.Tensor]:
        cond = self.encode_condition(images, actions)
        z_state = euler_integrate(self.state_flow, cond, self.cfg.state_latent_dim, steps=ode_steps)
        z_value = euler_integrate(self.value_flow, cond, self.cfg.value_latent_dim, steps=ode_steps)
        state = self.heads.decode_state(cond, z_state)
        value_score = self.heads.predict_value(cond, z_value)
        return {
            "future_state": state,
            # Raw scalar. With the default MSE setup this is trained to match your saved -0.5 / 0.5 labels.
            "value": value_score,
            # Useful only if you train with 0 / 1 labels and value_loss=bce.
            "value_probability": torch.sigmoid(value_score),
        }
