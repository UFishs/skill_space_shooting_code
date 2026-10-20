from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import yaml

from value_model_codebase.datasets.lerobot_chunk_dataset import LeRobotChunkDataset
from value_model_codebase.models.value_model import ValueFlowModel, ValueModelConfig
from value_model_codebase.training.checkpoint import save_checkpoint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def build_dataset(cfg: Dict[str, Any]) -> LeRobotChunkDataset:
    d = cfg["data"]
    return LeRobotChunkDataset(
        repo_id=d.get("repo_id"),
        local_path=d.get("local_path"),
        split=d.get("split", "train"),
        image_keys=d.get("image_keys"),
        action_chunk_key=d.get("action_chunk_key", "observation.state"),
        future_state_key=d.get("future_state_key", "observation.future_state"),
        value_key=d.get("value_key", "observation.value"),
        chunk_size=int(d.get("chunk_size", 16)),
        action_dim=int(d.get("action_dim", 10)),
        image_size=int(d.get("image_size", 224)),
        max_episodes=d.get("max_episodes"),
    )


def split_dataset(dataset: LeRobotChunkDataset, val_ratio: float, seed: int) -> Tuple[Subset, Subset]:
    """Shuffle once, then split 9:1 by default."""
    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_size = max(1, int(round(n * val_ratio))) if n > 1 and val_ratio > 0 else 0
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if len(train_indices) == 0:
        train_indices, val_indices = indices, []
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def build_model_config(cfg: Dict[str, Any], target_state_dim: int, target_value_dim: int) -> ValueModelConfig:
    m = dict(cfg["model"])
    m["target_state_dim"] = int(m.get("target_state_dim") or target_state_dim)
    m["target_value_dim"] = int(m.get("target_value_dim") or target_value_dim)
    return ValueModelConfig(**m)


def init_wandb(cfg: Dict[str, Any], output_dir: Path):
    wandb_cfg = cfg.get("wandb", {})
    enabled = bool(wandb_cfg.get("enabled", False))
    if not enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb.enabled=true but wandb is not installed. Run `uv pip install wandb`.") from exc

    run = wandb.init(
        project=wandb_cfg.get("project", "value_model"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name"),
        tags=wandb_cfg.get("tags"),
        notes=wandb_cfg.get("notes"),
        dir=str(output_dir),
        config=cfg,
        mode=wandb_cfg.get("mode", "online"),
    )
    return run


def make_loss_kwargs(train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "lambda_state": float(train_cfg.get("lambda_state", 1.0)),
        "lambda_flow_state": float(train_cfg.get("lambda_flow_state", 1.0)),
        "lambda_flow_value": float(train_cfg.get("lambda_flow_value", 0.25)),
        "lambda_value": float(train_cfg.get("lambda_value", 1.0)),
        "value_loss": str(train_cfg.get("value_loss", "mse")),
    }


@torch.no_grad()
def validate(
    model: ValueFlowModel,
    loader: DataLoader,
    device: torch.device,
    train_cfg: Dict[str, Any],
) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    count = 0
    loss_kwargs = make_loss_kwargs(train_cfg)
    for batch in tqdm(loader, desc="validation", leave=False):
        batch = move_batch(batch, device)
        losses = model.forward_train(
            images=batch["images"],
            actions=batch["actions"],
            target_state=batch["target_state"],
            value=batch["value"],
            **loss_kwargs,
        )
        bs = int(batch["images"].shape[0])
        count += bs
        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach().cpu()) * bs
    if count == 0:
        return {}
    return {f"val/{k}": v / count for k, v in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    output_dir = Path(cfg.get("output_dir", "runs/value_fm"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    dataset = build_dataset(cfg)
    target_state_dim = dataset.target_state_dim
    target_value_dim = dataset.target_value_dim
    model_cfg = build_model_config(cfg, target_state_dim=target_state_dim, target_value_dim=target_value_dim)

    train_cfg = cfg["train"]
    val_ratio = float(cfg["data"].get("val_ratio", 0.1))
    train_set, val_set = split_dataset(dataset, val_ratio=val_ratio, seed=seed)

    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=bool(train_cfg.get("drop_last", True)),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(train_cfg.get("val_batch_size", train_cfg.get("batch_size", 64))),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    ) if len(val_set) > 0 else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ValueFlowModel(model_cfg).to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-2)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda")

    wandb_run = init_wandb(cfg, output_dir)
    if wandb_run is not None:
        import wandb
        wandb.define_metric("train/step")
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("val/*", step_metric="train/step")
        wandb.log({
            "data/num_samples": len(dataset),
            "data/num_train": len(train_set),
            "data/num_val": len(val_set),
            "data/val_ratio": val_ratio,
        }, step=0)

    global_step = 0
    metrics_path = output_dir / "metrics.jsonl"
    best_val_loss = float("inf")
    loss_kwargs = make_loss_kwargs(train_cfg)

    for epoch in range(1, int(train_cfg.get("epochs", 50)) + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        running: Dict[str, float] = {}
        running_count = 0

        for batch in pbar:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                losses = model.forward_train(
                    images=batch["images"],
                    actions=batch["actions"],
                    target_state=batch["target_state"],
                    value=batch["value"],
                    **loss_kwargs,
                )
            scaler.scale(losses["loss"]).backward()
            if float(train_cfg.get("grad_clip_norm", 0.0)) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
            scaler.step(opt)
            scaler.update()

            bs = int(batch["images"].shape[0])
            running_count += bs
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach().cpu()) * bs

            global_step += 1
            if global_step % int(train_cfg.get("log_every", 50)) == 0:
                row = {"epoch": epoch, "step": global_step, "lr": opt.param_groups[0]["lr"]}
                row.update({f"train/{k}": float(v.detach().cpu()) for k, v in losses.items()})
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                if wandb_run is not None:
                    import wandb
                    wandb.log({**row, "train/step": global_step}, step=global_step)
                pbar.set_postfix({k: f"{v:.4f}" for k, v in row.items() if k.endswith("loss") or k.endswith("/loss")})

        epoch_metrics = {f"train_epoch/{k}": v / max(1, running_count) for k, v in running.items()}
        epoch_metrics.update({"epoch": epoch, "step": global_step})

        val_metrics: Dict[str, float] = {}
        if val_loader is not None and (epoch % int(train_cfg.get("validate_every", 1)) == 0):
            val_metrics = validate(model, val_loader, device, train_cfg)
            epoch_metrics.update(val_metrics)

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_metrics) + "\n")
        if wandb_run is not None:
            import wandb
            wandb.log({**epoch_metrics, "train/step": global_step}, step=global_step)

        if "val/loss" in val_metrics and val_metrics["val/loss"] < best_val_loss:
            best_val_loss = val_metrics["val/loss"]
            save_checkpoint(
                output_dir / "checkpoint_best.pt",
                model=model.state_dict(),
                optimizer=opt.state_dict(),
                model_config=model_cfg.__dict__,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
            )

        if epoch % int(train_cfg.get("save_every", 5)) == 0 or epoch == int(train_cfg.get("epochs", 50)):
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch}.pt",
                model=model.state_dict(),
                optimizer=opt.state_dict(),
                model_config=model_cfg.__dict__,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
            )

    save_checkpoint(
        output_dir / "checkpoint_last.pt",
        model=model.state_dict(),
        optimizer=opt.state_dict(),
        model_config=model_cfg.__dict__,
        epoch=epoch,
        global_step=global_step,
        best_val_loss=best_val_loss,
    )
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
