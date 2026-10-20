from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from value_model_codebase.datasets.lerobot_chunk_dataset import LeRobotChunkDataset
from value_model_codebase.models.value_model import ValueFlowModel, ValueModelConfig
from value_model_codebase.training.checkpoint import load_checkpoint


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _to_python_scalar(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.detach().cpu().item()
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return x.item()
    return x


def to_hwc_uint8(img: Any) -> np.ndarray:
    """Convert PIL / torch / numpy image to RGB uint8 HWC for visualization."""
    if isinstance(img, Image.Image):
        arr = np.asarray(img.convert("RGB"))
    elif isinstance(img, torch.Tensor):
        x = img.detach().cpu()
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0)
        arr = x.numpy()
    elif isinstance(img, dict) and "path" in img:
        arr = np.asarray(Image.open(img["path"]).convert("RGB"))
    elif isinstance(img, str):
        arr = np.asarray(Image.open(img).convert("RGB"))
    else:
        arr = np.asarray(img)

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if np.nanmax(arr) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset(cfg: Dict[str, Any]) -> LeRobotChunkDataset:
    data_cfg = cfg["data"]
    return LeRobotChunkDataset(
        repo_id=data_cfg.get("repo_id"),
        local_path=data_cfg.get("local_path"),
        split=data_cfg.get("split", "train"),
        image_keys=data_cfg.get("image_keys"),
        action_chunk_key=data_cfg.get("action_chunk_key", "observation.state"),
        future_state_key=data_cfg.get("future_state_key", "observation.future_state"),
        value_key=data_cfg.get("value_key", "observation.value"),
        chunk_size=int(data_cfg.get("chunk_size", 16)),
        action_dim=int(data_cfg.get("action_dim", 10)),
        image_size=int(data_cfg.get("image_size", 224)),
        max_episodes=data_cfg.get("max_episodes"),
    )


def load_model(checkpoint_path: str | Path, device: torch.device) -> ValueFlowModel:
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    if "model_config" not in ckpt:
        raise KeyError(
            "Checkpoint must contain `model_config`. Please use checkpoints saved by "
            "value_model_codebase.training.train."
        )
    model_cfg = ValueModelConfig(**ckpt["model_config"])
    model = ValueFlowModel(model_cfg).to(device)

    state_dict = ckpt.get("model", ckpt)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def get_episode_id(row: Dict[str, Any]) -> Optional[int]:
    for key in ("episode_index", "episode_id", "episode", "traj_id"):
        if key in row:
            val = _to_python_scalar(row[key])
            try:
                return int(val)
            except Exception:
                return val
    return None

def build_episode_table(dataset):
    """
    Fast episode table builder.

    Supports:
    1. LeRobotDataset with episode_data_index
    2. Wrapped dataset with .lerobot_dataset
    3. Viva-style dataset with _subdataset_episode_lengths
    """
    # unwrap if needed
    ds = getattr(dataset, "lerobot_dataset", dataset)

    if hasattr(dataset, "_subdataset_episode_lengths"):
        table = []
        global_offset = 0
        for sub_ds_idx, sub_lengths in enumerate(dataset._subdataset_episode_lengths):
            for ep_idx, ep_len in enumerate(sub_lengths):
                table.append({
                    "sub_ds_idx": sub_ds_idx,
                    "episode_index": ep_idx,
                    "episode_length": int(ep_len),
                    "global_start": int(global_offset),
                    "global_end": int(global_offset + ep_len),
                })
                global_offset += int(ep_len)
        return table

    # LeRobotDataset fast path
    if hasattr(ds, "episode_data_index"):
        edi = ds.episode_data_index

        # common LeRobot format:
        # episode_data_index["from"] / ["to"]
        if isinstance(edi, dict) and "from" in edi and "to" in edi:
            starts = edi["from"]
            ends = edi["to"]

            if hasattr(starts, "tolist"):
                starts = starts.tolist()
            if hasattr(ends, "tolist"):
                ends = ends.tolist()

            return [
                {
                    "episode_index": i,
                    "episode_length": int(end - start),
                    "global_start": int(start),
                    "global_end": int(end),
                }
                for i, (start, end) in enumerate(zip(starts, ends))
            ]

        # fallback for list-like episode_data_index
        table = []
        for i, item in enumerate(edi):
            start = int(item["from"])
            end = int(item["to"])
            table.append({
                "episode_index": i,
                "episode_length": end - start,
                "global_start": start,
                "global_end": end,
            })
        return table

    raise AttributeError(
        "Cannot build episode table quickly: dataset has neither "
        "`_subdataset_episode_lengths` nor `episode_data_index`."
    )

def find_episode(table: List[Dict[str, Any]], episode: int) -> Dict[str, Any]:
    for ep in table:
        if int(ep["episode_index"]) == int(episode):
            return ep
    available = [ep["episode_index"] for ep in table[:50]]
    raise ValueError(f"Episode {episode} not found. First available episodes: {available}")


def extract_frame(dataset: LeRobotChunkDataset, local_idx: int, image_key: str) -> np.ndarray:
    raw_idx = dataset.indices[local_idx]
    row = dataset._row(raw_idx)
    if image_key not in row:
        image_key = dataset.image_keys[0]
    return to_hwc_uint8(row[image_key])


@torch.no_grad()
def infer_episode(
    model: ValueFlowModel,
    dataset: LeRobotChunkDataset,
    ep_info: Dict[str, Any],
    device: torch.device,
    ode_steps: int,
    frame_key: str,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run frame-by-frame inference.

    Returns:
        pred_values: [T]
        gt_values: [T]
        pred_targets: [T, target_state_dim]
        frames: [T, H, W, 3]
    """
    start = int(ep_info["global_start"])
    end = int(ep_info["global_end"])
    if max_frames is not None:
        end = min(end, start + int(max_frames))

    pred_values: List[float] = []
    gt_values: List[float] = []
    pred_targets: List[np.ndarray] = []
    frames: List[np.ndarray] = []

    for local_idx in tqdm(range(start, end), desc=f"episode {ep_info['episode_index']} inference"):
        sample = dataset[local_idx]
        images = sample["images"].unsqueeze(0).to(device=device, dtype=torch.float32)
        actions = sample["actions"].unsqueeze(0).to(device=device, dtype=torch.float32)

        out = model.predict(images, actions, ode_steps=ode_steps)
        pred_value = out["value"].reshape(-1)[0].detach().float().cpu()
        # pred_value is logits
        pred_value = torch.sigmoid(pred_value).item()
        pred_target = out["future_state"].reshape(1, -1)[0].detach().float().cpu().numpy()

        pred_values.append(float(pred_value))
        gt_values.append(float(sample["value"].reshape(-1)[0].detach().float().cpu().item()))
        pred_targets.append(pred_target)
        frames.append(extract_frame(dataset, local_idx, frame_key))

    return (
        np.asarray(pred_values, dtype=np.float32),
        np.asarray(gt_values, dtype=np.float32),
        np.asarray(pred_targets, dtype=np.float32),
        np.asarray(frames),
    )


def render_video(
    frames: np.ndarray,
    pred_values: np.ndarray,
    output_path: str | Path,
    fps: int = 30,
    title: str = "",
    gt_values: Optional[np.ndarray] = None,
    y_min: float = -1.0,
    y_max: float = 1.0,
) -> None:
    """Render original frames on top and an evolving value curve below."""
    output_path = str(output_path)
    T = min(len(frames), len(pred_values))
    if T == 0:
        raise ValueError("No frames/predictions to render.")
    frames = frames[:T]
    pred_values = pred_values[:T]
    if gt_values is not None:
        gt_values = gt_values[:T]

    h_img, w_img = frames[0].shape[:2]
    fig_h = (h_img + h_img // 2) / 100
    fig_w = w_img / 100
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    fig.subplots_adjust(right=0.84)
    gs = fig.add_gridspec(2, 1, hspace=0.25, height_ratios=[2, 1])

    ax_video = fig.add_subplot(gs[0, 0])
    ax_video.axis("off")
    img_handle = ax_video.imshow(frames[0])
    if title:
        ax_video.set_title(title, fontsize=10)

    ax_plot = fig.add_subplot(gs[1, 0])
    xs = np.arange(T)
    ax_plot.set_xlim(0, max(1, T - 1))
    ax_plot.set_ylim(y_min, y_max)
    ax_plot.set_xlabel("Timestep", fontsize=8)
    ax_plot.set_ylabel("Value", fontsize=8)
    ax_plot.grid(True, linewidth=0.3, alpha=0.4)

    line_pred, = ax_plot.plot([], [], color="green", linewidth=1.6, label="pred")
    line_gt = None
    if gt_values is not None:
        ax_plot.plot(xs, gt_values, color="lightgray", linewidth=0.8, alpha=0.7, label="gt full")
        line_gt, = ax_plot.plot([], [], color="black", linewidth=1.0, alpha=0.8, label="gt")
        ax_plot.legend(fontsize=7, loc="lower left")

    value_text = ax_plot.text(
        1.01,
        0.5,
        "",
        transform=ax_plot.transAxes,
        fontsize=8,
        va="center",
        ha="left",
    )
    vline = ax_plot.axvline(0, linestyle="--", color="gray", linewidth=0.8, alpha=0.6)

    fig.canvas.draw()
    W, H = fig.canvas.get_width_height()
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        plt.close(fig)
        raise RuntimeError(f"Cannot open VideoWriter for {output_path}")

    for t in tqdm(range(T), desc="render video"):
        img_handle.set_data(frames[t])
        line_pred.set_data(xs[: t + 1], pred_values[: t + 1])
        if line_gt is not None and gt_values is not None:
            line_gt.set_data(xs[: t + 1], gt_values[: t + 1])
        vline.set_xdata([t, t])

        if gt_values is None:
            value_text.set_text(f"pred={pred_values[t]:.3f}")
        else:
            value_text.set_text(f"pred={pred_values[t]:.3f}\ngt={gt_values[t]:.3f}")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        writer.write(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))

    writer.release()
    plt.close(fig)
    logger.info("Video saved to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize ValueFlowModel value predictions on one LeRobot episode."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to train/resolved config yaml.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint_best.pt or checkpoint_epoch_xxx.pt.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to visualize.")
    parser.add_argument("--output", type=str, default=None, help="Output mp4 path.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ode_steps", type=int, default=16)
    parser.add_argument("--frame_key", type=str, default="observation.images.cam_high")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--plot_gt", action="store_true", help="Also plot dataset observation.value as ground truth.")
    parser.add_argument("--save_npz", type=str, default=None, help="Optional path to save predictions/targets as npz.")
    parser.add_argument("--y_min", type=float, default=-1.0)
    parser.add_argument("--y_max", type=float, default=1.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = build_dataset(cfg)
    logger.info("Loaded dataset with %d rows", len(dataset))

    table = build_episode_table(dataset)
    logger.info("Found %d episodes", len(table))
    ep_info = find_episode(table, args.episode)
    logger.info(
        "Processing episode %s, length=%d",
        ep_info["episode_index"],
        ep_info["episode_length"],
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_model(args.checkpoint, device=device)

    pred_values, gt_values, pred_targets, frames = infer_episode(
        model=model,
        dataset=dataset,
        ep_info=ep_info,
        device=device,
        ode_steps=args.ode_steps,
        frame_key=args.frame_key,
        max_frames=args.max_frames,
    )
    logger.info(
        "Inference done. pred value range: [%.4f, %.4f]",
        float(pred_values.min()),
        float(pred_values.max()),
    )

    if args.save_npz:
        npz_path = Path(args.save_npz)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            pred_values=pred_values,
            gt_values=gt_values,
            pred_targets=pred_targets,
            episode=np.asarray([args.episode], dtype=np.int64),
        )
        logger.info("Saved npz to %s", npz_path)

    if args.output is None:
        output_dir = Path("result")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"episode_{args.episode:06d}_value.mp4"
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    render_video(
        frames=frames,
        pred_values=pred_values,
        gt_values=gt_values if args.plot_gt else None,
        output_path=output_path,
        fps=args.fps,
        title=f"Episode {args.episode}",
        y_min=args.y_min,
        y_max=args.y_max,
    )


if __name__ == "__main__":
    main()
