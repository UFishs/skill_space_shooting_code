"""
Usage:
uv run examples/umi/export_umi_data_to_videos.py \
  --data_dir /path/to/your/data \
  --output_dir /path/to/output_preview \
  --camera_key camera0_rgb \
  --fps 30
"""

import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
import tyro
import zarr
from imagecodecs_numcodecs import register_codecs
from umi_replay_buffer import ReplayBuffer

register_codecs()


def draw_frame_idx(img_rgb: np.ndarray, frame_idx: int) -> np.ndarray:
    img = img_rgb.copy()
    h, w = img.shape[:2]

    text = f"frame {frame_idx}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = w - tw - 16
    y = th + 16

    cv2.rectangle(img, (x - 8, y - th - 8), (x + tw + 8, y + 8), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img

import imageio.v2 as imageio


def make_preview_gif(frames_dir: Path, gif_path: Path, fps: int):
    frame_paths = sorted(frames_dir.glob("*.png"))

    if len(frame_paths) == 0:
        raise RuntimeError(f"No frames found in {frames_dir}")

    duration = 1.0 / fps

    frames = []

    for frame_path in frame_paths:
        frame = imageio.imread(frame_path)
        frames.append(frame)

    imageio.mimsave(
        gif_path,
        frames,
        duration=duration,
        loop=0,
    )

def export_one_dataset(
    dataset_path: str,
    output_dir: Path,
    total_start_idx: int = 0,
    camera_key: str = "camera0_rgb",
    fps: int = 30,
):
    with zarr.ZipStore(dataset_path, mode="r") as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store,
            store=zarr.MemoryStore(),
        )

    episode_ends = replay_buffer.episode_ends[:]
    images = replay_buffer[camera_key]

    for ep_idx in range(len(episode_ends)):
        start_idx = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
        end_idx = int(episode_ends[ep_idx])

        global_start_idx = total_start_idx + start_idx
        global_end_idx = total_start_idx + end_idx - 1

        ep_dir = output_dir / f"episode_{ep_idx+1}_{global_start_idx:09d}_{global_end_idx:09d}"
        frames_dir = ep_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        for current_idx in range(start_idx, end_idx):
            global_frame_idx = total_start_idx + current_idx
            local_frame_idx = current_idx - start_idx

            img_rgb = images[current_idx]
            img_with_text = draw_frame_idx(img_rgb, local_frame_idx)

            image_path = frames_dir / f"{local_frame_idx:06d}.png"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(img_with_text, cv2.COLOR_RGB2BGR),
            )

        gif_path = ep_dir / "preview.gif"
        make_preview_gif(frames_dir, gif_path, fps)

        print(
            f"Saved episode {ep_idx}: "
            f"global frames {global_start_idx} -> {global_end_idx}, "
            f"video: {gif_path}"
        )

    total_start_idx += int(episode_ends[-1])
    return total_start_idx


def main(
    data_dir: str,
    output_dir: str,
    camera_key: str = "camera0_rgb",
    fps: int = 30,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    maybe_dataset_path = os.path.join(data_dir, "dataset.zarr.zip")

    total_start_idx = 0

    if os.path.exists(maybe_dataset_path):
        total_start_idx = export_one_dataset(
            dataset_path=maybe_dataset_path,
            output_dir=output_dir,
            total_start_idx=total_start_idx,
            camera_key=camera_key,
            fps=fps,
        )
    else:
        if "+" in data_dir:
            data_dir_list = data_dir.split("+")
            subdirs = []
            for one_data_dir in data_dir_list:
                subdirs.extend(
                    [os.path.join(one_data_dir, subdir) for subdir in os.listdir(one_data_dir)]
                )
                # extend itself
                subdirs.extend([one_data_dir])
        else:
            subdirs = [os.path.join(data_dir, subdir) for subdir in os.listdir(data_dir)]
            subdirs.extend([data_dir])

        for subdir in subdirs:
            maybe_dataset_path = os.path.join(subdir, "dataset.zarr.zip")
            if os.path.exists(maybe_dataset_path):
                total_start_idx = export_one_dataset(
                    dataset_path=maybe_dataset_path,
                    output_dir=output_dir,
                    total_start_idx=total_start_idx,
                    camera_key=camera_key,
                    fps=fps,
                )
            else:
                print(f"Warning: {maybe_dataset_path} does not exist.")

    print(f"Done. Total frames: {total_start_idx}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)