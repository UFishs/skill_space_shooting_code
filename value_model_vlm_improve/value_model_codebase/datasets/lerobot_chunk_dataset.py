from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from value_model_codebase.utils.image_transforms import build_image_transform


@dataclass
class ChunkSample:
    images: torch.Tensor        # [num_cams, 3, H, W]
    actions: torch.Tensor       # [chunk_size, action_dim]
    target_state: torch.Tensor  # [future_state_dim]
    value: torch.Tensor         # [1]


class LeRobotChunkDataset(Dataset):
    """Reader for the LeRobot dataset produced by convert_umi_data_to_viva.py.

    Your converter stores the model inputs/targets in each single LeRobot frame:

    - observation.images.cam_high
    - observation.images.cam_left_wrist
    - observation.images.cam_right_wrist
    - observation.state: flattened action chunk, shape [action_horizon * 10]
    - observation.future_state: saved future state target, shape [10]
    - observation.value: saved scalar value target, shape [1]

    Important: this dataset class does NOT reconstruct future_state/value by looking
    at neighboring rows. It uses the values already saved by the converter.
    """

    def __init__(
        self,
        repo_id: Optional[str] = None,
        local_path: Optional[str] = None,
        split: str = "train",
        image_keys: Optional[List[str]] = None,
        action_chunk_key: str = "observation.state",
        future_state_key: str = "observation.future_state",
        value_key: str = "observation.value",
        chunk_size: int = 16,
        action_dim: int = 10,
        image_size: int = 224,
        max_episodes: Optional[int] = None,
    ) -> None:
        super().__init__()
        if image_keys is None:
            image_keys = [
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            ]

        self.image_keys = image_keys
        self.action_chunk_key = action_chunk_key
        self.future_state_key = future_state_key
        self.value_key = value_key
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.transform = build_image_transform(image_size)

        self.dataset = self._load_dataset(repo_id=repo_id, local_path=local_path, split=split)
        self.indices = self._build_indices(max_episodes=max_episodes)

        if len(self.indices) == 0:
            raise RuntimeError("No rows found in the LeRobot dataset.")

        self._subdataset_episode_lengths = []
        max_episode_length = 0
        sub_lengths = []
        meta = self.dataset.meta
        total_episodes = meta.info["total_episodes"]
        for episode_idx in range(total_episodes):
            length = int(meta.episodes[episode_idx]['length'])
            sub_lengths.append(length)
            max_episode_length = max(max_episode_length, length)
        self._subdataset_episode_lengths.append(sub_lengths)


        self._validate_first_sample()

    def _load_dataset(self, repo_id: Optional[str], local_path: Optional[str], split: str):
        """Load either a native LeRobotDataset or a Hugging Face datasets table.

        Prefer LeRobotDataset because the converter created the dataset with
        `lerobot.common.datasets.lerobot_dataset.LeRobotDataset.create(...)`.
        Fall back to `datasets` for exported/pushed datasets.
        """
        if repo_id is None and local_path is None:
            raise ValueError("Either repo_id or local_path must be provided.")

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            if local_path:
                root = Path(local_path).expanduser()
                # LeRobotDataset expects a repo_id even for local datasets in many versions.
                # Use the directory name as a harmless fallback if the caller did not pass one.
                lr_repo_id = repo_id or root.name
                return LeRobotDataset(repo_id=lr_repo_id, root=root)
            return LeRobotDataset(repo_id=repo_id)
        except Exception as lerobot_exc:
            try:
                from datasets import load_dataset, load_from_disk
            except ImportError as datasets_exc:
                raise ImportError(
                    "Install `lerobot` or `datasets` to read this dataset."
                ) from datasets_exc

            try:
                if local_path:
                    path = Path(local_path).expanduser()
                    if path.exists() and (path / "dataset_info.json").exists():
                        return load_from_disk(str(path))
                    ds = load_dataset(str(path))
                else:
                    ds = load_dataset(repo_id)
                if isinstance(ds, dict):
                    if split in ds:
                        return ds[split]
                    if "train" in ds:
                        return ds["train"]
                return ds
            except Exception as datasets_exc:
                raise RuntimeError(
                    "Failed to load dataset with both LeRobotDataset and datasets. "
                    f"LeRobot error: {lerobot_exc}; datasets error: {datasets_exc}"
                ) from datasets_exc

    def _row(self, idx: int) -> Dict[str, Any]:
        row = self.dataset[idx]
        if not isinstance(row, dict):
            raise TypeError(f"Expected dataset row to be a dict, got {type(row)}")
        return row

    def _episode_id(self, row: Dict[str, Any]) -> Any:
        for key in ("episode_index", "episode_id", "episode", "traj_id"):
            if key in row:
                return row[key]
        return None

    def _build_indices(self, max_episodes: Optional[int]) -> List[int]:
        if max_episodes is None:
            return list(range(len(self.dataset)))

        valid: List[int] = []
        seen = set()
        for i in range(len(self.dataset)):
            ep = self._episode_id(self._row(i))
            if ep is not None:
                seen.add(ep)
                if len(seen) > max_episodes:
                    break
            valid.append(i)
        return valid

    def __len__(self) -> int:
        return len(self.indices)

    def _to_image(self, value: Any) -> Image.Image:
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, dict) and "path" in value:
            return Image.open(value["path"]).convert("RGB")
        if isinstance(value, str):
            return Image.open(value).convert("RGB")
        arr = np.asarray(value)
        if arr.ndim == 3:
            if arr.dtype != np.uint8:
                # Support both [0, 1] and [0, 255] arrays.
                if np.nanmax(arr) <= 1.0:
                    arr = arr * 255.0
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.shape[0] == 3:
                arr = arr.transpose(1, 2, 0)  # [3, H, W] -> [H, W, 3]
            return Image.fromarray(arr).convert("RGB")
        raise TypeError(f"Unsupported image value type: {type(value)}")

    def _to_float_tensor(self, value: Any) -> torch.Tensor:
        # Some HF image/array features come back as lists; numpy handles both.
        return torch.as_tensor(np.asarray(value), dtype=torch.float32)

    def _get_required(self, row: Dict[str, Any], key: str) -> Any:
        if key not in row:
            available = ", ".join(list(row.keys())[:40])
            raise KeyError(f"Missing key `{key}`. First available keys: {available}")
        return row[key]

    def _validate_first_sample(self) -> None:
        row = self._row(self.indices[0])
        for key in self.image_keys:
            self._get_required(row, key)
        action_chunk = self._to_float_tensor(self._get_required(row, self.action_chunk_key)).view(-1)
        expected = self.chunk_size * self.action_dim
        if action_chunk.numel() != expected:
            raise ValueError(
                f"`{self.action_chunk_key}` must have {expected} values "
                f"for chunk_size={self.chunk_size}, action_dim={self.action_dim}; "
                f"got {action_chunk.numel()}."
            )
        self._get_required(row, self.future_state_key)
        self._get_required(row, self.value_key)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self._row(self.indices[idx])

        images = []
        for key in self.image_keys:
            images.append(self.transform(self._to_image(self._get_required(row, key))))
        images_t = torch.stack(images, dim=0)

        # Converter stores the already-built action chunk in observation.state.
        # Shape in the LeRobot feature is [160]; model expects [16, 10].
        action_chunk = self._to_float_tensor(self._get_required(row, self.action_chunk_key)).view(-1)
        expected = self.chunk_size * self.action_dim
        if action_chunk.numel() != expected:
            raise ValueError(
                f"Expected flattened action chunk with {expected} values at row {self.indices[idx]}, "
                f"got {action_chunk.numel()}."
            )
        actions_t = action_chunk.view(self.chunk_size, self.action_dim)

        # Use future_state and value exactly as saved in the dataset.
        target_state = self._to_float_tensor(self._get_required(row, self.future_state_key)).view(-1)
        value = self._to_float_tensor(self._get_required(row, self.value_key)).view(-1)
        # value = torch.where(value > 0.0, torch.ones_like(value), torch.zeros_like(value))
        # print(f"Loaded value: {value.item()} at row {self.indices[idx]}")
        # if value.numel() != 1:
        #     raise ValueError(f"`{self.value_key}` must contain one scalar, got shape {tuple(value.shape)}")

        return {
            "images": images_t,
            "actions": actions_t,
            "target_state": target_state,
            "value": value.float(),
        }

    @property
    def target_state_dim(self) -> int:
        row = self._row(self.indices[0])
        return int(self._to_float_tensor(self._get_required(row, self.future_state_key)).numel())
    
    @property
    def target_value_dim(self) -> int:
        row = self._row(self.indices[0])
        return int(self._to_float_tensor(self._get_required(row, self.value_key)).numel())
