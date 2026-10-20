# OpenPI: UMI Data Processing and Policy Fine-Tuning

This repository extends [OpenPI](https://github.com/Physical-Intelligence/openpi) with data-processing and training utilities for UMI-style robot demonstrations. It supports converting recorded demonstrations into LeRobot datasets, extracting intervention/repair segments, preparing value-supervision datasets, and fine-tuning π₀.₅ policies on UMI data.

**Upstream attribution.** The underlying OpenPI framework and models are from [Physical Intelligence's OpenPI](https://github.com/Physical-Intelligence/openpi). Parts of the UMI integration and related code are adapted from [OneTwoVLA](https://github.com/Fanqi-Lin/OneTwoVLA).


## Environment setup

Python 3.11 and an NVIDIA CUDA-capable GPU are recommended for the training pipeline; exact memory requirements depend on the model, checkpoint, and training setup. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

Run commands from the repository root. Set `OPENPI_DATA_HOME` to a writable cache location if desired.

## Prepare UMI data

The UMI converters expect recordings in the form of `dataset.zarr.zip` with corresponding task instructions stored in `instruction.txt`. For example:

```text
/path/to/umi_recording/
├── dataset.zarr.zip
└── instruction.txt
```

### Export recorded camera frames

```bash
uv run examples/umi/export_umi_data_to_videos.py \
  --data_dir /path/to/umi_recording \
  --output_dir /path/to/preview \
  --camera_key camera0_rgb \
  --fps 30
```

### Convert demonstrations to LeRobot

```bash
uv run examples/umi/convert_umi_data_to_lerobot.py \
  --data_dir /path/to/umi_recording \
  --repo_name umi/example_base
```

To use only a selection of demonstrations, pass `--episode_start_idx` and `--episode_end_idx` to the same converter.

### Convert repair segments only

```bash
uv run examples/umi/convert_umi_data_to_lerobot_only_fix.py \
  --data_dir /path/to/repair_recording \
  --repo_name umi/example_repairs
```

When `fix_mask` is present in the replay buffer, this converter treats each contiguous run of marked frames as a repair segment. Otherwise, it falls back to its timestamp-discontinuity heuristic.

### Prepare value-supervision data

```bash
uv run examples/umi/convert_umi_data_to_value_list.py \
  --data_dir /path/to/umi_recording \
  --repo_name umi/example_value_list \
  --action_down_sample_steps 3
```

`action_down_sample_steps` controls the temporal down-sampling of the action sequence. It is usually set to 3 for UMI data, and 1 for rollout data.

## Train a π₀.₅ UMI policy

Training configurations are defined in `src/openpi/training/config.py`. The provided UMI base example is named `pi05_umi_stackthree_base` and uses the dataset ID `umi/stackthree`. **Before training**, replace that dataset ID with your accessible LeRobot dataset, and ensure the checkpoint and other hyperparameters are appropriate for your hardware and experiment.

```bash
# Compute dataset normalization statistics.
uv run scripts/compute_norm_stats.py --config-name pi05_umi_stackthree_base

# Train or fine-tune the policy.
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_umi_stackthree_base \
  --exp-name=example_base
```

For an iterative fine-tuning example, `pi05_umi_stackthree_vm_iter1` combines a base dataset and a repair dataset with configuration-specific sampling settings. **Edit its `repo_id` and `weight_loader` first**: its checkpoint path is only a placeholder and will not work unchanged.

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_umi_stackthree_vm_iter1 \
  --exp-name=example_iteration_1
```

Do not use `--overwrite` unless you intend to replace existing training outputs. The example memory fraction and device selection may need adjustment.

## Acknowledgments

This work builds on **[OpenPI](https://github.com/Physical-Intelligence/openpi)** by Physical Intelligence and adapts portions of **[OneTwoVLA](https://github.com/Fanqi-Lin/OneTwoVLA)** for UMI-related functionality. 