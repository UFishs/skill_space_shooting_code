# Value Model for Robotic Manipulation

A conditional flow-matching model that predicts future state and value from three camera images and a candidate action chunk. This repository provides a LeRobot dataset reader, a training entry point, an HTTP prediction server, and an episode visualization tool.

## Installation

Run all commands from the repository root, where `pyproject.toml` is located.

With `uv` installed:

```bash
uv sync --locked
```


## Train with the bundled configuration

Edit `value_model_codebase/configs/train.yaml`:

```yaml
output_dir: runs/value_model

data:
  local_path: data/example_dataset

wandb:
  enabled: false
```

Train with the bundled configuration:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m value_model_codebase.training.train \
  --config value_model_codebase/configs/train.yaml
```

## Serve a trained checkpoint

The server command is the same for all tasks and iterations. `RUN_DIR` must match the training YAML's actual `output_dir`. For the bundled configuration:

```bash
RUN_DIR=runs/value_model
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m value_model_codebase.server \
  --config "$RUN_DIR/resolved_config.yaml" \
  --checkpoint "$RUN_DIR/checkpoint_best.pt" \
  --host 0.0.0.0 \
  --port 8000
```

## Visualize one episode

Visualization requires access to the dataset specified in the YAML:

```bash
RUN_DIR=runs/value_model
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m value_model_codebase.inference.predict \
  --config "$RUN_DIR/resolved_config.yaml" \
  --checkpoint "$RUN_DIR/checkpoint_best.pt" \
  --episode 0 \
  --output result/episode_000000_value.mp4 \
  --plot_gt \
  --save_npz result/episode_000000_predictions.npz
```
