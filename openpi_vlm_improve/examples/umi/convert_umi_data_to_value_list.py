"""
Usage:
uv run examples/umi/convert_umi_data_to_value_list.py \
  --data_dir /path/to/your/data \
  --repo_name umi/your-viva-repo-name
"""

import os
import shutil

import numpy as np
import tyro
import zarr
from imagecodecs_numcodecs import register_codecs
import torch

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi.policies.pose_repr_util import convert_pose_mat_rep
from openpi.policies.pose_util import mat_to_pose10d, pose_to_mat
from umi_replay_buffer import ReplayBuffer

register_codecs()

REPO_NAME = "umi/your-viva-repo-name"


def get_action_chunk(
    replay_buffer,
    idx: int,
    end_idx: int,
    action_horizon: int = 16,
    action_down_sample_steps: int = 3,
) -> np.ndarray:
    slice_end = min(
        end_idx,
        idx + (action_horizon - 1) * action_down_sample_steps + 1,
    )

    actions = []
    for j in range(idx + 1, slice_end, action_down_sample_steps):
        action = np.concatenate(
            [
                replay_buffer["robot0_eef_pos"][j],
                replay_buffer["robot0_eef_rot_axis_angle"][j],
                replay_buffer["robot0_gripper_width"][j],
            ],
            axis=0,
        )
        actions.append(action.astype(np.float32))

    if actions == []:
        action = np.concatenate(
            [
                replay_buffer["robot0_eef_pos"][idx],
                replay_buffer["robot0_eef_rot_axis_angle"][idx],
                replay_buffer["robot0_gripper_width"][idx],
            ],
            axis=0,
        )
        actions.append(action.astype(np.float32))

    actions = np.stack(actions, axis=0)  # [action_horizon, 7]
    action_is_pad = torch.tensor([False] * actions.shape[0] + [True] * (action_horizon - actions.shape[0]))
    padding = np.repeat(actions[-1:], action_horizon - actions.shape[0], axis=0)
    actions = np.concatenate([actions, padding], axis=0)
    pose_mat = pose_to_mat(np.concatenate([replay_buffer['robot0_eef_pos'][idx], replay_buffer['robot0_eef_rot_axis_angle'][idx]]))
    action_mat = pose_to_mat(actions[..., :6])
    action_pose_mat = convert_pose_mat_rep(
        action_mat, 
        base_pose_mat=pose_mat,
        pose_rep='relative',
        backward=False)
    action_pose = mat_to_pose10d(action_pose_mat)    
    action_gripper = actions[..., 6:7]
    final_action = np.concatenate([action_pose, action_gripper], axis=-1) # [action_horizon, 10]

    return final_action.reshape(-1).astype(np.float32)  # [action_horizon * 10]


def get_future_state(
    replay_buffer,
    idx: int,
    end_idx: int,
    action_horizon: int = 16,
    action_down_sample_steps: int = 3,
) -> np.ndarray:
    pose_mat = pose_to_mat(np.concatenate([replay_buffer['robot0_eef_pos'][idx], replay_buffer['robot0_eef_rot_axis_angle'][idx]]))
    future_idx = min(end_idx - 1, idx + (action_horizon - 1) * action_down_sample_steps + 1 + action_down_sample_steps)
    future_mat = pose_to_mat(np.concatenate([replay_buffer['robot0_eef_pos'][future_idx], replay_buffer['robot0_eef_rot_axis_angle'][future_idx]]))
    future_pose_mat = convert_pose_mat_rep(
        future_mat,
        base_pose_mat=pose_mat,
        pose_rep='relative',
        backward=False)
    future_pose = mat_to_pose10d(future_pose_mat)
    future_pose = np.concatenate([future_pose, replay_buffer['robot0_gripper_width'][future_idx]], axis=-1)
    return future_pose.reshape(-1).astype(np.float32)  # [10]

def get_value_list(
    replay_buffer,
    idx: int,
    start_idx: int,
    end_idx: int,
    action_horizon: int = 16,
    action_down_sample_steps: int = 3,
    this_value: int = -1,
) -> np.ndarray:

    # local_idx = idx - start_idx
    # 1 ... this_value - action_horizon: 0.5, good action
    # this_value - action_horizon ... end: -0.5, bad action
    value_list = []

    sample_start_idx = idx
    sample_end_idx = min(end_idx, idx + (action_horizon - 1) * action_down_sample_steps + 1)
    for sample_idx in range(sample_start_idx, sample_end_idx, action_down_sample_steps):
        
        if this_value == -1:
            value = 0.5
        else:
            if sample_idx - start_idx < this_value:
                value = 0.5
            else:
                value = -0.5
        value_list.append(value)
    # If the episode is shorter than the action horizon, we pad the value with last value
    if len(value_list) < action_horizon:
        value_list.extend([value_list[-1]] * (action_horizon - len(value_list)))

    return np.asarray(value_list, dtype=np.float32)


def convert_umi_data_to_viva(
    dataset_path: str,
    dataset: LeRobotDataset,
    data_dir: str,
    total_start_idx: int = 0,
    action_horizon: int = 16,
    action_down_sample_steps: int = 3,
):
    with zarr.ZipStore(dataset_path, mode="r") as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store,
            store=zarr.MemoryStore(),
        )

    instruction_path = os.path.join(data_dir, "instruction.txt")
    if os.path.exists(instruction_path):
        with open(instruction_path, "r") as f:
            instructions = f.readlines()
    else:
        instructions = []
    
    value_path = os.path.join(data_dir, "value.txt")
    if os.path.exists(value_path):
        with open(value_path, "r") as f:
            value_list = f.readlines()
    else:
        value_list = []

    episode_ends = replay_buffer.episode_ends[:]
    last_instruction = ""
    

    for episode_idx in range(len(episode_ends)):
        start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx - 1]
        end_idx = episode_ends[episode_idx]

        this_instruction = (
            instructions[episode_idx].strip()
            if episode_idx < len(instructions)
            else last_instruction
        )
        last_instruction = this_instruction
        this_value = (
            float(value_list[episode_idx].strip())
            if episode_idx < len(value_list)
            else -1
        )



        for current_idx in range(start_idx, end_idx):
            image = replay_buffer["camera0_rgb"][current_idx]

            state = get_action_chunk(
                replay_buffer=replay_buffer,
                idx=current_idx,
                end_idx=end_idx,
                action_horizon=action_horizon,
                action_down_sample_steps=action_down_sample_steps,
            )
            future_state = get_future_state(
                replay_buffer=replay_buffer,
                idx=current_idx,
                end_idx=end_idx,
                action_horizon=action_horizon,
                action_down_sample_steps=action_down_sample_steps,
            )
            value = get_value_list(
                replay_buffer=replay_buffer,
                idx=current_idx,
                start_idx=start_idx,
                end_idx=end_idx,
                action_horizon=action_horizon,
                action_down_sample_steps=action_down_sample_steps,
                this_value=this_value,
            )

            dataset.add_frame(
                {
                    "observation.images.cam_high": image,
                    "observation.images.cam_left_wrist": image,
                    "observation.images.cam_right_wrist": image,
                    "observation.state": state,
                    "observation.future_state": future_state,
                    "observation.value": value,
                    "task": this_instruction,
                }
            )

        dataset.save_episode()

    total_start_idx += int(episode_ends[-1])
    return dataset, total_start_idx


def main(
    data_dir: str,
    repo_name: str = REPO_NAME,
    action_horizon: int = 16,
    action_down_sample_steps: int = 1,
    *,
    push_to_hub: bool = False,
):
    output_path = HF_LEROBOT_HOME / repo_name
    print("Output path:", output_path)

    # if output_path.exists():
    #     shutil.rmtree(output_path)

    state_dim = action_horizon * 10

    if output_path.exists():
        dataset = LeRobotDataset(
            repo_id=repo_name,
            root=output_path,
        )
        total_start_idx = dataset.meta.total_frames
        print("Append to existing dataset")
        print("Existing num_frames:", dataset.meta.total_frames)
        print("Existing num_episodes:", dataset.meta.total_episodes)
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo_name,
            robot_type="panda",
            fps=60,
            features={
                "observation.images.cam_high": {
                    "dtype": "image",
                    "shape": (224, 224, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.images.cam_left_wrist": {
                    "dtype": "image",
                    "shape": (224, 224, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.images.cam_right_wrist": {
                    "dtype": "image",
                    "shape": (224, 224, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (state_dim,),
                    "names": ["action_chunk"],
                },
                "observation.future_state": {
                    "dtype": "float32",
                    "shape": (10,),
                    "names": ["action_dim"],
                },
                "observation.value": {
                    "dtype": "float32",
                    "shape": (action_horizon,),
                    "names": ["value"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )

    maybe_dataset_path = os.path.join(data_dir, "dataset.zarr.zip")

    if os.path.exists(maybe_dataset_path):
        convert_umi_data_to_viva(
            dataset_path=maybe_dataset_path,
            dataset=dataset,
            data_dir=data_dir,
            action_horizon=action_horizon,
            action_down_sample_steps=action_down_sample_steps,
        )
    else:
        total_start_idx = 0

        if "+" in data_dir:
            data_dir_list = data_dir.split("+")
            subdirs = []
            for root in data_dir_list:
                subdirs.extend(
                    [os.path.join(root, subdir) for subdir in os.listdir(root)]
                )
                subdirs.extend([root])

        else:
            subdirs = [os.path.join(data_dir, subdir) for subdir in os.listdir(data_dir)]
            subdirs.extend([data_dir])
            

        for subdir in subdirs:
            maybe_dataset_path = os.path.join(subdir, "dataset.zarr.zip")
            if not os.path.exists(maybe_dataset_path):
                print(f"Warning: {maybe_dataset_path} does not exist.")
                continue

            dataset, total_start_idx = convert_umi_data_to_viva(
                dataset_path=maybe_dataset_path,
                dataset=dataset,
                data_dir=subdir,
                total_start_idx=total_start_idx,
                action_horizon=action_horizon,
                action_down_sample_steps=action_down_sample_steps,
            )
            print(
                f"Finished processing {maybe_dataset_path}, "
                f"total frames so far: {total_start_idx}"
            )



if __name__ == "__main__":
    tyro.cli(main)