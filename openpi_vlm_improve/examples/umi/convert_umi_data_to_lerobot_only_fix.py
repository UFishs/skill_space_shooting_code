"""
Usage:
uv run examples/umi/convert_umi_data_to_lerobot_only_fix.py --data_dir /path/to/your/data --repo_name ${REPO_NAME} 

"""

import os
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from umi_replay_buffer import ReplayBuffer
from imagecodecs_numcodecs import register_codecs
import numpy as np
import tyro
import zarr

from pathlib import Path
import imageio.v2 as imageio

register_codecs()

REPO_NAME = "umi/your-repo-name"  # Name of the output dataset, also used for the Hugging Face Hub

def export_episode_visualization(
    replay_buffer,
    frame_start: int,
    frame_end: int,
    output_dir: str | Path,
    episode_idx: int,
    fps: int = 20,
    image_key: str = "camera0_rgb",
    fmt: str = "mp4",  # "mp4" or "gif"
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert fmt in ["mp4", "gif"], f"Unsupported fmt: {fmt}"

    output_path = output_dir / f"episode_{episode_idx:06d}.{fmt}"

    frames = []
    for idx in range(frame_start, frame_end):
        frame = replay_buffer[image_key][idx]

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        frames.append(frame)

    if fmt == "mp4":
        imageio.mimsave(output_path, frames, fps=fps, macro_block_size=1)
    else:
        imageio.mimsave(output_path, frames, fps=fps)

    return output_path


def convert_umi_data_to_lerobot(dataset_path: str, dataset: LeRobotDataset, data_dir: str, total_start_idx: int = 0):

    with zarr.ZipStore(dataset_path, mode='r') as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store, 
            store=zarr.MemoryStore()
        )

    with open(os.path.join(data_dir, "instruction.txt"), "r") as f:
        instructions = f.readlines()

    episode_ends = replay_buffer.episode_ends[:]
    last_instruction = None
    for i in range(len(episode_ends)):
        start_idx = 0 if i == 0 else episode_ends[i-1]
        end_idx = episode_ends[i]
        this_instruction = instructions[i].strip() if i < len(instructions) else last_instruction
        last_instruction = this_instruction
        
        episode_len = int(end_idx - start_idx)

        global_start_idx = int(total_start_idx)
        global_end_idx = int(total_start_idx + episode_len)
        if this_instruction == 'fail':
            continue


        # Frame-level repair flags written by the robot environment.

        segments = []

        if "is_fix" in replay_buffer:
            fix_mask = np.asarray(
                replay_buffer["is_fix"][start_idx:end_idx]
            ).reshape(-1).astype(bool)

            # Pad with 0 on both sides so that transitions are easy
            # to detect even when the segment starts/ends at episode boundary.
            padded = np.concatenate([
                [False],
                fix_mask,
                [False],
            ])

            transitions = np.diff(padded.astype(np.int8))

            # 0 -> 1: segment starts
            local_starts = np.where(transitions == 1)[0]

            # 1 -> 0: segment ends
            local_ends = np.where(transitions == -1)[0]

            assert len(local_starts) == len(local_ends)

            for local_start, local_end in zip(local_starts, local_ends):
                seg_start = start_idx + int(local_start)
                seg_end = start_idx + int(local_end)

                if seg_end > seg_start:
                    segments.append((seg_start, seg_end))
        
        
        else:
            times = replay_buffer["timestamp"][start_idx:end_idx] - replay_buffer["timestamp"][start_idx]
            times_diff = np.diff(times)

            print("total_len: ", episode_len, 'max_time_diff: ', np.max(times_diff))
            fix_local_idxs = np.where(times_diff > 1.99)[0] + 1
            if len(fix_local_idxs) == 0:
                continue

            # [idx-32, idx+64)
            segments = []
            for local_idx in fix_local_idxs:
                seg_start = max(start_idx, start_idx + int(local_idx) - 32)
                seg_end = min(end_idx, start_idx + int(local_idx) + 64)
                if seg_end > seg_start:
                    segments.append((seg_start, seg_end))

        segments.sort()
        merged_segments = []
        for seg_start, seg_end in segments:
            if not merged_segments or seg_start > merged_segments[-1][1]:
                merged_segments.append([seg_start, seg_end])
            else:
                merged_segments[-1][1] = max(merged_segments[-1][1], seg_end)

        for seg_start, seg_end in merged_segments:
            episode_len = int(seg_end - seg_start)

            global_start_idx = int(total_start_idx)
            global_end_idx = int(total_start_idx + episode_len)

            for current_idx in range(seg_start, seg_end):
                eef_pos = replay_buffer["robot0_eef_pos"][current_idx]
                eef_rot_axis_angle = replay_buffer["robot0_eef_rot_axis_angle"][current_idx]
                gripper_width = replay_buffer["robot0_gripper_width"][current_idx]

                actions = np.concatenate([
                    replay_buffer["robot0_eef_pos"][current_idx],
                    replay_buffer["robot0_eef_rot_axis_angle"][current_idx],
                    replay_buffer["robot0_gripper_width"][current_idx],
                ])

                if "robot0_demo_start_pose" not in replay_buffer:
                    demo_start_pose = np.concatenate([
                        replay_buffer["robot0_eef_pos"][start_idx],
                        replay_buffer["robot0_eef_rot_axis_angle"][start_idx],
                    ])
                    demo_start_pose = np.asarray(demo_start_pose, dtype=np.float32)
                else:
                    demo_start_pose = np.asarray(
                        replay_buffer["robot0_demo_start_pose"][current_idx],
                        dtype=np.float32,
                    )

                dataset.add_frame(
                    {
                        "image": replay_buffer["camera0_rgb"][current_idx],
                        "eef_pos": np.asarray(eef_pos, dtype=np.float32),
                        "eef_rot_axis_angle": np.asarray(eef_rot_axis_angle, dtype=np.float32),
                        "gripper_width": np.asarray(gripper_width, dtype=np.float32),
                        "demo_start_pose": np.asarray(demo_start_pose, dtype=np.float32),
                        "actions": np.asarray(actions, dtype=np.float32),
                        "episode_start_idx": np.asarray([global_start_idx], dtype=np.int64),
                        "episode_end_idx": np.asarray([global_end_idx], dtype=np.int64),
                        "is_partial": np.asarray([True], dtype=bool),
                        "task": this_instruction,
                    }
                )

            dataset.save_episode()
            total_start_idx += episode_len


            # export_episode_visualization(
            #     replay_buffer=replay_buffer,
            #     frame_start=seg_start,
            #     frame_end=seg_end,
            #     output_dir=Path(data_dir) / "debug_episode_videos",
            #     episode_idx=global_start_idx,
            #     fps=20,
            #     fmt="gif",
            # )

    return dataset, total_start_idx

def main(data_dir: str | None = None, vl_data_path_list: str | None = None, vl_number_per_task: str | None = None, vl_task_list: str | None = None, repo_name: str = REPO_NAME, *, push_to_hub: bool = False):
    """
    Convert the UMI dataset to the LeRobot format.

    Args:
        data_dir: Path to the UMI dataset directory.
        repo_name: Name of the output dataset, also used for the Hugging Face Hub.
        push_to_hub: Whether to push the dataset to the Hugging Face Hub.
    """


    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / repo_name
    print("Output path: ", output_path)
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=60,
        features={
            "image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "eef_pos": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["eef_pos"],
            },
            "eef_rot_axis_angle": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["eef_rot_axis_angle"],
            },
            "gripper_width": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_width"],
            },
            "demo_start_pose": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["demo_start_pose"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
            "episode_start_idx": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["episode_start_idx"],
            },
            "episode_end_idx": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["episode_end_idx"],
            },
            "is_partial": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["is_partial"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    maybe_dataset_path = os.path.join(data_dir, 'dataset.zarr.zip')
    if os.path.exists(maybe_dataset_path):
        dataset, total_start_idx = convert_umi_data_to_lerobot(maybe_dataset_path, dataset, data_dir)
        
    else:
        # data_dir has more subdir to get the dataset.zarr.zip
        total_start_idx = 0
        if '+' in data_dir:
            data_dir_list = data_dir.split('+')
            subdirs = []
            for data_dir in data_dir_list:
                subdirs.extend([os.path.join(data_dir, subdir) for subdir in os.listdir(data_dir)])
        else:
            subdirs = [os.path.join(data_dir, subdir) for subdir in os.listdir(data_dir)]

        for subdir in subdirs:
            maybe_dataset_path = os.path.join(subdir, 'dataset.zarr.zip')
            if os.path.exists(maybe_dataset_path):
                dataset, total_start_idx = convert_umi_data_to_lerobot(maybe_dataset_path, dataset, subdir, total_start_idx)
                print(f"Finished processing {maybe_dataset_path}, total frames so far: {total_start_idx}")
            else:
                print(f"Warning: {maybe_dataset_path} does not exist.")


    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )



if __name__ == "__main__":
    tyro.cli(main)
