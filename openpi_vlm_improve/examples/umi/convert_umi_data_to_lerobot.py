"""
Usage:
uv run examples/umi/convert_umi_data_to_lerobot.py --data_dir /path/to/your/data --repo_name ${REPO_NAME} 

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

register_codecs()

REPO_NAME = "umi/your-repo-name"  # Name of the output dataset, also used for the Hugging Face Hub

def convert_umi_data_to_lerobot(dataset_path: str, dataset: LeRobotDataset, data_dir: str, total_start_idx: int = 0, episode_start_idx: int = -1, episode_end_idx: int = -1):

    with zarr.ZipStore(dataset_path, mode='r') as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store, 
            store=zarr.MemoryStore()
        )

    with open(os.path.join(data_dir, "instruction.txt"), "r") as f:
        instructions = f.readlines()

    episode_ends = replay_buffer.episode_ends[:]
    last_instruction = instructions[0].strip() if len(instructions) > 0 else "fail"

    index_start_idx = 0
    index_end_idx = len(episode_ends)
    if episode_start_idx >= 0:
        index_start_idx = episode_start_idx
    if episode_end_idx >= 0:
        index_end_idx = episode_end_idx

    for i in range(index_start_idx, index_end_idx):
        start_idx = 0 if i == 0 else episode_ends[i-1]
        end_idx = episode_ends[i]
        this_instruction = instructions[i].strip() if i < len(instructions) else last_instruction
        last_instruction = this_instruction
        
        episode_len = int(end_idx - start_idx)

        global_start_idx = int(total_start_idx)
        global_end_idx = int(total_start_idx + episode_len)
        if this_instruction == 'fail':
            continue

        for current_idx in range(start_idx, end_idx):
            eef_pos, eef_rot_axis_angle, gripper_width = replay_buffer["robot0_eef_pos"][current_idx], replay_buffer["robot0_eef_rot_axis_angle"][current_idx], replay_buffer["robot0_gripper_width"][current_idx]
            actions = np.concatenate([replay_buffer["robot0_eef_pos"][current_idx], replay_buffer["robot0_eef_rot_axis_angle"][current_idx], replay_buffer["robot0_gripper_width"][current_idx]])

            if 'robot0_demo_start_pose' not in replay_buffer:
                demo_start_pose = np.concatenate([replay_buffer["robot0_eef_pos"][start_idx], replay_buffer["robot0_eef_rot_axis_angle"][start_idx]])
                demo_start_pose = np.asarray(demo_start_pose, dtype=np.float32)
            else:
                demo_start_pose = np.asarray(replay_buffer["robot0_demo_start_pose"][current_idx], dtype=np.float32)



            dataset.add_frame(
                {
                    "image": replay_buffer["camera0_rgb"][current_idx],  # Image.fromarray(replay_buffer["camera0_rgb"][current_idx]).save('bb.png')
                    "eef_pos": np.asarray(eef_pos, dtype=np.float32),
                    "eef_rot_axis_angle": np.asarray(eef_rot_axis_angle, dtype=np.float32),
                    "gripper_width": np.asarray(gripper_width, dtype=np.float32),
                    "demo_start_pose": np.asarray(demo_start_pose, dtype=np.float32),
                    "actions": np.asarray(actions, dtype=np.float32),
                    "episode_start_idx": np.asarray([global_start_idx], dtype=np.int64),
                    "episode_end_idx": np.asarray([global_end_idx], dtype=np.int64),
                    "task": this_instruction,
                }
            )
        dataset.save_episode()
        total_start_idx += episode_len

    return dataset, total_start_idx

def main(data_dir: str | None = None, vl_data_path_list: str | None = None, vl_number_per_task: str | None = None, vl_task_list: str | None = None, repo_name: str = REPO_NAME, *, push_to_hub: bool = False, episode_start_idx: str = '-1', episode_end_idx: str = '-1', **kwargs):
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
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    maybe_dataset_path = os.path.join(data_dir, 'dataset.zarr.zip')
    if os.path.exists(maybe_dataset_path):
        if '+' in episode_start_idx or '+' in episode_end_idx:
            raise ValueError("episode_start_idx and episode_end_idx should not contain '+' when data_dir is a single dataset.zarr.zip file.")
        episode_start_idx = int(episode_start_idx)
        episode_end_idx = int(episode_end_idx)

        dataset, total_start_idx = convert_umi_data_to_lerobot(maybe_dataset_path, dataset, data_dir, total_start_idx=0, episode_start_idx=episode_start_idx, episode_end_idx=episode_end_idx)
        
    else:
        # data_dir has more subdir to get the dataset.zarr.zip
        total_start_idx = 0
        if '+' in data_dir:
            data_dir_list = data_dir.split('+')
            subdirs = []
            for data_dir in data_dir_list:


                for subdir in os.listdir(data_dir):
                    should_in_list_dir = os.path.join(data_dir, subdir)
                    if os.path.exists(os.path.join(should_in_list_dir, 'dataset.zarr.zip')):
                        subdirs.append(should_in_list_dir)
                    print(f"Checking {should_in_list_dir}, exists: {os.path.exists(os.path.join(should_in_list_dir, 'dataset.zarr.zip'))}")
                # check data_dir itself
                if os.path.exists(os.path.join(data_dir, 'dataset.zarr.zip')):
                    subdirs.append(data_dir)


            assert '+' in episode_start_idx and '+' in episode_end_idx, "episode_start_idx and episode_end_idx should contain '+' when data_dir contains multiple subdirectories."
            episode_start_idx_list = [int(x) for x in episode_start_idx.split('+')]
            episode_end_idx_list = [int(x) for x in episode_end_idx.split('+')]
            assert len(subdirs) == len(episode_start_idx_list) == len(episode_end_idx_list), "The number of subdirectories in data_dir should match the number of episode_start_idx and episode_end_idx. len(subdirs): {}, len(episode_start_idx_list): {}, len(episode_end_idx_list): {}".format(len(subdirs), len(episode_start_idx_list), len(episode_end_idx_list))
        else:
            subdirs = [os.path.join(data_dir, subdir) for subdir in os.listdir(data_dir)]
            assert episode_start_idx == '-1' and episode_end_idx == '-1', "episode_start_idx and episode_end_idx should be '-1' when data_dir contains multiple subdirectories."
            episode_start_idx_list = [-1] * len(subdirs)
            episode_end_idx_list = [-1] * len(subdirs)

        for subdir in subdirs:
            maybe_dataset_path = os.path.join(subdir, 'dataset.zarr.zip')
            if os.path.exists(maybe_dataset_path):
                episode_start_idx = episode_start_idx_list[subdirs.index(subdir)]
                episode_end_idx = episode_end_idx_list[subdirs.index(subdir)]
                dataset, total_start_idx = convert_umi_data_to_lerobot(maybe_dataset_path, dataset, subdir, total_start_idx, episode_start_idx=episode_start_idx, episode_end_idx=episode_end_idx)
                print(f"Finished processing {maybe_dataset_path}, total frames so far: {total_start_idx}, subdir: {subdir}, episode_start_idx: {episode_start_idx}, episode_end_idx: {episode_end_idx}")
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
