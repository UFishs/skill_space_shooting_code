import dataclasses

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
import os
import json
import torch
import copy
import scipy.interpolate as si
import scipy.spatial.transform as st
from openpi.policies.pose_util import pose_to_mat, mat_to_pose10d, mat_to_pose
from openpi.policies.pose_repr_util import convert_pose_mat_rep
from openpi.training.config import UMIDataConfig

def get_interp1d(t, x):
    gripper_interp = si.interp1d(
        t, x, 
        axis=0, bounds_error=False, 
        fill_value=(x[0], x[-1]))
    return gripper_interp


class PoseInterpolator:
    def __init__(self, t, x):
        pos = x[:,:3]
        rot = st.Rotation.from_rotvec(x[:,3:])
        self.pos_interp = get_interp1d(t, pos)
        self.rot_interp = st.Slerp(t, rot)
    
    @property
    def x(self):
        return self.pos_interp.x
    
    def __call__(self, t):
        min_t = self.pos_interp.x[0]
        max_t = self.pos_interp.x[-1]
        t = np.clip(t, min_t, max_t)

        pos = self.pos_interp(t)
        rot = self.rot_interp(t)
        rvec = rot.as_rotvec()
        pose = np.concatenate([pos, rvec], axis=-1)
        return pose



class UMIDataset(LeRobotDataset):
    def __init__(self, data_config: UMIDataConfig, action_horizon: int):
        super().__init__(data_config.repo_id)
        self.data_config = data_config

        # umi related variables
        self.image_hisory_length = len(data_config.image_down_sample_steps) + 1
        self.image_down_sample_steps = data_config.image_down_sample_steps
        self.state_hisory_length = len(data_config.state_down_sample_steps) + 1
        self.state_down_sample_steps = data_config.state_down_sample_steps
        self.action_horizon = action_horizon
        self.action_down_sample_steps = data_config.action_down_sample_steps

        # how to calculate the state
        # we merge historical state into one single vector
        self.getitem_type = data_config.getitem_type
        self.low_dim_keys = ["eef_pos", "eef_rot_axis_angle", "gripper_width", "demo_start_pose"]
        self.low_dim_features = {}
        for key in self.low_dim_keys:
            self.low_dim_features[key] = torch.stack(self.hf_dataset[key]).numpy().astype(np.float32)
        self.low_dim_features['gripper_width'] = self.low_dim_features['gripper_width'][:, None]
        self.actions = torch.stack(self.hf_dataset['actions']).numpy().astype(np.float32)

        episode_ends = self.hf_dataset['episode_end_idx']
        self.episode_ends = np.unique(episode_ends)

        # self.indices = list(range(len(self.hf_dataset)))
        self.indices = []

        episode_ends = torch.stack(self.hf_dataset["episode_end_idx"]).numpy().reshape(-1)

        if "is_partial" in self.hf_dataset.features:
            is_partials = torch.stack(self.hf_dataset["is_partial"]).numpy().reshape(-1).astype(bool)
        else:
            is_partials = np.zeros(len(self.hf_dataset), dtype=bool)

        step = float(self.action_down_sample_steps)
        max_action_span = (self.action_horizon - 1) * step

        all_indices = np.arange(len(self.hf_dataset))
        valid_partial = all_indices + max_action_span <= episode_ends - 1

        self.indices = all_indices[(~is_partials) | valid_partial].tolist()

        self.rdm = np.random.RandomState(self.data_config.seed)

   
    def __len__(self):
        return len(self.indices)
  

    def __getitem__(self, idx: int) -> dict:
        """
        return_dict['thought'] is a list of strings
            - if the length is 1, it only contains the latest reasoning content
            - if the length is 2, it contains the latest reasoning content and the updated reasoning content
        """
        idx = self.indices[idx]
        low_dim_dict, return_dict = {}, {}
        current_idx_item = self.hf_dataset[idx]
        start_idx, end_idx = current_idx_item['episode_start_idx'].item(), current_idx_item['episode_end_idx'].item()
        
        freeze_action = False

        # get image and image history
        image_target_idx = np.array([idx] + [idx - self.image_down_sample_steps[history_idx] for history_idx in range(self.image_hisory_length - 1)])
        image_target_idx = np.clip(image_target_idx[::-1], start_idx, end_idx - 1)
        for i in range(self.image_hisory_length):
            return_dict['image_{}'.format(i + 1)] = self.hf_dataset[int(image_target_idx[i])]['image']
        
        # get state features and state history
        state_target_idx = np.array([idx] + [idx - self.state_down_sample_steps[history_idx] for history_idx in range(self.state_hisory_length - 1)])
        
        state_target_idx = np.clip(state_target_idx[::-1], start_idx, end_idx - 1)
        interpolation_start = max(int(state_target_idx[0]) - 5, start_idx)
        interpolation_end = min(int(state_target_idx[-1]) + 2 + 5, end_idx)
        for key in self.low_dim_keys:
            input_arr = self.low_dim_features[key]
            if 'rot' in key:
                slerp = st.Slerp(
                    times=np.arange(interpolation_start, interpolation_end),
                    rotations=st.Rotation.from_rotvec(input_arr[interpolation_start: interpolation_end]))
                output = st.Rotation.as_rotvec(slerp(state_target_idx))
            else:
                interp = si.interp1d(
                    x=np.arange(interpolation_start, interpolation_end),
                    y=input_arr[interpolation_start: interpolation_end],
                    axis=0, assume_sorted=True)
                output = interp(state_target_idx)
            low_dim_dict[key] = output


        step = self.action_down_sample_steps

        if float(step).is_integer():
            step = int(step)
            slice_end = min(
                end_idx,
                idx + (self.action_horizon - 1) * step + 1
            )
            actions = self.actions[idx:slice_end:step]
        else:
            sample_t = idx + np.arange(self.action_horizon) * step
            sample_t = sample_t[sample_t <= end_idx - 1]

            src_start = int(np.floor(sample_t[0]))
            src_end = int(np.ceil(sample_t[-1])) + 1
            src_end = min(src_end, end_idx)

            src_t = np.arange(src_start, src_end, dtype=np.float32)
            src_actions = self.actions[src_start:src_end]

            if len(src_actions) == 1:
                actions = src_actions.copy()
            else:
                pose_interp = PoseInterpolator(src_t, src_actions[:, :6])
                interp_pose = pose_interp(sample_t)

                gripper_interp = get_interp1d(src_t, src_actions[:, 6:])
                interp_gripper = gripper_interp(sample_t)

                actions = np.concatenate([interp_pose, interp_gripper], axis=-1)


        action_is_pad = torch.tensor([False] * actions.shape[0] + [True] * (self.action_horizon - actions.shape[0]))
        return_dict['action_is_pad'] = action_is_pad
        padding = np.repeat(actions[-1:], self.action_horizon - actions.shape[0], axis=0)
        actions = np.concatenate([actions, padding], axis=0)

        # calculate relative pose to start pose
        pose_mat = pose_to_mat(np.concatenate([low_dim_dict['eef_pos'], low_dim_dict['eef_rot_axis_angle']], axis=-1))
        start_pose = low_dim_dict['demo_start_pose'][0]
        start_pose += np.random.normal(scale=[0.05,0.05,0.05,0.05,0.05,0.05],size=start_pose.shape)
        start_pose_mat = pose_to_mat(start_pose)
        rel_obs_pose_mat = convert_pose_mat_rep(
            pose_mat,
            base_pose_mat=start_pose_mat,
            pose_rep='relative',
            backward=False)
        rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
        low_dim_dict['eef_rot_axis_angle_wrt_start'] = rel_obs_pose[:,3:]
        del low_dim_dict['demo_start_pose']

        # calculate relative pose to previous pose for obs and action
        action_mat = pose_to_mat(actions[..., :6])
        obs_pose_mat = convert_pose_mat_rep(
            pose_mat, 
            base_pose_mat=pose_mat[-1],
            pose_rep='relative',
            backward=False)
        action_pose_mat = convert_pose_mat_rep(
            action_mat, 
            base_pose_mat=pose_mat[-1],
            pose_rep='relative',
            backward=False)

        obs_pose = mat_to_pose10d(obs_pose_mat)
        action_pose = mat_to_pose10d(action_pose_mat)    
        action_gripper = actions[..., 6:7]
        final_action = np.concatenate([action_pose, action_gripper], axis=-1)
        if freeze_action:
            # repeat final_action[0] for the rest of the action horizon
            return_dict['actions'] = torch.from_numpy(np.repeat(final_action[:1], self.action_horizon, axis=0).astype(np.float32))
        else:
            return_dict['actions'] = torch.from_numpy(final_action.astype(np.float32))

        low_dim_dict['eef_pos'] = obs_pose[:,:3]
        low_dim_dict['eef_rot_axis_angle'] = obs_pose[:,3:]

        # concat low dim features to get state
        if self.getitem_type == 'default':
            key_sequence = ['eef_pos', 'eef_rot_axis_angle', 'eef_rot_axis_angle_wrt_start', 'gripper_width']
            return_dict['state'] = torch.from_numpy(np.concatenate([low_dim_dict[key].flatten() for key in key_sequence], axis=-1).astype(np.float32))
        elif self.getitem_type == 'necessary':
            return_dict['state'] = torch.from_numpy(np.concatenate([low_dim_dict['eef_pos'][:-1].flatten(), low_dim_dict['eef_rot_axis_angle'][:-1].flatten(), \
            low_dim_dict['eef_rot_axis_angle_wrt_start'][-1], low_dim_dict['gripper_width'].flatten()], axis=-1).astype(np.float32))
        elif self.getitem_type == 'shortest':
            history_rel_pose = mat_to_pose(obs_pose_mat)[:-1]
            history_rel_pose = np.concatenate([history_rel_pose, low_dim_dict['gripper_width'][:-1]], axis=-1)
            current_start_rel_pose = mat_to_pose(rel_obs_pose_mat)[-1]
            return_dict['state'] = torch.from_numpy(np.concatenate([history_rel_pose.flatten(), current_start_rel_pose, low_dim_dict['gripper_width'][-1]], axis=-1).astype(np.float32))
        else:
            raise ValueError('getitem_type should be one of default, necessary, shortest')
        
        copy_key = ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']
        for key in copy_key:
            return_dict[key] = current_idx_item[key]

        return return_dict

class SingleUMIDataset(UMIDataset):
    def __init__(self, data_config, action_horizon, repo_id=None, action_down_sample_steps=None):
        cfg = copy.copy(data_config)
        if repo_id is not None:
            cfg.repo_id = repo_id
        if action_down_sample_steps is not None:
            cfg.action_down_sample_steps = action_down_sample_steps
        super().__init__(cfg, action_horizon)


class MultiRepoUMIDataset(torch.utils.data.Dataset):
    def __init__(self, data_config, action_horizon: int):
        self.repo_ids = data_config.repo_id
        self.action_down_sample_steps = data_config.action_down_sample_steps
        self.weighted_sample = data_config.weighted_sample
        self.datasets = []
        self.tasks = {}
        self.task_id_maps = []

        global_task_id = 0

        for repo_id, action_down_sample_steps in zip(
            self.repo_ids,
            self.action_down_sample_steps,
        ):
            cfg = dataclasses.replace(
                data_config,
                repo_id=repo_id,
                action_down_sample_steps=action_down_sample_steps,
            )

            ds = UMIDataset(cfg, action_horizon)
            self.datasets.append(ds)

            meta = LeRobotDatasetMetadata(repo_id)

            local_to_global = {}
            for local_task_id, task in meta.tasks.items():
                local_to_global[int(local_task_id)] = global_task_id
                self.tasks[global_task_id] = task
                global_task_id += 1

            self.task_id_maps.append(local_to_global)

        lengths = np.array([len(ds) for ds in self.datasets], dtype=np.float64)

        if self.weighted_sample == -1:
            self.repo_probs = lengths / lengths.sum()
        else:
            probs = np.array(self.weighted_sample, dtype=np.float64)
            self.repo_probs = probs / probs.sum()

        self.total_len = int(lengths.sum())
        self.rdm = np.random.RandomState(data_config.seed)

        self.sample_map = []

        for _ in range(self.total_len):
            repo_idx = self.rdm.choice(len(self.datasets), p=self.repo_probs)
            inner_idx = self.rdm.randint(0, len(self.datasets[repo_idx]))
            self.sample_map.append((repo_idx, inner_idx))

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        # repo_idx, inner_idx = self.sample_map[int(index)]
        repo_idx = self.rdm.choice(len(self.datasets), p=self.repo_probs)
        ds = self.datasets[repo_idx]
        inner_idx = self.rdm.randint(0, len(ds))
        item = ds[inner_idx]

        task_index = item["task_index"]
        if hasattr(task_index, "item"):
            task_index = task_index.item()

        item["task_index"] = self.task_id_maps[repo_idx][int(task_index)]
        item["repo_id"] = self.repo_ids[repo_idx]
        item["repo_index"] = repo_idx

        return item

    def get_all_norm_stats(self):
        """
        calculate all repo merged norm stats.
        not weighted sample.
        """
        merged = {}

        keys = list(self.datasets[0].low_dim_features.keys())
        for key in keys:
            arr = np.concatenate(
                [ds.low_dim_features[key] for ds in self.datasets],
                axis=0,
            )
            merged[key] = {
                "mean": arr.mean(axis=0),
                "std": arr.std(axis=0),
                "min": arr.min(axis=0),
                "max": arr.max(axis=0),
            }

        actions = np.concatenate([ds.actions for ds in self.datasets], axis=0)
        merged["actions"] = {
            "mean": actions.mean(axis=0),
            "std": actions.std(axis=0),
            "min": actions.min(axis=0),
            "max": actions.max(axis=0),
        }

        return merged
