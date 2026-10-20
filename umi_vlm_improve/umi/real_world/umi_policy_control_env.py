"""Hardware-side environment used by the DSRL remote-control server.

This module intentionally owns every UMI/Franka-specific operation.  The DSRL
process only sees already-packed policy observations and sends raw pi0.5 action
chunks back through ``step``.
"""

from __future__ import annotations

import time
import cv2
import numpy as np
import scipy.spatial.transform as st
from pynput.keyboard import Key, KeyCode
from scipy.spatial.transform import Rotation as R

from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from umi.common.pose_util import mat_to_pose, mat_to_pose10d, pose_to_mat
from umi.common.precise_sleep import precise_wait
from umi.real_world.real_inference_util import get_real_umi_action


class OperatorExit(RuntimeError):
    """Raised when the robot-side operator presses Q."""


def solve_table_collision(ee_pose, gripper_width, height_threshold):
    finger_thickness = 25.5 / 1000
    keypoints = np.asarray(
        [(dx * gripper_width / 2, dy * finger_thickness / 2, 0) for dx in (-1, 1) for dy in (-1, 1)]
    )
    rotation = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed = np.transpose(rotation @ np.transpose(keypoints)) + ee_pose[:3]
    ee_pose[2] += max(height_threshold - np.min(transformed[:, 2]), 0)


def solve_sphere_collision(ee_poses, robots_config):
    if len(robots_config) < 2:
        return
    this_that_mat = np.identity(4)
    this_that_mat[:3, 3] = np.array([0, 0.89, 0])
    for this_idx in range(len(robots_config)):
        for that_idx in range(this_idx + 1, len(robots_config)):
            this_ee = pose_to_mat(ee_poses[this_idx][:6])
            this_local = np.identity(4)
            this_local[:3, 3] = np.asarray(robots_config[this_idx]["sphere_center"])
            this_global = this_ee @ this_local

            that_ee = pose_to_mat(ee_poses[that_idx][:6])
            that_local = np.identity(4)
            that_local[:3, 3] = np.asarray(robots_config[that_idx]["sphere_center"])
            that_global = this_that_mat @ that_ee @ that_local

            delta = that_global[:3, 3] - this_global[:3, 3]
            distance = np.linalg.norm(delta)
            threshold = robots_config[this_idx]["sphere_radius"] + robots_config[that_idx]["sphere_radius"]
            if distance < threshold:
                if distance < 1e-8:
                    raise RuntimeError("Two robot collision spheres have indistinguishable centers")
                correction = (threshold - distance) / 2 * delta / distance
                this_global[:3, 3] -= correction
                that_global[:3, 3] += correction
                ee_poses[this_idx][:6] = mat_to_pose(this_global @ np.linalg.inv(this_local))
                ee_poses[that_idx][:6] = mat_to_pose(
                    np.linalg.inv(this_that_mat) @ that_global @ np.linalg.inv(that_local)
                )


def get_pi_obs_dict(env_obs, getitem_type, episode_start_pose, instruction):
    """Build exactly the UMI observation schema consumed by the OpenPI server."""
    low_dim, result = {}, {"prompt": instruction}
    history_length = env_obs["camera0_rgb"].shape[0]
    for i in range(history_length):
        result[f"image_{i + 1}"] = env_obs["camera0_rgb"][i]

    pose_mat = pose_to_mat(
        np.concatenate([env_obs["robot0_eef_pos"], env_obs["robot0_eef_rot_axis_angle"]], axis=-1)
    )
    start_pose_mat = pose_to_mat(episode_start_pose[0])
    start_relative_mat = convert_pose_mat_rep(
        pose_mat, base_pose_mat=start_pose_mat, pose_rep="relative", backward=False
    )
    start_relative_pose = mat_to_pose10d(start_relative_mat)
    low_dim["eef_rot_axis_angle_wrt_start"] = start_relative_pose[:, 3:]

    observation_pose_mat = convert_pose_mat_rep(
        pose_mat, base_pose_mat=pose_mat[-1], pose_rep="relative", backward=False
    )
    observation_pose = mat_to_pose10d(observation_pose_mat)
    low_dim["eef_pos"] = observation_pose[:, :3]
    low_dim["eef_rot_axis_angle"] = observation_pose[:, 3:]
    low_dim["gripper_width"] = env_obs["robot0_gripper_width"]

    if getitem_type == "default":
        keys = ["eef_pos", "eef_rot_axis_angle", "eef_rot_axis_angle_wrt_start", "gripper_width"]
        state = np.concatenate([low_dim[key].flatten() for key in keys], axis=-1)
    elif getitem_type == "necessary":
        state = np.concatenate(
            [
                low_dim["eef_pos"][:-1].flatten(),
                low_dim["eef_rot_axis_angle"][:-1].flatten(),
                low_dim["eef_rot_axis_angle_wrt_start"][-1],
                low_dim["gripper_width"].flatten(),
            ],
            axis=-1,
        )
    elif getitem_type == "shortest":
        history = mat_to_pose(observation_pose_mat)[:-1]
        history = np.concatenate([history, low_dim["gripper_width"][:-1]], axis=-1)
        current_from_start = mat_to_pose(start_relative_mat)[-1]
        state = np.concatenate(
            [history.flatten(), current_from_start, low_dim["gripper_width"][-1]], axis=-1
        )
    else:
        raise ValueError(f"Unknown getitem_type: {getitem_type!r}")
    result["state"] = state.astype(np.float32)
    return result


def _as_uint8(image):
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating) and image.max(initial=0.0) <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


class UmiPolicyControlEnv:
    """Stateful Gym-like control environment around ``BimanualUmiEnv``."""

    def __init__(
        self,
        env,
        spacemouse,
        key_counter,
        *,
        robots_config,
        instruction,
        getitem_type="necessary",
        frequency=5.0,
        steps_per_inference=1,
        max_timesteps=5000,
        max_duration=2_000_000.0,
        action_exec_latency=0.01,
        temporal_agg=False,
        fix_gripper=False,
        save_umi_episode=False,
        show_camera=True,
    ):
        self.env = env
        self.spacemouse = spacemouse
        self.key_counter = key_counter
        self.robots_config = robots_config
        self.instruction = instruction
        self.getitem_type = getitem_type
        self.frequency = frequency
        self.steps_per_inference = steps_per_inference
        self.max_timesteps = max_timesteps
        self.max_duration = max_duration
        self.action_exec_latency = action_exec_latency
        self.temporal_agg = temporal_agg
        self.fix_gripper = fix_gripper
        self.save_umi_episode = save_umi_episode
        self.show_camera = show_camera
        self.num_robots = len(robots_config)
        self.raw_action_dim = 10 * self.num_robots
        self.real_action_dim = 7 * self.num_robots
        self._active = False
        self._episode_start_pose = None
        self._last_env_obs = None
        self._prediction_history = []
        self._step_index = 0
        self._episode_wall_start = None
        self._control_start = None

    @property
    def metadata(self):
        return {
            "protocol_version": 1,
            "environment": "umi_franka",
            "num_robots": self.num_robots,
            "raw_action_dim": self.raw_action_dim,
            "real_action_dim": self.real_action_dim,
            "frequency": self.frequency,
            "getitem_type": self.getitem_type,
            "instruction": self.instruction,
            "pixel_shape": [224, 224, 3],
            "operations": ["sample_observation", "reset", "step", "end_episode", "ping", "close"],
        }

    def _episode_start_poses(self, obs):
        return [
            np.concatenate(
                [obs[f"robot{i}_eef_pos"], obs[f"robot{i}_eef_rot_axis_angle"]], axis=-1
            )[-1]
            for i in range(self.num_robots)
        ]

    def _observation(self, obs):
        return {
            "policy_observation": get_pi_obs_dict(
                obs, self.getitem_type, self._episode_start_pose, self.instruction
            ),
            "pixels": _as_uint8(obs["camera0_rgb"][-1]),
            "step_index": np.int64(self._step_index),
        }

    def _show_reset_frame(self):
        if not self.show_camera:
            return
        obs = self.env.get_obs()
        image = _as_uint8(obs["camera0_rgb"][-1]).copy()
        cv2.putText(image, "RESET: C=start, Q=exit", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("DSRL UMI control server", image[..., ::-1])
        cv2.pollKey()

    # def _human_control_until_start(self):
    #     print("Human in control. Press C to start the remote rollout; Q exits.")
    #     target_pose = np.stack([state["ActualTCPPose"] for state in self.env.get_robot_state()])
    #     gripper = np.asarray([state["gripper_position"] for state in self.env.get_gripper_state()])
    #     controlled = [0]
    #     start = time.monotonic()
    #     step = 0
    #     while True:
    #         cycle_end = start + (step + 1) / self.frequency
    #         self._show_reset_frame()
    #         for event in self.key_counter.get_press_events():
    #             if event == KeyCode(char="q"):
    #                 raise OperatorExit("Operator requested shutdown")
    #             if event == KeyCode(char="c"):
    #                 self.key_counter.clear()
    #                 return
    #             if event == KeyCode(char="a"):
    #                 controlled = list(range(len(target_pose)))
    #             if event == KeyCode(char="1"):
    #                 controlled = [0]
    #             if event == KeyCode(char="2") and len(target_pose) > 1:
    #                 controlled = [1]

    #         motion = self.spacemouse.get_motion_state_transformed()
    #         dpos = motion[:3] * (0.5 / self.frequency) / 2.0
    #         drot = R.from_euler("xyz", motion[3:] * (1.5 / self.frequency) / 2.0)
    #         for robot_idx in controlled:
    #             target_pose[robot_idx, :3] += dpos
    #             target_pose[robot_idx, 3:] = (
    #                 drot * R.from_rotvec(target_pose[robot_idx, 3:])
    #             ).as_rotvec()

    #         dgrip = 0.0
    #         if self.spacemouse.is_button_pressed(0):
    #             dgrip = -0.2 / self.frequency
    #         if self.spacemouse.is_button_pressed(1):
    #             dgrip = 0.2 / self.frequency
    #         for robot_idx in controlled:
    #             gripper[robot_idx] = np.clip(gripper[robot_idx] + dgrip, 0.0, 0.09)
    #             solve_table_collision(
    #                 target_pose[robot_idx], gripper[robot_idx], self.robots_config[robot_idx]["height_threshold"]
    #             )
    #         solve_sphere_collision(target_pose, self.robots_config)

    #         command = np.zeros(self.real_action_dim, dtype=np.float64)
    #         for robot_idx in range(self.num_robots):
    #             command[7 * robot_idx : 7 * robot_idx + 6] = target_pose[robot_idx]
    #             command[7 * robot_idx + 6] = gripper[robot_idx]
    #         self.env.exec_actions(
    #             [command],
    #             [cycle_end - time.monotonic() + time.time()],
    #             compensate_latency=False,
    #         )
    #         precise_wait(cycle_end)
    #         step += 1

    def reset(self):
        if self._active:
            self.end_episode()
        self.env.robot_reset_to_init()
        time.sleep(5.0)
        # self._human_control_until_start()
        start_time = time.time() + 1.0
        self.env.start_episode(start_time)
        self._active = True
        self.key_counter.clear()
        print(f"RESET!!!! Episode started at {start_time:.3f}, wall time {time.time():.3f}")
        try:
            obs = self.env.get_obs()
            self._episode_start_pose = self._episode_start_poses(obs)
            self._step_index = 0
            self._episode_wall_start = time.time()
            self._control_start = start_time
            self._prediction_history = []

            self._step_index += 1
            zero_action = np.array([0,0,0,1,0,0,0,1,0,0])
            real_zero_action = get_real_umi_action(zero_action, obs, "relative")
            self._prediction_history.append((self._step_index, real_zero_action[None, :]))

            precise_wait(start_time - 1 / 60, time_func=time.time)
            self._last_env_obs = self.env.get_obs()
            return self._observation(self._last_env_obs), {"episode_started": True}
        except Exception:
            self.end_episode()
            raise

    def sample_observation(self):
        """Return a shape-valid observation without starting or resetting an episode."""
        obs = self.env.get_obs()
        self._episode_start_pose = self._episode_start_poses(obs)
        self._last_env_obs = obs
        return self._observation(obs)

    def _temporal_aggregate(self, horizon):
        targets = []
        for offset in range(horizon):
            absolute_step = self._step_index + offset
            actions = []
            for prediction_step, prediction in self._prediction_history:
                prediction_offset = absolute_step - prediction_step
                if 0 <= prediction_offset < len(prediction):
                    actions.append(prediction[prediction_offset])
            actions = np.asarray(actions)
            if not len(actions):
                continue
            weights = np.exp(-0.01 * np.arange(len(actions)))
            weights /= weights.sum()
            weighted = (actions * weights[:, None]).sum(axis=0, keepdims=True)
            for robot_idx in range(self.num_robots):
                rotation_slice = slice(7 * robot_idx + 3, 7 * robot_idx + 6)
                weighted[0, rotation_slice] = R.from_rotvec(actions[:, rotation_slice]).mean(weights=weights).as_rotvec()
            targets.append(weighted)
        if not targets:
            raise RuntimeError("Temporal aggregation found no valid action")
        return np.concatenate(targets, axis=0)

    def _apply_safety(self, targets):
        for target in targets:
            for robot_idx, config in enumerate(self.robots_config):
                solve_table_collision(
                    target[7 * robot_idx : 7 * robot_idx + 6],
                    target[7 * robot_idx + 6],
                    config["height_threshold"],
                )
            solve_sphere_collision(target.reshape(self.num_robots, -1), self.robots_config)

    def step(self, raw_action):
        if not self._active:
            raise RuntimeError("step() called before reset()")
        raw_action = np.asarray(raw_action)
        if raw_action.ndim != 2 or raw_action.shape[1] != self.raw_action_dim:
            raise ValueError(
                f"Expected raw action shape (horizon, {self.raw_action_dim}), got {raw_action.shape}"
            )
        action = get_real_umi_action(raw_action, self._last_env_obs, "relative")
        if self.fix_gripper:
            for robot_idx in range(self.num_robots):
                grip_idx = 7 * robot_idx + 6
                action[action[:, grip_idx] < 0.04, grip_idx] -= 0.03

        horizon = len(action)
        self._prediction_history.append((self._step_index, action.copy()))
        self._prediction_history = self._prediction_history[-8:]
        targets = self._temporal_aggregate(horizon) if self.temporal_agg else action
        self._apply_safety(targets)

        timestamps = np.arange(len(targets), dtype=np.float64) / self.frequency + self._last_env_obs["timestamp"][-1]
        now = time.time()
        keep = timestamps > now + self.action_exec_latency
        if np.any(keep):
            targets, timestamps = targets[keep], timestamps[keep]
        else:
            targets = targets[[-1]]
            timestamps = np.asarray([now + 1 / self.frequency])
        self.env.exec_actions(targets, timestamps, compensate_latency=True)
        self._step_index += 1

        terminated = False
        truncated = False
        stop_reason = None
        for event in self.key_counter.get_press_events():
            if event == KeyCode(char="q"):
                terminated, stop_reason = True, "operator_exit"
            elif event == KeyCode(char="s"):
                terminated, stop_reason = True, "operator_stop"
        if self._step_index >= self.max_timesteps:
            truncated, stop_reason = True, "max_timesteps"
        if time.time() - self._episode_wall_start >= self.max_duration:
            truncated, stop_reason = True, "max_duration"

        precise_wait(self._control_start + self._step_index / self.frequency, time_func=time.time)
        self._last_env_obs = self.env.get_obs()
        observation = self._observation(self._last_env_obs)
        if terminated or truncated:
            self.end_episode()
        return observation, -1.0, terminated, truncated, {"stop_reason": stop_reason}

    def end_episode(self):
        if self._active:
            self.env.end_episode(save=self.save_umi_episode)
            self._active = False
        return {"episode_active": False}

    def close(self):
        self.end_episode()
        cv2.destroyAllWindows()
