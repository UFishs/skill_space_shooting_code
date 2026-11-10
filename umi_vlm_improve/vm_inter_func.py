"""
Usage:
(umi): python umi_client.py \
--robot_config=example/eval_robots_config.yaml \
-o /path/to/output/dir/ \
--frequency 5 -j \
--temporal_agg -si 1 \
-ins "$instruction" \
-state_horizon 3,15 -action_down_sample_steps 3 -getitem_type necessary \
--remote_port 8000

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly!

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import base64
import os
from bisect import bisect_left, bisect_right

import json
import time
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import numpy as np
import requests
import scipy.spatial.transform as st
import yaml
from omegaconf import OmegaConf
from openpi_client import websocket_client_policy
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from PIL import ImageDraw, Image
import copy

from umi.common.cv_util import FisheyeRectConverter, parse_fisheye_intrinsics
from umi.common.pose_util import mat_to_pose, pose_to_mat, mat_to_pose10d
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from umi.common.precise_sleep import precise_wait
from umi.real_world.bimanual_umi_env import BimanualUmiEnv
from umi.real_world.keystroke_counter import Key, KeyCode, KeystrokeCounter
from umi.real_world.real_inference_util import get_real_umi_action
from umi.real_world.spacemouse_shared_memory import Spacemouse

from vlm_request.gemini_request_genai import request_check_stage, request_new_instruction, request_check_completion
from vlm_request.prompts import check_completion_prompt, new_instruction_prompt, check_stage_repair_prompt

OmegaConf.register_new_resolver("eval", eval, replace=True)


SERVER = "http://localhost:9000/predict_json"
VALUE_THRESHOLD = -0

from instr_dict import predefined_instr_dict, task_instruction_to_name

def encode_image(img):
    # encode PIL.Image to base64 string
    from io import BytesIO
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def request_monitor(instruction, cam_high, cam_left, cam_right, action_chunk):
    '''
    ref_start, ref_end: PIL.Image
    before, after: List[PIL.Image]
    '''
    payload = {
        "cam_high": encode_image(cam_high),
        "cam_left": encode_image(cam_left),
        "cam_right": encode_image(cam_right),
        "state": action_chunk.reshape(-1).tolist()
    }
    payload = {
        'images': [encode_image(cam_high), encode_image(cam_left), encode_image(cam_right)],
        'state': action_chunk.tolist(),
    }


    begin_time = time.time()
    resp = requests.post(SERVER, json=payload)
    print(f"Response received in {time.time() - begin_time:.2f} seconds")

    return resp.json()

def check_value(env, instruction, raw_actions, verbose=False):
    obs = env.get_obs()
    img = np.array(obs['camera0_rgb'][-1]*255, dtype=np.uint8)
    img = Image.fromarray(img)


    check_value_response = request_monitor(
        instruction,
        cam_high=img,
        cam_left=img,
        cam_right=img,
        action_chunk=raw_actions
    )
    value = check_value_response['value']
    if isinstance(value, list):
        value = value[-1]

    return value


def check_stage_repair(env, instruction, predefined_instr_dict, task_instruction):
    obs = env.get_obs()
    img = np.array(obs['camera0_rgb'][-1]*255, dtype=np.uint8)
    img = Image.fromarray(img)

    check_stage_response = request_check_stage(
        prompt_template=check_stage_repair_prompt,
        prompt_content={'task_desc': predefined_instr_dict[task_instruction]['1'], 'stage_desc': instruction},
        images=[img],
        images_desc=['Current frame']
    )
    response = check_stage_response['completed']
    if isinstance(response, list):
        response = response[0]
    is_stage_repair = (response == 'Yes')

    return is_stage_repair

def check_task_completion(env, predefined_instr_dict, task_instruction):    
    obs = env.get_obs()
    img = np.array(obs['camera0_rgb'][-1]*255, dtype=np.uint8)
    img = Image.fromarray(img)

    check_completion_response = request_check_completion(
        prompt_template=check_completion_prompt,
        prompt_content={'task_desc': predefined_instr_dict[task_instruction]['1']},
        images=[img],
        images_desc=['Current frame']
    )
    response = check_completion_response['completed']
    if isinstance(response, list):
        response = response[0]
    is_task_completed = (response == 'Yes')

    return is_task_completed

def update_fix_status(
    env: BimanualUmiEnv,
    instruction: str,
    predefined_instr_dict,
):
    """
    Base task and terminal states are not fix.
    Other instructions are considered fix/recovery behavior.
    """

    non_fix_instructions = {
        predefined_instr_dict['1'],
        'The task is successfully completed.',
        'The task directly succeed.',
        'The task is failed, please try again.',
    }

    is_fix = instruction not in non_fix_instructions

    env.set_fix_status(
        is_fix=is_fix,
        timestamp=time.time(),
    )

def solve_table_collision(ee_pose, gripper_width, height_threshold):
    finger_thickness = 25.5 / 1000
    keypoints = list()
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed_keypoints = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    ee_pose[2] += delta


def solve_sphere_collision(ee_poses, robots_config):
    num_robot = len(robots_config)
    this_that_mat = np.identity(4)
    this_that_mat[:3, 3] = np.array([0, 0.89, 0]) # TODO: very hacky now!!!!

    for this_robot_idx in range(num_robot):
        for that_robot_idx in range(this_robot_idx + 1, num_robot):
            this_ee_mat = pose_to_mat(ee_poses[this_robot_idx][:6])
            this_sphere_mat_local = np.identity(4)
            this_sphere_mat_local[:3, 3] = np.asarray(robots_config[this_robot_idx]['sphere_center'])
            this_sphere_mat_global = this_ee_mat @ this_sphere_mat_local
            this_sphere_center = this_sphere_mat_global[:3, 3]

            that_ee_mat = pose_to_mat(ee_poses[that_robot_idx][:6])
            that_sphere_mat_local = np.identity(4)
            that_sphere_mat_local[:3, 3] = np.asarray(robots_config[that_robot_idx]['sphere_center'])
            that_sphere_mat_global = this_that_mat @ that_ee_mat @ that_sphere_mat_local
            that_sphere_center = that_sphere_mat_global[:3, 3]

            distance = np.linalg.norm(that_sphere_center - this_sphere_center)
            threshold = robots_config[this_robot_idx]['sphere_radius'] + robots_config[that_robot_idx]['sphere_radius']
            # print(that_sphere_center, this_sphere_center)
            if distance < threshold:
                print('avoid collision between two arms')
                half_delta = (threshold - distance) / 2
                normal = (that_sphere_center - this_sphere_center) / distance
                this_sphere_mat_global[:3, 3] -= half_delta * normal
                that_sphere_mat_global[:3, 3] += half_delta * normal

                ee_poses[this_robot_idx][:6] = mat_to_pose(this_sphere_mat_global @ np.linalg.inv(this_sphere_mat_local))
                ee_poses[that_robot_idx][:6] = mat_to_pose(np.linalg.inv(this_that_mat) @ that_sphere_mat_global @ np.linalg.inv(that_sphere_mat_local))

def get_pi_obs_dict(env_obs, start_image, bbox_list, getitem_type, obs_pose_repr, tx_robot1_robot0, episode_start_pose, instruction):
    low_dim_dict, return_dict = {}, {'prompt': instruction}
    history_length = env_obs['camera0_rgb'].shape[0]

    for i in range(history_length):
        return_dict[f'image_{i + 1}'] = env_obs['camera0_rgb'][i]


    if start_image is not None:
        return_dict['condition'] = {
            'detect': bbox_list,
            'episode_start_image': start_image
        }

    pose_mat = pose_to_mat(np.concatenate([env_obs['robot0_eef_pos'], env_obs['robot0_eef_rot_axis_angle']], axis=-1))
    start_pose = episode_start_pose[0]
    start_pose_mat = pose_to_mat(start_pose)
    rel_obs_pose_mat = convert_pose_mat_rep(
        pose_mat,
        base_pose_mat=start_pose_mat,
        pose_rep=obs_pose_repr,
        backward=False)
    rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
    low_dim_dict['eef_rot_axis_angle_wrt_start'] = rel_obs_pose[:,3:]

    obs_pose_mat = convert_pose_mat_rep(
        pose_mat, 
        base_pose_mat=pose_mat[-1],
        pose_rep=obs_pose_repr,
        backward=False)

    obs_pose = mat_to_pose10d(obs_pose_mat)

    low_dim_dict['eef_pos'] = obs_pose[:,:3]
    low_dim_dict['eef_rot_axis_angle'] = obs_pose[:,3:]
    low_dim_dict['gripper_width'] = env_obs['robot0_gripper_width']

    # concat low dim features to get state
    if getitem_type == 'default':
        key_sequence = ['eef_pos', 'eef_rot_axis_angle', 'eef_rot_axis_angle_wrt_start', 'gripper_width']
        return_dict['state'] = np.concatenate([low_dim_dict[key].flatten() for key in key_sequence], axis=-1).astype(np.float32)
    elif getitem_type == 'necessary':
        return_dict['state'] = np.concatenate([low_dim_dict['eef_pos'][:-1].flatten(), low_dim_dict['eef_rot_axis_angle'][:-1].flatten(), \
                                                                low_dim_dict['eef_rot_axis_angle_wrt_start'][-1], low_dim_dict['gripper_width'].flatten()], axis=-1).astype(np.float32)
    elif getitem_type == 'shortest':
        history_rel_pose = mat_to_pose(obs_pose_mat)[:-1]
        history_rel_pose = np.concatenate([history_rel_pose, low_dim_dict['gripper_width'][:-1]], axis=-1)
        current_start_rel_pose = mat_to_pose(rel_obs_pose_mat)[-1]
        return_dict['state'] = np.concatenate([history_rel_pose.flatten(), current_start_rel_pose, low_dim_dict['gripper_width'][-1]], axis=-1).astype(np.float32)
    else:
        raise ValueError('getitem_type should be one of default, necessary, shortest')
    return return_dict


def get_images_history(env, camera_key='camera0_rgb'):
    accumulator = env.obs_accumulator
    if accumulator is not None and accumulator.timestamps.get(camera_key):
        return {
            'images': accumulator.data[camera_key],
            'timestamps': accumulator.timestamps[camera_key],
        }

    camera_idx = int(camera_key.removeprefix('camera').removesuffix('_rgb'))
    camera_data = (env.last_camera_data or {}).get(camera_idx)
    if camera_data is None:
        return {'images': [], 'timestamps': []}
    return {
        'images': camera_data['color'],
        'timestamps': camera_data['timestamp'],
    }


def get_nearest_history_image(images_history, seconds_ago, current_timestamp=None):
    seconds_ago = float(seconds_ago)
    current_timestamp = time.time() if current_timestamp is None else float(current_timestamp)
    if not np.isfinite(seconds_ago) or seconds_ago < 0:
        raise ValueError('seconds_ago must be finite and non-negative')
    if not np.isfinite(current_timestamp):
        raise ValueError('current_timestamp must be finite')

    timestamps = images_history['timestamps']
    if len(timestamps) != len(images_history['images']):
        raise ValueError('History images and timestamps must have equal lengths')
    end = bisect_right(timestamps, current_timestamp)
    target_timestamp = current_timestamp - seconds_ago
    if end == 0 or target_timestamp < timestamps[0] or target_timestamp > timestamps[end - 1]:
        return None

    right = bisect_left(timestamps, target_timestamp, 0, end)
    index = right
    if right > 0 and target_timestamp - timestamps[right - 1] <= timestamps[right] - target_timestamp:
        index = right - 1
    image = np.asarray(images_history['images'][index])
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(image).convert('RGB'), float(timestamps[index])


def change_instruction(old_instruction, key_counter: KeystrokeCounter, actual_in_fix_dict, env: BimanualUmiEnv, force_new_instruction=None, predefined_instr_dict=None, task_instruction=None):

    if force_new_instruction is not None:
        update_fix_status(
            env,
            force_new_instruction,
            predefined_instr_dict=predefined_instr_dict[task_instruction],
        )
        return force_new_instruction, actual_in_fix_dict
    
    print("Change instruction!")
    print("Predefined instructions:")
    for key, instr in predefined_instr_dict[task_instruction].items():
        print(f'{key}: {instr}')


    env.get_obs()  # Refresh the camera buffers and this episode's saved history.
    images_history = get_images_history(env)
    if len(images_history['timestamps']) == 0:
        raise ValueError('No camera frames available for skill selection')
    current_timestamp = float(images_history['timestamps'][-1])
    current_image, _ = get_nearest_history_image(
        images_history, seconds_ago=0, current_timestamp=current_timestamp,
    )
    images = [current_image]
    images_desc = ['Current frame']

    # Optional history example: uncomment to send frames near 5, 3 and 1 seconds
    # before the current image. Use time.time() as current_timestamp instead if
    # the offsets should be relative to wall-clock now. History is collected by
    # env.get_obs(); it only contains frames retained by the environment.
    # for seconds_ago in (5.0, 3.0, 1.0):
    #     historical_frame = get_nearest_history_image(
    #         images_history, seconds_ago, current_timestamp=current_timestamp,
    #     )
    #     if historical_frame is None:
    #         continue  # The requested time is outside the available history.
    #     history_image, history_timestamp = historical_frame
    #     images.append(history_image)
    #     images_desc.append(
    #         f'{current_timestamp - history_timestamp:.3f} seconds ago before current frame'
    #     )

    json_response = request_new_instruction(
        prompt_template=new_instruction_prompt,
        prompt_content={'task_desc': predefined_instr_dict[task_instruction]['1']},
        images=images,
        images_desc=images_desc,
        predefined_instr_dict=predefined_instr_dict[task_instruction]
    )

    response_code = str(json_response['next_skill_id']).strip()

    # human control
    # import sys
    # import termios
    # termios.tcflush(sys.stdin, termios.TCIFLUSH)
    # response_code = input('Please input instruction code: ')


    if response_code in {'s', 'f', 'd'} or response_code not in predefined_instr_dict[task_instruction]:
        raise ValueError(f"Invalid next_skill_id: {json_response['next_skill_id']!r}")
    now_instruction = predefined_instr_dict[task_instruction][response_code]
    print("Current instruction:", now_instruction)  
    key_counter.clear() 

    update_fix_status(
        env,
        now_instruction,
        predefined_instr_dict=predefined_instr_dict[task_instruction],
    )


    if response_code == 's':
        temp_keys = list(env.record_ins_timestamp_dict.keys())
        for ins in temp_keys:
            if ins in actual_in_fix_dict:
                env.set_record_ins_timestamp(
                    ins=ins,
                    timestamp=time.time(),
                    type='end',
                )
    elif response_code == 'f':
        env.set_record_ins_timestamp(
            ins="fail",
            timestamp=time.time(),
            type='end',
        )
    elif response_code == 'd':
        env.set_record_ins_timestamp(
            ins="direct success",
            timestamp=time.time(),
            type='end',
        )
    
    return now_instruction, actual_in_fix_dict

def run_robot_with_recall(
    env: BimanualUmiEnv,
    dt,
    key_counter,
    instruction,
    actual_in_fix_dict,
    recall_steps=8,
):
    action_accumulator = env.action_accumulator

    actions = np.asarray(action_accumulator.actions)

    if actions is None or len(actions) == 0:
        print("[Recall] No actions available.")
        return instruction, actual_in_fix_dict

    if actions.ndim == 3:
        actions = actions[:, -1, :]

    if actions.ndim != 2:
        raise ValueError(
            f"[Recall] Unexpected action shape: {actions.shape}"
        )

    current_idx = len(actions) - 1

    print(
        f"[Recall] Start recall, "
        f"total actions = {len(actions)}"
    )

    while current_idx >= 0:

        _ = cv2.pollKey()
        press_events = key_counter.get_press_events()

        for key_stroke in press_events:
            if key_stroke == Key.f2:
                print("[Recall] F2 pressed. Changing instruction.")

                now_instruction, actual_in_fix_dict = change_instruction(
                    instruction,
                    key_counter,
                    actual_in_fix_dict,
                    env,
                )

                return now_instruction, actual_in_fix_dict

        actions = np.asarray(action_accumulator.actions)

        if actions.ndim == 3:
            actions = actions[:, -1, :]

        if actions.ndim != 2:
            raise ValueError(
                f"[Recall] Unexpected action shape: {actions.shape}"
            )

        current_idx = min(
            current_idx,
            len(actions) - 1
        )

        if current_idx < 0:
            break

        start_idx = max(
            0,
            current_idx - recall_steps + 1
        )

        recall_actions = actions[
            start_idx:current_idx + 1
        ][::-1].copy()

        print(
            f"[Recall] "
            f"{current_idx} -> {start_idx}, "
            f"num={len(recall_actions)}"
        )

        start_time = time.time()

        action_timestamps = (
            start_time
            + np.arange(
                len(recall_actions),
                dtype=np.float64
            ) * dt
        )

        env.exec_actions(
            actions=recall_actions,
            timestamps=action_timestamps,
            compensate_latency=True,
        )

        target_time = (
            start_time
            + len(recall_actions) * dt
        )

        while True:
            _ = cv2.pollKey()
            press_events = key_counter.get_press_events()

            f2_pressed = False

            for key_stroke in press_events:
                if key_stroke == Key.f2:
                    f2_pressed = True
                    break

            if f2_pressed:
                print(
                    "[Recall] F2 pressed during recall. "
                    "Changing instruction."
                )

                now_instruction, actual_in_fix_dict = change_instruction(
                    instruction,
                    key_counter,
                    actual_in_fix_dict,
                    env,
                )

                return now_instruction, actual_in_fix_dict

            remaining = target_time - time.time()

            if remaining <= 0:
                break

            time.sleep(min(remaining, 0.01))

        current_idx = start_idx - 1

    print("[Recall] Finished.")

    return instruction, actual_in_fix_dict



def run_robot_with_instruction(env: BimanualUmiEnv, steps_per_inference, dt, start_image, bbox_list, getitem_type, obs_pose_rep, tx_robot1_robot0, instruction, policy_client, robots_config, temporal_agg, fix_gripper, action_pose_repr, all_time_actions, max_timesteps, key_counter: KeystrokeCounter, max_duration, action_horizon, actual_in_fix_dict, episode_start_pose, task_instruction, predefined_instr_dict, beginning=False):


    # start episode
    start_delay = 1.0
    eval_t_start = time.time() + start_delay
    t_start = time.monotonic() + start_delay

    # get current pose
    obs = env.get_obs()
    if episode_start_pose is None:
        episode_start_pose = list()
        for robot_id in range(len(robots_config)):
            pose = np.concatenate([
                obs[f'robot{robot_id}_eef_pos'],
                obs[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1)[-1]
            episode_start_pose.append(pose)

    episode_id = env.replay_buffer.n_episodes
    obs_left_img = obs['camera0_rgb'][-1]
    obs_right_img = obs['camera0_rgb'][-1]
    vis_img = np.concatenate([obs_left_img, obs_right_img], axis=1)
    text = 'Episode: {}, Time: {:.1f}'.format(
        episode_id, time.monotonic() - t_start
    )
    cv2.putText(
        vis_img,
        text,
        (10,20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        thickness=1,
        color=(255,255,255)
    )
    cv2.imshow('default', vis_img[...,::-1])

    # wait for 1/30 sec to get the closest frame actually
    # reduces overall latency
    frame_latency = 1/60
    precise_wait(eval_t_start - frame_latency, time_func=time.time)
    print("Started!")

    iter_idx = 0
    inference_idx = steps_per_inference
    cycle_idx = 0

    continuous_positive_value_cnt = 0
    if beginning:
        # smooth start
        for temp_zero_id in range(1):
            keep_gripper = obs['robot0_gripper_width'][-1][0]
            zero_action = np.array([0,0,0,1,0,0,0,1,0,keep_gripper])
            real_zero_action = get_real_umi_action(zero_action, obs, action_pose_repr)
            all_time_actions[[iter_idx], iter_idx:iter_idx + action_horizon] = real_zero_action
            iter_idx += 1

    while True:
        # calculate timing
        t_cycle_end = t_start + (cycle_idx + 1) * dt

        

        # get obs
        obs = env.get_obs()
        obs_timestamps = obs['timestamp']

        # run inference
        s = time.time()
        obs_dict_np = get_pi_obs_dict(
            env_obs=obs, start_image=start_image, bbox_list=bbox_list, getitem_type=getitem_type,
            obs_pose_repr=obs_pose_rep,
            tx_robot1_robot0=tx_robot1_robot0,
            episode_start_pose=episode_start_pose, instruction=instruction)
        policy_return_dict = policy_client.infer(obs_dict_np)
        
        raw_action = policy_return_dict['actions']

        value = check_value(env, instruction, raw_action, verbose=True)
        print(f"Value: {value:.4f}, Time for inference and value check: {time.time() - s:.2f} seconds")
        # human control
        # value = 0.5
        if instruction == predefined_instr_dict[task_instruction]['1']:

            # check task completion
            if check_task_completion(env, predefined_instr_dict, task_instruction):
                print("--"*30)
                print(f"Task completed, returning to base instruction.")
                print("--"*30)
                now_instruction, actual_in_fix_dict = change_instruction(
                    instruction, key_counter, actual_in_fix_dict, env, 
                    predefined_instr_dict=predefined_instr_dict, task_instruction=task_instruction,
                    force_new_instruction=predefined_instr_dict[task_instruction]['s']
                )
                return now_instruction, actual_in_fix_dict



            if value < VALUE_THRESHOLD:
                now_instruction, actual_in_fix_dict = change_instruction(
                    instruction, key_counter, actual_in_fix_dict, env,
                    predefined_instr_dict=predefined_instr_dict, task_instruction=task_instruction,
                )

                if now_instruction != instruction:
                    return now_instruction, actual_in_fix_dict
                else:
                    # replan
                    continue
        else:
            if value > 0:
                continuous_positive_value_cnt += 1
            else:
                continuous_positive_value_cnt = 0

            # use stage repair check
            if check_stage_repair(env, instruction, predefined_instr_dict, task_instruction):
                print("--"*30)
                print(f"Stage repair done, returning to base instruction.")
                print("--"*30)
                now_instruction, actual_in_fix_dict = change_instruction(
                    instruction, key_counter, actual_in_fix_dict, env, 
                    predefined_instr_dict=predefined_instr_dict, task_instruction=task_instruction,
                    force_new_instruction=predefined_instr_dict[task_instruction]['1']
                )
                return now_instruction, actual_in_fix_dict

            # use consecutive positive value check
            if continuous_positive_value_cnt >= 30:
                print("--"*30)
                print(f"Value has been positive for {continuous_positive_value_cnt} consecutive checks, returning to base instruction.")
                print("--"*30)
                now_instruction, actual_in_fix_dict = change_instruction(
                    instruction, key_counter, actual_in_fix_dict, env, 
                    predefined_instr_dict=predefined_instr_dict, task_instruction=task_instruction,
                    force_new_instruction=predefined_instr_dict[task_instruction]['1']
                )
                return now_instruction, actual_in_fix_dict

        _ = cv2.pollKey()
        press_events = key_counter.get_press_events()
        for key_stroke in press_events:
            old_instruction = instruction
            if key_stroke == Key.f2:
                now_instruction, actual_in_fix_dict = change_instruction(
                    old_instruction, key_counter, actual_in_fix_dict, env,
                    predefined_instr_dict=predefined_instr_dict, task_instruction=task_instruction
                )
                return now_instruction, actual_in_fix_dict



        action = get_real_umi_action(raw_action, obs, action_pose_repr)  # (16, 7)
        all_time_actions[[iter_idx], iter_idx:iter_idx + action_horizon] = action

        if inference_idx == steps_per_inference:
            inference_idx = 0

            if temporal_agg:
                # temporal ensemble
                action_seq_for_curr_step = all_time_actions[:, iter_idx:iter_idx + action_horizon]
                target_pose_list = []
                for i in range(action_horizon):
                    ensemble_num = 8
                    actions_for_curr_step = action_seq_for_curr_step[max(0, iter_idx - ensemble_num + 1):iter_idx + 1, i]
                    actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]

                    k = -0.01
                    exp_weights = np.exp(k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    weighted_rotvec = R.from_rotvec(np.array(actions_for_curr_step)[:, 3:6]).mean(weights=exp_weights).as_rotvec()
                    weighted_action = (actions_for_curr_step * exp_weights[:, np.newaxis]).sum(axis=0, keepdims=True)
                    weighted_action[0][3:6] = weighted_rotvec
                    target_pose_list.append(weighted_action)
                this_target_poses = np.concatenate(target_pose_list, axis=0)
            else:
                this_target_poses = action

            assert this_target_poses.shape[1] == len(robots_config) * 7
            for target_pose in this_target_poses:
                for robot_idx in range(len(robots_config)):
                    solve_table_collision(
                        ee_pose=target_pose[robot_idx * 7: robot_idx * 7 + 6],
                        gripper_width=target_pose[robot_idx * 7 + 6],
                        height_threshold=robots_config[robot_idx]['height_threshold']
                    )

                # solve collison between two robots
                solve_sphere_collision(
                    ee_poses=target_pose.reshape([len(robots_config), -1]),
                    robots_config=robots_config
                )

            # deal with timing
            # the same step actions are always the target for
            action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1]
            action_exec_latency = 0.01
            curr_time = time.time()
            is_new = action_timestamps > (curr_time + action_exec_latency)
            if np.sum(is_new) == 0:
                # exceeded time budget, still do something
                this_target_poses = this_target_poses[[-1]]  # (1, 7)
                # schedule on next available step
                next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                action_timestamp = eval_t_start + (next_step_idx) * dt
                print('Over budget', action_timestamp - curr_time)
                action_timestamps = np.array([action_timestamp])
            else:
                this_target_poses = this_target_poses[is_new]
                action_timestamps = action_timestamps[is_new]


            # execute actions
            env.exec_actions(
                actions=this_target_poses,
                timestamps=action_timestamps,
                compensate_latency=True,
            )

        # visualize
        episode_id = env.replay_buffer.n_episodes
        obs_left_img = obs['camera0_rgb'][-1]
        obs_right_img = obs['camera0_rgb'][-1]
        vis_img = np.concatenate([obs_left_img, obs_right_img], axis=1)
        text = 'Episode: {}, Time: {:.1f}'.format(
            episode_id, time.monotonic() - t_start
        )
        cv2.putText(
            vis_img,
            text,
            (10,20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            thickness=1,
            color=(255,255,255)
        )
        cv2.imshow('default', vis_img[...,::-1])

        stop_episode = False

        t_since_start = time.time() - eval_t_start
        if t_since_start > max_duration:
            print("Max Duration reached.")
            stop_episode = True
        if iter_idx + 1 >= max_timesteps:
            print("Max Timesteps reached.")
            stop_episode = True
        if stop_episode:
            env.get_obs()
            # env.end_episode()
            break

        # wait for execution
        precise_wait(t_cycle_end - frame_latency)
        cycle_idx += 1
        iter_idx += 1
        inference_idx += 1



def open_gripper(env: BimanualUmiEnv, steps_per_inference, dt):
    # base 
    this_primivie_action = np.array([0, 0, 0, 0, 0, 0,  0.5], dtype=np.float32)
    obs = env.get_obs()

    action = this_primivie_action[None, :].repeat(steps_per_inference, axis=0)

    original_pose = np.concatenate([
        obs[f'robot{0}_eef_pos'][-1],
        obs[f'robot{0}_eef_rot_axis_angle'][-1]
    ], axis=-1)
    original_gripper = obs[f'robot{0}_gripper_width'][-1]
    print("now_pose:", original_pose, " gripper:", original_gripper)

    this_target_pose = []
    # interpolate to target
    interpolation_steps = steps_per_inference - 2
    for i in range(1, interpolation_steps + 1):
        new_pose = original_pose.copy() + action[0, :6] * (i / interpolation_steps) # [0, 0.1, 0, 0, 0, 0]
        new_gripper = original_gripper + action[0, 6] * (i / interpolation_steps)

        this_target_pose.append(
            np.concatenate([new_pose, new_gripper])
        )

    for i in range(interpolation_steps + 1, steps_per_inference + 1):
        new_pose = original_pose + action[0, :6]
        new_gripper = original_gripper + action[0, 6]
        this_target_pose.append(
            np.concatenate([new_pose, new_gripper])
        )

    this_target_poses = np.stack(this_target_pose, axis=0)
    
    action = this_target_poses

    action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + time.time()

    env.exec_actions(
        actions=this_target_poses,
        timestamps=action_timestamps,
        compensate_latency=True
    )
    time.sleep(3.0)

@click.command()
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--camera_reorder', '-cr', default='0')
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--init_gripper_close', '-igc', is_flag=True, default=False, help="Whether to initialize gripper to open in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=2000000, help='Max duration for each epoch in seconds.')
@click.option('--max_timesteps', '-mt', default=5_000, help='Max steps for each epoch.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)
@click.option('--mirror_swap', is_flag=True, default=False)
@click.option('--temporal_agg', is_flag=True, default=False)
@click.option('--instruction', '-ins', default='', help='Instruction for the task')
@click.option('--remote_host', '-rh', default='localhost', help='Remote host for the policy server')
@click.option('--remote_port', '-rp', default=8000, help='Remote port for the policy server')
@click.option('--state_horizon', '-state_horizon', type=str, default='3,15', help='state down sample steps')
@click.option('--image_horizon', '-image_horizon', type=str, default='', help='image down sample steps')
@click.option('--action_down_sample_steps', '-action_down_sample_steps', type=int, default=3, help='action down sample steps')
@click.option('--getitem_type', '-getitem_type', type=str, default='necessary', help='`getitem_type` in the dataset config of the training code')
@click.option('--fix_gripper', '-fix_gripper', is_flag=True, default=False, help='Whether to fix the gripper width manually.')
def main(output, robot_config, match_camera,
    camera_reorder,
    init_joints, init_gripper_close,
    steps_per_inference, max_duration, max_timesteps,
    frequency, command_latency,
    no_mirror, sim_fov, camera_intrinsics,
    mirror_swap, temporal_agg, instruction, remote_host, remote_port, state_horizon, image_horizon, action_down_sample_steps, getitem_type, fix_gripper):
    max_gripper_width = 0.09
    gripper_speed = 0.2

    # load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))

    # load left-right robot relative transform
    tx_left_right = np.array(robot_config_data['tx_left_right'])
    tx_robot1_robot0 = tx_left_right

    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['grippers']

    state_horizon = [int(x) for x in state_horizon.split(',')]
    state_down_sample_steps = [0] + [x // action_down_sample_steps for x in state_horizon]
    state_down_sample_steps = state_down_sample_steps[::-1]

    image_horizon =  [] if image_horizon == '' else [int(x) for x in image_horizon.split(',')]
    image_down_sample_steps = [0] + [x // action_down_sample_steps for x in image_horizon]
    image_down_sample_steps = image_down_sample_steps[::-1]
    
    print("Instruction:", instruction)

    # setup experiment
    dt = 1/frequency

    obs_res = (224, 224)
    # load fisheye converter
    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )

    print("steps_per_inference:", steps_per_inference)
    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager) as sm, \
            KeystrokeCounter() as key_counter, \
            BimanualUmiEnv(
                output_dir=output,
                robots_config=robots_config,
                grippers_config=grippers_config,
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_reorder=[int(x) for x in camera_reorder],
                init_joints=init_joints,
                init_gripper_open=(not init_gripper_close),

                enable_multi_cam_vis=True,
                multi_cam_vis_resolution=(1080, 1080),
                # latency
                camera_obs_latency=0.17,

                # downsample
                camera_down_sample_steps=image_down_sample_steps,
                robot_down_sample_steps=state_down_sample_steps,
                gripper_down_sample_steps=state_down_sample_steps,

                # obs
                camera_obs_horizon=image_horizon,
                robot_obs_horizon=state_horizon,
                gripper_obs_horizon=state_horizon,
                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                mirror_swap=mirror_swap,
                # action
                max_pos_speed=2.0,
                max_rot_speed=6.0,
                shm_manager=shm_manager) as env:
            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0)

            print("Connecting to policy server")
            policy_client = websocket_client_policy.WebsocketClientPolicy(remote_host, remote_port)

            obs_pose_rep = 'relative'
            action_pose_repr = 'relative'
            print('obs_pose_rep', obs_pose_rep)
            print('action_pose_repr', action_pose_repr)

            print("Warming up policy inference")
            obs = env.get_obs()
            print(env.robots[0].get_all_state()['ActualQ'][-1])
            # import ipdb; ipdb.set_trace()
            episode_start_pose = list()

            start_image, bbox_list = None, None

            for robot_id in range(len(robots_config)):
                pose = np.concatenate([
                    obs[f'robot{robot_id}_eef_pos'],
                    obs[f'robot{robot_id}_eef_rot_axis_angle']
                ], axis=-1)[-1]
                episode_start_pose.append(pose)
            # compile mode like 'reduce-overhead' might need more than one iteration to warm up
            for _ in tqdm(range(1), desc='Warming up', leave=False):
                obs_dict_np = get_pi_obs_dict(
                    env_obs=obs, start_image=start_image, bbox_list=bbox_list, getitem_type=getitem_type,
                    obs_pose_repr=obs_pose_rep,
                    tx_robot1_robot0=tx_robot1_robot0,
                    episode_start_pose=episode_start_pose, instruction=instruction)
                obs_dict_np['is_warm_up'] = np.array([True])
                # for warm up, policy will always return an action
                action = policy_client.infer(obs_dict_np)['actions']
                assert action.shape[-1] == 10 * len(robots_config)
                action = get_real_umi_action(action, obs, action_pose_repr)
                action_horizon = action.shape[0]
                action_dim = action.shape[-1]
                assert action.shape[-1] == 7 * len(robots_config)

            print('Ready!')
            all_time_actions = np.zeros((max_timesteps, max_timesteps + action_horizon, action_dim))

            run_config = {
                'steps_per_inference': steps_per_inference,
                # 'dt': dt,
                'start_image': start_image,
                'bbox_list': bbox_list,
                'getitem_type': getitem_type,
                'obs_pose_rep': obs_pose_rep,
                'action_pose_repr': action_pose_repr,
                'tx_robot1_robot0': tx_robot1_robot0,
                'policy_client': policy_client,
                'robots_config': robots_config,
                'temporal_agg': temporal_agg,
                'all_time_actions': all_time_actions,
                'max_timesteps': max_timesteps,
                'action_horizon': action_horizon,
                'key_counter': key_counter,
                'task_instruction': task_instruction_to_name[instruction],
                'predefined_instr_dict': predefined_instr_dict,
            }
            actual_in_fix_dict = {instruction: True}
            # actual_in_fix_dict = {}
            now_instruction = instruction
            env.start_episode(time.time() + 1.0)
            env.set_record_ins_timestamp(
                ins=instruction,
                timestamp=time.time(),
                type='begin',
            )
            env.set_record_ins_timestamp(
                ins="fail",
                timestamp=time.time(),
                type='begin',
            )
            env.set_record_ins_timestamp(
                ins="direct success",
                timestamp=time.time(),
                type='begin',
            )
            beginning = True
            while True:
                # ========== policy control loop ==============
                try:
                    
                    dt_used = dt

                    now_instruction, actual_in_fix_dict = run_robot_with_instruction(
                        env=env,
                        instruction=now_instruction,
                        max_duration=1000,
                        fix_gripper=fix_gripper,
                        actual_in_fix_dict=actual_in_fix_dict,
                        dt=dt_used,
                        episode_start_pose=None,
                        beginning=beginning,
                        **run_config
                    )
                    if now_instruction == 'The task is successfully completed.' or now_instruction == 'The task directly succeed.':
                        print("Task completed. Stopping.")
                        env.end_episode()
                        break
                    elif now_instruction == 'The task is failed, please try again.':
                        print("Task failed. Stopping.")
                        env.end_episode()
                        break
                    elif now_instruction == 'recall' or now_instruction == 'r' :
                        print("Recall last actions.")
                        now_instruction, actual_in_fix_dict = run_robot_with_recall(
                            env=env,
                            dt=dt_used,
                            key_counter=key_counter,
                            instruction=now_instruction,
                            actual_in_fix_dict=actual_in_fix_dict,
                            recall_steps=2
                        )

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode(save=False)
                    break
                
                print("Stopped.")


# %%
if __name__ == '__main__':
    main()