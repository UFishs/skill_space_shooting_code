from typing import Optional, List
import pathlib
import numpy as np
import time
import shutil
import math
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.rtde_interpolation_controller import RTDEInterpolationController
from umi.real_world.wsg_controller import WSGController
from umi.real_world.franka_interpolation_controller import FrankaInterpolationController
from umi.real_world.multi_uvc_camera import MultiUvcCamera, VideoRecorder
from diffusion_policy.real_world.video_recorder import VideoRecorder as RealsenseVideoRecorder
from diffusion_policy.common.timestamp_accumulator import (
    TimestampActionAccumulator,
    ObsAccumulator
)
from umi.common.cv_util import draw_predefined_mask
from umi.real_world.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)
from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from umi.common.pose_util import pose_to_pos_rot
from umi.common.interpolation_util import get_interp1d, PoseInterpolator
from diffusion_policy.real_world.multi_realsense import MultiRealsense, SingleRealsense
import scipy.interpolate as si

DEFAULT_OBS_KEY_MAP = {
    # robot
    'ActualTCPPose': 'robot_eef_pose',
    'ActualTCPSpeed': 'robot_eef_pose_vel',
    'ActualQ': 'robot_joint',
    'ActualQd': 'robot_joint_vel',
    # timestamps
    'step_idx': 'step_idx',
    'timestamp': 'timestamp'
}

class BimanualUmiEnv:
    def __init__(self, 
            # required params
            output_dir,
            robots_config, # list of dict[{robot_type: 'ur5', robot_ip: XXX, obs_latency: 0.0001, action_latency: 0.1, tcp_offset: 0.21}]
            grippers_config, # list of dict[{gripper_ip: XXX, gripper_port: 1000, obs_latency: 0.01, , action_latency: 0.1}]
            # env params
            frequency=20,
            # obs
            obs_image_resolution=(224,224),
            max_obs_buffer_size=60,
            obs_float32=False,
            camera_reorder=None,
            no_mirror=False,
            fisheye_converter=None,
            mirror_swap=False,
            # this latency compensates receive_timestamp
            # all in seconds
            camera_obs_latency=0.125,
            # all in steps (relative to frequency)
            camera_down_sample_steps=1,
            robot_down_sample_steps=1,
            gripper_down_sample_steps=1,
            # all in steps (relative to frequency)
            camera_obs_horizon=2,
            robot_obs_horizon=2,
            gripper_obs_horizon=2,
            # action
            max_pos_speed=0.25,
            max_rot_speed=0.6,
            init_joints=False,
            init_gripper_open=True,
            # vis params
            enable_multi_cam_vis=True,
            multi_cam_vis_resolution=(960, 960),
            # shared memory
            shm_manager=None,

            # realsense
            use_realsense=False,
            camera_serial_numbers=None,
            obs_key_map=DEFAULT_OBS_KEY_MAP,
            video_capture_fps=30,
            video_capture_resolution=(640,480),
            record_raw_video=True,
            thread_per_video=2,
            video_crf=21,
            realsense_obs_resolution=(224,224),

            ):
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')
        
        ins_txt_path = str(output_dir.joinpath('instructions.txt').absolute())
        ins_txt_buffer = open(ins_txt_path, 'a', buffering=1)



        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        # Find and reset all Elgato capture cards.
        # Required to workaround a firmware bug.
        reset_all_elgato_devices()

        # Wait for all v4l cameras to be back online
        time.sleep(0.1)
        v4l_paths = get_sorted_v4l_paths()
        if camera_reorder is not None:
            paths = [v4l_paths[i] for i in camera_reorder]
            v4l_paths = paths

        if use_realsense:
            if camera_serial_numbers is None:
                camera_serial_numbers = SingleRealsense.get_connected_devices_serial()

            color_tf = get_image_transform(
            input_res=video_capture_resolution,
            output_res=realsense_obs_resolution, 
            # obs output rgb
            bgr_to_rgb=True)

            color_transform = color_tf
            if obs_float32:
                color_transform = lambda x: color_tf(x).astype(np.float32) / 255

            def transform(data):
                data['color'] = color_transform(data['color'])
                return data
            
            rw, rh, col, row = optimal_row_cols(
                n_cameras=len(camera_serial_numbers),
                in_wh_ratio=realsense_obs_resolution[0]/realsense_obs_resolution[1],
                max_resolution=multi_cam_vis_resolution
            )
            vis_color_transform = get_image_transform(
                input_res=video_capture_resolution,
                output_res=(rw,rh),
                bgr_to_rgb=False
            )
            def vis_transform(data):
                data['color'] = vis_color_transform(data['color'])
                return data

            recording_transfrom = None
            recording_fps = video_capture_fps
            recording_pix_fmt = 'bgr24'
            if not record_raw_video:
                recording_transfrom = transform
                recording_fps = frequency
                recording_pix_fmt = 'rgb24'
            

            realsense_video_recorder = RealsenseVideoRecorder.create_h264(
                fps=recording_fps, 
                codec='h264',
                input_pix_fmt=recording_pix_fmt, 
                crf=video_crf,
                thread_type='FRAME',
                thread_count=thread_per_video)

            

            realsense = MultiRealsense(
                serial_numbers=camera_serial_numbers,
                shm_manager=shm_manager,
                resolution=video_capture_resolution,
                capture_fps=video_capture_fps,
                put_fps=video_capture_fps,
                # send every frame immediately after arrival
                # ignores put_fps
                put_downsample=False,
                record_fps=recording_fps,
                enable_color=True,
                enable_depth=True,
                enable_infrared=False,
                get_max_k=max_obs_buffer_size,
                transform=transform,
                vis_transform=vis_transform,
                recording_transform=recording_transfrom,
                video_recorder=realsense_video_recorder,
                verbose=False
            )

            
            self.realsense = realsense



        # compute resolution for vis
        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(v4l_paths),
            in_wh_ratio=1,
            max_resolution=multi_cam_vis_resolution
        )

        # HACK: Separate video setting for each camera
        # Elagto Cam Link 4k records at 4k 30fps
        # Other capture card records at 720p 60fps
        resolution = list()
        capture_fps = list()
        cap_buffer_size = list()
        video_recorder = list()
        transform = list()
        vis_transform = list()
        for path in v4l_paths:
            if 'Cam_Link_4K' in path:
                res = (3840, 2160)
                fps = 30
                buf = 3
                bit_rate = 6000*1000
                def tf4k(data, input_res=res):
                    img = data['color']
                    f = get_image_transform(
                        input_res=input_res,
                        output_res=obs_image_resolution, 
                        # obs output rgb
                        bgr_to_rgb=True)
                    img = f(img)
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                    data['color'] = img
                    return data
                transform.append(tf4k)
            else:
                res = (1920, 1080)
                fps = 60
                buf = 1
                bit_rate = 3000*1000

                is_mirror = None
                if mirror_swap:
                    mirror_mask = np.ones((224,224,3),dtype=np.uint8)
                    mirror_mask = draw_predefined_mask(
                        mirror_mask, color=(0,0,0), mirror=True, gripper=False, finger=False)
                    is_mirror = (mirror_mask[...,0] == 0)
                
                def tf(data, input_res=res):
                    img = data['color']
                    if fisheye_converter is None:
                        f = get_image_transform(
                            input_res=input_res,
                            output_res=obs_image_resolution, 
                            # obs output rgb
                            bgr_to_rgb=True)
                        img = np.ascontiguousarray(f(img))
                        if is_mirror is not None:
                            img[is_mirror] = img[:,::-1,:][is_mirror]
                        img = draw_predefined_mask(img, color=(0,0,0), 
                            mirror=no_mirror, gripper=True, finger=False, use_aa=True)
                    else:
                        img = fisheye_converter.forward(img)
                        img = img[...,::-1]
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                    data['color'] = img
                    return data
                transform.append(tf)

            resolution.append(res)
            capture_fps.append(fps)
            cap_buffer_size.append(buf)
            video_recorder.append(VideoRecorder.create_hevc_nvenc(
                fps=fps,
                input_pix_fmt='bgr24',
                bit_rate=bit_rate
            ))

            def vis_tf(data, input_res=res):
                img = data['color']
                f = get_image_transform(
                    input_res=input_res,
                    output_res=(rw,rh),
                    bgr_to_rgb=False
                )
                img = f(img)
                data['color'] = img
                return data
            vis_transform.append(vis_tf)

        camera = MultiUvcCamera(
            dev_video_paths=v4l_paths,
            shm_manager=shm_manager,
            resolution=resolution,
            capture_fps=capture_fps,
            # send every frame immediately after arrival
            # ignores put_fps
            put_downsample=False,
            get_max_k=max_obs_buffer_size,
            receive_latency=camera_obs_latency,
            cap_buffer_size=cap_buffer_size,
            transform=transform,
            vis_transform=vis_transform,
            video_recorder=video_recorder,
            verbose=False
        )

        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                camera=camera,
                row=row,
                col=col,
                rgb_to_bgr=False
            )

        cube_diag = np.linalg.norm([1,1,1])
        j_init = np.array([0,-90,-90,-90,90,0]) / 180 * np.pi

        if not init_joints:
            j_init = None

        assert len(robots_config) == len(grippers_config)
        robots: List[RTDEInterpolationController] = list()
        grippers: List[WSGController] = list()
        for rc in robots_config:
            if rc['robot_type'].startswith('ur5'):
                assert rc['robot_type'] in ['ur5', 'ur5e']
                this_robot = RTDEInterpolationController(
                    shm_manager=shm_manager,
                    robot_ip=rc['robot_ip'],
                    frequency=500 if rc['robot_type'] == 'ur5e' else 125,
                    lookahead_time=0.1,
                    gain=300,
                    max_pos_speed=max_pos_speed*cube_diag,
                    max_rot_speed=max_rot_speed*cube_diag,
                    launch_timeout=3,
                    tcp_offset_pose=[0, 0, rc['tcp_offset'], 0, 0, 0],
                    payload_mass=None,
                    payload_cog=None,
                    joints_init=j_init,
                    joints_init_speed=1.05,
                    soft_real_time=False,
                    verbose=False,
                    receive_keys=None,
                    receive_latency=rc['robot_obs_latency']
                )
            elif rc['robot_type'].startswith('franka'):
                this_robot = FrankaInterpolationController(
                    shm_manager=shm_manager,
                    robot_ip=rc['robot_ip'],
                    frequency=200,
                    Kx_scale=1.0,
                    Kxd_scale=np.array([2.0,1.5,2.0,1.0,1.0,1.0]),
                    joints_init=j_init,
                    joints_init_duration=4,
                    verbose=False,
                    receive_latency=rc['robot_obs_latency']
                )
            else:
                raise NotImplementedError()
            robots.append(this_robot)

        for gc in grippers_config:
            this_gripper = WSGController(
                shm_manager=shm_manager,
                hostname=gc['gripper_ip'],
                port=gc['gripper_port'],
                receive_latency=gc['gripper_obs_latency'],
                use_meters=True,
                home_to_open=init_gripper_open
            )

            grippers.append(this_gripper)

        self.camera = camera
        self.use_realsense = use_realsense
        if not use_realsense:
            self.realsense = None

        self.robots = robots
        self.robots_config = robots_config
        self.grippers = grippers
        self.grippers_config = grippers_config

        self.multi_cam_vis = multi_cam_vis
        self.frequency = frequency
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        # timing
        self.camera_obs_latency = camera_obs_latency
        self.camera_down_sample_steps = camera_down_sample_steps
        self.robot_down_sample_steps = robot_down_sample_steps
        self.gripper_down_sample_steps = gripper_down_sample_steps
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon
        self.gripper_obs_horizon = gripper_obs_horizon
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        # temp memory buffers
        self.last_camera_data = None
        if use_realsense:
            self.last_realsense_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None

        self.start_time = None
        self.last_time_step = 0

        self.record_ins_timestamp_dict = {}
        self.ins_txt_buffer = ins_txt_buffer

        # fix interval recording
        self.fix_intervals = []
        self.fix_active_since = None
    
    # ======== start-stop API =============
    @property
    def is_ready(self):
        ready_flag = self.camera.is_ready
        if self.use_realsense:
            ready_flag = ready_flag and self.realsense.is_ready
        for robot in self.robots:
            ready_flag = ready_flag and robot.is_ready
        for gripper in self.grippers:
            ready_flag = ready_flag and gripper.is_ready

        if not ready_flag:
            print('Not ready:  ', {
                'camera': self.camera.is_ready,
                'realsense': self.realsense.is_ready if self.use_realsense else 'N/A',
                'robots': [robot.is_ready for robot in self.robots],
                'grippers': [gripper.is_ready for gripper in self.grippers]
            })
        return ready_flag
    
    def start(self, wait=True):
        self.camera.start(wait=False)
        if self.use_realsense:
            self.realsense.start(wait=False)
        for robot in self.robots:
            robot.start(wait=False)
        for gripper in self.grippers:
            gripper.start(wait=False)

        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.end_episode()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        for robot in self.robots:
            robot.stop(wait=False)
        for gripper in self.grippers:
            gripper.stop(wait=False)
        self.camera.stop(wait=False)
        if self.use_realsense:
            self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.camera.start_wait()
        if self.use_realsense:
            self.realsense.start_wait()
        for robot in self.robots:
            robot.start_wait()
        for gripper in self.grippers:
            gripper.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()
    
    def stop_wait(self):
        for robot in self.robots:
            robot.stop_wait()
        for gripper in self.grippers:
            gripper.stop_wait()
        self.camera.stop_wait()
        if self.use_realsense:
            self.realsense.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self) -> dict:
        """
        Timestamp alignment policy
        We assume the cameras used for obs are always [0, k - 1], where k is the number of robots
        All other cameras, find corresponding frame with the nearest timestamp
        All low-dim observations, interpolate with respect to 'current' time
        """

        "observation dict"
        while not self.is_ready:
            print('Waiting for all devices to be ready...')
            time.sleep(0.1)
        assert self.is_ready

        # get data
        # 60 Hz, camera_calibrated_timestamp
        if type(self.camera_down_sample_steps) == int:
            k = math.ceil(
                self.camera_obs_horizon * self.camera_down_sample_steps \
                * (60 / self.frequency)) + 2 # here 2 is adjustable, typically 1 should be enough
        elif type(self.camera_down_sample_steps) == list:
            k = math.ceil(max(self.camera_down_sample_steps) * (60 / self.frequency)) + 2
        # print('==>k  ', k, self.camera_obs_horizon, self.camera_down_sample_steps, self.frequency)
        self.last_camera_data = self.camera.get(
            k=k, 
            out=self.last_camera_data)
        if self.use_realsense:
            self.last_realsense_data = self.realsense.get(
                k=k,
                out=self.last_realsense_data)

        # both have more than n_obs_steps data
        last_robots_data = list()
        last_grippers_data = list()
        # 125/500 hz, robot_receive_timestamp
        for robot in self.robots:
            last_robots_data.append(robot.get_all_state())
        # 30 hz, gripper_receive_timestamp
        for gripper in self.grippers:
            last_grippers_data.append(gripper.get_all_state())

        # select align_camera_idx
        num_obs_cameras = len(self.robots)
        if self.use_realsense:
            # attach all realsense data to camera data
            num_realsense_cameras = len(self.last_realsense_data.keys())
            for realsense_camera_idx in range(num_realsense_cameras):
                self.last_camera_data[num_obs_cameras + realsense_camera_idx] = self.last_realsense_data[realsense_camera_idx]
            num_obs_cameras += num_realsense_cameras



        align_camera_idx = None
        running_best_error = np.inf
   
        for camera_idx in range(num_obs_cameras):
            this_error = 0
            this_timestamp = self.last_camera_data[camera_idx]['timestamp'][-1]
            for other_camera_idx in range(num_obs_cameras):
                if other_camera_idx == camera_idx:
                    continue
                other_timestep_idx = -1
                while True:
                    if self.last_camera_data[other_camera_idx]['timestamp'][other_timestep_idx] < this_timestamp:
                        this_error += this_timestamp - self.last_camera_data[other_camera_idx]['timestamp'][other_timestep_idx]
                        break
                    other_timestep_idx -= 1
            if align_camera_idx is None or this_error < running_best_error:
                running_best_error = this_error
                align_camera_idx = camera_idx

        last_timestamp = self.last_camera_data[align_camera_idx]['timestamp'][-1]
        dt = 1 / self.frequency

        # align camera obs timestamps
        if type(self.camera_down_sample_steps) == int:
            camera_obs_timestamps = last_timestamp - (
                np.arange(self.camera_obs_horizon)[::-1] * self.camera_down_sample_steps * dt)
        elif type(self.camera_down_sample_steps) == list:
            camera_obs_timestamps = last_timestamp - np.array(self.camera_down_sample_steps) * dt
        camera_obs = dict()
        for camera_idx, value in self.last_camera_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in camera_obs_timestamps:
                nn_idx = np.argmin(np.abs(this_timestamps - t))
                # if np.abs(this_timestamps - t)[nn_idx] > 1.0 / 120 and camera_idx != 3:
                #     print('ERROR!!!  ', camera_idx, len(this_timestamps), nn_idx, (this_timestamps - t)[nn_idx-1: nn_idx+2])
                this_idxs.append(nn_idx)
            # remap key
            camera_obs[f'camera{camera_idx}_rgb'] = value['color'][this_idxs]

        # obs_data to return (it only includes camera data at this stage)
        obs_data = dict(camera_obs)

        # include camera timesteps
        obs_data['timestamp'] = camera_obs_timestamps

        # align robot obs
        if type(self.robot_down_sample_steps) == int:
            robot_obs_timestamps = last_timestamp - (
                np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt)
        elif type(self.robot_down_sample_steps) == list:
            robot_obs_timestamps = last_timestamp - np.array(self.robot_down_sample_steps) * dt
        for robot_idx, last_robot_data in enumerate(last_robots_data):
            robot_pose_interpolator = PoseInterpolator(
                t=last_robot_data['robot_timestamp'], 
                x=last_robot_data['ActualTCPPose'])
            robot_pose = robot_pose_interpolator(robot_obs_timestamps)
            robot_obs = {
                f'robot{robot_idx}_eef_pos': robot_pose[...,:3],
                f'robot{robot_idx}_eef_rot_axis_angle': robot_pose[...,3:]
            }
            # update obs_data
            obs_data.update(robot_obs)

        # align gripper obs
        if type(self.gripper_down_sample_steps) == int:
            gripper_obs_timestamps = last_timestamp - (
                np.arange(self.gripper_obs_horizon)[::-1] * self.gripper_down_sample_steps * dt)
        elif type(self.gripper_down_sample_steps) == list:
            gripper_obs_timestamps = last_timestamp - np.array(self.gripper_down_sample_steps) * dt
        for robot_idx, last_gripper_data in enumerate(last_grippers_data):
            # align gripper obs
            gripper_interpolator = get_interp1d(
                t=last_gripper_data['gripper_timestamp'],
                x=last_gripper_data['gripper_position'][...,None]
            )
            gripper_obs = {
                f'robot{robot_idx}_gripper_width': gripper_interpolator(gripper_obs_timestamps)
            }

            # update obs_data
            obs_data.update(gripper_obs)

        # accumulate obs
        if self.obs_accumulator is not None:
            for robot_idx, last_robot_data in enumerate(last_robots_data):
                self.obs_accumulator.put(
                    data={
                        f'robot{robot_idx}_eef_pose': last_robot_data['ActualTCPPose'],
                        f'robot{robot_idx}_joint_pos': last_robot_data['ActualQ'],
                        f'robot{robot_idx}_joint_vel': last_robot_data['ActualQd'],
                    },
                    timestamps=last_robot_data['robot_timestamp']
                )

            for robot_idx, last_gripper_data in enumerate(last_grippers_data):
                self.obs_accumulator.put(
                    data={
                        f'robot{robot_idx}_gripper_width': last_gripper_data['gripper_position'][...,None]
                    },
                    timestamps=last_gripper_data['gripper_timestamp']
                )
            
            for camera_idx, value in self.last_camera_data.items():
                image = np.array(value['color']*255, dtype=np.uint8)
                self.obs_accumulator.put(
                    data={
                        f'camera{camera_idx}_rgb': image,
                    },
                    timestamps=value['timestamp']
                )

        return obs_data
    
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray,
            compensate_latency=False,
            start_action_flag=False):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]

        assert new_actions.shape[1] // len(self.robots) == 7
        assert new_actions.shape[1] % len(self.robots) == 0

        # if len(new_actions) > 0:
        #     for robot in self.robots:
        #         # clean the action queue to guarantee real-time control
        #         robot.clear_buffer(target_time=new_timestamps[-1] - 0.01)

        # schedule waypoints
        for i in range(len(new_actions)):
            for robot_idx, (robot, gripper, rc, gc) in enumerate(zip(self.robots, self.grippers, self.robots_config, self.grippers_config)):
                r_latency = rc['robot_action_latency'] if compensate_latency else 0.0
                g_latency = gc['gripper_action_latency'] if compensate_latency else 0.0
                r_actions = new_actions[i, 7 * robot_idx + 0: 7 * robot_idx + 6]
                g_actions = new_actions[i, 7 * robot_idx + 6]
                if start_action_flag and i == 0:
                    robot.schedule_reset_waypoint(
                        pose=r_actions,
                        target_time=new_timestamps[i] - r_latency
                    )
                else:
                    robot.schedule_waypoint(
                        pose=r_actions,
                        target_time=new_timestamps[i] - r_latency
                    )
                gripper.schedule_waypoint(
                    pos=g_actions,
                    target_time=new_timestamps[i] - g_latency
                )

        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
    
    def get_robot_state(self):
        return [robot.get_state() for robot in self.robots]
    
    def get_gripper_state(self):
        return [gripper.get_state() for gripper in self.grippers]

    # recording API

    def set_fix_status(self, is_fix: bool, timestamp=None):
        """
        Update whether the current episode is inside a fix segment.

        A fix segment is represented internally as:
            [begin_timestamp, end_timestamp)

        Repeated calls with the same status are safe.
        """

        if timestamp is None:
            timestamp = time.time()

        timestamp = float(timestamp)

        if is_fix:
            # 0 -> 1
            if self.fix_active_since is None:
                self.fix_active_since = timestamp
                print(f"[Fix] Begin at {timestamp:.6f}")

        else:
            # 1 -> 0
            if self.fix_active_since is not None:
                begin_time = self.fix_active_since
                end_time = timestamp

                if end_time > begin_time:
                    self.fix_intervals.append(
                        (begin_time, end_time)
                    )

                    print(
                        f"[Fix] End at {end_time:.6f}, "
                        f"interval=({begin_time:.6f}, {end_time:.6f})"
                    )

                self.fix_active_since = None


    def get_is_fix(self, timestamps):
        """
        Convert recorded fix time intervals into a frame-level bool array.

        timestamps:
            shape (N,)

        returns:
            is_fix:
                shape (N,)
                False -> normal frame
                True  -> fix frame
        """

        timestamps = np.asarray(timestamps)

        is_fix = np.zeros(
            len(timestamps),
            dtype=bool,
        )

        # Finished fix intervals
        for begin_time, end_time in self.fix_intervals:
            is_fix |= (
                (timestamps >= begin_time)
                & (timestamps < end_time)
            )

        # Fix is still active when data is being saved
        if self.fix_active_since is not None:
            is_fix |= timestamps >= self.fix_active_since

        return is_fix

    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

         # ========================================================
        # Reset fix recording
        # ========================================================
        self.fix_intervals = []
        self.fix_active_since = None

        assert self.is_ready

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.camera.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))
        
        # start recording on camera
        self.camera.restart_put(start_time=start_time)
        self.camera.start_recording(video_path=video_paths, start_time=start_time)

        if self.use_realsense:
            realsense_video_paths = list()
            for i in range(self.realsense.n_cameras):
                realsense_video_paths.append(
                    str(this_video_dir.joinpath(f'realsense_{i}.mp4').absolute()))
            self.realsense.restart_put(start_time=start_time)
            self.realsense.start_recording(
                video_path=realsense_video_paths,
                start_time=start_time
            )


        # create accumulators
        self.obs_accumulator = ObsAccumulator()
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        # print(f'Episode {episode_id} started!')
    
    def end_episode(self, save=False):
        "Stop recording"
        assert self.is_ready
        
        # stop video recorder
        self.camera.stop_recording()
        if self.use_realsense:
            self.realsense.stop_recording()

        # TODO
        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None

            if save:
                # Since the only way to accumulate obs and action is by calling
                # get_obs and exec_actions, which will be in the same thread.
                # We don't need to worry new data come in here.
                end_time = float('inf')
                for key, value in self.obs_accumulator.timestamps.items():
                    end_time = min(end_time, value[-1])
                end_time = min(end_time, self.action_accumulator.timestamps[-1])

                actions = self.action_accumulator.actions
                action_timestamps = self.action_accumulator.timestamps
                n_steps = 0
                if np.sum(self.action_accumulator.timestamps <= end_time) > 0:
                    n_steps = np.nonzero(self.action_accumulator.timestamps <= end_time)[0][-1]+1

                if n_steps > 0:
                    timestamps = action_timestamps[:n_steps]
                    episode = {
                        'timestamp': timestamps,
                        'actions': actions[:n_steps],
                        'is_fix': self.get_is_fix(timestamps),
                    }
                    for robot_idx in range(len(self.robots)):
                        robot_pose_interpolator = PoseInterpolator(
                            t=np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_eef_pose']),
                            x=np.array(self.obs_accumulator.data[f'robot{robot_idx}_eef_pose'])
                        )
                        robot_pose = robot_pose_interpolator(timestamps)
                        episode[f'robot{robot_idx}_eef_pos'] = robot_pose[:,:3]
                        episode[f'robot{robot_idx}_eef_rot_axis_angle'] = robot_pose[:,3:]
                        joint_pos_interpolator = get_interp1d(
                            np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_joint_pos']),
                            np.array(self.obs_accumulator.data[f'robot{robot_idx}_joint_pos'])
                        )
                        joint_vel_interpolator = get_interp1d(
                            np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_joint_vel']),
                            np.array(self.obs_accumulator.data[f'robot{robot_idx}_joint_vel'])
                        )
                        episode[f'robot{robot_idx}_joint_pos'] = joint_pos_interpolator(timestamps)
                        episode[f'robot{robot_idx}_joint_vel'] = joint_vel_interpolator(timestamps)

                        # np.all(np.abs(np.diff(episode['robot0_joint_vel']))<1e-2, axis=-1)
                        # np.all(np.abs(np.diff(actions[:n_steps]))<1e-2, axis=-1)

                        gripper_interpolator = get_interp1d(
                            t=np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_gripper_width']),
                            x=np.array(self.obs_accumulator.data[f'robot{robot_idx}_gripper_width'])
                        )
                        episode[f'robot{robot_idx}_gripper_width'] = gripper_interpolator(timestamps)
                        

                        index_interp = si.interp1d(
                            self.obs_accumulator.timestamps['camera0_rgb'], 
                            np.arange(len(self.obs_accumulator.timestamps['camera0_rgb'])), 
                            kind='nearest', bounds_error=False, fill_value=(0, len(self.obs_accumulator.timestamps['camera0_rgb'])-1))
                        indices = np.array(index_interp(timestamps), dtype=int)
                        selected_images = np.array(self.obs_accumulator.data['camera0_rgb'])[indices]
                        episode['camera0_rgb'] = selected_images

                        # filter idle data
                        is_idle_from_joint_vel = np.all(np.abs(np.diff(episode['robot0_joint_vel'], axis=0))<1e-2, axis=-1)
                        is_idle_from_image = np.abs(np.diff(indices))<1e-2

                        is_idle = is_idle_from_joint_vel | is_idle_from_image
                        is_idle = np.concatenate([is_idle, [False]])



                        for k, v in episode.items():
                            episode[k] = v[~is_idle]

                        # episode = {k: v[~is_idle] for k, v in episode.items()}

                        demo_start_pose = np.concatenate([
                            episode[f'robot{robot_idx}_eef_pos'][0],
                            episode[f'robot{robot_idx}_eef_rot_axis_angle'][0]
                        ])
                        episode['demo_start_pose'] = np.repeat(demo_start_pose[None], len(episode[f'robot{robot_idx}_eef_pos']), axis=0)


                    self.replay_buffer.add_episode(episode, compressors='disk')
                    episode_id = self.replay_buffer.n_episodes - 1
                    # print(f'Episode {episode_id} saved!')
            
            self.obs_accumulator = None
            self.action_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')

    def robot_reset_to_init(self):
        for robot in self.robots:
            robot.reset_to_init()
        for gripper in self.grippers:
            gripper.reset_to_init()

    def set_record_ins_timestamp(self, ins, timestamp, type):
        '''
        type: begin/end
        '''
        if ins not in self.record_ins_timestamp_dict:
            self.record_ins_timestamp_dict[ins] = {}
        
        if type == 'begin':
            self.record_ins_timestamp_dict[ins]['begin'] = timestamp
        elif type == 'end':
            self.record_ins_timestamp_dict[ins]['end'] = timestamp
        else:
            raise ValueError(f'Invalid type {type} for set_record_ins_timestamp')
        
        print(f'Set record instruction {ins} timestamp: {self.record_ins_timestamp_dict[ins]}')

        if type == 'end' and 'begin' in self.record_ins_timestamp_dict[ins]:
            self.save_ins_timestamp_to_disk(ins)

    
    def delete_ins_from_record(self, ins):
        if ins in self.record_ins_timestamp_dict:
            del self.record_ins_timestamp_dict[ins]
            print(f'Deleted record instruction {ins} from record_ins_timestamp_dict')

    def save_ins_timestamp_to_disk(self, ins):
        
        this_begin_time = self.record_ins_timestamp_dict[ins]['begin']
        this_end_time = self.record_ins_timestamp_dict[ins]['end']


        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            end_time = float('inf')
            for key, value in self.obs_accumulator.timestamps.items():
                end_time = min(end_time, value[-1])
            end_time = min(end_time, self.action_accumulator.timestamps[-1])
            end_time = min(end_time, this_end_time)

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps

            n_steps = 0
            
            if np.sum(self.action_accumulator.timestamps <= end_time) > 0:
                end_steps = np.nonzero(self.action_accumulator.timestamps <= end_time)[0][-1]+1
            if np.sum(self.action_accumulator.timestamps >= this_begin_time) > 0:
                begin_steps = np.nonzero(self.action_accumulator.timestamps >= this_begin_time)[0][0]
                if end_steps > begin_steps:
                    n_steps = end_steps - begin_steps

            if n_steps > 0:
                timestamps = action_timestamps[begin_steps:end_steps]
                episode = {
                    'timestamp': timestamps,
                    'actions': actions[begin_steps:end_steps],
                    # frame-level fix mask
                    'is_fix': self.get_is_fix(timestamps),
                }
                for robot_idx in range(len(self.robots)):
                    robot_pose_interpolator = PoseInterpolator(
                        t=np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_eef_pose']),
                        x=np.array(self.obs_accumulator.data[f'robot{robot_idx}_eef_pose'])
                    )
                    robot_pose = robot_pose_interpolator(timestamps)
                    episode[f'robot{robot_idx}_eef_pos'] = robot_pose[:,:3]
                    episode[f'robot{robot_idx}_eef_rot_axis_angle'] = robot_pose[:,3:]
                    joint_pos_interpolator = get_interp1d(
                        np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_joint_pos']),
                        np.array(self.obs_accumulator.data[f'robot{robot_idx}_joint_pos'])
                    )
                    joint_vel_interpolator = get_interp1d(
                        np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_joint_vel']),
                        np.array(self.obs_accumulator.data[f'robot{robot_idx}_joint_vel'])
                    )
                    episode[f'robot{robot_idx}_joint_pos'] = joint_pos_interpolator(timestamps)
                    episode[f'robot{robot_idx}_joint_vel'] = joint_vel_interpolator(timestamps)


                    gripper_interpolator = get_interp1d(
                        t=np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_gripper_width']),
                        x=np.array(self.obs_accumulator.data[f'robot{robot_idx}_gripper_width'])
                    )
                    episode[f'robot{robot_idx}_gripper_width'] = gripper_interpolator(timestamps)
                    
                    

                    index_interp = si.interp1d(
                        self.obs_accumulator.timestamps['camera0_rgb'], 
                        np.arange(len(self.obs_accumulator.timestamps['camera0_rgb'])), 
                        kind='nearest', bounds_error=False, fill_value=(0, len(self.obs_accumulator.timestamps['camera0_rgb'])-1))
                    indices = np.array(index_interp(timestamps), dtype=int)
                    selected_images = np.array(self.obs_accumulator.data['camera0_rgb'])[indices]
                    episode['camera0_rgb'] = selected_images

                    # filter idle data
                    is_idle_from_joint_vel = np.all(np.abs(np.diff(episode['robot0_joint_vel'], axis=0))<1e-2, axis=-1)
                    is_idle_from_gripper_width = np.all(np.abs(np.diff(episode[f'robot{robot_idx}_gripper_width'], axis=0))<1e-2, axis=-1)
                    is_idle_from_image = np.abs(np.diff(indices))<1e-2
                    is_idle = is_idle_from_joint_vel & is_idle_from_gripper_width
                    is_idle = np.concatenate([is_idle, [False]])



                    for k, v in episode.items():
                        episode[k] = v[~is_idle]

                    # episode = {k: v[~is_idle] for k, v in episode.items()}

                    demo_start_pose = np.concatenate([
                        episode[f'robot{robot_idx}_eef_pos'][0],
                        episode[f'robot{robot_idx}_eef_rot_axis_angle'][0]
                    ])
                    episode['demo_start_pose'] = np.repeat(demo_start_pose[None], len(episode[f'robot{robot_idx}_eef_pos']), axis=0)


                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                # print(f'Episode {episode_id} saved!')
                self.ins_txt_buffer.write(ins+'\n')
                self.ins_txt_buffer.flush()
                self.delete_ins_from_record(ins)
                
            