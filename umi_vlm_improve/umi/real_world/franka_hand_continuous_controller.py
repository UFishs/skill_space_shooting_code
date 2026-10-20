import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.precise_sleep import precise_wait
from umi.real_world.wsg_binary_driver import WSGBinaryDriver
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
import zerorpc


class Command(enum.Enum):
    STOP = 0
    OPEN_GRIPPER = 1
    CLOSE_GRIPPER = 2
    RESTART_PUT = 3
    SCHEDULE_WAYPOINT = 4


class FrankaHandInterface:
    def __init__(self, ip='172.16.0.1', port=4243):
        self.client = zerorpc.Client(heartbeat=20)
        self.client.connect(f"tcp://{ip}:{port}")

    def get_gripper_position(self):
        return self.client.get_gripper_position()

    def open_gripper(self):
        self.client.open_gripper()

    def close_gripper(self):
        self.client.close_gripper()

    def gripper_continuous_control(self, target_pos, target_vel):
        self.client.gripper_continuous_control(target_pos, target_vel)

    def close(self):
        self.client.close()


class FrankaHandController(mp.Process):
    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 robot_ip,
                 hand_port=4243,
                 frequency=1000,
                 launch_timeout=3,
                 move_max_speed=200.0,
                 verbose=False,
                 get_max_k=None,
                 command_queue_size=1024,
                 receive_latency=0.0
                 ):
        """
        robot_ip: the ip of the middle-layer controller (NUC)
        frequency: 1000 for franka
        """
        super().__init__(name="FrankaHandController")
        self.robot_ip = robot_ip
        self.hand_port = hand_port
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.verbose = verbose
        self.move_max_speed = move_max_speed

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )

        # build ring buffer
        example = {
            'gripper_position': 0.0,
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[FrankaHandController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def open_gripper(self):
        message = {
            'cmd': Command.OPEN_GRIPPER.value,
        }
        self.input_queue.put(message)

    def close_gripper(self):
        message = {
            'cmd': Command.CLOSE_GRIPPER.value,
        }
        self.input_queue.put(message)

    def restart_put(self, start_time):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })

    def schedule_waypoint(self, pos: float, target_time: float):
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= main loop in process ============
    def run(self):
        # start polymetis interface
        franka_hand = FrankaHandInterface(self.robot_ip, self.hand_port)
        try:
            if self.verbose:
                print(f"[FrankaHandController] Connect to robot hand: {self.robot_ip}")

            # main loop
            dt = 1. / self.frequency
            curr_pos = franka_hand.get_gripper_position()
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[[curr_pos,0,0,0,0,0]]
            )
            keep_running = True
            t_start = time.monotonic()
            iter_idx = 0
            while keep_running:
                # send command to robot
                t_now = time.monotonic()
                t_target = t_now
                target_pos = pose_interp(t_target)[0]
                # TODO
                target_vel = (target_pos - pose_interp(t_target - dt)[0]) / dt
                
                franka_hand.gripper_continuous_control(target_pos, target_vel)
                # update robot state
                state = dict()
                state['gripper_position'] = franka_hand.get_gripper_position()
                state['gripper_receive_timestamp'] = time.time()
                state['gripper_timestamp'] = time.time() - self.receive_latency
                print('controller', target_pos, target_vel, state['gripper_position'] * 1000)
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.OPEN_GRIPPER.value:
                        franka_hand.open_gripper()
                    elif cmd == Command.CLOSE_GRIPPER.value:
                        franka_hand.close_gripper()
                    elif cmd == Command.RESTART_PUT.value:
                        t_start = command['target_time'] - time.time() + time.monotonic()
                        iter_idx = 1
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pos = command['target_pos']
                        target_time = command['target_time']
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=[target_pos, 0, 0, 0, 0, 0],
                            time=target_time,
                            max_pos_speed=self.move_max_speed,
                            max_rot_speed=self.move_max_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                # regulate frequency
                t_wait_util = t_start + iter_idx* dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # if self.verbose:
                #     print(f"[FrankaHandController] Actual frequency {1 / (time.monotonic() - t_now)}")

        finally:
            del franka_hand
            self.ready_event.set()
            if self.verbose:
                print(f"[FrankaHandController] Disconnected from robot hand: {self.robot_ip}")