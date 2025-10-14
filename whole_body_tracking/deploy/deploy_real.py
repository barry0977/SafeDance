from typing import Union
import numpy as np
import time
import torch # type: ignore

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap

import onnxruntime as ort

isaaclab_joint_names = [
    'left_hip_pitch_joint', 
    'right_hip_pitch_joint',
    'waist_yaw_joint',
    'left_hip_roll_joint',
    'right_hip_roll_joint',
    'waist_roll_joint',
    'left_hip_yaw_joint',
    'right_hip_yaw_joint',
    'waist_pitch_joint',
    'left_knee_joint',
    'right_knee_joint',
    'left_shoulder_pitch_joint',
    'right_shoulder_pitch_joint',
    'left_ankle_pitch_joint',
    'right_ankle_pitch_joint',
    'left_shoulder_roll_joint',
    'right_shoulder_roll_joint',
    'left_ankle_roll_joint',
    'right_ankle_roll_joint',
    'left_shoulder_yaw_joint',
    'right_shoulder_yaw_joint',
    'left_elbow_joint',
    'right_elbow_joint',
    'left_wrist_roll_joint',
    'right_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'right_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_wrist_yaw_joint'
]

real_joint_names = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

isaaclab_to_real_reindex = [isaaclab_joint_names.index(name) for name in real_joint_names]
real_to_isaaclab_reindex = [real_joint_names.index(name) for name in isaaclab_joint_names]

class MotionLoader:
    def __init__(self, motion_file):
        data = np.load(motion_file)
        self.joint_pos = data['joint_pos']  # [T, 29]
        self.joint_vel = data['joint_vel']  # [T, 29]
        self.body_pos = data['body_pos_w']  # [T, N, 3]
        self.body_ori = data['body_quat_w'] # [T, N, 4]
        self.body_ang_vel_w = data['body_ang_vel_w'] 
        self.fps = data['fps']
        self.T = self.joint_pos.shape[0]

class Controller:
    def __init__(self) -> None:
        self.remote_controller = RemoteController()
        self.num_actions = 29
        
        
        motion_path = '/home/bcj/whole_body_tracking/motions/lafan_dance1_subject2.npz'
        self.motion_loader =  MotionLoader(motion_path)
        print('Motion Loaded')

        policy_path = '/home/bcj/whole_body_tracking/logs/rsl_rl/g1_flat/2025-08-19_20-41-53_dance1_subject2-new_obs_add_prog-motion_global_root_pos_2/exported/policy.onnx'
        self.session = ort.InferenceSession(policy_path)
        self.obs_name = self.session.get_inputs()[0].name
        self.time_step_name = self.session.get_inputs()[1].name
        print('Policy Loaded')

        self.qj = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj = np.zeros(self.num_actions, dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)

        self.kps = np.array([
            40.18, 99.10, 40.18, 99.10, 28.50, 28.50, 
            40.18, 99.10, 40.18, 99.10, 28.50, 28.50, 
            40.18, 28.50, 28.50,
            14.25, 14.25, 14.25, 14.25, 14.25, 16.78, 16.78,
            14.25, 14.25, 14.25, 14.25, 14.25, 16.78, 16.78
        ], dtype=np.float32)
        self.kds = np.array([
            2.56, 6.31, 2.56, 6.31, 1.81, 1.81, 
            2.56, 6.31, 2.56, 6.31, 1.81, 1.81,
            2.56, 1.81, 1.81,
            0.91, 0.91, 0.91, 0.91, 0.91, 1.07, 1.07,
            0.91, 0.91, 0.91, 0.91, 0.91, 1.07, 1.07
        ], dtype=np.float32)
        self.action_scales = np.array([
            0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
            0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
            0.55, 0.44, 0.44, 
            0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07, 
            0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07
        ], dtype=np.float32)
        self.default_angles = np.array([
            -0.312, 0., 0., 0.669, -0.363, 0.,
            -0.312, 0., 0., 0.669, -0.363, 0.,
            0., 0., 0., 
            0.2, 0.2, 0., 0.6, 0., 0., 0.,
            0.2, -0.2, 0., 0.6, 0., 0., 0.
        ], dtype=np.float32)
        
        self.control_dt = 0.02
        
        self.num_obs = 154
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        
        self.counter = 0

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0
        
        self.lowcmd_topic = "rt/lowcmd"
        self.lowstate_topic = "rt/lowstate"

        self.lowcmd_publisher_ = ChannelPublisher(self.lowcmd_topic, LowCmdHG)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(self.lowstate_topic, LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: LowCmdHG):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.control_dt)
        
        kps = self.kps
        kds = self.kds
        default_pos = self.default_angles
        
        # record the current pos
        init_dof_pos = np.zeros(29, dtype=np.float32)
        for i in range(29):
            init_dof_pos[i] = self.low_state.motor_state[i].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(29):
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[j].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[j].qd = 0
                self.low_cmd.motor_cmd[j].kp = kps[j]
                self.low_cmd.motor_cmd[j].kd = kds[j]
                self.low_cmd.motor_cmd[j].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(29):
                self.low_cmd.motor_cmd[i].q = self.default_angles[i]
                self.low_cmd.motor_cmd[i].qd = 0
                self.low_cmd.motor_cmd[i].kp = self.kps[i]
                self.low_cmd.motor_cmd[i].kd = self.kds[i]
                self.low_cmd.motor_cmd[i].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)
        
    def get_command(self, motion_loader, t):
        return np.concatenate([motion_loader.joint_pos[t], motion_loader.joint_vel[t]], axis=0)

    def get_ref_ang_vel(self, motion_loader, t):
        return motion_loader.body_ang_vel_w[t][9].reshape(1, 3)
        
    def run(self):
        start_time = time.time()
        # Get the current joint position and velocity
        for i in range(29):
            self.qj[i] = self.low_state.motor_state[i].q
            self.dqj[i] = self.low_state.motor_state[i].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        # if self.imu_type == "torso":
        #     # h1 and h1_2 imu is on the torso
        #     # imu data needs to be transformed to the pelvis frame
        #     waist_yaw = self.low_state.motor_state[self.arm_waist_joint2motor_idx[0]].q
        #     waist_yaw_omega = self.low_state.motor_state[self.arm_waist_joint2motor_idx[0]].dq
        #     quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
 
        self.obs[:58] = self.get_command(self.motion_loader, self.counter)
        self.obs[58:61] = gravity_orientation
        self.obs[61:64] = self.get_ref_ang_vel(self.motion_loader, self.counter)
        self.obs[64:67] = ang_vel
        self.obs[67:96] = qj_obs[real_to_isaaclab_reindex]
        self.obs[96:125] = dqj_obs[real_to_isaaclab_reindex]
        self.obs[125:154] = self.action
        
        # Get the action from the policy network
        obs_tensor = np.array(self.obs, dtype=np.float32).reshape(1, -1)  # obs 为一维或二维
        time_step = np.array([[self.counter]], dtype=np.float32)  # 或 float，根据模型要求

        output = self.session.run(None, {self.obs_name: obs_tensor, self.time_step_name: time_step})
        self.action = output[0].squeeze()
        
        # transform action to target_dof_pos
        target_dof_pos = self.default_angles + self.action[isaaclab_to_real_reindex] * self.action_scales

        # Build low cmd
        for i in range(29):
            self.low_cmd.motor_cmd[i].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[i].qd = 0
            self.low_cmd.motor_cmd[i].kp = self.kps[i]
            self.low_cmd.motor_cmd[i].kd = self.kds[i]
            self.low_cmd.motor_cmd[i].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)

        self.counter += 1
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        if self.control_dt - elapsed_time > 0:
            time.sleep(self.control_dt - elapsed_time)
            print(elapsed_time)
        else:
            print(f'Warning: Control loop takes {elapsed_time} s')
            
        

if __name__ == "__main__":
    
    # Initialize DDS communication
    ChannelFactoryInitialize(0, 'eno1')

    controller = Controller()

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")