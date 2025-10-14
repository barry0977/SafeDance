import re
import time
import mujoco, mujoco_viewer, mujoco.viewer
import numpy as np
import torch
import yaml
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms
import onnxruntime as ort
import argparse

isaaclab_joint_names = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

mujoco_joint_names = [
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

stiffness_dict = {
    ".*_hip_pitch_joint": 40.17923847137318,
    ".*_hip_roll_joint": 99.09842777666113,
    ".*_hip_yaw_joint": 40.17923847137318,
    ".*_knee_joint": 99.09842777666113,
    ".*_ankle_pitch_joint": 28.50124619574858,
    ".*_ankle_roll_joint": 28.50124619574858,
    "waist_roll_joint": 28.50124619574858,
    "waist_pitch_joint": 28.50124619574858,
    "waist_yaw_joint": 40.17923847137318,
    ".*_shoulder_pitch_joint": 14.25062309787429,
    ".*_shoulder_roll_joint": 14.25062309787429,
    ".*_shoulder_yaw_joint": 14.25062309787429,
    ".*_elbow_joint": 14.25062309787429,
    ".*_wrist_roll_joint": 14.25062309787429,
    ".*_wrist_pitch_joint": 16.77832748089279,
    ".*_wrist_yaw_joint": 16.77832748089279,
}

damping_dict = {
    ".*_hip_pitch_joint": 2.5578897650279457,
    ".*_hip_roll_joint": 6.3088018534966395,
    ".*_hip_yaw_joint": 2.5578897650279457,
    ".*_knee_joint": 6.3088018534966395,
    ".*_ankle_pitch_joint": 1.814445686584846,
    ".*_ankle_roll_joint": 1.814445686584846,
    "waist_roll_joint": 1.814445686584846,
    "waist_pitch_joint": 1.814445686584846,
    "waist_yaw_joint": 2.5578897650279457,
    ".*_shoulder_pitch_joint": 0.907222843292423,
    ".*_shoulder_roll_joint": 0.907222843292423,
    ".*_shoulder_yaw_joint": 0.907222843292423,
    ".*_elbow_joint": 0.907222843292423,
    ".*_wrist_roll_joint": 0.907222843292423,
    ".*_wrist_pitch_joint": 1.06814150219,
    ".*_wrist_yaw_joint": 1.06814150219,
}
scale_dict = {
    ".*_hip_yaw_joint": 0.5475464652142303,
    ".*_hip_roll_joint": 0.3506614663788243,
    ".*_hip_pitch_joint": 0.5475464652142303,
    ".*_knee_joint": 0.3506614663788243,
    ".*_ankle_pitch_joint": 0.43857731392336724,
    ".*_ankle_roll_joint": 0.43857731392336724,
    "waist_roll_joint": 0.43857731392336724,
    "waist_pitch_joint": 0.43857731392336724,
    "waist_yaw_joint": 0.5475464652142303,
    ".*_shoulder_pitch_joint": 0.43857731392336724,
    ".*_shoulder_roll_joint": 0.43857731392336724,
    ".*_shoulder_yaw_joint": 0.43857731392336724,
    ".*_elbow_joint": 0.43857731392336724,
    ".*_wrist_roll_joint": 0.43857731392336724,
    ".*_wrist_pitch_joint": 0.07450087032950714,
    ".*_wrist_yaw_joint": 0.07450087032950714,
}


def get_param(joint_name, param_dict):
    for pattern, value in param_dict.items():
        if pattern.startswith(".*"):
            if joint_name.startswith("left") or joint_name.startswith("right"):
                if joint_name.endswith(pattern[3:]):
                    return value
        else:
            if joint_name == pattern:
                return value
    raise ValueError(f"No value found for joint: {joint_name}")


def get_action_scale(joint_name):
    for pattern, scale in scale_dict.items():
        # 用 left/right 替换 .*
        if pattern.startswith(".*"):
            if joint_name.startswith("left") or joint_name.startswith("right"):
                if joint_name.endswith(pattern[3:]):
                    return scale
        else:
            if joint_name == pattern:
                return scale
    raise ValueError(f"No scale found for joint: {joint_name}")


# 根据 isaaclab joint_pos 配置设置默认关节位置
joint_pos_config = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}


def get_joint_default_pos(joint_name):
    for pattern, pos in joint_pos_config.items():
        if pattern.startswith(".*"):
            if joint_name.startswith("left") or joint_name.startswith("right"):
                if joint_name.endswith(pattern[3:]):
                    return pos
        else:
            if joint_name == pattern:
                return pos
    return 0.0  # 默认值为0


isaaclab_to_mujoco_reindex = [isaaclab_joint_names.index(name) for name in mujoco_joint_names]
mujoco_to_isaaclab_reindex = [mujoco_joint_names.index(name) for name in isaaclab_joint_names]
joint_names = mujoco_joint_names
kps = [get_param(name, stiffness_dict) for name in joint_names]
kds = [get_param(name, damping_dict) for name in joint_names]
kps = np.array(kps, dtype=np.float32)
kds = np.array(kds, dtype=np.float32)
action_scale = np.array([get_action_scale(name) for name in joint_names], dtype=np.float32)
default_angles = np.array([get_joint_default_pos(name) for name in mujoco_joint_names], dtype=np.float32)

print("isaaclab_to_mujoco_reindex")
print(isaaclab_to_mujoco_reindex)
print("mujoco_to_isaaclab_reindex")
print(mujoco_to_isaaclab_reindex)

print("kp:")
print(kps)
print("kd:")
print(kds)

print("action_scale")
print(action_scale)

print("default joint pos:")
print(default_angles)


# Fix bug for mujoco_viewer:
def __fix__add_marker_to_scene(self, marker):
    if self.scn.ngeom >= self.scn.maxgeom:
        raise RuntimeError("Ran out of geoms. maxgeom: %d" % self.scn.maxgeom)

    g = self.scn.geoms[self.scn.ngeom]
    # default values.
    g.dataid = -1
    g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
    g.objid = -1
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    # g.texid = -1
    # g.texuniform = 0
    # g.texrepeat[0] = 1
    # g.texrepeat[1] = 1
    g.emission = 0
    g.specular = 0.5
    g.shininess = 0.5
    g.reflectance = 0
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size[:] = np.ones(3) * 0.1
    g.mat[:] = np.eye(3)
    g.rgba[:] = np.ones(4)

    for key, value in marker.items():
        if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
            setattr(g, key, value)
        elif isinstance(value, (tuple, list, np.ndarray)):
            attr = getattr(g, key)
            attr[:] = np.asarray(value).reshape(attr.shape)
        elif isinstance(value, str):
            assert key == "label", "Only label is a string in mjtGeom."
            if value is None:
                g.label[0] = 0
            else:
                g.label = value
        elif hasattr(g, key):
            raise ValueError("mjtGeom has attr {} but type {} is invalid".format(key, type(value)))
        else:
            raise ValueError("mjtGeom doesn't have field %s" % key)

    self.scn.ngeom += 1

    return


mujoco_viewer.MujocoViewer._add_marker_to_scene = __fix__add_marker_to_scene


def update_joint_visualization(viewer, motion_loader, t):
    """Update joint visualization spheres using body positions"""
    if t < 0 or t >= motion_loader.T:
        return

    # Get body positions from motion data
    body_pos = motion_loader.body_pos[t]  # [N, 3]

    # Update spheres for each body (excluding the first one which is pelvis)
    for i in range(0, len(body_pos)):
        viewer.add_marker(
            pos=body_pos[i],
            size=[0.02, 0.02, 0.02],
            rgba=[0.8, 0.5, 0.3, 1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            label=""
        )

    # Update anchor body sphere (larger) - use torso_link position
    anchor_pos = body_pos[motion_loader.anchor_body_index]  # torso_link position
    viewer.add_marker(
        pos=anchor_pos, size=[0.05, 0.05, 0.05], rgba=[0.8, 0.5, 0.3, 1], type=mujoco.mjtGeom.mjGEOM_SPHERE, label=""
    )


# ====== MotionLoader 类（参考 isaaclab） ======
class MotionLoader:
    def __init__(self, motion_file):
        data = np.load(motion_file)
        self.joint_pos = data["joint_pos"]  # [T, 29]
        self.joint_vel = data["joint_vel"]  # [T, 29]
        self.body_pos = data["body_pos_w"]  # [T, N, 3]
        self.body_ori = data["body_quat_w"]  # [T, N, 4]
        self.body_ang_vel_w = data["body_ang_vel_w"]
        self.fps = data["fps"]
        self.T = self.joint_pos.shape[0]

        # G1 body names from the config
        self.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]
        self.anchor_body_name = "torso_link"
        self.anchor_body_index = 9

        # Future steps configuration
        self.future_steps = 5


def quat_rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a quaternion along the last dimension of q and v (NumPy version).

    Args:
        q: The quaternion in (w, x, y, z). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    q_w = q[..., 0]
    q_vec = q[..., 1:]

    # Component a: v * (2.0 * q_w^2 - 1.0)
    a = v * np.expand_dims(2.0 * q_w**2 - 1.0, axis=-1)

    # Component b: cross(q_vec, v) * q_w * 2.0
    b = np.cross(q_vec, v, axis=-1) * np.expand_dims(q_w, axis=-1) * 2.0

    # Component c: q_vec * dot(q_vec, v) * 2.0
    # For efficient computation, handle different dimensionalities
    if q_vec.ndim == 2:
        # For 2D case: use matrix multiplication for better performance
        dot_product = np.sum(q_vec * v, axis=-1, keepdims=True)
        c = q_vec * dot_product * 2.0
    else:
        # For general case: use Einstein summation
        dot_product = np.expand_dims(np.einsum('...i,...i->...', q_vec, v), axis=-1)
        c = q_vec * dot_product * 2.0

    return a - b + c


# ====== observation 计算函数 ======
def get_command(motion_loader, t):
    """Get command (joint_pos + joint_vel) for future steps - matching Isaac Lab order"""
    if t < 0:
        return np.zeros(290, dtype=np.float32)  # 29 * 2 * 5 = 290

    # Get future joint positions and velocities - batch process like Isaac Lab
    future_joint_pos = []
    future_joint_vel = []

    for i in range(motion_loader.future_steps):
        step_idx = min(t + i, motion_loader.T - 1)
        future_joint_pos.append(motion_loader.joint_pos[step_idx])
        future_joint_vel.append(motion_loader.joint_vel[step_idx])

    # Stack to create [future_steps, 29] then flatten to [future_steps * 29]
    joint_pos_future = np.stack(future_joint_pos, axis=0).flatten()  # [29 * 5]
    joint_vel_future = np.stack(future_joint_vel, axis=0).flatten()  # [29 * 5]
    # breakpoint()

    cmd = np.concatenate([joint_pos_future, joint_vel_future], axis=0)
    # breakpoint()
    return cmd  # [290]


def motion_anchor_pos_b_future(sim_data, motion_loader, t):
    """Future N-step motion anchor position in body frame - matching Isaac Lab order"""
    if t < 0:
        return np.zeros(15, dtype=np.float32)  # 3 * 5 = 15

    # Get robot anchor pose
    robot_pos = sim_data.body(motion_loader.anchor_body_name).xpos.copy().reshape(1, 3)
    robot_quat = sim_data.body(motion_loader.anchor_body_name).xquat.copy().reshape(1, 4)

    # Get future motion anchor poses - batch process like Isaac Lab
    future_positions = []
    future_orientations = []
    for i in range(motion_loader.future_steps):
        step_idx = min(t + i, motion_loader.T - 1)
        ref_pos = motion_loader.body_pos[step_idx][motion_loader.anchor_body_index].reshape(1, 3)
        ref_quat = motion_loader.body_ori[step_idx][motion_loader.anchor_body_index].reshape(1, 4)
        future_positions.append(ref_pos)
        future_orientations.append(ref_quat)

    # Stack to create [future_steps, 1, 3] then reshape to [future_steps, 3]
    future_anchor_pos_w = np.stack(future_positions, axis=0).squeeze(1)  # [5, 3]
    future_anchor_quat_w = np.stack(future_orientations, axis=0).squeeze(1)  # [5, 4]

    # Expand robot anchor for broadcasting: [future_steps, 3] and [future_steps, 4]
    robot_anchor_pos_w_exp = robot_pos.repeat(motion_loader.future_steps, axis=0)  # [5, 3]
    robot_anchor_quat_w_exp = robot_quat.repeat(motion_loader.future_steps, axis=0)  # [5, 4]

    # Transform all future steps at once
    pos_b, _ = subtract_frame_transforms(
        torch.from_numpy(robot_anchor_pos_w_exp).float(),
        torch.from_numpy(robot_anchor_quat_w_exp).float(),
        torch.from_numpy(future_anchor_pos_w).float(),
        torch.from_numpy(future_anchor_quat_w).float(),
    )

    pos_b_flat = np.array(pos_b, dtype=np.float32).flatten()  # [15]
    return pos_b_flat  # [15]


def motion_anchor_ori_b_future(sim_data, motion_loader, t):
    """Future N-step motion anchor orientation in body frame - matching Isaac Lab order"""
    if t < 0:
        return np.zeros(30, dtype=np.float32)  # 6 * 5 = 30

    # Get robot anchor pose
    robot_pos = sim_data.body(motion_loader.anchor_body_name).xpos.copy().reshape(1, 3)
    robot_quat = sim_data.body(motion_loader.anchor_body_name).xquat.copy().reshape(1, 4)

    # Get future motion anchor orientations - batch process like Isaac Lab
    future_positions = []
    future_orientations = []
    for i in range(motion_loader.future_steps):
        step_idx = min(t + i, motion_loader.T - 1)
        ref_pos = motion_loader.body_pos[step_idx][motion_loader.anchor_body_index].reshape(1, 3)
        ref_quat = motion_loader.body_ori[step_idx][motion_loader.anchor_body_index].reshape(1, 4)
        future_positions.append(ref_pos)
        future_orientations.append(ref_quat)
    # Stack to create [future_steps, 1, 3] then reshape to [future_steps, 3]
    future_anchor_pos_w = np.stack(future_positions, axis=0).squeeze(1)  # [5, 3]
    future_anchor_quat_w = np.stack(future_orientations, axis=0).squeeze(1)  # [5, 4]

    # Expand robot anchor for broadcasting: [future_steps, 3] and [future_steps, 4]
    robot_anchor_pos_w_exp = robot_pos.repeat(motion_loader.future_steps, axis=0)  # [5, 3]
    robot_anchor_quat_w_exp = robot_quat.repeat(motion_loader.future_steps, axis=0)  # [5, 4]

    # Transform all future steps at once
    pos_b, ori_b = subtract_frame_transforms(
        torch.from_numpy(robot_anchor_pos_w_exp).float(),
        torch.from_numpy(robot_anchor_quat_w_exp).float(),
        torch.from_numpy(future_anchor_pos_w).float(),
        torch.from_numpy(future_anchor_quat_w).float(),
    )

    # Convert to rotation matrix and take first 2 rows
    mat = matrix_from_quat(ori_b)
    mat_flat = mat[..., :2].reshape(-1)
    # breakpoint()
    ori_b_flat = np.array(mat_flat, dtype=np.float32).flatten()  # [30]
    # breakpoint()
    return ori_b_flat  # [30]


def robot_body_pos_b(sim_data, motion_loader, t):
    """Robot body positions in body frame"""
    # Get robot anchor pose
    robot_anchor_pos = sim_data.body(motion_loader.anchor_body_name).xpos.copy().reshape(1, 3)
    robot_anchor_quat = sim_data.body(motion_loader.anchor_body_name).xquat.copy().reshape(1, 4)

    # Get all robot body poses
    robot_body_positions = []
    robot_body_orientations = []

    for body_name in motion_loader.body_names:
        body_pos = sim_data.body(body_name).xpos.copy().reshape(1, 3)
        body_quat = sim_data.body(body_name).xquat.copy().reshape(1, 4)
        robot_body_positions.append(body_pos)
        robot_body_orientations.append(body_quat)

    robot_body_pos = np.concatenate(robot_body_positions, axis=0)  # [14, 3]
    robot_body_quat = np.concatenate(robot_body_orientations, axis=0)  # [14, 4]

    # Transform to body frame - expand robot_anchor to match robot_body shape
    robot_anchor_pos_expanded = robot_anchor_pos.repeat(len(motion_loader.body_names), axis=0)  # [14, 3]
    robot_anchor_quat_expanded = robot_anchor_quat.repeat(len(motion_loader.body_names), axis=0)  # [14, 4]

    pos_b, _ = subtract_frame_transforms(
        torch.from_numpy(robot_anchor_pos_expanded).float(),
        torch.from_numpy(robot_anchor_quat_expanded).float(),
        torch.from_numpy(robot_body_pos).float(),
        torch.from_numpy(robot_body_quat).float(),
    )

    return np.array(pos_b, dtype=np.float32).flatten()  # [42]


def robot_body_ori_b(sim_data, motion_loader, t):
    """Robot body orientations in body frame"""
    # Get robot anchor pose
    robot_anchor_pos = sim_data.body(motion_loader.anchor_body_name).xpos.copy().reshape(1, 3)
    robot_anchor_quat = sim_data.body(motion_loader.anchor_body_name).xquat.copy().reshape(1, 4)

    # Get all robot body poses
    robot_body_positions = []
    robot_body_orientations = []

    for body_name in motion_loader.body_names:
        body_pos = sim_data.body(body_name).xpos.copy().reshape(1, 3)
        body_quat = sim_data.body(body_name).xquat.copy().reshape(1, 4)
        robot_body_positions.append(body_pos)
        robot_body_orientations.append(body_quat)

    robot_body_pos = np.concatenate(robot_body_positions, axis=0)  # [14, 3]
    robot_body_quat = np.concatenate(robot_body_orientations, axis=0)  # [14, 4]

    # Transform to body frame - expand robot_anchor to match robot_body shape
    robot_anchor_pos_expanded = robot_anchor_pos.repeat(len(motion_loader.body_names), axis=0)  # [14, 3]
    robot_anchor_quat_expanded = robot_anchor_quat.repeat(len(motion_loader.body_names), axis=0)  # [14, 4]

    _, ori_b = subtract_frame_transforms(
        torch.from_numpy(robot_anchor_pos_expanded).float(),
        torch.from_numpy(robot_anchor_quat_expanded).float(),
        torch.from_numpy(robot_body_pos).float(),
        torch.from_numpy(robot_body_quat).float(),
    )

    # Convert to rotation matrix and take first 2 rows
    mat = matrix_from_quat(ori_b)
    mat_flat = mat[..., :2].reshape(-1)

    return np.array(mat_flat, dtype=np.float32).flatten()  # [84]


def get_base_lin_vel(sim_data):
    """Get base linear velocity"""
    linvel_b = quat_rotate_inverse_np(sim_data.qpos[3:7], sim_data.qvel[0:3])
    # return sim_data.qvel[0:3]  # shape [3]
    return linvel_b


def get_base_ang_vel(sim_data):
    """Get base angular velocity
    
    Pay special attention: 
    - The angular velocity sim_data.qvel[3:6] is in the body frame, not the world frame.
    - The angular velocity sim_data.qvel[3:6] is in the body frame, not the world frame.
    """

    return sim_data.qvel[3:6]  # shape [3]


def get_joint_pos_rel(sim_data):
    """Get relative joint positions"""
    return sim_data.qpos[7:36] - default_angles  # shape [29]


def get_joint_vel_rel(sim_data):
    """Get relative joint velocities"""
    return sim_data.qvel[6:35]  # shape [29]


def get_last_action(last_actions):
    """Get last actions"""
    return last_actions  # shape [29]


def compute_observation(sim_data, motion_loader, t, last_actions):
    """Compute observation matching PrivPrivObservationsCfg structure"""
    # obs_dim = 554
    obs_dim = 554 - 42 - 84
    # obs_dim = 160
    if t < 0:
        return np.zeros(obs_dim, dtype=np.float32)

    obs = np.zeros(obs_dim, dtype=np.float32)
    idx = 0

    # 0. command (290,) - future joint pos + vel
    command = get_command(motion_loader, t)
    dim_command = command.shape[0]
    obs[idx:idx + dim_command] = command
    idx += dim_command

    # 1. motion_anchor_pos_b (15,) - future anchor pos in body frame
    motion_anchor_pos = motion_anchor_pos_b_future(sim_data, motion_loader, t)
    dim_motion_anchor_pos = motion_anchor_pos.shape[0]
    obs[idx:idx + dim_motion_anchor_pos] = motion_anchor_pos
    idx += dim_motion_anchor_pos

    # 2. motion_anchor_ori_b (30,) - future anchor ori in body frame
    motion_anchor_ori = motion_anchor_ori_b_future(sim_data, motion_loader, t)
    dim_motion_anchor_ori = motion_anchor_ori.shape[0]
    obs[idx:idx + dim_motion_anchor_ori] = motion_anchor_ori
    idx += dim_motion_anchor_ori

    # breakpoint()

    # if True:
    if False:
        # 3. body_pos (42,) - robot body pos in body frame
        body_pos = robot_body_pos_b(sim_data, motion_loader, t)
        dim_body_pos = body_pos.shape[0]
        obs[idx:idx + dim_body_pos] = body_pos
        idx += dim_body_pos

        # 4. body_ori (84,) - robot body ori in body frame
        body_ori = robot_body_ori_b(sim_data, motion_loader, t)
        dim_body_ori = body_ori.shape[0]
        obs[idx:idx + dim_body_ori] = body_ori
        idx += dim_body_ori

    # 5. base_lin_vel (3,) - base linear velocity
    base_lin_vel = get_base_lin_vel(sim_data)
    dim_base_lin_vel = base_lin_vel.shape[0]
    obs[idx:idx + dim_base_lin_vel] = base_lin_vel
    idx += dim_base_lin_vel

    # 6. base_ang_vel (3,) - base angular velocity
    base_ang_vel = get_base_ang_vel(sim_data)
    dim_base_ang_vel = base_ang_vel.shape[0]
    obs[idx:idx + dim_base_ang_vel] = base_ang_vel
    idx += dim_base_ang_vel

    # 7. joint_pos (29,) - relative joint positions
    joint_pos = get_joint_pos_rel(sim_data)[mujoco_to_isaaclab_reindex]
    dim_joint_pos = joint_pos.shape[0]
    obs[idx:idx + dim_joint_pos] = joint_pos
    idx += dim_joint_pos

    # 8. joint_vel (29,) - relative joint velocities
    joint_vel = get_joint_vel_rel(sim_data)[mujoco_to_isaaclab_reindex]
    dim_joint_vel = joint_vel.shape[0]
    obs[idx:idx + dim_joint_vel] = joint_vel
    idx += dim_joint_vel

    # 9. actions (29,) - last actions
    last_action = get_last_action(last_actions)
    dim_last_action = last_action.shape[0]
    obs[idx:idx + dim_last_action] = last_action
    idx += dim_last_action

    return obs


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    VIEW_MOTION = False

    parser = argparse.ArgumentParser(description="Deploy MuJoCo simulation with motion tracking")
    parser.add_argument(
        "--motion_path",
        type=str,
        default="/home/bcj/whole_body_tracking/motions/lafan_dance1_subject2.npz",
        help="Path to motion file (.npz)"
    )
    parser.add_argument(
        "--policy_path",
        type=str,
        default=
        "/home/bcj/whole_body_tracking/logs/rsl_rl/g1_flat/2025-08-20_21-45-11_dance1_subject2-new_obs_add_prog-add_baselinvel-add_baseposz/exported/policy.onnx",
        help="Path to policy file (.onnx)",
    )

    args = parser.parse_args()

    policy_path = args.policy_path
    motion_path = args.motion_path

    xml_path = "./source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1_act.xml"

    simulation_duration = 1000
    simulation_dt = 0.002
    control_decimation = 10

    num_actions = 29

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()

    counter = 0
    inner_counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    viewer = mujoco_viewer.MujocoViewer(m, d)
    viewer.cam.lookat[:] = np.array([0, 0, 0.7])
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 0
    viewer.cam.elevation = -20  # 负值表示从上往下看viewer
    m.opt.timestep = simulation_dt

    # load policy
    session = ort.InferenceSession(policy_path)
    obs_name = session.get_inputs()[0].name

    motion_loader = MotionLoader(motion_path)
    T = motion_loader.T
    print("T: ", T)

    d.qpos[7:] = default_angles
    d.qpos[:3] = motion_loader.body_pos[0][motion_loader.anchor_body_index]
    # ori_yaw = motion_loader.body_ori[0][motion_loader.anchor_body_index]
    # ori_yaw[1:3] = 0
    # ori_yaw = ori_yaw / np.linalg.norm(ori_yaw)
    # d.qpos[3:7] = ori_yaw
    # breakpoint()
    d.qpos[3:7] = motion_loader.body_ori[0][motion_loader.anchor_body_index]
    mujoco.mj_step(m, d)

    # with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    while viewer.is_alive and time.time() - start < simulation_duration:
        step_start = time.time()
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        if VIEW_MOTION:
            d.qpos[:3] = motion_loader.body_pos[inner_counter][0]
            d.qpos[3:7] = motion_loader.body_ori[inner_counter][0]
            d.qpos[7:] = motion_loader.joint_pos[inner_counter][isaaclab_to_mujoco_reindex]
            d.qvel[6:] = motion_loader.joint_vel[inner_counter][isaaclab_to_mujoco_reindex]
        else:
            d.ctrl[:] = tau

        mujoco.mj_step(m, d)
        counter += 1
        if counter % control_decimation == 0:
            # create observation
            # breakpoint()

            print("t: ", inner_counter, "/", T)
            obs = compute_observation(d, motion_loader, inner_counter, action)
            obs_tensor = np.array(obs, dtype=np.float32).reshape(1, -1)

            output = session.run(None, {obs_name: obs_tensor})
            action = output[0].squeeze()

            # transform action to target_dof_pos
            target_dof_pos = action[isaaclab_to_mujoco_reindex] * action_scale + default_angles

            inner_counter += 1

        if inner_counter >= motion_loader.T:
            inner_counter = 0
            # breakpoint()
            break

        # Update joint visualization
        update_joint_visualization(viewer, motion_loader, inner_counter)

        viewer.render()
