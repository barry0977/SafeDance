import os
import argparse
import numpy as np
import torch
import joblib
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot
from scipy.interpolate import CubicSpline


class PKLMotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Load motion from pkl file."""
        assert self.frame_range is None, "Don't use frame_range, it's stupid."
        assert self.motion_file[-4:] == ".pkl", f"Only allow .pkl motion file, got {self.motion_file}"
        _motion_data = joblib.load(self.motion_file)
        assert len(_motion_data) == 1, "Only allow motion file containing single motion for now."
        motion_data = next(iter(_motion_data.values()))

        assert motion_data["fps"] == self.input_fps, f"Input FPS Mismatch!, {motion_data['fps']} != {self.input_fps}"

        self.motion_base_poss_input = torch.from_numpy(motion_data["root_trans_offset"]).to(torch.float32).to(self.device)
        self.motion_base_rots_input = torch.from_numpy(motion_data["root_rot"]).to(torch.float32).to(self.device)
        # motion_data['root_rot']: XYZW
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # convert xyzw to wxyz

        num_dof = motion_data["dof"].shape[1]
        self.motion_dof_poss_input = torch.from_numpy(motion_data["dof"]).to(torch.float32).to(self.device)
        if num_dof == 29:
            # Everything good
            pass
        elif num_dof == 23:
            # 29 = 15 + ( 4 + __3__)*2
            #    = 19 + 3 + 4 + 3
            self.motion_dof_poss_input = torch.cat(
                [
                    self.motion_dof_poss_input[:, :19],
                    torch.zeros_like(self.motion_dof_poss_input[:, :3]),
                    self.motion_dof_poss_input[:, 19:23],
                    torch.zeros_like(self.motion_dof_poss_input[:, :3]),
                ],
                dim=1,
            )
            assert self.motion_dof_poss_input.shape[1] == 29
        else:
            raise NotImplementedError

        self.input_frames = self.motion_dof_poss_input.shape[0]
        self.duration = (self.input_frames) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolate motion to output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)

        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            # Convert to numpy for processing
            q1 = a[i].cpu().numpy()
            q2 = b[i].cpu().numpy()
            t = blend[i].cpu().numpy()

            # Use scipy for slerp
            r1 = sRot.from_quat(q1)
            r2 = sRot.from_quat(q2)
            r_slerp = r1.inv() * r2
            r_result = r1 * sRot.from_rotvec(r_slerp.as_rotvec() * t)
            slerped_quats[i] = torch.from_numpy(r_result.as_quat()).to(self.device)
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Compute frame blend for motion interpolation."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Compute velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Compute derivative of SO3 rotations."""
        q_prev, q_next = rotations[:-2], rotations[2:]
        # Convert to scipy format for quaternion operations
        q_prev_scipy = sRot.from_quat(q_prev.cpu().numpy())
        q_next_scipy = sRot.from_quat(q_next.cpu().numpy())
        q_rel = q_next_scipy * q_prev_scipy.inv()
        omega = q_rel.as_rotvec() / (2.0 * dt)
        omega = torch.from_numpy(omega).to(self.device)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(self):
        """Get next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_mujoco_forward_kinematics(motion_loader, model_path, joint_names, output_name, output_fps):
    """Run MuJoCo forward kinematics and collect data."""
    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Set timestep
    model.opt.timestep = 1.0 / output_fps

    # Get joint indices
    joint_indices = []
    for joint_name in joint_names:
        try:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            joint_indices.append(joint_id)
        except:
            print(f"Warning: Joint {joint_name} not found in model")

    # Convert motion to numpy arrays for MuJoCo
    motion_base_poss = motion_loader.motion_base_poss.cpu().numpy()
    motion_base_rots = motion_loader.motion_base_rots.cpu().numpy()
    motion_dof_poss = motion_loader.motion_dof_poss.cpu().numpy()
    motion_base_lin_vels = motion_loader.motion_base_lin_vels.cpu().numpy()
    motion_base_ang_vels = motion_loader.motion_base_ang_vels.cpu().numpy()
    motion_dof_vels = motion_loader.motion_dof_vels.cpu().numpy()

    # Convert to qpos and qvel format - now using full 29 DOF
    qpos_list = np.concatenate(
        [
            motion_base_poss,
            motion_base_rots,
            motion_dof_poss,  # Use all 29 DOF
        ],
        axis=1,
    )

    qvel_list = np.concatenate(
        [
            motion_base_lin_vels,
            motion_base_ang_vels,
            motion_dof_vels,  # Use all 29 DOF
        ],
        axis=1,
    )

    print(f"qpos_list shape: {qpos_list.shape}, expected: {model.nq}")
    print(f"qvel_list shape: {qvel_list.shape}, expected: {model.nv}")

    # Collect simulation data
    log = {
        "fps": [output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    # Run forward kinematics for each frame

    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.lookat[:] = np.array([0, 0, 0.7])
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -30  # 负值表示从上往下看viewer

    isaac_robot_joint_indexes = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
    isaac_robot_body_indexes = [0] + [i + 1 for i in isaac_robot_joint_indexes]

    for t in range(len(qpos_list)):
        # Set joint positions and velocities
        data.qpos[:] = qpos_list[t]
        data.qvel[:] = qvel_list[t]

        # Run forward kinematics (no dynamics, just kinematics)
        mujoco.mj_forward(model, data)
        viewer.sync()

        # Extract joint positions and velocities - match IsaacLab format (29 DOF)
        joint_pos = np.zeros(29)  # Match IsaacLab format
        joint_vel = np.zeros(29)  # Match IsaacLab format

        # Fill in all 29 DOF from MuJoCo (skip floating base joint)
        for i, isaac_idx in enumerate(isaac_robot_joint_indexes):
            joint_idx = joint_indices[i]
            joint_pos[isaac_idx] = data.qpos[model.jnt_qposadr[joint_idx]]
            joint_vel[isaac_idx] = data.qvel[model.jnt_dofadr[joint_idx]]

        # Extract body positions and orientations - match IsaacLab format (30 bodies)
        # IsaacLab returns positions for 30 bodies, MuJoCo has 31 bodies (including world)
        # We skip the world body (index 0) and take the first 30 bodies
        body_pos_w = np.zeros((30, 3))  # Match IsaacLab format
        body_quat_w = np.zeros((30, 4))  # Match IsaacLab format
        body_lin_vel_w = np.zeros((30, 3))  # Match IsaacLab format
        body_ang_vel_w = np.zeros((30, 3))  # Match IsaacLab format

        # # Fill in all body positions and orientations (skip world body)
        # for i in range(1, min(31, model.nbody)):  # Skip world body (index 0)
        #     body_idx = i - 1  # Adjust index for IsaacLab format
        #     if body_idx < 30:
        #         body_pos_w[body_idx] = data.xpos[i]
        #         body_quat_w[body_idx] = data.xquat[i]
        #         body_lin_vel_w[body_idx] = data.cvel[i][3:]
        #         body_ang_vel_w[body_idx] = data.cvel[i][:3]
        for i, isaac_idx in enumerate(isaac_robot_body_indexes):
            mjc_idx = i + 1
            res = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, mjc_idx, res, 1)
            angvel = res[:3]
            linvel = res[3:]
            # cvel = data.cvel[mjc_idx].reshape(6)  # (ω_body, v_body) in body frame
            # R = data.xmat[mjc_idx].reshape(3, 3)  # rotation matrix world->body
            # angvel = R.T @ cvel[:3]
            # linvel = R.T @ cvel[3:]

            body_pos_w[isaac_idx] = data.xpos[mjc_idx]
            body_quat_w[isaac_idx] = data.xquat[mjc_idx]
            body_lin_vel_w[isaac_idx] = linvel
            body_ang_vel_w[isaac_idx] = angvel
            # print(f"DEBUG: isaac_idx: {isaac_idx}, i: {i}, mjc_idx: {mjc_idx}")

        # breakpoint()

        log["joint_pos"].append(joint_pos)
        log["joint_vel"].append(joint_vel)
        log["body_pos_w"].append(body_pos_w)
        log["body_quat_w"].append(body_quat_w)
        log["body_lin_vel_w"].append(body_lin_vel_w)
        log["body_ang_vel_w"].append(body_ang_vel_w)
        # if t % 20 == 0:
        #     breakpoint()

    # Convert lists to numpy arrays
    for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        log[k] = np.stack(log[k], axis=0)

    # Save data
    save_path = f"./artifacts/{output_name}/motion.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, **log)

    print(f"Motion data saved to: {save_path}")
    return log


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Replay motion from pkl file and output to npz file using MuJoCo forward kinematics (29 DOF).")
    parser.add_argument("--input_file", type=str, required=True, help="The path to the input motion pkl file.")
    parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
    parser.add_argument("--output_name", type=str, required=True, help="The name of the motion npz file.")
    parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml",
        help="Path to MuJoCo model file.",
    )

    args = parser.parse_args()

    # Create motion loader
    motion_loader = PKLMotionLoader(
        motion_file=args.input_file,
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        device=torch.device("cpu"),  # MuJoCo doesn't use GPU
        frame_range=None,
    )

    # Joint names (29 DOF - all joints except floating base)
    joint_names = [
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

    print("[INFO]: Setup complete...")

    # Run MuJoCo forward kinematics
    log = run_mujoco_forward_kinematics(
        motion_loader=motion_loader, model_path=args.model_path, joint_names=joint_names, output_name=args.output_name, output_fps=args.output_fps
    )

    print("[INFO]: Forward kinematics completed successfully!")


if __name__ == "__main__":
    main()
    os._exit(0)
