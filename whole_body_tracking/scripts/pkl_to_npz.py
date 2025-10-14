import os
import torch

import joblib
from csv_to_npz import MotionLoader, ReplayMotionsSceneCfg, run_simulator, sim_utils, args_cli
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext


class PKLMotionLoader(MotionLoader):
    def _load_motion(self):
        assert self.frame_range is None, "Don't use frame_range, it's stupid."
        assert self.motion_file[-4:] == ".pkl", f"Only allow .pkl motion file, got {self.motion_file}"
        _motion_data = joblib.load(self.motion_file)
        assert len(_motion_data) == 1, "Only allow motion file containing single motion for now."
        motion_data = next(iter(_motion_data.values()))

        assert motion_data["fps"] == self.input_fps, f'Input FPS Mismatch!, {{motion_data["fps"]}} != {self.input_fps}'

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
            ...
        else:
            print(f"num_dof is{num_dof}")
            raise NotImplementedError

        self.input_frames = self.motion_dof_poss_input.shape[0]
        self.duration = (self.input_frames) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

        ...


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(
        sim,
        scene,
        joint_names=[
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
        ],
        MotionLoaderCls=PKLMotionLoader,
    )


if __name__ == "__main__":
    # run the main function
    main()
    os._exit(0)  # type: ignore
