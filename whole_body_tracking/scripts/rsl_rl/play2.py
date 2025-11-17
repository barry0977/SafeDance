"""Script to play a checkpoint if an RL agent from RSL-RL."""
"""Launch Isaac Sim Simulator first."""
EXPORT_ONNX = True  # 临时禁用，因为使用的是不兼容的旧检查点

import argparse
import sys
from pathlib import Path
from isaaclab.app import AppLauncher
import glob
import numpy as np

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--resume_path", type=str, default=None, help="Path to the resume checkpoint.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch

# 使用与训练时相同的 runner（支持 MY_PPO + Encoder）
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from isaaclab.utils.io.pkl import load_pickle


def load_config(resume_path: str) -> tuple[ManagerBasedRLEnvCfg, RslRlOnPolicyRunnerCfg]:
    """Load the config from the resume path."""
    param_dir = Path(resume_path).parent / "params"
    env_cfg = load_pickle(str(param_dir / "env.pkl"))
    agent_cfg = load_pickle(str(param_dir / "agent.pkl"))
    return env_cfg, agent_cfg


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        raise NotImplementedError("Wandb is not supported")
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    elif args_cli.resume_path:
        assert args_cli.motion_file is not None, "Motion file is required when resume_path is provided"
        resume_path = args_cli.resume_path
        # env_cfg.commands.motion.motion_file = str(Path("./artifacts") / Path(args_cli.motion_file) / "motion.npz")
        # env_cfg, agent_cfg = load_config(resume_path)

        motion_files = glob.glob(str(Path("./artifacts") / Path(args_cli.motion_file) / "motion.npz"))
        if not motion_files:
            raise FileNotFoundError(f"No motion.npz found in {Path('./artifacts') / Path(args_cli.motion_file)}")
        env_cfg.commands.motion.motion_files = motion_files  # List[str]

        print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
        print(f"[INFO]: Using resume path from CLI: {args_cli.resume_path}")
    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    if EXPORT_ONNX:
        # export policy to onnx/jit
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

        # 如果使用了 MY_PPO，传入 encoder
        encoder = getattr(ppo_runner.alg, 'encoder', None)
        
        export_motion_policy_as_onnx(
            env.unwrapped,
            ppo_runner.alg.policy,
            normalizer=ppo_runner.obs_normalizer,
            encoder=encoder,
            path=export_model_dir,
            filename="policy.onnx",
        )
        attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    print("Start playing...")
    print("Available sensors:", env.unwrapped.scene.sensors.keys())
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
    print(f"Contact sensor: {contact_sensor.body_names}")
    
    # 获取目标身体部位的索引
    target_bodies = ["right_wrist_yaw_link", "left_wrist_yaw_link"]
    target_body_indices = [contact_sensor.body_names.index(body) for body in target_bodies]
    
    # 用于记录接触力
    force_log = []
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            
            # 获取所有目标身体部位的受力
            forces = {}
            total_force = 0.0
            for body_name, idx in zip(target_bodies, target_body_indices):
                force_vector = contact_sensor.data.net_forces_w[0, idx]  # (3,) [x, y, z]
                force_magnitude = torch.norm(force_vector).item()
                forces[body_name] = {
                    "vector": force_vector.cpu().numpy(),
                    "magnitude": force_magnitude
                }
                total_force += force_magnitude
            
            # 打印接触力信息（每10步打印一次避免刷屏）
            print(f"\n--- Step {timestep} ---")
            for body_name, force_data in forces.items():
                print(f"{body_name}: {force_data['magnitude']:.2f} N, vector: {force_data['vector']}")
            
            # 记录力的大小用于后续分析
            force_log.append({
                "timestep": timestep,
                "forces": {k: v["magnitude"] for k, v in forces.items()},
                "total": total_force
            })
            
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        
        timestep += 1

    # 打印接触力统计信息
    if force_log:
        print("\n" + "="*50)
        print("接触力统计信息")
        print("="*50)
        
        for body_name in target_bodies:
            forces_array = np.array([log["forces"][body_name] for log in force_log])
            print(f"\n{body_name}:")
            print(f"  平均力: {forces_array.mean():.2f} N")
            print(f"  最大力: {forces_array.max():.2f} N")
            print(f"  最小力: {forces_array.min():.2f} N")
            print(f"  标准差: {forces_array.std():.2f} N")
        
        total_forces_array = np.array([log["total"] for log in force_log])
        print(f"\n总接触力:")
        print(f"  平均力: {total_forces_array.mean():.2f} N")
        print(f"  最大力: {total_forces_array.max():.2f} N")
        print(f"  最小力: {total_forces_array.min():.2f} N")
        print(f"  标准差: {total_forces_array.std():.2f} N")
        print("="*50)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
