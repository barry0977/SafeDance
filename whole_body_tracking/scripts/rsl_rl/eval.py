"""
定量评测脚本：统计给定外力幅值下的 motion 完成失败率
使用与训练/回放相同的 ISAAC-Lab + RSL-RL 基础设施
"""
import argparse
import sys
from pathlib import Path
from isaaclab.app import AppLauncher
import glob

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
parser.add_argument("--num_trials", type=int, default=1000)
parser.add_argument("--force_list", type=float, nargs="+",
                    default=[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])         # 以 N 为单位


# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)                # headless/renderer 等
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args                    # 交给 hydra

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch

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
def evaluate(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
             agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # 指定 motion 数据
    motion_files = glob.glob(str(Path("./artifacts") / Path(args_cli.motion_file) / "motion.npz"))
    assert motion_files, "motion.npz 未找到"
    env_cfg.commands.motion.motion_files = motion_files

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(args_cli.resume_path)


    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # 获取 motion command term
    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    
    # 检查是否有 force command 并获取引用
    has_force_cmd = False
    force_cmd = None
    if hasattr(env.unwrapped.cfg.commands, 'force'):
        try:
            force_cmd = env.unwrapped.command_manager.get_term("force")
            has_force_cmd = True
            print(f"[INFO] 检测到 ForceCommand，将控制外力幅值")
        except:
            print(f"[WARN] cfg 中有 force 配置但无法获取 force command term")
    else:
        print(f"[WARN] 未检测到 force command，将跳过外力控制")

    # 6-0. ─────────── 预热：设置从第0帧开始 ───────────
    env.unwrapped.cfg.commands.motion.start_from_zero_step = True
    motion_cmd.cfg.start_from_zero_step = True
    print(f"[INFO] 已设置从第0帧开始")
    
    print(f"[INFO] 正在初始化环境...")
    env.unwrapped.reset()
    obs, _ = env.get_observations()
    print(f"[INFO] 环境初始化完成")

    # 6. ─────────── 评测不同外力幅值 ───────────
    results = {}
    for F in args_cli.force_list:
        print(f"\n[EVAL] 开始评测 Force = {F:.1f} N ...")
        
        # 6-1 修改外力幅值 (如果使用 force command)
        if has_force_cmd and force_cmd is not None:
            # 根据 commands_force.py 的实现，需要修改三处：
            # 1. cfg 配置
            env.unwrapped.cfg.commands.force.force_magnitude_range = (F, F)
            # 2. force_cmd 实例的 force_min 和 force_max 属性
            force_cmd.cfg.force_magnitude_range = (F, F)
            force_cmd.force_min = F
            force_cmd.force_max = F
            print(f"[INFO] 已设置外力幅值范围为 [{F:.1f}, {F:.1f}] N")
            
            # 3. 强制触发一次重新采样，让所有环境立即使用新的力幅值
            all_env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
            force_cmd._resample_command(all_env_ids)
            print(f"[INFO] 已触发所有环境的外力重新采样")

        # 6-2 重置所有环境到初始状态（会根据 start_from_zero_step 参数自动设置起始帧）
        env.unwrapped.reset()
        obs, _ = env.get_observations()

        # 6-5 统计变量（每次新的力度评测都重置）
        total_episodes = 0
        failed = 0
        
        # 记录上一步的 time_steps，用于检测 motion 完成后的重置
        prev_time_steps = motion_cmd.time_steps.clone()

        while total_episodes < args_cli.num_trials :
            # 使用 no_grad 而不是 inference_mode，避免影响环境重置
            with torch.no_grad():
                actions = policy(obs)
                obs, rew, dones, extras = env.step(actions)
            
            # 情况1：检查是否有环境完整完成了 motion
            # motion 完成后，_resample_command 会被调用，time_steps 会被重置为 0（或很小的值）
            # 检测条件：当前 time_steps < 上一步 time_steps（说明被重置了）且没有 done
            curr_time_steps = motion_cmd.time_steps
            finish = curr_time_steps < prev_time_steps
            
            if finish.any():
                completed_ids = torch.nonzero(finish, as_tuple=False).squeeze(-1)
                for env_id in completed_ids:
                    total_episodes += 1      
                    failed += dones[env_id].item()
                    print(f"[DONE] Env {env_id.item()} 终止\t  (failed={dones[env_id].item()}, time_step={prev_time_steps[env_id].item()}/{motion_cmd.motion_length[env_id].item()})")

                    if total_episodes >= args_cli.num_trials:
                        break
            
            # 更新 prev_time_steps
            prev_time_steps = curr_time_steps.clone()
            

        # 6-6 输出结果
        success = total_episodes - failed
        failure_rate = failed / total_episodes if total_episodes > 0 else 0.0
        success_rate = success / total_episodes if total_episodes > 0 else 0.0
        results[F] = failure_rate
        print(f"[EVAL] Force={F:>5.1f} N -> 成功={success}/{total_episodes} ({100*success_rate:.2f}%), "
              f"失败={failed}/{total_episodes} ({100*failure_rate:.2f}%)")

    # close the simulator
    env.close()
    print("\n" + "="*60)
    print("最终评测结果:")
    print_dict(results, nesting=2)
    return results

if __name__ == "__main__":
    evaluate()
    simulation_app.close()