# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch

import onnx

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

from whole_body_tracking.tasks.tracking.mdp import MotionCommand


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    encoder: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, encoder, verbose)
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, encoder=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)
        # 保存 encoder（如果使用了 MY_PPO）
        if encoder is not None:
            import copy
            self.encoder = copy.deepcopy(encoder)
            self.encoder.cpu()
        else:
            self.encoder = None
        
        # cmd: MotionCommand = env.command_manager.get_term("motion")
        # self.joint_pos = cmd.motion.joint_pos.to("cpu")
        # self.joint_vel = cmd.motion.joint_vel.to("cpu")
        # self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        # self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        # self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        # self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        # self.time_step_total = self.joint_pos.shape[0]

    def forward(self, x):
        """推理时的前向流程应与训练一致：先归一化 → 再编码 → 最后进入 Actor。"""

        # 1. 归一化（若提供 normalizer）
        if self.normalizer is not None:
            x = self.normalizer(x)

        # 2. 编码（若使用了 MY_PPO 的 encoder）
        if self.encoder is not None:
            x = self.encoder(x)

        # 3. Actor 输出动作
        return (self.actor(x), )

    def _forward_bak(self, x, time_step):
        # time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(self.normalizer(x)),
            # self.joint_pos[time_step_clamped],
            # self.joint_vel[time_step_clamped],
            # self.body_pos_w[time_step_clamped],
            # self.body_quat_w[time_step_clamped],
            # self.body_lin_vel_w[time_step_clamped],
            # self.body_ang_vel_w[time_step_clamped],
        )

    def export(self, path, filename):
        self.to("cpu")
        # 输入维度应该始终是“原始观测维度”，即 normalizer 的输入维度
        if self.normalizer is not None:
            input_dim = self.normalizer._mean.shape[1]
        else:
            # fallback：从 encoder 或 actor 推断
            if self.encoder is not None:
                input_dim = self.encoder.encoder[0].in_features
            else:
                input_dim = self.actor[0].in_features
        
        obs = torch.zeros(1, input_dim)
        torch.onnx.export(
            self,
            (obs, ),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            # input_names=["obs", "time_step"],
            output_names=[
                "actions",
                # "joint_pos",
                # "joint_vel",
                # "body_pos_w",
                # "body_quat_w",
                # "body_lin_vel_w",
                # "body_ang_vel_w",
            ],
            dynamic_axes={},
        )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)
    metadata = {
        "run_path": run_path,
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": env.observation_manager.active_terms["policy"],
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "body_names": env.command_manager.get_term("motion").cfg.body_names,
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
