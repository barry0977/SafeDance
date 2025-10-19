from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ForceCommand(CommandTerm):
    cfg: ForceCommandCfg

    def __init__(self, cfg: ForceCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_indexes = torch.tensor(self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device)
        self.num_links = len(self.body_indexes)

        self.current_forces = torch.zeros(self.num_envs, len(self.cfg.body_names), 3, device=self.device)
        self.time_to_next_sample = torch.zeros(self.num_envs, device=self.device) # 到下一次sample剩余步数
        self.duration_left = torch.zeros(self.num_envs, device=self.device) # 当前force持续剩余步数
       
        self.interval_steps = cfg.interval_steps
        self.duration_steps = cfg.duration_steps
        self.force_min, self.force_max = cfg.force_magnitude_range
        self.direction_mode = cfg.direction_mode

        self.metrics["force_active"]        = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force_norm_avg"]      = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force_norm_max"]      = torch.zeros(self.num_envs, device=self.device)
        self.metrics["applied_links_count"] = torch.zeros(self.num_envs, device=self.device)

        # 直接将 current_forces 设置为环境的属性，方便其他地方访问
        env.unwrapped.current_forces = self.current_forces
        print(f"[DEBUG ForceCommand.__init__] 设置 env.unwrapped.current_forces, shape: {self.current_forces.shape}, device: {self.current_forces.device}")

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return self.current_forces.view(self.num_envs, -1)    # shape [E, 3*L]

    def _update_metrics(self):
        force_norm = torch.linalg.vector_norm(self.current_forces, dim=-1)   # [E, L]
        active = self.duration_left > 0                                      # [E]
        self.metrics["force_active"]        = active.float()
        self.metrics["force_norm_avg"]      = force_norm.mean(dim=-1)
        self.metrics["force_norm_max"]      = force_norm.max(dim=-1).values
        self.metrics["applied_links_count"] = (force_norm > 0.0).sum(dim=-1).float()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        # 每个env随机一个方向
        direction = torch.randn(len(env_ids), 3, device=self.device)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-6)
        # 每个env每个link随机一个力大小
        magnitude = torch.rand(len(env_ids), self.num_links, 1, device=self.device)
        magnitude = magnitude * (self.force_max - self.force_min) + self.force_min

        self.current_forces[env_ids] = direction.unsqueeze(1) * magnitude # 要把direction的维度扩展到和magnitude一致
        self.duration_left[env_ids] = self.duration_steps
        self.time_to_next_sample[env_ids] = self.interval_steps + self.duration_steps


    def _update_command(self):
        self.time_to_next_sample -= 1
        self.duration_left -= 1
        env_ids = torch.where(self.time_to_next_sample <= 0)[0]
        self._resample_command(env_ids)

        env_active = torch.where(self.duration_left > 0)[0]
        env_inactive = torch.where(self.duration_left <= 0)[0]
        # 对激活环境施加力
        if env_active.numel() > 0:
            self.robot.set_external_force_and_torque(
                forces=self.current_forces[env_active],
                torques=torch.zeros_like(self.current_forces[env_active]),
                body_ids=self.body_indexes,
                env_ids=env_active,
            )
        # 对非激活环境施加0力
        if env_inactive.numel() > 0:
            self.current_forces[env_inactive] = 0.0
            self.robot.set_external_force_and_torque(
                forces=self.current_forces[env_inactive],
                torques=torch.zeros_like(self.current_forces[env_inactive]),
                body_ids=self.body_indexes,
                env_ids=env_inactive,
            )



    def compute(self, dt: float):
        self._update_metrics()
        self._update_command()



@configclass
class ForceCommandCfg(CommandTermCfg):
    """Configuration for the force command."""

    class_type: type = ForceCommand

    asset_name: str = MISSING

    body_names: list[str] =  [
            "pelvis",                    # 躯干下部/骨盆
            "left_hip_roll_link",        # 左大腿
            "left_knee_link",            # 左小腿
            "right_hip_roll_link",       # 右大腿
            "right_knee_link",           # 右小腿
            #"torso_link",                # 躯干上部(好像被作为锚点了)
            "left_shoulder_roll_link",   # 左大臂
            "left_elbow_link",           # 左小臂
            "right_shoulder_roll_link",  # 右大臂
            "right_elbow_link"           # 右小臂
        ]

    interval_steps: int = MISSING
    duration_steps: int = MISSING
    force_magnitude_range: tuple[float, float] = MISSING
    direction_mode: str = MISSING

