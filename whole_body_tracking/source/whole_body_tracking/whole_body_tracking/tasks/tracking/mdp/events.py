from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)



def apply_gradual_external_force(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    key_links: list[str] | None = None,
    max_force_range: tuple[float, float] = (-40.0, 40.0),
    duration_range: tuple[float, float] = (1.0, 3.0),
    ramp_ratio: float = 0.3,
    hold_ratio: float = 0.4,
    probability: float = 0.2,
):
    """
    施加线性渐变的持续外力（线性增长 -> 保持 -> 线性衰减）
    同时为环境添加get_forces方法
    
    力的变化模式：
    - 前30%时间：从0线性增长到最大值
    - 中间40%时间：保持最大值
    - 后30%时间：从最大值线性衰减到0
    
    Args:
        env: 环境实例
        env_ids: 环境ID列表
        asset_cfg: 资产配置
        key_links: 关键关节名称列表
        max_force_range: 最大力的大小范围 (N)
        duration_range: 总持续时间范围 (s)
        ramp_ratio: 上升/下降阶段占总时间的比例（默认0.3）
        hold_ratio: 保持阶段占总时间的比例（默认0.4）
        probability: 每步应用的概率
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    
    # 获取key links的body索引
    if key_links is None:
        key_links = [
            "pelvis",                    # 躯干下部/骨盆
            "left_hip_roll_link",        # 左大腿
            "left_knee_link",            # 左小腿
            "right_hip_roll_link",       # 右大腿
            "right_knee_link",           # 右小腿
            "torso_link",                # 躯干上部
            "left_shoulder_roll_link",   # 左大臂
            "left_elbow_link",           # 左小臂
            "right_shoulder_roll_link",  # 右大臂
            "right_elbow_link"           # 右小臂
        ]
    
    body_indices = asset.find_bodies(key_links, preserve_order=True)[0]
    num_key_links = len(body_indices)
    
    # 初始化渐变力状态（每个环境的每个key link都有独立的力）
    if not hasattr(asset, 'gradual_forces'):
        asset.gradual_forces = {
            'active': torch.zeros(env.scene.num_envs, num_key_links, dtype=torch.bool, device=asset.device),
            'progress': torch.zeros(env.scene.num_envs, num_key_links, device=asset.device),  # 0-1的进度
            'total_duration': torch.zeros(env.scene.num_envs, num_key_links, device=asset.device),
            'max_forces': torch.zeros(env.scene.num_envs, num_key_links, 3, device=asset.device),
            'current_forces': torch.zeros(env.scene.num_envs, num_key_links, 3, device=asset.device),  # 当前施加的力
        }
    
    # 为环境提供get_forces接口
    if not hasattr(env, 'get_forces'):
        def get_forces() -> torch.Tensor:
            return asset.gradual_forces['current_forces'].clone()
        
        env.get_forces = get_forces
    
    dt = env.cfg.sim.dt * env.cfg.decimation
    
    # 更新渐变力的进度
    active_mask = asset.gradual_forces['active']  # (num_envs, num_key_links)
    if active_mask.any():
        asset.gradual_forces['progress'][active_mask] += dt / asset.gradual_forces['total_duration'][active_mask]
        
        # 检查哪些力应该结束
        finished_mask = (asset.gradual_forces['progress'] >= 1.0) & active_mask
        if finished_mask.any():
            asset.gradual_forces['active'][finished_mask] = False
            asset.gradual_forces['progress'][finished_mask] = 0
            asset.gradual_forces['current_forces'][finished_mask] = 0
    
    available_mask = ~asset.gradual_forces['active'].any(dim=1)  
    available_envs = env_ids[available_mask[env_ids]]
    if len(available_envs) == 0:
            return

    for env_idx in available_envs:
        if(torch.rand(1, device=asset.device) < probability):
            force_direction = torch.randn(3, device=asset.device)  #
            force_direction = force_direction / torch.norm(force_direction, dim=0, keepdim=True)
            
            # 为这个环境的每个key link生成外力（方向相同，大小不同）
            for link_idx in range(num_key_links):
                max_force_magnitude = max_force_range[0] + (max_force_range[1] - max_force_range[0]) * torch.rand(1, device=asset.device)
                max_force = max_force_magnitude * force_direction
                
                # 生成持续时间
                duration = (duration_range[0] + (duration_range[1] - duration_range[0]) * torch.rand(1, device=asset.device)).item()
                
                # 设置渐变力状态
                asset.gradual_forces['active'][env_idx, link_idx] = True
                asset.gradual_forces['progress'][env_idx, link_idx] = 0
                asset.gradual_forces['total_duration'][env_idx, link_idx] = duration
                asset.gradual_forces['max_forces'][env_idx, link_idx] = max_force
                asset.gradual_forces['current_forces'][env_idx, link_idx] = 0
        
    
    # 计算并施加当前活跃的渐变力
    active_mask = asset.gradual_forces['active']  # (num_envs, num_key_links)
    
    # 先计算所有力的缩放因子
    progress = asset.gradual_forces['progress']  # (num_envs, num_key_links)
    force_scale = torch.zeros_like(progress)
    
    # 线性增长阶段：0 -> 1
    ramp_mask = (progress < ramp_ratio) & active_mask
    force_scale[ramp_mask] = progress[ramp_mask] / ramp_ratio
    
    # 保持阶段：1
    hold_mask = (progress >= ramp_ratio) & (progress < (ramp_ratio + hold_ratio)) & active_mask
    force_scale[hold_mask] = 1.0
    
    # 线性衰减阶段：1 -> 0
    decay_mask = (progress >= (ramp_ratio + hold_ratio)) & active_mask
    force_scale[decay_mask] = 1.0 - (progress[decay_mask] - ramp_ratio - hold_ratio) / ramp_ratio
    
    # 计算当前力 (num_envs, num_key_links, 3)
    current_forces = asset.gradual_forces['max_forces'] * force_scale.unsqueeze(-1)
    
    # 更新当前力记录
    asset.gradual_forces['current_forces'] = current_forces
    
    # 批量施加外力到所有环境的所有body
    # set_external_force_and_torque需要: (num_envs, num_bodies, 3)
    # 我们需要将current_forces映射到完整的body数组
    num_bodies = asset.num_bodies
    all_forces = torch.zeros(env.scene.num_envs, num_bodies, 3, device=asset.device)
    all_torques = torch.zeros(env.scene.num_envs, num_bodies, 3, device=asset.device)
    
    # 将key link的力填充到对应的body位置
    for link_idx, body_idx in enumerate(body_indices):
        all_forces[:, body_idx, :] = current_forces[:, link_idx, :]
    
    # 施加外力（在body的local frame）
    # 注意：set_external_force_and_torque只是填充buffer
    # 真正的力会在下一个simulation step时由环境的write_data_to_sim()自动施加
    asset.set_external_force_and_torque(
        forces=all_forces,
        torques=all_torques,
        body_ids=None,  # None表示所有bodies
        env_ids=None    # None表示所有envs
    )