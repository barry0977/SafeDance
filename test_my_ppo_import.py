#!/usr/bin/env python3
"""
测试脚本：验证本地安装的 rsl_rl 包是否正确
包括 MY_PPO, Encoder, ForceEstimator, My_RolloutStorage 等模块
"""

import sys
import torch

print("="*80)
print("测试 rsl_rl 本地安装")
print("="*80)

# 测试 1: 导入基础算法类
print("\n[测试 1] 导入算法类...")
try:
    from rsl_rl.algorithms import PPO, Distillation, MY_PPO
    print("✅ 成功导入: PPO, Distillation, MY_PPO")
    print(f"   MY_PPO 类位置: {MY_PPO.__module__}")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 测试 2: 导入网络模块
print("\n[测试 2] 导入网络模块...")
try:
    from rsl_rl.modules import ActorCritic, Encoder, ForceEstimator
    print("✅ 成功导入: ActorCritic, Encoder, ForceEstimator")
    print(f"   Encoder 类位置: {Encoder.__module__}")
    print(f"   ForceEstimator 类位置: {ForceEstimator.__module__}")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 测试 3: 导入存储类
print("\n[测试 3] 导入存储类...")
try:
    from rsl_rl.storage import My_RolloutStorage
    print("✅ 成功导入: My_RolloutStorage")
    print(f"   My_RolloutStorage 类位置: {My_RolloutStorage.__module__}")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 测试 4: 导入 Runner
print("\n[测试 4] 导入 Runner...")
try:
    from rsl_rl.runners import OnPolicyRunner1
    print("✅ 成功导入: OnPolicyRunner1")
    print(f"   OnPolicyRunner1 类位置: {OnPolicyRunner1.__module__}")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 测试 5: 实例化 Encoder
print("\n[测试 5] 实例化 Encoder...")
try:
    encoder = Encoder(num_obs=50, feature_dim=128)
    print("✅ Encoder 实例化成功")
    print(f"   is_recurrent: {encoder.is_recurrent}")
    
    # 测试前向传播
    test_obs = torch.randn(4, 50)
    features = encoder(test_obs)
    print(f"   输入形状: {test_obs.shape} → 输出形状: {features.shape}")
    assert features.shape == (4, 128), "特征维度不正确"
    print("✅ Encoder 前向传播正确")
except Exception as e:
    print(f"❌ Encoder 测试失败: {e}")
    sys.exit(1)

# 测试 6: 实例化 ForceEstimator
print("\n[测试 6] 实例化 ForceEstimator...")
try:
    force_estimator = ForceEstimator(feature_dim=128, num_force=9)
    print("✅ ForceEstimator 实例化成功")
    print(f"   is_recurrent: {force_estimator.is_recurrent}")
    
    # 测试前向传播
    test_features = torch.randn(4, 128)
    estimated_forces = force_estimator(test_features)
    print(f"   输入形状: {test_features.shape} → 输出形状: {estimated_forces.shape}")
    assert estimated_forces.shape == (4, 27), "力估计维度不正确 (应该是 9*3=27)"
    print("✅ ForceEstimator 前向传播正确")
except Exception as e:
    print(f"❌ ForceEstimator 测试失败: {e}")
    sys.exit(1)

# 测试 7: 实例化 ActorCritic
print("\n[测试 7] 实例化 ActorCritic...")
try:
    policy = ActorCritic(
        num_actor_obs=128,  # 使用 feature_dim
        num_critic_obs=128,
        num_actions=12
    )
    print("✅ ActorCritic 实例化成功")
    
    # 测试前向传播
    test_features = torch.randn(4, 128)
    actions = policy.act(test_features)
    values = policy.evaluate(test_features)
    print(f"   特征形状: {test_features.shape} → 动作形状: {actions.shape}, 值形状: {values.shape}")
    assert actions.shape == (4, 12), "动作维度不正确"
    assert values.shape == (4, 1), "值维度不正确"
    print("✅ ActorCritic 前向传播正确")
except Exception as e:
    print(f"❌ ActorCritic 测试失败: {e}")
    sys.exit(1)

# 测试 8: 实例化 MY_PPO
print("\n[测试 8] 实例化 MY_PPO...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   使用设备: {device}")
    
    policy = ActorCritic(
        num_actor_obs=128,
        num_critic_obs=128,
        num_actions=12
    ).to(device)
    
    alg = MY_PPO(
        policy=policy,
        device=device,
        feature_dim=128,
        raw_obs_dim=50,  # 原始观测维度
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3
    )
    print("✅ MY_PPO 实例化成功")
    print(f"   Encoder: {alg.encoder}")
    print(f"   ForceEstimator: {alg.force_estimator}")
    print(f"   优化器包含参数组数: {len(list(alg.optimizer.param_groups))}")
except Exception as e:
    print(f"❌ MY_PPO 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 9: 初始化存储
print("\n[测试 9] 初始化 RolloutStorage...")
try:
    alg.init_storage(
        training_type="rl",
        num_envs=10,
        num_transitions_per_env=24,
        actor_obs_shape=[50],
        critic_obs_shape=[50],
        actions_shape=[12]
    )
    print("✅ RolloutStorage 初始化成功")
    print(f"   存储类型: {type(alg.storage)}")
    print(f"   latent_feature 形状: {alg.storage.latent_feature.shape}")
    print(f"   real_forces 形状: {alg.storage.real_forces.shape}")
except Exception as e:
    print(f"❌ RolloutStorage 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 10: 测试 act 方法
print("\n[测试 10] 测试 MY_PPO.act() 方法...")
try:
    obs = torch.randn(10, 50).to(device)
    critic_obs = torch.randn(10, 50).to(device)
    
    actions = alg.act(obs, critic_obs)
    print("✅ act() 方法执行成功")
    print(f"   观测形状: {obs.shape}")
    print(f"   动作形状: {actions.shape}")
    print(f"   latent_feature 形状: {alg.transition.latent_feature.shape}")
    assert actions.shape == (10, 12), "动作维度不正确"
    assert alg.transition.latent_feature.shape == (10, 128), "latent_feature 维度不正确"
    print("✅ act() 输出维度正确")
except Exception as e:
    print(f"❌ act() 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 11: 测试 process_env_step 方法
print("\n[测试 11] 测试 MY_PPO.process_env_step() 方法...")
try:
    rewards = torch.randn(10).to(device)
    dones = torch.zeros(10).to(device)
    real_forces = torch.randn(10, 27).to(device)  # 9 links * 3
    infos = {"real_forces": real_forces}
    
    alg.process_env_step(rewards, dones, infos)
    print("✅ process_env_step() 方法执行成功")
    print(f"   存储的 real_forces 形状: {alg.storage.real_forces[alg.storage.step-1].shape}")
except Exception as e:
    print(f"❌ process_env_step() 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 12: 检查优化器参数
print("\n[测试 12] 检查优化器包含的参数...")
try:
    total_params = 0
    encoder_params = sum(p.numel() for p in alg.encoder.parameters())
    force_estimator_params = sum(p.numel() for p in alg.force_estimator.parameters())
    policy_params = sum(p.numel() for p in alg.policy.parameters())
    
    print(f"   Encoder 参数量: {encoder_params:,}")
    print(f"   ForceEstimator 参数量: {force_estimator_params:,}")
    print(f"   Policy 参数量: {policy_params:,}")
    print(f"   总参数量: {encoder_params + force_estimator_params + policy_params:,}")
    
    # 验证优化器包含所有参数
    optimizer_params = sum(p.numel() for group in alg.optimizer.param_groups for p in group['params'])
    print(f"   优化器管理的参数量: {optimizer_params:,}")
    
    if optimizer_params == encoder_params + force_estimator_params + policy_params:
        print("✅ 优化器正确包含所有网络的参数")
    else:
        print("⚠️ 警告: 优化器参数量不匹配")
except Exception as e:
    print(f"❌ 参数检查失败: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("🎉 所有测试通过！rsl_rl 本地安装正确，MY_PPO 及相关模块工作正常！")
print("="*80)

