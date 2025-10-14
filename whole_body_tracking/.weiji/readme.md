

Data
```bash
python scripts/csv_to_npz.py --input_file '/home/bai/LAFAN1_Retargeting_Dataset/g1/run2_subject1.csv' --input_fps 30 --output_name run2_subject1 --headless


python scripts/pkl_to_npz.py --input_file '/home/bai/PBHC-Internal/PBHC-Motion/g1robot/long-wushu/Fight.pkl' --input_fps 30 --output_name fight --headless


python scripts/replay_npz.py --motion_file=dance1_subject2
python scripts/replay_npz.py --motion_file=fight


python scripts/pklpack_to_npz.py --input_file /home/bai/PBHC-Internal/PBHC-Motion/g1robot/package/Data10k/Data10k.pkl --output_dir ./artifacts/Data10k --input_fps 30 --output_fps 50 --headless
```

Vis
```bash


python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 \
--resume_path=/home/bai/BeyondMimic/logs/rsl_rl/DebugMJC-LargeAAC/2025-09-11_22-31-51_F5Obs-1024-Data10kT/model_20500.pt \
--motion_file=walk_contact_maskMPI_HDM05dgHDM_dg_01-01_04_120_posespkl \
env.commands.motion.future_steps=5 \
agent.policy.actor_hidden_dims=[1024,512,256] \
agent.policy.critic_hidden_dims=[1024,512,256] \
--num_envs=10 env.episode_length_s=100.0 env.commands.motion.start_from_zero_step=True env.events.push_robot=null \
--kit_args "--/log/level=error --/log/outputStreamLevel=error --/log/fileLogLevel=error" 



python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0-PrivPrivObs-v0 \
--resume_path=/home/bai/BeyondMimic/logs/rsl_rl/LargeNetAbl-Priv/2025-09-10_21-31-50_1536-8K-Data10kT/model_21000.pt \
--motion_file=Data10k-train/homejrhanprojectsPBHC-InternalPBHC-Motiong1robotlafanrun2_subject1_0_7345_cont_mask_inter05_S00-30_E \
env.terminations.ee_body_pos=null \
env.terminations.anchor_pos=null \
--num_envs=10 env.episode_length_s=100.0 env.commands.motion.start_from_zero_step=True env.events.push_robot=null \
--kit_args "--/log/level=error --/log/outputStreamLevel=error --/log/fileLogLevel=error" \
env.commands.motion.future_steps=5 \
agent.policy.actor_hidden_dims=[1536,768,384] \
agent.policy.critic_hidden_dims=[1536,768,384]



python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=10 --resume_path=logs/rsl_rl/g1_flat/2025-09-09_16-04-31_v0-Data10k_74-future5/model_29999.pt --motion_file=Data10k_74/*



deploy_mujoco
```bash

_EM() {
    python deploy/deploy_mujoco.py --motion_path=/home/bai/BeyondMimic/artifacts/$1/motion.npz --policy_path=/home/bai/BeyondMimic/logs/rsl_rl/DebugMJC-LargeAAC/2025-09-11_22-31-51_F5Obs-1024-Data10kT/exported/policy_20.5k.onnx
}

_EL(){
    python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 \
    --resume_path=/home/bai/BeyondMimic/logs/rsl_rl/DebugMJC-LargeAAC/2025-09-11_22-31-51_F5Obs-1024-Data10kT/model_20500.pt \
    --motion_file=$1 \
    env.commands.motion.future_steps=5 \
    agent.policy.actor_hidden_dims=[1024,512,256] \
    agent.policy.critic_hidden_dims=[1024,512,256] \
    --num_envs=10 env.episode_length_s=100.0 env.commands.motion.start_from_zero_step=True env.events.push_robot=null \
    --kit_args "--/log/level=error --/log/outputStreamLevel=error --/log/fileLogLevel=error" 
}

python deploy/deploy_mujoco.py --motion_path=/home/bai/BeyondMimic/artifacts/walk_contact_maskMPI_HDM05dgHDM_dg_01-01_04_120_posespkl/motion.npz --policy_path=/home/bai/BeyondMimic/logs/rsl_rl/DebugMJC-LargeAAC/2025-09-11_22-31-51_F5Obs-1024-Data10kT/exported/policy_20.5k.onnx





python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0-NoisePrivObs-v0 \
--resume_path=/home/bai/BeyondMimic/logs/rsl_rl/DebugMJC-LargeNPO/2025-09-11_22-40-33_F5Obs-1024-Data10kT/model_19500.pt \
--motion_file=dance1_subject2 \
env.commands.motion.future_steps=5 \
agent.policy.actor_hidden_dims=[1024,512,256] \
agent.policy.critic_hidden_dims=[1024,512,256] \
--num_envs=10 env.episode_length_s=100.0 env.commands.motion.start_from_zero_step=True env.events.push_robot=null \
--kit_args "--/log/level=error --/log/outputStreamLevel=error --/log/fileLogLevel=error" 

python deploy/deploy_mujoco.py --motion_path=/home/bai/BeyondMimic/artifacts/hand_contact_maskKIT1229hand_to_mouth_right_arm_05_posespkl/motion.npz --policy_path=/home/bai/BeyondMimic/logs/rsl_rl/LargeNetAbl-Priv/2025-09-10_21-31-50_1536-8K-Data10kT/exported/policy.onnx


```


多motion
```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--motion_file=multi_2/* \
--headless --log_project_name BeyondMimic --run_name v0-multi_2-future

python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 --headless --log_project_name BeyondMimic \
--motion_file=Data10k_74/* \
--run_name v0-Data10k_74-future5 \
env.commands.motion.future_steps=5

python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0-PropPropObs-v0 --headless --log_project_name BeyondMimic \
--motion_file=Data10k_74/* \
--run_name v0PropPOv0-Data10k_74-future5 \
env.commands.motion.future_steps=5

python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0-PrivPrivObs-v0 --headless --log_project_name BeyondMimic \
--motion_file=Data10k_74/* \
--run_name v0PrivPOv0-Data10k_74-future5 \
agent.experiment_name=debug \
env.commands.motion.future_steps=5

# ---

python scripts/rsl_rl/train.py --headless --log_project_name BeyondMimic \
--task=Tracking-Flat-G1-v0-PrivPrivObs-v0 \
--motion_file=Data10k-train/* \
--run_name v0PrivPOv0-Data10kT-future5 \
agent.experiment_name=debug \
env.commands.motion.future_steps=5 \
--device=cuda:0 \
--seed=1


python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 --headless --log_project_name BeyondMimic \
--motion_file=Data10k-train/* \
--run_name v0-Data10kT-future5 \
env.commands.motion.future_steps=5

```

