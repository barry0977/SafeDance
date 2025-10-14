####################################################################################################

#################
: <<'HEAD'
Taskname: LargeNetAbl-Priv
Date: 20250910-21181757510322

Comment: 




Total: 2

Motions: {'Data10kT': 'Data10k-train/*'}

Settings: {'1024-8K': '--num_envs=8192 \\\n'
            'env.commands.motion.future_steps=5 \\\n'
            'agent.policy.actor_hidden_dims=[1024,512,256] \\\n'
            'agent.policy.critic_hidden_dims=[1024,512,256] \\',
 '1536-8K': '--num_envs=8192 \\\n'
            'env.commands.motion.future_steps=5 \\\n'
            'agent.policy.actor_hidden_dims=[1536,768,384] \\\n'
            'agent.policy.critic_hidden_dims=[1536,768,384] \\'}

Template:

# {taskname} {motion} {setting}  ({counter} / {total})
python scripts/rsl_rl/train.py --headless --log_project_name BeyondMimic \
--task=Tracking-Flat-G1-v0-PrivPrivObs-v0 \
--motion_file={motion_file} \
--run_name {setting}-{motion} \
agent.experiment_name={taskname} \
{extra} \
--seed=1 \
--device=cuda:0


HEAD
#################


# LargeNetAbl-Priv Data10kT 1024-8K  (1 / 2)
python scripts/rsl_rl/train.py --headless --log_project_name BeyondMimic \
--task=Tracking-Flat-G1-v0-PrivPrivObs-v0 \
--motion_file=Data10k-train/* \
--run_name 1024-8K-Data10kT \
agent.experiment_name=LargeNetAbl-Priv \
--num_envs=8192 \
env.commands.motion.future_steps=5 \
agent.policy.actor_hidden_dims=[1024,512,256] \
agent.policy.critic_hidden_dims=[1024,512,256] \
 \
--seed=1 \
--device=cuda:0

sleep 0.4s

################################################################################


# LargeNetAbl-Priv Data10kT 1536-8K  (2 / 2)
python scripts/rsl_rl/train.py --headless --log_project_name BeyondMimic \
--task=Tracking-Flat-G1-v0-PrivPrivObs-v0 \
--motion_file=Data10k-train/* \
--run_name 1536-8K-Data10kT \
agent.experiment_name=LargeNetAbl-Priv \
--num_envs=8192 \
env.commands.motion.future_steps=5 \
agent.policy.actor_hidden_dims=[1536,768,384] \
agent.policy.critic_hidden_dims=[1536,768,384] \
 \
--seed=1 \
--device=cuda:0

sleep 0.4s

################################################################################

# Total commands: 2
####################################################################################################
