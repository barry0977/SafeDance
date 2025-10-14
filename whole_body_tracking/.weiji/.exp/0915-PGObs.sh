####################################################################################################

#################
: <<'HEAD'
Taskname: PGObs
Date: 20250915-15051757919917

Comment: 




Total: 1

Motions: {'Data10kT': 'Data10k-train/*'}

Settings: {'F5Obs-1024': '--num_envs=8192 \\\n'
               'env.commands.motion.future_steps=5 \\\n'
               'agent.policy.actor_hidden_dims=[1024,512,256] \\\n'
               'agent.policy.critic_hidden_dims=[1024,512,256] \\'}

Template:

# {taskname} {motion} {setting}  ({counter} / {total})
python scripts/rsl_rl/train.py --headless --log_project_name BeyondMimic \
--task=Tracking-Flat-G1-v0-ProjGravObs-v0 \
--motion_file={motion_file} \
--run_name {setting}-{motion} \
agent.experiment_name={taskname} \
{extra} \
--seed=1 \
--device=cuda:0


HEAD
#################


# PGObs Data10kT F5Obs-1024  (1 / 1)
python scripts/rsl_rl/train.py --headless --log_project_name BeyondMimic \
--task=Tracking-Flat-G1-v0-ProjGravObs-v0 \
--motion_file=Data10k-train/* \
--run_name F5Obs-1024-Data10kT \
agent.experiment_name=PGObs \
--num_envs=8192 \
env.commands.motion.future_steps=5 \
agent.policy.actor_hidden_dims=[1024,512,256] \
agent.policy.critic_hidden_dims=[1024,512,256] \
 \
--seed=1 \
--device=cuda:0

sleep 0.4s

################################################################################

# Total commands: 1
####################################################################################################
