####################################################################################################

#################
: <<'HEAD'
Taskname: DebugMJC
Date: 20250911-22291757600962

Comment: 




Total: 2

Motions: {'Charleston': 'dance1_subject2'}

Settings: {'F5Obs': '--num_envs=8192 \\\nenv.commands.motion.future_steps=5 \\',
 'OldObs': '--num_envs=8192 \\\nenv.commands.motion.future_steps=1 \\'}

Template:

# {taskname} {motion} {setting}  ({counter} / {total})
python scripts/rsl_rl/train.py --headless --log_project_name BeyondMimic \
--task=Tracking-Flat-G1-v0 \
--motion_file={motion_file} \
--run_name {setting}-{motion} \
agent.experiment_name={taskname} \
{extra} \
--seed=1 \
--device=cuda:0


HEAD
#################


# DebugMJC Charleston OldObs  (1 / 2)
python scripts/rsl_rl/train.py --headless --log_project_name BeyondMimic \
--task=Tracking-Flat-G1-v0 \
--motion_file=dance1_subject2 \
--run_name OldObs-Charleston \
agent.experiment_name=DebugMJC \
--num_envs=8192 \
env.commands.motion.future_steps=1 \
 \
--seed=1 \
--device=cuda:0

sleep 0.4s

################################################################################


# DebugMJC Charleston F5Obs  (2 / 2)
python scripts/rsl_rl/train.py --headless --log_project_name BeyondMimic \
--task=Tracking-Flat-G1-v0 \
--motion_file=dance1_subject2 \
--run_name F5Obs-Charleston \
agent.experiment_name=DebugMJC \
--num_envs=8192 \
env.commands.motion.future_steps=5 \
 \
--seed=1 \
--device=cuda:0

sleep 0.4s

################################################################################

# Total commands: 2
####################################################################################################
