import os
from datetime import datetime
from pprint import pformat

taskname = "PGObs"
date = datetime.now().strftime("%Y%m%d-%H%M%s")
comment = """


"""
motions = {
    # "Charleston": "dance1_subject2",
    "Data10kT": "Data10k-train/*",
}

settings = {
    # "": ("env.commands.motion.future_steps=5 \\"),
    "F5Obs-1024":
        (
            "--num_envs=8192 \\\n"
            "env.commands.motion.future_steps=5 \\\n"
            "agent.policy.actor_hidden_dims=[1024,512,256] \\\n"
            "agent.policy.critic_hidden_dims=[1024,512,256] \\"
        ),
}
counter = 0
total = len(motions) * len(settings)

base = """
# {taskname} {motion} {setting}  ({counter} / {total})
python scripts/rsl_rl/train.py --headless --log_project_name BeyondMimic \\
--task=Tracking-Flat-G1-v0-ProjGravObs-v0 \\
--motion_file={motion_file} \\
--run_name {setting}-{motion} \\
agent.experiment_name={taskname} \\
{extra} \\
--seed=1 \\
--device=cuda:0
"""

headline = f"""
#################
: <<'HEAD'
Taskname: {taskname}
Date: {date}

Comment: {comment}

Total: {total}

Motions: {pformat(motions)}

Settings: {pformat(settings)}

Template:
{(base)}

HEAD
#################
"""
print(f"#" * 100)
print(headline)

for motion, motion_file in motions.items():
    # Check File exists - skip check for wildcard patterns
    if not motion_file.endswith("/*") and not os.path.exists(f"artifacts/{motion_file}"):
        print(f"File {motion_file} does not exist")
        exit(1)
    for setting, extra in settings.items():
        cmd = base.format(
            counter=counter + 1,
            total=total,
            taskname=taskname,
            motion=motion,
            setting=setting,
            extra=(extra + "\n" if extra else ""),
            motion_file=motion_file,
        )
        print(cmd)
        print("sleep 0.4s")  # sleep to avoid order issue
        print("\n" + "#" * 80 + "\n")
        counter += 1

print(f"# Total commands: {counter}")
print(f"#" * 100)
