import os
from typing import Optional

from rsl_rl.env import VecEnv

from rsl_rl.runners.on_policy_runner1 import OnPolicyRunner1 as OnPolicyRunner
# from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

# class MyOnPolicyRunner(OnPolicyRunner):
#     def save(self, path: str, infos=None):
#         """Save the model and training information."""
#         super().save(path, infos)
#         if self.logger_type in ["wandb"]:
#             policy_path = path.split("model")[0]
#             filename = policy_path.split("/")[-2] + ".onnx"
#             export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
#             attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
#             wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device="cpu",
        registry_name: Optional[str] = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            
            # 如果使用了 MY_PPO，传入 encoder
            encoder = getattr(self.alg, 'encoder', None)
            
            export_motion_policy_as_onnx(
                self.env.unwrapped,
                self.alg.policy,
                normalizer=self.obs_normalizer,
                encoder=encoder,
                path=policy_path,
                filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
