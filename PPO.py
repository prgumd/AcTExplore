import os
import hydra
from stable_baselines3 import PPO
from env import TactoEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from omegaconf import OmegaConf
import numpy as np
from sb3_contrib import RecurrentPPO

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.locals.get("dones")[-1]:
            self.logger.record("episode/coverage", self.locals.get("infos")[0]["coverage"])
            self.logger.record("episode/reward", self.locals.get("infos")[0]["acc_reward"])
            self.logger.record("episode/length", self.locals.get("infos")[0]["horizon_counter"])

        return True

@hydra.main(config_path="conf", config_name="RL")
def train(cfg):
    env = TactoEnv(cfg)
    save_dir = os.path.join('Training', 'Logs', cfg.RL.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=cfg.RL.save_dir == "PPO_TEST")
    OmegaConf.save(cfg, os.path.join(save_dir, "RL.yaml"))
    
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.RL.save_freq,
        save_path=save_dir,
        name_prefix="model",
    )
    if cfg.RL.pretrain_model_path:
        if cfg.RL.algorithm == "RecurrentPPO":
            model = RecurrentPPO.load(cfg.RL.pretrain_model_path, env, verbose=1, tensorboard_log=save_dir)
        else:
            model = PPO.load(cfg.RL.pretrain_model_path, env, verbose=1, tensorboard_log=save_dir)
    else:
        if cfg.RL.algorithm == "RecurrentPPO":
            model = RecurrentPPO(cfg.RL.policy_network, env, verbose=1, tensorboard_log=save_dir,
                        policy_kwargs=dict(normalize_images=False), n_steps=1024)
        else:
            model = PPO(cfg.RL.policy_network, env, verbose=1, tensorboard_log=save_dir,
                        policy_kwargs=dict(normalize_images=False), n_steps=1024)

    coverage_callback = TensorboardCallback()

    model.learn(total_timesteps=100e4, callback=[checkpoint_callback, coverage_callback])

if __name__ == "__main__":
    train()
