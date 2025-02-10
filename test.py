import os
import hydra
from stable_baselines3 import PPO
from env import TactoEnv
from sb3_contrib import RecurrentPPO

@hydra.main(config_path="conf", config_name="test")
def test(cfg, num_repeat=1):
    env = TactoEnv(cfg)
    input("press any  key to start")
    if cfg.RL.algorithm == "RecurrentPPO":
        model = RecurrentPPO.load(cfg.RL.pretrain_model_path, env)
    else:
        model = PPO.load(cfg.RL.pretrain_model_path, env)

    for i in range(num_repeat):
        print("##### Iteration: ", i)
        model.learn(total_timesteps=5000)

if __name__ == "__main__":
    test(num_repeat=1)
