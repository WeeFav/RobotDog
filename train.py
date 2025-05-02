import gymnasium as gym
from stable_baselines3 import PPO
import os

import RobotDog.custom_env

models_dir = "models/PPO7"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)
env = RobotDog.custom_env.RobotDogEnv(7,0)
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 20000
iters = 0
for i in range(30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO7")
    model.save(f"{models_dir}/{TIMESTEPS*i}")