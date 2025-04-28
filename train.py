import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
import os
import time
register(
    id='RobotDog-v0',
    entry_point='custom_env:RobotDogEnv',
)
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})


models_dir = f"./models/{int(time.time())}/"
log_dir = f"./logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(log_dir):
	os.makedirs(log_dir)

env = gym.make('RobotDog-v0', x_ref=3.0, y_ref=0.0)
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, device="cpu")

TIMESTEPS = 10000
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")