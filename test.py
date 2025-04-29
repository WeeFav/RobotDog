import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
register(
    id='RobotDog-v0',
    entry_point='custom_env:RobotDogEnv',
)
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})

env = gym.make('RobotDog-v0', x_ref=3.0, y_ref=0.0)
env.reset()

model = PPO.load("./models/15700000.zip", env=env, device="cpu")

while True:
    observation, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(observation)
        print(action)
        observation, reward, terminated, truncated, info = env.step(action)