import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

register(
    id='RobotDog-v0',
    entry_point='custom_env:RobotDogEnv',
)

np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})

env = gym.make('RobotDog-v0', x_ref=3.0, y_ref=3.0)
env.reset()
while True:
    random_action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(random_action)
    print("observation", observation)
    env.render()