from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id='RobotDog-v0',
    entry_point='custom_env:RobotDogEnv',
)

env = gym.make('RobotDog-v0', x_ref=3.0, y_ref=0.0)
check_env(env)