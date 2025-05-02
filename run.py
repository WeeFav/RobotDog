import gymnasium as gym
from stable_baselines3 import PPO

from RobotDog import custom_env

models_dir = "models/PPO7"

env = custom_env.RobotDogEnv(7,0)
env.reset()

model_path = f"{models_dir}/580000.zip"
model = PPO.load(model_path, env=env)

episodes = 100

for ep in range(episodes):
    obs = env.reset()[0]
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, Truncated, info = env.step(action)
        env.render()
        print(rewards)