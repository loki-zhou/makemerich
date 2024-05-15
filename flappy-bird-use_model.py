# pip install flappy-bird-gymnasium
import pygame
import flappy_bird_gymnasium
import gymnasium
from stable_baselines3 import PPO

model = PPO.load("model_dir/FlappyBird_best.zip")
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

done, truncated = False, False
observation, info = env.reset()

while not done and not truncated:
    action, _states = model.predict(observation)
    observation, reward, done, truncated, info = env.step(action)
    print("reward = ", reward)
    env.render()