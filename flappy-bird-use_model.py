# pip install flappy-bird-gymnasium
# import pygame
import flappy_bird_gymnasium
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import gymnasium as gym
from stable_baselines3 import PPO


env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
# env = gym.wrappers.NormalizeObservation(env)
#
model = PPO.load("model_dir/FlappyBird_best.zip")
# model = PPO.load("model_dir/FlappyBird/rl_model_3000000_steps.zip")
done, truncated = False, False
observation, info = env.reset()
total_reward = 0

while not done and not truncated:
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, done, truncated, info = env.step(action)
    total_reward = total_reward + reward
    print("reward = ", reward)
    env.render()

print("total_reward = ", total_reward)