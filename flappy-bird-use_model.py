# pip install flappy-bird-gymnasium
# import pygame
import flappy_bird_gymnasium
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack

env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
# env = gym.make("FlappyBird-v0",  use_lidar=False)
# env = gym.wrappers.NormalizeObservation(env)

# envs = gym.make("FlappyBird-v0",use_lidar=False)

# envs = DummyVecEnv([make_env() for _ in range(num_envs)])
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env,4,channels_order='last')
#
model = PPO.load("model_dir/FlappyBird_bestV3.zip")
# model = PPO.load("model_dir/FlappyBird_best.zip")
# model = PPO.load("model_dir/FlappyBird/rl_model_3000000_steps.zip")

reward_list = []

# for item in range(100):
#     total_reward = 0
#     done, truncated = False, False
#     observation, info = env.reset()
#     while not done and not truncated:
#         action, _states = model.predict(observation, deterministic=True)
#         observation, reward, done, truncated, info = env.step(action)
#         total_reward = total_reward + reward
#         # env.render()
#     print("total_reward = ", total_reward)
#     reward_list.append(total_reward)
#
# import statistics
# print("mean(reward) = ", statistics.mean(reward_list))

total_reward = 0
done, truncated = False, False
observation= env.reset()
while not done and not truncated:
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, done,  info = env.step(action)
    total_reward = total_reward + reward
    env.render()
print("total_reward = ", total_reward)

