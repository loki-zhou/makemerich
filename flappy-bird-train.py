# pip install flappy-bird-gymnasium
import pygame
import flappy_bird_gymnasium
import gymnasium as gym
import os
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# 创建单个环境的工厂函数
def make_env():
    def _init():
        env = gym.make("FlappyBird-v0",use_lidar=False)
        return env
    return _init
num_envs = 8
# pygame.init()
# clock = pygame.time.Clock()
# env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
envs = gym.make("FlappyBird-v0",use_lidar=False)

# envs = DummyVecEnv([make_env() for _ in range(num_envs)])

monitor_dir = "./model_dir/FlappyBird/"
os.makedirs(monitor_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path=monitor_dir,
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)
import torch as th
policy_kwargs = dict(
                     net_arch=dict(pi=[256, 256], vf=[256, 256]))

model = PPO("MlpPolicy", envs, policy_kwargs=policy_kwargs, tensorboard_log="./tlog/FlappyBird/ppo/",
            batch_size=256,
            gamma=0,
            verbose=1)

# rl_model_3235000_steps.zip
# model.set_parameters("model_dir/FlappyBird_best.zip")

model.learn(total_timesteps=3_000_000, callback=checkpoint_callback)

model.save("model_dir/FlappyBird_bestV2.zip")
