# pip install flappy-bird-gymnasium
import pygame
import flappy_bird_gymnasium
import gymnasium
import os
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO

pygame.init()
clock = pygame.time.Clock()
# env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
env = gymnasium.make("FlappyBird-v0",use_lidar=False)

monitor_dir = "./model_dir/FlappyBird/"
os.makedirs(monitor_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path=monitor_dir,
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)
model = PPO("MlpPolicy", env, tensorboard_log="./tlog/FlappyBird/ppo/", verbose=1)

model.learn(total_timesteps=5_000_000, callback=checkpoint_callback)

model.save("model_dir/FlappyBird_best.zip")
env.close()