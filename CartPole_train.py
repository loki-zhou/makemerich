import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

monitor_dir = "./model_dir"
os.makedirs(monitor_dir, exist_ok=True)

# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1")

checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path=monitor_dir,
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)
model = PPO("MlpPolicy", env, tensorboard_log="./tlog/ppo/", verbose=1)

model.learn(total_timesteps=1_00000, callback=checkpoint_callback)

model.save("model_dir/best.zip")
env.close()