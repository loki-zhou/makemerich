import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.envs import FakeImageEnv
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()