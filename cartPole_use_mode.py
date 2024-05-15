import time

from stable_baselines3 import PPO
import gymnasium as gym


model = PPO.load("best.zip")
env = gym.make("CartPole-v1", render_mode="human")
done, truncated = False, False
observation, info = env.reset()

while not done and not truncated:
    action, _states = model.predict(observation)
    observation, reward, done, truncated, info = env.step(action)
    print("reward = ", reward)
    env.render()