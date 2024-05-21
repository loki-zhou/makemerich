import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import flappy_bird_gymnasium


# 创建环境
# env = gym.make('CartPole-v1')
env = gym.make("FlappyBird-v0",  use_lidar=False)
env = DummyVecEnv([lambda: env])

import torch as th
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 128], vf=[128, 128]))

# 创建默认模型
model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)

# model = PPO('MlpPolicy', env,  verbose=1)

# 获取策略网络和价值网络
policy = model.policy

# 打印策略网络和价值网络结构
print("Policy Network (Actor):")
print(policy.mlp_extractor.policy_net)

print("\nValue Network (Critic):")
print(policy.mlp_extractor.value_net)