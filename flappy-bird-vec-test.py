import os
import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

env_id = "FlappyBird-v0"
n_training_envs = 1
n_eval_envs = 5

# Create log dir where evaluation results will be saved
eval_log_dir = "./eval_logs/"
os.makedirs(eval_log_dir, exist_ok=True)
# env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
# Initialize a vectorized training environment with default parameters
# train_env = make_vec_env(env_id, n_envs=n_training_envs, seed=0, env_kwargs={"use_lidar": False, "render_mode":"human"})
train_env = make_vec_env(env_id, n_envs=n_training_envs, seed=0, env_kwargs={"use_lidar": False})
train_env = VecFrameStack(train_env, n_stack=4)
# Separate evaluation env, with different parameters passed via env_kwargs
# Eval environments can be vectorized to speed up evaluation.
# eval_env = make_vec_env(env_id, n_envs=n_eval_envs, seed=0,
#                         env_kwargs={'g':0.7})

# Create callback that evaluates agent for 5 episodes every 500 training environment steps.
# When using multiple training environments, agent will be evaluated every
# eval_freq calls to train_env.step(), thus it will be evaluated every
# (eval_freq * n_envs) training steps. See EvalCallback doc for more information.
# eval_callback = EvalCallback(eval_env, best_model_save_path=eval_log_dir,
#                               log_path=eval_log_dir, eval_freq=max(500 // n_training_envs, 1),
#                               n_eval_episodes=5, deterministic=True,
#                               render=False)

# model = PPO("MlpPolicy", train_env, verbose=1)
# # model.learn(5000, callback=eval_callback)
# model.learn(int(5e5))
# model.save()
model = PPO.load("eval_logs/FlappyBird/best_model.zip")


total_reward = 0
done, truncated = False, False
observation= train_env.reset()
while not done and not truncated:
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, done,  info = train_env.step(action)
    total_reward = total_reward + reward
    print("total_reward = ", total_reward)
    # train_env.render()
# print("total_reward = ", total_reward)