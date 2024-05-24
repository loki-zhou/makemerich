import os
import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack

env_id = "FlappyBird-v0"
n_training_envs = 4
n_eval_envs = 2

# Create log dir where evaluation results will be saved
eval_log_dir = "./eval_logs/FlappyBird"
os.makedirs(eval_log_dir, exist_ok=True)
# env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
# Initialize a vectorized training environment with default parameters


train_env = make_vec_env(env_id, n_envs=n_training_envs, seed=0, env_kwargs={"use_lidar": False})
train_env = VecFrameStack(train_env, n_stack=4)

# Separate evaluation env, with different parameters passed via env_kwargs
# Eval environments can be vectorized to speed up evaluation.
eval_env = make_vec_env(env_id, n_envs=n_eval_envs, seed=0,
                        env_kwargs={"use_lidar": False})
eval_env = VecFrameStack(eval_env, n_stack=4)

# Create callback that evaluates agent for 5 episodes every 500 training environment steps.
# When using multiple training environments, agent will be evaluated every
# eval_freq calls to train_env.step(), thus it will be evaluated every
# (eval_freq * n_envs) training steps. See EvalCallback doc for more information.
eval_callback = EvalCallback(eval_env, best_model_save_path=eval_log_dir,
                              log_path=eval_log_dir, eval_freq=max(50000 // n_training_envs, 1),
                              n_eval_episodes=5, deterministic=True,
                              render=False)

model = PPO("MlpPolicy", train_env, verbose=1)
model.learn(int(1e6), callback=eval_callback)
# model.learn(int(1e6))
