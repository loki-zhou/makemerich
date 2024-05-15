要获取 `CartPole-v1` 的示范数据，可以通过手动编写一个专家策略（或近似的专家策略），然后运行该策略以生成数据。以下是一个示例代码，它演示了如何使用一个简单的专家策略来生成 `CartPole-v1` 环境的示范数据。

### 1. 导入必要的库
```python
import gym
import numpy as np
import pickle
```

### 2. 定义一个简单的专家策略
我们使用一个基于策略的专家，这个策略比较简单，当杆子向左倾斜时向左移动，当杆子向右倾斜时向右移动。
```python
def expert_policy(state):
    # state[2] 是杆子的角度
    return 0 if state[2] < 0 else 1
```

### 3. 收集示范数据
运行专家策略以生成示范数据，并将数据保存到列表中。
```python
env = gym.make('CartPole-v1')
num_episodes = 100  # 生成 100 个 episodes 的数据

demo_data = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = expert_policy(state)
        next_state, reward, done, info = env.step(action)
        demo_data.append((state, action, reward, next_state, done))
        state = next_state

env.close()

# 将示范数据保存到文件
with open('cartpole_demo_data.pkl', 'wb') as f:
    pickle.dump(demo_data, f)
```

### 4. 加载示范数据
当你需要使用这些示范数据时，可以将其从文件中加载出来。
```python
with open('cartpole_demo_data.pkl', 'rb') as f:
    demo_data = pickle.load(f)
```

### 5. 使用示范数据预训练PPO模型
结合之前的代码，将示范数据用于预训练PPO模型。以下是完整的代码：

```python
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# 1. 定义一个简单的专家策略
def expert_policy(state):
    return 0 if state[2] < 0 else 1

# 2. 收集示范数据
env = gym.make('CartPole-v1')
num_episodes = 100
demo_data = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = expert_policy(state)
        next_state, reward, done, info = env.step(action)
        demo_data.append((state, action, reward, next_state, done))
        state = next_state

env.close()

# 保存示范数据到文件
import pickle
with open('cartpole_demo_data.pkl', 'wb') as f:
    pickle.dump(demo_data, f)

# 加载示范数据
with open('cartpole_demo_data.pkl', 'rb') as f:
    demo_data = pickle.load(f)

# 3. 自定义回调以加载示范数据
class PretrainWithDemoCallback(BaseCallback):
    def __init__(self, demo_data, verbose=0):
        super(PretrainWithDemoCallback, self).__init__(verbose)
        self.demo_data = demo_data

    def _on_training_start(self):
        for state, action, reward, next_state, done in self.demo_data:
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            self.model.rollout_buffer.add(state, next_state, action, reward, done, [0.99])
        
        self.model.rollout_buffer.compute_returns_and_advantage(self.model.policy, last_values=self.model.policy.predict_values(next_state))

callback = PretrainWithDemoCallback(demo_data)

# 4. 创建和训练PPO模型
env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000, callback=callback)

# 保存模型
model.save("ppo_cartpole")

# 5. 加载和评估模型
model = PPO.load("ppo_cartpole")

episodes = 10
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(state, deterministic=True)
        state, reward, done, info = env.step(action)
        total_reward += reward
    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

env.close()
```

### 说明
- 这个示例代码展示了如何生成 `CartPole-v1` 的示范数据，并利用这些数据进行PPO算法的预训练。
- 预训练的步骤包括将示范数据添加到 `rollout_buffer` 中，并计算回报和优势函数。
- 使用 `Stable-Baselines3` 库训练模型，并保存和评估模型的性能。

通过这种方式，你可以有效地利用示范数据来初始化PPO模型，并通过与环境的交互进一步训练模型。