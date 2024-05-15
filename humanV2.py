import gymnasium as gym
import pygame
import numpy as np

# 初始化 LunarLander-v2 环境
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)

# 初始化 Pygame
pygame.init()
win = pygame.display.set_mode((600, 400))
pygame.display.set_caption("LunarLander Manual Control")

clock = pygame.time.Clock()

# 定义动作
# 0: 不动作, 1: 向左, 2: 向上, 3: 向右
action = 0

# 重置环境
state = env.reset()

done = False

# 手动控制循环
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                action = 1
            elif event.key == pygame.K_UP:
                action = 2
            elif event.key == pygame.K_RIGHT:
                action = 3
        elif event.type == pygame.KEYUP:
            # 当按键释放时停止动作
            action = 0

    # 在环境中执行动作
    observation, reward, terminated, truncated, info = env.step(action)

    # 渲染环境
    env.render()

    if terminated or truncated:
        observation, info = env.reset()
    # 设置每秒帧数
    clock.tick(30)

# 退出 Pygame
pygame.quit()

# 关闭环境
env.close()
