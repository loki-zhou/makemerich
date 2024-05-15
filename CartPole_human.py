import gymnasium as gym
import pygame
import numpy as np

# 初始化 CartPole-v1 环境
env = gym.make('CartPole-v1',render_mode="human")


# 初始化 Pygame
pygame.init()
win = pygame.display.set_mode((400, 300))
pygame.display.set_caption("CartPole Manual Control")

clock = pygame.time.Clock()

# 定义动作
action = 0  # 初始化动作（0 是向左，1 是向右）

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
                action = 0
            elif event.key == pygame.K_RIGHT:
                action = 1

    # 在环境中执行动作
    observation, reward, terminated, truncated, info = env.step(action)
    # 渲染环境
    env.render()

    # 设置每秒帧数
    clock.tick(30)
    if terminated or truncated:
        observation, info = env.reset()

# 退出 Pygame
pygame.quit()

# 关闭环境
env.close()
