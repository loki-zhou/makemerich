# pip install flappy-bird-gymnasium
import pygame
import flappy_bird_gymnasium
import gymnasium

pygame.init()
clock = pygame.time.Clock()
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

observation, _ = env.reset()

action = 0  # 初始化动作（0 是向左，1 是向右）

done = False
while not done:
    action = 0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1
                pygame.event.clear()
                break
        elif event.type == pygame.KEYUP:
            # 当按键释放时停止动作
            action = 0
        else:
            action = 0

        print(" event.type = ", event.type )


    print("action = ", action)
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

env.close()