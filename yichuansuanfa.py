import numpy as np
import random
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box

# 创建示例多边形 (Shapely Polygon)
def create_polygon():
    # 示例多边形，可以替换为任何其他形状
    return Polygon([(1, 1), (5, 1), (6, 3), (4, 6), (1, 4)])

# 适应度函数：计算矩形的面积，如果矩形在多边形内则返回面积，否则返回0
def fitness(rect, polygon):
    rect_polygon = box(rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])
    if polygon.contains(rect_polygon):
        return rect[2] * rect[3]  # 面积 = 宽度 * 高度
    return 0  # 如果矩形不在多边形内，返回0

# 初始化种群：生成N个随机矩形
def initialize_population(polygon, population_size, min_size, max_size):
    population = []
    min_x, min_y, max_x, max_y = polygon.bounds  # 多边形的边界
    for _ in range(population_size):
        w = random.uniform(min_size, max_size)  # 随机宽度
        h = random.uniform(min_size, max_size)  # 随机高度
        x = random.uniform(min_x, max_x - w)    # 随机 x 坐标
        y = random.uniform(min_y, max_y - h)    # 随机 y 坐标
        population.append((x, y, w, h))
    return population

# 选择：基于适应度选择个体（轮盘赌选择法）
def selection(population, fitnesses, num_parents):
    total_fitness = sum(fitnesses)
    selected = []
    for _ in range(num_parents):
        pick = random.uniform(0, total_fitness)
        current = 0
        for i in range(len(population)):
            current += fitnesses[i]
            if current > pick:
                selected.append(population[i])
                break
    return selected

# 交叉：基于两个父代个体生成子代
def crossover(parent1, parent2):
    # 简单的二进制交叉法
    x = (parent1[0] + parent2[0]) / 2
    y = (parent1[1] + parent2[1]) / 2
    w = (parent1[2] + parent2[2]) / 2
    h = (parent1[3] + parent2[3]) / 2
    return (x, y, w, h)

# 变异：随机调整个体的某些属性
def mutate(rect, mutation_rate, polygon):
    if random.uniform(0, 1) < mutation_rate:
        min_x, min_y, max_x, max_y = polygon.bounds
        x, y, w, h = rect
        x += random.uniform(-0.1, 0.1)  # 随机调整位置和大小
        y += random.uniform(-0.1, 0.1)
        w += random.uniform(-0.1, 0.1)
        h += random.uniform(-0.1, 0.1)
        # 保证矩形仍在多边形边界内
        x = np.clip(x, min_x, max_x - w)
        y = np.clip(y, min_y, max_y - h)
        w = np.clip(w, 0, max_x - x)
        h = np.clip(h, 0, max_y - y)
        return (x, y, w, h)
    return rect

# 遗传算法主流程
def genetic_algorithm(polygon, population_size=100, generations=100, mutation_rate=0.01, num_parents=10, min_size=0.1, max_size=2):
    population = initialize_population(polygon, population_size, min_size, max_size)

    for generation in range(generations):
        # 计算每个个体的适应度
        fitnesses = [fitness(rect, polygon) for rect in population]

        # 输出当前最优个体的信息
        best_fitness = max(fitnesses)
        best_individual = population[np.argmax(fitnesses)]
        print(f"Generation {generation}: Best Fitness = {best_fitness}, Best Individual = {best_individual}")

        # 选择适应度最高的个体作为父代
        parents = selection(population, fitnesses, num_parents)

        # 生成下一代
        next_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = random.sample(parents, 2)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent1, parent2)
            next_population.extend([mutate(child1, mutation_rate, polygon), mutate(child2, mutation_rate, polygon)])

        population = next_population

    # 返回最终的最优个体
    final_fitnesses = [fitness(rect, polygon) for rect in population]
    best_fitness = max(final_fitnesses)
    best_individual = population[np.argmax(final_fitnesses)]
    return best_individual

# 可视化结果
def visualize(polygon, rect):
    x, y, w, h = rect
    fig, ax = plt.subplots()
    x_poly, y_poly = polygon.exterior.xy
    ax.plot(x_poly, y_poly, color="blue", linewidth=2, label="Polygon")
    rect_patch = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none', label="Best Rectangle")
    ax.add_patch(rect_patch)
    plt.xlim([0, 7])
    plt.ylim([0, 7])
    ax.set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()

# 测试遗传算法
if __name__ == "__main__":
    polygon = create_polygon()
    best_rect = genetic_algorithm(polygon, population_size=100, generations=1000, mutation_rate=0.05)
    print(f"Best Rectangle: {best_rect}")
    visualize(polygon, best_rect)
