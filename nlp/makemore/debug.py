# 等高线图
import numpy as np
import matplotlib.pyplot as plt

# 模拟海拔高度
def fz(x, y):
  z = (1 -x / 2 + x**5 + y**3) * np.exp(-x**2-y**2)
  return z

w = np.linspace(-4, 4, 100)
h = np.linspace(-2, 2, 100)

grid_x, grid_y = np.meshgrid(w, h)
z = fz(grid_x, grid_y)



plt.figure('Contour Chart',facecolor='lightgray')
plt.title('contour',fontsize=16)
plt.grid(linestyle=':')

cntr = plt.contour(
    grid_x, # 网格坐标矩阵的x坐标（2维数组）
    grid_y, # 网格坐标矩阵的y坐标（2维数组）
    z,      # 网格坐标矩阵的z坐标（2维数组）
    8,      # 等高线绘制8部分
    colors = 'black', # 等高线图颜色
    linewidths = 0.5 # 等高线图线宽
)
# 设置标签
plt.clabel(cntr, inline_spacing = 1, fmt='%.2f', fontsize=10)
# 填充颜色  大的是红色  小的是蓝色
plt.contourf(grid_x, grid_y, z, 8, cmap='jet')

plt.legend()
plt.show()
