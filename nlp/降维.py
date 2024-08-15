import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 生成身高和体重数据
data = np.array([
    [170, 65],
    [165, 59],
    [180, 75],
    [175, 70],
    [160, 58],
    [168, 62],
    [172, 68],
    [178, 73],
    [166, 60],
    [182, 78]
])

# 数据可视化（原始二维数据）
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], color='blue')
plt.title('Original 2D Data (Height vs Weight)')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.grid(True)

# 执行PCA
pca = PCA(n_components=1)  # 降维到1维
data_pca = pca.fit_transform(data)

# 数据可视化（降维后的数据）
plt.subplot(1, 2, 2)
plt.scatter(data_pca, np.zeros_like(data_pca), color='red')
plt.title('1D Data after PCA')
plt.xlabel('Principal Component')
plt.grid(True)

plt.show()
