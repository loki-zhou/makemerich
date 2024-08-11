import numpy as np

# 定义输入数据和目标输出
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
y = np.array([[0, 1, 1, 0]]).T

# 初始化权重矩阵
syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

# 训练神经网络
for j in range(60000):
    # 前向传播
    l1 = 1 / (1 + np.exp(-np.dot(X, syn0)))
    l2 = 1 / (1 + np.exp(-np.dot(l1, syn1)))

    # 计算误差
    l2_error = y - l2
    l2_delta = l2_error * (l2 * (1 - l2))

    # 反向传播
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * (l1 * (1 - l1))

    # 更新权重
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)

# 训练完成后，可以添加代码来测试神经网络的输出

print(syn0)
print(syn1)