import numpy as np

# sigmoid 函数
def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# 输入数据集
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# 输出数据集
y = np.array([[0, 0, 1, 1]]).T

# 设定随机数种子以确保计算的确定性
np.random.seed(1)

# 随机初始化权重，均值为0
syn0 = 2 * np.random.random((3, 1)) - 1

for iter in range(10000):
    # 前向传播
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    # 计算误差
    l1_error = y - l1

    # 乘以 l1 处的 sigmoid 函数的斜率
    l1_delta = l1_error * nonlin(l1, True)

    # 更新权重
    syn0 += np.dot(l0.T, l1_delta)
    #syn0 -= np.dot(l0.T, l1_delta)

print("训练后的输出:")
print(l1)
print(syn0)
# y = x1*9.6 - 0.2*x2 + -0.4*x3
#  y = ax1 + bx2 + cx3
#
#  y = a*x1 + b*x2 + c* x3

print(np.dot(X[1,:],syn0))
print(nonlin(np.dot(X[1,:],syn0)))


