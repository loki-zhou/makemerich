# 三元一次方程
import numpy as np

X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0, 0, 1, 1]]).T
syn0 = 2 * np.random.random((3, 1)) - 1
lr = 0.1
for j in range(1000):
    l0 = X
    l1 = np.dot(l0, syn0)  # 4*3 3*1 =  4*1
    l1_error = y - l1
    if j % 100 == 0:
        print(f"Iteration {j}, Error: {np.mean(l1_error**2):.10f}")
    print(l1_error)
    print(l0.T)
    print(syn0)
    syn0 += lr * np.dot(l0.T, l1_error)  #3*4 * 4*1  = 3*1

print(syn0)
print(f"Y = {syn0[0,0]: .5f} * X1 + {syn0[1,0]: .5f} * X2  + {syn0[2,0]: .5f} * X3 ")