# 三元一次方程
import numpy as np


def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值

    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad

def function_src(x):
    return


X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [1,1,0]])
y = np.array([[0, 0, 0, 1, 1]]).T
syn0 = 2 * np.random.random((3, 1)) - 1
lr = 0.001
for j in range(10000):
    l0 = X
    l1 = np.dot(l0, syn0)  # 4*3 3*1 =  4*1
    l1_error = y - l1
    if j % 100 == 0:
        print(f"Iteration {j}, Error: {np.mean(l1_error**2):.10f}")
    syn0 += lr * np.dot(l0.T, l1_error)  #3*4 * 4*1  = 3*1

print(syn0)
print(np.dot(X, syn0))
print(f"Y = {syn0[0,0]: .5f} * X1 + {syn0[1,0]: .5f} * X2  + {syn0[2,0]: .5f} * X3 ")