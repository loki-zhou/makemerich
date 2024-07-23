import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 生成两组数据
np.random.seed(0)
group_A = np.random.normal(50, 10, 100)  # A组数据，均值50，标准差10，样本数100
group_B = np.random.normal(55, 10, 100)  # B组数据，均值55，标准差10，样本数100

# 计算T检验的P值
t_stat, p_value = stats.ttest_ind(group_A, group_B)

# 打印P值
print(f"P值: {p_value}")

# 作图
plt.figure(figsize=(10, 6))
plt.hist(group_A, bins=20, alpha=0.5, label='A组')
plt.hist(group_B, bins=20, alpha=0.5, label='B组')
plt.axvline(np.mean(group_A), color='blue', linestyle='dashed', linewidth=2)
plt.axvline(np.mean(group_B), color='orange', linestyle='dashed', linewidth=2)
plt.legend()
plt.title('A组和B组数据分布')
plt.xlabel('值')
plt.ylabel('频率')
plt.show()

# 显示P值
if p_value < 0.05:
    print("结果显著，A组和B组的平均值有显著差异。")
else:
    print("结果不显著，A组和B组的平均值没有显著差异。")

