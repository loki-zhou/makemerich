import torch
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
column_sum = torch.sum(x, dim=0)  # 计算每一列的总和
row_sum = torch.sum(x, dim=1)    # 计算每一行的总和A
print(column_sum)  # 输出: tensor([5, 7, 9])
print(row_sum)     # 输出: tensor([ 6, 15])