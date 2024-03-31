
import torch
import torch.nn as nn

# 假设你有一个类别特征，取值范围为 0 到 9
# 我们定义了一个嵌入层，输入大小为 10（类别的总数），输出大小为 3（嵌入向量的维度）
embedding = nn.Embedding(6, 3)

# 创建一个示例输入，其中包含了 5 个样本，每个样本的特征值都在 0 到 9 之间
input_data = torch.LongTensor([[1, 2, 3, 4, 5]])

# 将输入传递给嵌入层
embedded_data = embedding(input_data)

# 输出嵌入后的数据
print("Embedded Data:")
print(embedded_data)
