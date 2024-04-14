
import torch
import torch.nn as nn

embedding_layer = nn.Embedding(3, 3)
print(embedding_layer.weight)

# 输入为一个形状为(batch_size, sequence_length)的张量
input_indices = torch.tensor([[0],[1],[2]])

# 得到嵌入向量
embedded_output = embedding_layer(input_indices)
print(embedded_output)
print(embedded_output.shape)


# 访问嵌入层的参数列表
embedding_params = list(embedding_layer.parameters())

# 获取嵌入矩阵（唯一的参数）
embedding_matrix = embedding_params[0]