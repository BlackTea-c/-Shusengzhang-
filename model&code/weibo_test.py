import pandas as pd

# 读取原始数据
weibo_extra_train_data = pd.read_csv('testdata/wei_bo_data/weibo_extra_train_data.csv')

# 新增三列，并将值全部设置为 0
weibo_extra_train_data['forward_count'] = 0
weibo_extra_train_data['comment_count'] = 0
weibo_extra_train_data['like_count'] = 0

# 保存新数据到 CSV 文件
weibo_extra_train_data.to_csv('weibo_extra_train_data_with_counts.csv', index=False)

print("新的数据已保存到 weibo_extra_train_data_with_counts.csv 文件中。")
