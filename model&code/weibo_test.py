import pandas as pd
#用户有相同



# 读取两个数据集
df1 = pd.read_csv('testdata/wei_bo_data/weibo_predict_data.csv')
df2 = pd.read_csv('testdata/wei_bo_data/weibo_train_data.csv')

# 提取 uid 列并转换为 set
uid_set1 = set(df1['time'])
uid_set2 = set(df2['time'])


# 获取两个集合的交集
common_uids = uid_set1.intersection(uid_set2)

# 获取交集元素数量
common_uids_count = len(common_uids)

print(f"df1的uid列与df2的uid列共有{common_uids_count}个相同的元素。")