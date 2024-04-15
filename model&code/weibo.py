import pandas as pd

#列名：uid,mid,time,forward_count,comment_count,like_count,content
pd.set_option('display.float_format', '{:.2f}'.format)

data = pd.read_csv('testdata/wei_bo_data/weibo_train_data.csv')

# 设定content长度上限
n = 2

# 过滤content长度不超过n的数据
filtered_data = data[data['content'].apply(lambda x: len(str(x)) if pd.notnull(x) else 0) <= n]


# 逐个统计
for index, row in filtered_data.iterrows():
    content = row['content']
    forward_count = row['forward_count']
    comment_count = row['comment_count']
    like_count = row['like_count']

    if forward_count>2 or comment_count+like_count>2:
      print(f"Content: {content}")
      print(f"Forward Count: {forward_count}")
      print(f"Comment Count: {comment_count}")
      print(f"Like Count: {like_count}")
      print("--------------")

nan_content_rows = data[data['content'].isnull()]
