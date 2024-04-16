import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,VotingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime
from tqdm import tqdm,trange
# 读取数据集
print("读取数据集...")
data = pd.read_csv('testdata/wei_bo_data/weibo_train_data.csv')

print("处理缺失值...")
data = data.dropna(subset=['content'])

# 处理时间数据
print("处理时间数据...")
data['time'] = pd.to_datetime(data['time'])
data['hour'] = data['time'].dt.hour
data['weekday'] = data['time'].dt.weekday
# 计算文本长度
print("计算文本长度...")
data['text_length'] = data['content'].apply(len)

# 对类别特征进行编码
print("对类别特征进行编码...")
label_encoder = LabelEncoder()
data['uid_encoded'] = label_encoder.fit_transform(data['uid'])
data['mid_encoded'] = label_encoder.fit_transform(data['mid'])

# 特征选择
print("特征选择...")
X = data[['hour','weekday', 'uid_encoded', 'text_length']]
y = data[['forward_count', 'comment_count', 'like_count']]





print("训练随机森林回归模型...")
n_estimators = 5
rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
rf_model.fit(X, y['forward_count'])



# 训练AdaBoost回归模型
print("训练AdaBoost回归模型...")
ada_model = AdaBoostRegressor(n_estimators=3, random_state=42)
ada_model.fit(X, y['forward_count'])

voting_model = VotingRegressor([('rf', rf_model), ('ada', ada_model)])
voting_model.fit(X,y['forward_count'])





import pickle

'''
print("保存模型...")
with open(f'model_temp/voting_model.pkl', 'wb') as f:
    pickle.dump(voting_model, f)
'''
# 在整个数据集上进行预测
print("预测...")
y_pred_voting = voting_model.predict(X)
print(y_pred_voting)

# 计算加权平均并取整
y_pred = y_pred_voting.mean(axis=1).astype(int)


# 计算偏差
def calculate_bias(true_value, predicted_value):
    return abs(predicted_value - true_value) / (true_value + 5), abs(predicted_value - true_value) / (true_value + 3), abs(predicted_value - true_value) / (true_value + 3)

forward_bias, comment_bias, like_bias = calculate_bias(y['forward_count'], y_pred[:, 0]), calculate_bias(y['comment_count'], y_pred[:, 1]), calculate_bias(y['like_count'], y_pred[:, 2])

# 计算准确率
accuracy = 1 - 0.5 * forward_bias[0] - 0.25 * comment_bias[0] - 0.25 * like_bias[0]

# 计算整体的精度
total_interactions = y.sum(axis=1)
total_interactions[total_interactions > 100] = 100

precision_numerator = ((total_interactions + 1) * (accuracy > 0.8)).sum()
precision_denominator = total_interactions.sum() + len(y)
precision = precision_numerator / precision_denominator

print("整体的精度：", precision)
