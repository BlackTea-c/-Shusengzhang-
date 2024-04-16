import pandas as pd
from datetime import datetime
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
# 加载训练好的模型
with open('model_temp/random_forest_model_5.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# 读取预测数据集
weibo_predict_data = pd.read_csv('testdata/wei_bo_data/weibo_predict_data.csv')

# 处理时间数据
weibo_predict_data['time'] = pd.to_datetime(weibo_predict_data['time'])
weibo_predict_data['hour'] = weibo_predict_data['time'].dt.hour
weibo_predict_data['weekday'] = weibo_predict_data['time'].dt.weekday

#类别特征编码
label_encoder = LabelEncoder()
weibo_predict_data['uid_encoded'] = label_encoder.fit_transform(weibo_predict_data['uid'])
weibo_predict_data['mid_encoded'] = label_encoder.fit_transform(weibo_predict_data['mid'])

# 计算文本长度
print("计算文本长度...")
weibo_predict_data['text_length'] = weibo_predict_data['content'].apply(lambda x: len(str(x)) if pd.notnull(x) else 1)

# 选择特征
X_predict = weibo_predict_data[['hour','weekday', 'uid_encoded','text_length']]

# 使用加载的模型进行预测
predictions = loaded_model.predict(X_predict)

# 将预测结果保存到 txt 文件
with open('predictions.txt', 'w') as f:
    for i in range(len(predictions)):
        line = f"{weibo_predict_data.loc[i, 'uid']}\t{weibo_predict_data.loc[i, 'mid']}\t{int(predictions[i][0])},{int(predictions[i][1])},{int(predictions[i][2])}\n"
        f.write(line)

print("预测结果已保存到 predictions.txt 文件中。")
