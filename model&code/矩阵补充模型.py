

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from torch.nn import CosineSimilarity
import torch.optim as optim


moive_data=pd.read_csv('testdata/ml-latest-small/movies.csv')
user_data=pd.read_csv('testdata/ml-latest-small/ratings.csv')



user_ids = user_data['userId'].unique()
moive_ids = moive_data['movieId'].unique()


print(len(user_ids),len(moive_data))


class MovieDataset(Dataset):
    def __init__(self, user_data, moive_data):
        self.user_data = user_data
        self.moive_data = moive_data
        self.user_ids = user_data['userId'].unique()
        self.movie_ids = moive_data['movieId'].unique()

    def __len__(self):
        return len(self.user_data)

    def __getitem__(self, idx):
        user_id = self.user_data.iloc[idx]['userId']
        movie_id = self.user_data.iloc[idx]['movieId']
        rating = self.user_data.iloc[idx]['rating']

        movie_index = self.moive_data[self.moive_data['movieId'] == movie_id].index[0]

        if rating <= 1:
            label = -1
        elif rating <= 2:
            label = -0.5
        elif rating <= 3:
            label = 0
        elif rating <= 4:
            label = 0.5
        else:
            label = 1

        return {
            'user_id': torch.tensor(int(user_id)),
            'movie_id': torch.tensor(int(movie_index)),
            'label': torch.tensor(label),
        }



movie_dataset = MovieDataset(user_data, moive_data)

class MatrixCompletion(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim): #这里num_users,num_moives指的是最大的索引
        super(MatrixCompletion, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        self.cos_sim = CosineSimilarity(dim=1)

    def forward(self, user_id, movie_id):
        user_embedded = self.user_embedding(user_id)
        movie_embedded = self.movie_embedding(movie_id)


        similarity_score = self.cos_sim(user_embedded, movie_embedded)

        return similarity_score


num_users=max(user_ids)+1
num_moives=max(moive_ids)+1
model = MatrixCompletion(num_users,num_moives,4)



from sklearn.model_selection import train_test_split

# 划分数据集
train_data, val_data = train_test_split(movie_dataset, test_size=0.3, random_state=42)

# 定义数据加载器
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(val_data, batch_size=8, shuffle=False)



optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

def train_model(model, train_loader, optimizer, criterion, num_epochs=10):
    best_acc=0.0
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            user_ids = batch['user_id']
            movie_ids = batch['movie_id']
            labels = batch['label']

            optimizer.zero_grad()

            outputs = model(user_ids, movie_ids)

            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")
        acc=evaluate_model(model, test_loader)
        print(acc)

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict()

    return best_model_state
def evaluate_model(model, test_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            user_ids = batch['user_id']
            movie_ids = batch['movie_id']
            labels = batch['label']

            outputs = model(user_ids, movie_ids)

            predicted = torch.round(outputs)  # 将模型输出四舍五入到最接近的离散值
            print(predicted,labels)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy
# 训练模型并获取最佳模型状态
best_model_state = train_model(model, train_loader, optimizer, criterion)

# 使用最佳模型状态来加载模型
model.load_state_dict(best_model_state)

evaluate_model(model, test_loader)







