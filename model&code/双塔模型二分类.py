
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from sklearn.model_selection import train_test_split

moive_data=pd.read_csv('testdata/ml-latest-small/movies.csv')
user_data=pd.read_csv('testdata/ml-latest-small/ratings.csv')


user_ids = user_data['userId'].unique()
moive_ids = moive_data['movieId'].unique()
from sklearn.feature_extraction.text import CountVectorizer

moive_data['genres'] = moive_data['genres'].str.split('|')
moive_data['genres'] = moive_data['genres'].apply(lambda x: ' '.join(map(str, x)))



# 使用词袋模型进行编码
vectorizer = CountVectorizer()
genres_encoded = vectorizer.fit_transform(moive_data['genres']).toarray()
genres_dim=len(genres_encoded[0])

class MovieDataset(Dataset):
    def __init__(self, user_data, moive_data, vectorizer):
        self.user_data = user_data
        self.moive_data = moive_data
        self.vectorizer = vectorizer
        self.user_ids = user_data['userId'].unique()
        self.movie_ids = moive_data['movieId'].unique()

    def __len__(self):
        return len(self.user_data)

    def __getitem__(self, idx):
        user_id = self.user_data.iloc[idx]['userId']
        movie_id = self.user_data.iloc[idx]['movieId']
        rating = self.user_data.iloc[idx]['rating']

        # Encode genres using CountVectorizer
        movie_index = self.moive_data[self.moive_data['movieId'] == movie_id].index[0]
        genres_encoded = self.vectorizer.transform([self.moive_data.loc[movie_index, 'genres']]).toarray()[0]

        #Based on rating~
        if rating <= 3:
            label = -1
        else:
            label = 1

        return {
            'user_id': torch.tensor(int(user_id)),
            'movie_id': torch.tensor(int(movie_index)),
            'genres_encoded': torch.tensor(genres_encoded),
            'label': torch.tensor(label)
        }


movie_dataset = MovieDataset(user_data, moive_data, vectorizer)
batch_size = 16

train_data, val_data = train_test_split(movie_dataset, test_size=0.3, random_state=42)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)


class NeuralNetwork(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim):
        super(NeuralNetwork, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.fc_user = nn.Linear(embedding_dim, 456)
        self.fc_moive = nn.Linear(embedding_dim+genres_dim, 456)

        self.fc_user1 = nn.Linear(456, 216)
        self.fc_moive1 = nn.Linear(456, 216)


        self.dropout = nn.Dropout(0.5)

        self.cos_sim = CosineSimilarity(dim=1)

    def forward(self, user_id, movie_id, genres):


        user_embedded = self.user_embedding(user_id)


        movie_embedded = self.movie_embedding(movie_id)

        genres_encoded =genres

        moive_features = torch.cat([movie_embedded, genres_encoded], dim=1)

        x_moive = self.fc_moive(moive_features)
        x_moive = self.fc_moive1(x_moive)
        x_moive = self.dropout(x_moive)


        y_moive = self.fc_user(user_embedded)
        y_moive = self.fc_user1(y_moive)
        y_moive = self.dropout(y_moive)

        x_movie_normalized = F.normalize(x_moive, p=2, dim=1)
        y_user_normalized = F.normalize(y_moive, p=2, dim=1)

        similarity_score = self.cos_sim(x_movie_normalized, y_user_normalized)


        return similarity_score
num_users=max(user_ids)+1
num_moives=max(moive_ids)+1
embedding_dim=10

model = NeuralNetwork(num_users,num_moives,embedding_dim)
import torch.optim as optim



criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

def train_model(model, train_loader, optimizer, criterion, num_epochs=80):
    best_acc=0.0
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        acc = evaluate_model(model, test_loader)
        for batch in train_loader:
            user_ids = batch['user_id']
            movie_ids = batch['movie_id']
            labels = batch['label']
            genres = batch['genres_encoded']

            optimizer.zero_grad()

            outputs = model(user_ids, movie_ids,genres)

            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")

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
            genres = batch['genres_encoded']

            outputs = model(user_ids, movie_ids,genres)

            predicted = torch.where(outputs > 0, torch.tensor(1), torch.tensor(-1))

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

best_model_state = train_model(model, train_loader, optimizer, criterion)

model.load_state_dict(best_model_state)

evaluate_model(model, test_loader)