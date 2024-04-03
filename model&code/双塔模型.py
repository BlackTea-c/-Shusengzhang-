
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from torch.nn import CosineSimilarity

moive_data=pd.read_csv('testdata/ml-latest-small/movies.csv')
user_data=pd.read_csv('testdata/ml-latest-small/ratings.csv')

'''
moive_data示例:
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy'''

'''
rating.csv示例
userId,movieId,rating,timestamp
1,1,4.0,964982703
1,3,4.0,964981247

这里首先需要把所有的用户id归并出来；timestamp不作为特征，抛弃就行；然后rating可以作为双塔召回的label；大于3我们认为是正样本(1)
小于3认为是负样本(-1)


所以目前我们的双塔：
用户部分：特征就只有userID;  通过embeding来处理
item部分：特征有 moiveID,title,genres; moiveID通过embeding来处理;title不需要;而genres因为一个电影可能有多种genre或者只有一种
genres之间是用 "|"来隔开的；用词袋来装

'''

user_ids = user_data['userId'].unique()
from sklearn.feature_extraction.text import CountVectorizer

moive_data['genres'] = moive_data['genres'].str.split('|')
moive_data['genres'] = moive_data['genres'].apply(lambda x: ' '.join(map(str, x)))



# 使用词袋模型进行编码
vectorizer = CountVectorizer()
genres_encoded = vectorizer.fit_transform(moive_data['genres']).toarray()
genres_dim=len(genres_encoded[0])

class MovieDataset(Dataset):
    def __init__(self, user_data, moive_data, vectorizer, positive_threshold=3):
        self.user_data = user_data
        self.moive_data = moive_data
        self.vectorizer = vectorizer
        self.positive_threshold = positive_threshold
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
        label = 1 if rating >= self.positive_threshold else 0

        return {
            'user_id': torch.tensor(int(user_id)),
            'movie_id': torch.tensor(int(movie_index)),
            'genres_encoded': torch.tensor(genres_encoded),
            'label': torch.tensor(label)
        }


# Create Dataset instance
movie_dataset = MovieDataset(user_data, moive_data, vectorizer)

# Create DataLoader instance
batch_size = 64
movie_dataloader = DataLoader(movie_dataset, batch_size=batch_size, shuffle=True)



class NeuralNetwork(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim):
        super(NeuralNetwork, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.fc_user = nn.Linear(embedding_dim, 64)
        self.fc_moive = nn.Linear(embedding_dim+genres_dim, 64)
        self.cos_sim = CosineSimilarity(dim=1)

    def forward(self, user_id, movie_id, genres):
        #id embeding

        user_embedded = self.user_embedding(user_id)
        movie_embedded = self.movie_embedding(movie_id)

        genres_encoded =genres

        moive_features = torch.cat([movie_embedded, genres_encoded], dim=1)

        x_moive = torch.relu(self.fc_moive(moive_features))
        y_moive = torch.relu(self.fc_user(user_embedded))

        similarity_score = self.cos_sim(x_moive, y_moive)

        return similarity_score
num_users=max(user_ids)+1
num_moives=len(moive_data['movieId'])
embedding_dim=300

model = NeuralNetwork(num_users,num_moives,embedding_dim)
import torch.optim as optim
# Define the loss function
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, similarity_score, label):
        positive_mask = label.bool()
        negative_mask = ~positive_mask

        positive_loss = (1 - similarity_score[positive_mask]).pow(2).mean()
        negative_loss = similarity_score[negative_mask].pow(2).mean()

        total_loss = positive_loss + negative_loss
        return total_loss
criterion = CustomLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, criterion, optimizer, data_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in data_loader:
            optimizer.zero_grad()

            user_id = batch['user_id']
            movie_id = batch['movie_id']
            genres = batch['genres_encoded']
            label = batch['label']

            similarity_score = model(user_id, movie_id, genres)

            loss = criterion(similarity_score, label.float())

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * user_id.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))

# Start training
train_model(model, criterion, optimizer, movie_dataloader)