
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset


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

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, num_users, num_movies, num_genres, embedding_dim):
        super(NeuralNetwork, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.genre_fc = nn.Linear(num_genres, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, user_id, movie_id, genres):
        user_embedded = self.user_embedding(user_id)
        movie_embedded = self.movie_embedding(movie_id)
        genres_encoded = self.genre_fc(genres)

        # Concatenate user and movie embeddings
        x = torch.cat([user_embedded, movie_embedded], dim=1)
        x = torch.cat([x, genres_encoded], dim=1)  # Concatenate genres with user and movie embeddings

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()