

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from torch.nn import CosineSimilarity

moive_data=pd.read_csv('testdata/ml-latest-small/movies.csv')
user_data=pd.read_csv('testdata/ml-latest-small/ratings.csv')



user_ids = user_data['userId'].unique()
moive_ids = moive_data['movieId'].unique()


print(len(user_ids),len(moive_data))


class MovieDataset(Dataset):
    def __init__(self, user_data, moive_data, positive_threshold=3):
        self.user_data = user_data
        self.moive_data = moive_data
        self.positive_threshold = positive_threshold
        self.user_ids = user_data['userId'].unique()
        self.movie_ids = moive_data['movieId'].unique()

    def __len__(self):
        return len(self.user_data)

    def __getitem__(self, idx):
        user_id = self.user_data.iloc[idx]['userId']
        movie_id = self.user_data.iloc[idx]['movieId']
        rating = self.user_data.iloc[idx]['rating']

        movie_index = self.moive_data[self.moive_data['movieId'] == movie_id].index[0]

        #Based on rating~
        label = 1 if rating >= self.positive_threshold else 0

        return {
            'user_id': torch.tensor(int(user_id)),
            'movie_id': torch.tensor(int(movie_index)),
            'label': torch.tensor(label),
        }



movie_dataset = MovieDataset(user_data, moive_data)

