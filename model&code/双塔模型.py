
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


moive_data=pd.read_csv('testdata/ml-latest-small/movies.csv')
user_data=pd.read_csv('testdata/ml-latest-small/ratings.csv')





# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits