from Utils import *
import torch.nn as nn
import torch

TRAIN_IMG_DIR = "\\data\\train_input\\resnet_features\\"


class Model(nn.Module):
    def __init__(self, n=6, N=12, R=2):
        super().__init__()
        self.R = R
        self.conv1d = nn.Conv1d(N, N, 100)

        self.head = nn.Sequential(
            Flatten(),
            nn.Linear(R * 2, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Dropout(0.5),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Dropout(0.5),
            nn.Linear(100, n),
        )

    def forward(self, x):
        shape = x.shape
        n = shape[1]
        x = x.view(x.shape[0], x.shape[1], -1)  # => x: bs x N x C
        x = self.conv1d(x)  # => x: bs x N
        x, _ = torch.sort(x, dim=1, descending=True)
        x = torch.cat((x[:, : (self.R)], x[:, (n - self.R) :]), 1)  # keep min and max
        x = self.head(x)  # => x: bs x 6
        return x
