import torch
import torch.nn as nn


class DenseModel(nn.Module):
    def __init__(self, action_shape):
        super(DenseModel, self).__init__()

        self.in_to_lstm = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_shape[0]),
            nn.Tanh()
            )

    def forward(self, obs):
        h = self.in_to_lstm(obs)
        return h