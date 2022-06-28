import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn


def _build_obs_embed_model(obs_shape, embed_size, layers, hidden_size, activation):
    model = [nn.Linear(obs_shape, hidden_size)]
    model += [activation()]
    for i in range(layers - 1):
        model += [nn.Linear(hidden_size, hidden_size)]
        model += [activation()]
    model += [nn.Linear(hidden_size, embed_size)]
    return nn.Sequential(*model)


class ObservationEncoder(nn.Module):
    def __init__(self, obs_shape, embed_size, layers, hidden_size, activation=nn.ELU):
        super().__init__()
        self._embed_model = _build_obs_embed_model(obs_shape[0], embed_size, layers, hidden_size, activation)

    def forward(self, obs):
        return self._embed_model(obs)