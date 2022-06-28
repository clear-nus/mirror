import numpy as np
import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F

from models.distribution import TanhBijector, SampleDist


def _build_action_model(feature_size, action_size, layers, hidden_size, dist,
                           activation):
    model = [nn.Linear(feature_size, hidden_size)]
    model += [activation()]
    for i in range(layers - 1):
        model += [nn.Linear(hidden_size, hidden_size)]
        model += [activation()]
    if dist == 'tanh_normal':
        model += [nn.Linear(hidden_size, action_size * 2)]
    elif dist == 'one_hot' or dist == 'relaxed_one_hot' or dist == 'independent_binary':
        model += [nn.Linear(hidden_size, action_size)]
    return nn.Sequential(*model)

class BinaryReparameterization(nn.Module):
    def __init__(self, logit, temperature=0.1):
        super(BinaryReparameterization, self).__init__()
        self.p = torch.sigmoid(logit)
        self.temperature = temperature

    def rsample(self, temperature=None):
        # dist = td.independent.Independent(td.Bernoulli(probs=self.p), 1)
        epsilon = torch.rand(size=self.p.size()).to(self.p.device)
        z = torch.log(epsilon + 1e-4) - torch.log(-epsilon + 1.0 + 1e-4) \
            + torch.log(torch.sigmoid(self.p) + 1e-4) - torch.log(-torch.sigmoid(self.p) + 1.0 + 1e-4)

        if temperature is None:
            z = torch.sigmoid(z / self.temperature)
        else:
            z = torch.sigmoid(z / temperature)
        # print(temperature)
        return z

    def log_prob(self, x):
        return (x * (self.p + 1e-5).log() + (1.0 - x) * (1.0 - self.p + 1e-5).log()).mean(dim=-1)

class ActionDecoder(nn.Module):
    def __init__(self, feature_size: int, output_shape: tuple, layers: int, hidden_size: int, dist='tanh_normal',
                 activation=nn.ELU):
        super().__init__()
        self._action_model = _build_action_model(feature_size=feature_size,
                                                 action_size=output_shape,
                                                 layers=layers,
                                                 hidden_size=hidden_size,
                                                 dist=dist,
                                                 activation=activation)
        self._dist = dist
        self.log_std_min = -10
        self.log_std_max = 2

    def forward(self, features):
        dist_inputs = self._action_model(features)
        dist = None
        if self._dist == 'one_hot':
            dist = torch.distributions.OneHotCategorical(logits=dist_inputs)
        elif self._dist == 'tanh_normal':
            mean, log_std = torch.chunk(dist_inputs, 2, dim=-1)
            # constrain log_std inside [log_std_min, log_std_max]
            log_std = torch.tanh(log_std)
            log_std = self.log_std_min + 0.5 * (
                    self.log_std_max - self.log_std_min
            ) * (log_std + 1)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            # print(mean[0,0])
            # print(log_std[0,0], std[0,0])
            # print("=====================================")
        elif self._dist == 'independent_binary':
            dist = torch.distributions.Bernoulli(logits=dist_inputs)
            # dist = torch.distributions.independent.Independent(dist, 1)
            # dist = BinaryReparameterization(logit=dist_inputs)

        return dist

    @property
    def action_model(self):
        return self._action_model


class ActionDecoderOld(nn.Module):
    def __init__(self, feature_size, action_size, hidden_size, layers, dist='tanh_normal',
                 activation=nn.ELU, min_std=1e-4, init_std=5, mean_scale=5):
        super().__init__()
        self.action_size = action_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.dist = dist
        self.activation = activation
        self.min_std = min_std
        self.init_std = init_std
        self.log_std_min = -10
        self.log_std_max = 2
        self.mean_scale = mean_scale
        self.feedforward_model = self.build_model()
        self.raw_init_std = np.log(np.exp(self.init_std) - 1)

    def build_model(self):
        model = [nn.Linear(self.feature_size, self.hidden_size)]
        model += [self.activation()]
        for i in range(1, self.layers):
            model += [nn.Linear(self.hidden_size, self.hidden_size)]
            model += [self.activation()]
        if self.dist == 'tanh_normal':
            model += [nn.Linear(self.hidden_size, self.action_size * 2)]
        elif self.dist == 'one_hot' or self.dist == 'relaxed_one_hot':
            model += [nn.Linear(self.hidden_size, self.action_size)]
        else:
            raise NotImplementedError(f'{self.dist} not implemented')
        return nn.Sequential(*model)

    def forward(self, state_features):
        x = self.feedforward_model(state_features)
        dist = None
        if self.dist == 'tanh_normal':
            mean, log_std = torch.chunk(x, 2, -1)
            log_std = torch.tanh(log_std)
            log_std = self.log_std_min + 0.5 * (
                    self.log_std_max - self.log_std_min
            ) * (log_std + 1)
            std = log_std.exp()
            # mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
            # std = F.softplus(pre_std + self.raw_init_std) + self.min_std
            # std = torch.clamp(std, self.min_std, 5.0)
            dist = torch.distributions.Normal(mean, std)
            # print(mean.size(), std.size(), state_features.size())
            if state_features.dim() == 3:
                print(x[0,0])
            #     print(state_features[0,1], x[0,1], mean[0,1], std[0,1])
                print(x[1,0])
                print(x[-2,0], mean[-2,0], std[-2,0])
            #     print(state_features.size())
            #     print("=================")
            # else:
            #     print(mean, std)
            dist = torch.distributions.TransformedDistribution(dist, TanhBijector())
            dist = torch.distributions.Independent(dist, 1)
            dist = SampleDist(dist)
        elif self.dist == 'one_hot':
            if state_features.dim() == 3:
                print(x[0, 0])
                #     print(state_features[0,1], x[0,1], mean[0,1], std[0,1])
                print(x[1, 0])
                print(x[-2, 0])
                print("==================")
            dist = torch.distributions.OneHotCategorical(logits=x)
        elif self.dist == 'relaxed_one_hot':
            dist = torch.distributions.RelaxedOneHotCategorical(0.1, logits=x)
        return dist