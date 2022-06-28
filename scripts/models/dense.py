import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn


# class DenseModelNormal(nn.Module):
#     def __init__(self, feature_size: int, output_shape: tuple, layers: int, hidden_size: int,
#                  activation=nn.ELU):
#         super().__init__()
#         self._output_shape = output_shape
#         self._layers = layers
#         self._hidden_size = hidden_size
#         self.activation = activation
#         # For adjusting pytorch to tensorflow
#         self._feature_size = feature_size
#         self.std_model = self.build_std_model()
#         self.soft_plus = nn.Softplus()
#         # Defining the structure of the NN
#         self.model = self.build_model()
#
#     def build_std_model(self):
#         model = [nn.Linear(self._feature_size, self._hidden_size)]
#         model += [self.activation()]
#         for i in range(self._layers - 1):
#             model += [nn.Linear(self._hidden_size, self._hidden_size)]
#             model += [self.activation()]
#         model += [nn.Linear(self._hidden_size, int(np.prod(self._output_shape)))]
#         model += [nn.Softplus()]
#         return nn.Sequential(*model)
#
#     def build_model(self):
#         model = [nn.Linear(self._feature_size, self._hidden_size)]
#         model += [self.activation()]
#         for i in range(self._layers - 1):
#             model += [nn.Linear(self._hidden_size, self._hidden_size)]
#             model += [self.activation()]
#         model += [nn.Linear(self._hidden_size, int(np.prod(self._output_shape)))]
#         return nn.Sequential(*model)
#
#     def forward(self, features):
#         dist_inputs = self.model(features)
#         reshaped_inputs = torch.reshape(dist_inputs, features.shape[:-1] + self._output_shape)
#
#         dist_std = self.std_model(features)
#         reshaped_std = torch.reshape(dist_std, features.shape[:-1] + self._output_shape)
#         reshaped_std = torch.clamp(reshaped_std, min=1e-2, max=10)
#         return td.independent.Independent(td.Normal(reshaped_inputs, reshaped_std), len(self._output_shape))


class DenseModelDist(nn.Module):
    def __init__(self, feature_size: int, output_shape: tuple, layers: int, hidden_size: int, dist='normal',
                 activation=nn.ELU):
        super().__init__()
        self._output_shape = output_shape
        self._layers = layers
        self._hidden_size = hidden_size
        self._dist = dist
        self.activation = activation
        # For adjusting pytorch to tensorflow
        self._feature_size = feature_size
        self.std_model = self.build_std_model()
        self.soft_plus = nn.Softplus()
        # Defining the structure of the NN
        self.model = self.build_model()

    def build_std_model(self):
        model = [nn.Linear(self._feature_size, self._hidden_size)]
        model += [self.activation()]
        for i in range(self._layers - 1):
            model += [nn.Linear(self._hidden_size, self._hidden_size)]
            model += [self.activation()]
        model += [nn.Linear(self._hidden_size, int(np.prod(self._output_shape)))]
        model += [nn.Softplus()]
        return nn.Sequential(*model)

    def build_model(self):
        model = [nn.Linear(self._feature_size, self._hidden_size)]
        model += [self.activation()]
        for i in range(self._layers - 1):
            model += [nn.Linear(self._hidden_size, self._hidden_size)]
            model += [self.activation()]
        model += [nn.Linear(self._hidden_size, int(np.prod(self._output_shape)))]
        return nn.Sequential(*model)

    def forward(self, features):
        dist_inputs = self.model(features)
        reshaped_inputs = torch.reshape(dist_inputs, features.shape[:-1] + self._output_shape)

        dist_std = self.std_model(features)
        reshaped_std = torch.reshape(dist_std, features.shape[:-1] + self._output_shape)
        reshaped_std = torch.clamp(reshaped_std, min=1e-2, max=10)
        if self._dist == 'normal':
            return td.independent.Independent(td.Normal(reshaped_inputs, reshaped_std), len(self._output_shape))
        if self._dist == 'binary':
            return td.independent.Independent(td.Bernoulli(logits=reshaped_inputs), len(self._output_shape))
        raise NotImplementedError(self._dist)


class DenseModel(nn.Module):
    def __init__(self, feature_size: int, output_shape: tuple, layers: int, hidden_size: int, dist='normal',
                 activation=nn.ELU):
        super().__init__()
        self._output_shape = output_shape
        self._layers = layers
        self._hidden_size = hidden_size
        self._dist = dist
        self.activation = activation
        # For adjusting pytorch to tensorflow
        self._feature_size = feature_size
        # Defining the structure of the NN
        self.model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self._feature_size, self._hidden_size)]
        model += [self.activation()]
        for i in range(self._layers - 1):
            model += [nn.Linear(self._hidden_size, self._hidden_size)]
            model += [self.activation()]
        model += [nn.Linear(self._hidden_size, int(np.prod(self._output_shape)))]
        return nn.Sequential(*model)

    def forward(self, features):
        dist_inputs = self.model(features)
        reshaped_inputs = torch.reshape(dist_inputs, features.shape[:-1] + self._output_shape)
        if self._dist == 'normal':
            return td.independent.Independent(td.Normal(reshaped_inputs, 0.05), len(self._output_shape))
        if self._dist == 'binary':
            return td.independent.Independent(td.Bernoulli(logits=reshaped_inputs), len(self._output_shape))
        raise NotImplementedError(self._dist)


class CatDenseModel(nn.Module):
    def __init__(self, feature_size: int, output_shape: tuple, layers: int, hidden_size: int, dist='normal',
                 activation=nn.ELU):
        super().__init__()
        self._output_shape = output_shape
        self._layers = layers
        self._hidden_size = hidden_size
        self._dist = dist
        self.activation = activation
        # For adjusting pytorch to tensorflow
        self._feature_size = feature_size
        # Defining the structure of the NN
        self.model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self._feature_size, self._hidden_size)]
        model += [self.activation()]
        for i in range(self._layers - 1):
            model += [nn.Linear(self._hidden_size, self._hidden_size)]
            model += [self.activation()]
        model += [nn.Linear(self._hidden_size, int(np.prod(self._output_shape)))]
        return nn.Sequential(*model)

    def forward(self, feature_a, feature_b):
        features = torch.cat([feature_a, feature_b], dim=-1)
        dist_inputs = self.model(features)
        reshaped_inputs = torch.reshape(dist_inputs, features.shape[:-1] + self._output_shape)
        if self._dist == 'normal':
            return td.independent.Independent(td.Normal(reshaped_inputs, 0.05), len(self._output_shape))
        if self._dist == 'binary':
            return td.independent.Independent(td.Bernoulli(logits=reshaped_inputs), len(self._output_shape))
        raise NotImplementedError(self._dist)


class DenseModelNormal(nn.Module):
    def __init__(self, feature_size: int, output_shape: tuple, layers: int, hidden_size: int, activation=nn.ELU, min=1e-4, max=10.0):
        super().__init__()
        self._output_shape = output_shape
        self._layers = layers
        self._hidden_size = hidden_size
        self.activation = activation
        # For adjusting pytorch to tensorflow
        self._feature_size = feature_size
        # Defining the structure of the NN
        self.model = self.build_model()
        self.soft_plus = nn.Softplus()

        self._min = min
        self._max = max

    def build_model(self):
        model = [nn.Linear(self._feature_size, self._hidden_size)]
        model += [self.activation()]
        for i in range(self._layers - 1):
            model += [nn.Linear(self._hidden_size, self._hidden_size)]
            model += [self.activation()]
        model += [nn.Linear(self._hidden_size, 2 * int(np.prod(self._output_shape)))]
        return nn.Sequential(*model)

    def forward(self, features):
        if len(features.size()) > 1:
            dist_inputs = self.model(features)
            reshaped_inputs_mean = torch.reshape(dist_inputs[..., :np.prod(self._output_shape)],
                                                 features.shape[:-1] + self._output_shape)
            reshaped_inputs_std = torch.reshape(dist_inputs[..., np.prod(self._output_shape):],
                                                features.shape[:-1] + self._output_shape)

            reshaped_inputs_std = torch.clamp(self.soft_plus(reshaped_inputs_std), min=self._min, max=self._max)
            return td.independent.Independent(td.Normal(reshaped_inputs_mean, reshaped_inputs_std), len(self._output_shape))
        else:
            dist_inputs = self.model(features.unsqueeze(0))
            reshaped_inputs_mean = torch.reshape(dist_inputs[..., :np.prod(self._output_shape)],
                                                 features.shape[:-1] + self._output_shape)
            reshaped_inputs_std = torch.reshape(dist_inputs[..., np.prod(self._output_shape):],
                                                features.shape[:-1] + self._output_shape)

            reshaped_inputs_std = torch.clamp(self.soft_plus(reshaped_inputs_std), min=self._min, max=self._max)
            return td.independent.Independent(td.Normal(reshaped_inputs_mean.squeeze(0), reshaped_inputs_std.squeeze(0)), len(self._output_shape))


class ModelNormalStd(nn.Module):
    def __init__(self, feature_size: int = 4, hidden_size: int = 4, layers: int = 2, activation=nn.ELU):
        super(ModelNormalStd, self).__init__()
        # self.std = nn.Parameter(torch.ones(size=(feature_size,)))
        # self.std.requires_grad = True

        self._feature_size = feature_size
        self._hidden_size = hidden_size
        self._layers = layers
        self.activation = activation

        self.std_model = self.build_model()
        self.soft_plus = nn.Softplus()

    def build_model(self):
        model = [nn.Linear(self._feature_size, self._hidden_size)]
        model += [self.activation()]
        for i in range(self._layers - 1):
            model += [nn.Linear(self._hidden_size, self._hidden_size)]
            model += [self.activation()]
        model += [nn.Linear(self._hidden_size, self._feature_size)]
        return nn.Sequential(*model)

    def forward(self, features):
        std = self.std_model(features)
        std = self.soft_plus(std)
        std = torch.clamp(std, min=1e-4, max=10)
        return td.independent.Independent(td.Normal(features, std), len((self._feature_size,)))
