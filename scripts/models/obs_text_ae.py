import torch
import torch.nn as nn
import numpy as np
import torch.distributions as td


class DenseModelNormal(nn.Module):
    def __init__(self, feature_size: int, output_shape: tuple, layers: int, hidden_size: int, activation=nn.ELU,
                 min=1e-4, max=10.0):
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
        dist_inputs = self.model(features)
        reshaped_inputs_mean = torch.reshape(dist_inputs[..., :np.prod(self._output_shape)],
                                             features.shape[:-1] + self._output_shape)
        reshaped_inputs_std = torch.reshape(dist_inputs[..., np.prod(self._output_shape):],
                                            features.shape[:-1] + self._output_shape)

        reshaped_inputs_std = torch.clamp(self.soft_plus(reshaped_inputs_std), min=self._min, max=self._max)
        return td.independent.Independent(td.Normal(reshaped_inputs_mean, reshaped_inputs_std), len(self._output_shape))


class AutoEncoder(torch.nn.Module):
    def __init__(self, latent_size=4, hidden_size=32, layers=3):
        super(AutoEncoder, self).__init__()
        self.gpt_embed_size = 768
        self.num_of_words = 13
        self.encoder = DenseModelNormal(
            feature_size=2,
            output_shape=(latent_size,),
            hidden_size=hidden_size,
            layers=layers)

        self.decoder_text = DenseModelNormal(
            feature_size=latent_size,
            output_shape=(self.gpt_embed_size * self.num_of_words,),
            hidden_size=hidden_size,
            layers=layers)


class AutoEncoders(torch.nn.Module):
    def __init__(self, h_direction_list, v_direction_list, latent_size, hidden_size, layers):
        super(AutoEncoders, self).__init__()
        self.autoencoders = {}
        self.h_direction_list = h_direction_list
        self.v_direction_list = v_direction_list
        for h_direction in self.h_direction_list:
            for v_direction in self.v_direction_list:
                self.autoencoders[f'{h_direction}-{v_direction}'] = AutoEncoder(latent_size, hidden_size, layers)
        # self.autoencoders = torch.nn.ModuleDict(model_dict)

    def set_device(self, device):
        for h_direction in self.h_direction_list:
            for v_direction in self.v_direction_list:
                self.autoencoders[f'{h_direction}-{v_direction}'].to(device)

    def load_models(self, save_path='./model_save/'):
        for h_direction in self.h_direction_list:
            for v_direction in self.v_direction_list:
                self.autoencoders[f'{h_direction}-{v_direction}'] = torch.load(f'{save_path}autoencoder_{h_direction}-{v_direction}.pt')


class AutoEncodersUni(torch.nn.Module):
    def __init__(self, latent_size=4, hidden_size=32, layers=3):
        super(AutoEncodersUni, self).__init__()
        self.gpt_embed_size = 768
        self.num_of_words = 13
        self.encoder = DenseModelNormal(
            feature_size=2+2,
            output_shape=(latent_size,),
            hidden_size=hidden_size,
            layers=layers)

        self.decoder_text = DenseModelNormal(
            feature_size=latent_size,
            output_shape=(self.gpt_embed_size * self.num_of_words,),
            hidden_size=hidden_size,
            layers=layers)

        self.h_direction_list = {'front':1, 'rear':-1, 'left-front':2, 'right-front':-2, 'left-rear':3, 'right-rear':-3}
        self.v_direction_list = {'front':1, 'rear':-1}

    def get_latent_dist(self, lidar_obs, speed, h_direction, v_direction):
        h_value = torch.ones_like(speed).to(speed.device) * self.h_direction_list[h_direction]
        v_value = torch.ones_like(speed).to(speed.device) * self.v_direction_list[v_direction]

        feature = torch.cat([lidar_obs, speed, h_value, v_value], dim=-1)
        return self.encoder(feature)

    def set_device(self, device):
        self.encoder.to(device)
        self.decoder_text.to(device)

