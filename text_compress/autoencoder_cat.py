import torch
import torch.nn as nn
import numpy as np
from models.dense import DenseModelDist, DenseModelNormal, ModelNormalStd, DenseModel
# from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
import torch.distributions as td


def product_of_expert(mean: torch.Tensor, std: torch.Tensor, mask: torch.Tensor = None):
    '''
    :param mean: (modality, batch, input) mean of each modality
    :param std: (modality, batch, input) std of each modality
    :param mask: (modality, batch, input) position of missing modality
    :return: return the product of each independent gaussian expert
    '''

    var = std.pow(2) + 1e-4
    # Precision matrix of i-th Gaussian expert (T = 1/sigma^2)
    T = 1. / var * std.sign()
    # Set missing data to zero so they are excluded from calculation
    if mask is None:
        mask = 1 - torch.isnan(var).any(dim=-1).type(torch.FloatTensor)
    mask = mask.to(std.device)
    T = T * mask.float()
    mean = mean * mask.float()
    product_mean = torch.sum(mean * T, dim=0) / torch.sum(T, dim=0)
    product_mean[torch.isnan(product_mean)] = 0.0
    product_std = (1. / torch.sum(T, dim=0)).pow(0.5)
    return product_mean, product_std


class AutoEncoder(nn.Module):
    def __init__(self, latent_state_size=4, description_size=4, action_size=2, use_id_decoder=False):
        super(AutoEncoder, self).__init__()

        self.encoder = DenseModelNormal(feature_size=2 * description_size,
                                        output_shape=(latent_state_size,),
                                        hidden_size=48,
                                        layers=3)

        self.encoder_seq = DenseModelNormal(feature_size=2 * description_size + latent_state_size + action_size,
                                            output_shape=(latent_state_size,),
                                            hidden_size=48,
                                            layers=4)

        self.transition = DenseModelNormal(feature_size=latent_state_size+action_size,
                                           output_shape=(latent_state_size,),
                                           layers=3,
                                           hidden_size=48)

        self.decoder_normal = DenseModel(feature_size=latent_state_size,
                                         output_shape=(description_size,),
                                         layers=3,
                                         hidden_size=48,
                                         dist='normal')

        self.description_size = description_size
        self.latent_state_size = latent_state_size
        # self.transition = None

    def get_latent_state_dist(self, description,
                              pre_state: torch.Tensor = None,
                              action: torch.Tensor = None,
                              mask: torch.Tensor = None):
        # input (time, batch, feature)
        # T, B, _ = observation.shape

        if pre_state is not None and action is not None:
            latent_state_dist = self.encoder_seq(torch.cat([description, mask, pre_state, action], dim=-1))
        else:
            latent_state_dist = self.encoder(torch.cat([description, mask], dim=-1))
        return latent_state_dist
