import numpy as np
import torch
import torch.nn as nn
from rlpyt.utils.buffer import buffer_func
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims, to_onehot, from_onehot

from models.dense import DenseModel
from models.observation import ObservationEncoder
from models.action import ActionDecoder
from models.latent import BisimState, BisimRepresentation, BisimTransition, BisimRollout, get_feat, soft_update_from_to

ModelReturnSpec = namedarraytuple('ModelReturnSpec', ['action', 'state'])


class AutoEncoder(nn.Module):
    def __init__(self, latent_state_size=4, lidar_obs_size=4, text_obs_size=4, action_size=2, use_id_decoder=False):
        super(AutoEncoder, self).__init__()

        self.encoder = DenseModelNormal(feature_size=2 * lidar_obs_size + 2 * text_obs_size,
                                        output_shape=(latent_state_size,),
                                        hidden_size=64,
                                        layers=3)

        self.encoder_seq = DenseModelNormal(
            feature_size=2 * lidar_obs_size + 2 * text_obs_size + latent_state_size + action_size,
            output_shape=(latent_state_size,),
            hidden_size=64,
            layers=4)

        self.transition = DenseModelNormal(feature_size=latent_state_size + action_size,
                                           output_shape=(latent_state_size,),
                                           layers=3,
                                           hidden_size=64)

        self.decoder_lidar = DenseModel(feature_size=latent_state_size,
                                        output_shape=(lidar_obs_size,),
                                        layers=3,
                                        hidden_size=64,
                                        dist='normal')

        self.decoder_text = DenseModel(feature_size=latent_state_size,
                                       output_shape=(text_obs_size,),
                                       layers=3,
                                       hidden_size=64,
                                       dist='normal')

        self.lidar_obs_size = lidar_obs_size
        self.text_obs_size = text_obs_size
        self.latent_state_size = latent_state_size
        # self.transition = None

    def get_latent_state_dist(self, lidar_obs, text_obs, pre_state=None, action=None, lidar_mask=None, text_mask=None):
        # input (time, batch, feature)
        # T, B, _ = observation.shape

        if pre_state is not None and action is not None:
            latent_state_dist = self.encoder_seq(torch.cat([lidar_obs * lidar_mask,
                                                            text_obs * text_mask,
                                                            lidar_mask, text_mask,
                                                            pre_state, action,
                                                            ], dim=-1))
        else:
            latent_state_dist = self.encoder(torch.cat([lidar_obs * lidar_mask,
                                                        text_obs * text_mask,
                                                        lidar_mask, text_mask], dim=-1))
        return latent_state_dist


class BisimModel(nn.Module):
    def __init__(
            self,
            action_shape,
            latent_size=32,
            hidden_size=128,
            observation_shape=(19 + 5,),
            observation_layers=2,
            embed_size=64,
            action_dist='tanh_normal',
            reward_shape=(1,),
            reward_layers=3,
            reward_hidden=128,
            qf_shape=(1,),
            qf_layers=3,
            qf_hidden=128,
            actor_layers=3,
            actor_hidden=128,
            init_temp=0.2,
            dtype=torch.float,
            q_update_tau=0.005,
            encoder_update_tau=0.005,
            **kwargs,
    ):
        super().__init__()
        self.autoencoder = AutoEncoder(latent_state_size=latent_size, description_size=observation_shape[0])

        self.action_shape = action_shape
        output_size = np.prod(action_shape)
        action_size = output_size
        self.action_dist = action_dist
        feature_size = latent_size

        self.actor_model = ActionDecoder(feature_size, action_size, actor_layers, actor_hidden,
                                         dist='tanh_normal')
        self.log_alpha = torch.tensor(np.log(init_temp))
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -torch.prod(torch.Tensor(action_shape)).item()

        self.reward_model = DenseModel(feature_size, reward_shape, reward_layers, reward_hidden)

        self.qf1_model = DenseModel(feature_size + action_size, qf_shape, qf_layers, qf_hidden)
        self.qf2_model = DenseModel(feature_size + action_size, qf_shape, qf_layers, qf_hidden)
        self.target_qf1_model = DenseModel(feature_size + action_size, qf_shape, qf_layers, qf_hidden)
        self.target_qf2_model = DenseModel(feature_size + action_size, qf_shape, qf_layers, qf_hidden)

        self.dtype = dtype
        self.q_update_tau = q_update_tau
        self.encoder_update_tau = encoder_update_tau
        self.step_count = 0
        self.init_random_steps = 1000
        self.latent_size = latent_size

    def forward(self, observation: torch.Tensor, prev_action: torch.Tensor = None, prev_state: BisimState = None):
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 1)
        observation = observation.reshape(T * B, *img_shape).type(self.dtype)
        state = self.get_state_representation(observation, None, None)
        # state = BisimState(observation, observation, torch.ones(observation.size()))
        if self.step_count < self.init_random_steps:
            action = torch.rand(size=(1, 1)).cuda() * 2 - 1.0
            self.step_count += 1
        else:
            action, action_dist = self.policy(state)
            action = torch.tanh(action)
        return_spec = ModelReturnSpec(action, state)
        return_spec = buffer_func(return_spec, restore_leading_dims, lead_dim, T, B)
        return return_spec

    def update_target_networks(self):
        soft_update_from_to(target=self.target_qf1_model, source=self.qf1_model, tau=self.q_update_tau)
        soft_update_from_to(target=self.target_qf2_model, source=self.qf2_model, tau=self.q_update_tau)

    def update_encoder(self):
        soft_update_from_to(target=self.target_autoencoder, source=self.autoencoder, tau=self.encoder_update_tau)

    def policy(self, state: torch.Tensor):
        action_dist = self.actor_model(state)
        if self.action_dist == 'tanh_normal':
            action = action_dist.rsample()
        elif self.action_dist == 'one_hot':
            action = action_dist.sample()
            # This doesn't change the value, but gives us straight-through gradients
            action = action + action_dist.probs - action_dist.probs.detach()

        return action, action_dist

    def get_state_representation(self, lidar_obs, text_obs, action=None, pre_state=None, lidar_mask=None,
                                 text_mask=None):
        if lidar_mask is None:
            lidar_mask = torch.ones_like(lidar_obs).to(lidar_obs.device)
        if text_mask is None:
            text_mask = torch.ones_like(text_obs).to(text_obs.device)
        latent_state_dist = self.autoencoder.get_latent_state_dist(lidar_obs=lidar_obs,
                                                                   text_obs=text_obs,
                                                                   pre_state=pre_state,
                                                                   action=action,
                                                                   lidar_mask=lidar_mask,
                                                                   text_mask=text_mask)
        if self.training:
            return latent_state_dist.rsample()
        else:
            return latent_state_dist.sample()

    def get_state_transition(self, prev_action: torch.Tensor, prev_state: BisimState):
        """

        :param prev_action: size(batch, action_size)
        :param prev_state: RSSMState: size(batch, state_size)
        :return: RSSMState
        """
        state = self.transition(prev_action, prev_state)
        return state
