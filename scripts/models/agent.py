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


class BisimModel(nn.Module):
    def __init__(
            self,
            action_shape,
            latent_size=48,
            hidden_size=128,
            observation_shape=(231,),
            observation_layers=2,
            embed_size=64,
            action_dist='tanh_normal',
            reward_shape=(1,),
            reward_layers=3,
            reward_hidden=128,
            qf_shape=(1,),
            qf_layers=3,
            qf_hidden=128,
            actor_layers=2,
            actor_hidden=128,
            init_temp=0.2,
            dtype=torch.float,
            q_update_tau=0.005,
            encoder_update_tau=0.005,
            **kwargs,
    ):
        super().__init__()
        self.observation_encoder = ObservationEncoder(obs_shape=observation_shape,
                                              embed_size=embed_size,
                                              layers=observation_layers,
                                              hidden_size=hidden_size)
        self.target_observation_encoder = ObservationEncoder(obs_shape=observation_shape,
                                                      embed_size=embed_size,
                                                      layers=observation_layers,
                                                      hidden_size=hidden_size)
        self.action_shape = action_shape
        output_size = np.prod(action_shape)
        action_size = output_size
        self.action_dist = action_dist
        feature_size = latent_size
        self.transition = BisimTransition(action_size=output_size,
                                          latent_size=latent_size,
                                          hidden_size=hidden_size)
        self.representation = BisimRepresentation(obs_embed_size=embed_size,
                                                  latent_size=latent_size,
                                                  hidden_size=hidden_size)
        self.target_representation = BisimRepresentation(obs_embed_size=embed_size,
                                                  latent_size=latent_size,
                                                  hidden_size=hidden_size)
        self.rollout = BisimRollout(representation_model=self.representation,
                                    target_representation_model=self.target_representation,
                                    transition_model=self.transition)

        self.actor_model = ActionDecoder(feature_size, action_size, actor_layers, actor_hidden,
                                         dist='tanh_normal')
        self.log_alpha = torch.tensor(np.log(init_temp))
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -torch.prod(torch.Tensor(action_shape)).item()

        self.reward_model = DenseModel(feature_size, reward_shape, reward_layers, reward_hidden)

        self.qf1_model = DenseModel(feature_size+action_size, qf_shape, qf_layers, qf_hidden)
        self.qf2_model = DenseModel(feature_size+action_size, qf_shape, qf_layers, qf_hidden)
        self.target_qf1_model = DenseModel(feature_size + action_size, qf_shape, qf_layers, qf_hidden)
        self.target_qf2_model = DenseModel(feature_size + action_size, qf_shape, qf_layers, qf_hidden)

        self.dtype = dtype
        self.q_update_tau = q_update_tau
        self.encoder_update_tau = encoder_update_tau
        self.step_count = 0
        self.init_random_steps = 1000

    def forward(self, observation: torch.Tensor, prev_action: torch.Tensor = None, prev_state: BisimState = None):
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 1)
        observation = observation.reshape(T * B, *img_shape).type(self.dtype)
        state = self.get_state_representation(observation, None, None)
        # state = BisimState(observation, observation, torch.ones(observation.size()))
        if self.step_count < self.init_random_steps:
            action = torch.rand(size=(1,1)).cuda() * 2 - 1.0
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
        soft_update_from_to(target=self.target_observation_encoder, source=self.observation_encoder, tau=self.encoder_update_tau)
        soft_update_from_to(target=self.target_representation, source=self.representation, tau=self.encoder_update_tau)

    def policy(self, state: BisimState):
        if type(state) is BisimState:
            feat = get_feat(state)
        elif type(state) is torch.Tensor:
            feat = state
        else:
            raise NotImplementedError(type(state))

        action_dist = self.actor_model(feat)
        if self.action_dist == 'tanh_normal':
            action = action_dist.rsample()
        elif self.action_dist == 'one_hot':
            action = action_dist.sample()
            # This doesn't change the value, but gives us straight-through gradients
            action = action + action_dist.probs - action_dist.probs.detach()

        return action, action_dist

    def get_state_representation(self, observation: torch.Tensor, prev_action: torch.Tensor = None,
                                 prev_state: BisimState = None):
        """

        :param observation: size(batch, channels, width, height)
        :param prev_action: size(batch, action_size)
        :param prev_state: RSSMState: size(batch, state_size)
        :return: RSSMState
        """
        obs_embed = self.observation_encoder(observation)
        posterior = self.representation(obs_embed)
        return posterior

    def get_state_transition(self, prev_action: torch.Tensor, prev_state: BisimState):
        """

        :param prev_action: size(batch, action_size)
        :param prev_state: RSSMState: size(batch, state_size)
        :return: RSSMState
        """
        state = self.transition(prev_action, prev_state)
        return state
