import numpy as np
import torch
import torch.nn as nn
from rlpyt.utils.buffer import buffer_func
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims, to_onehot, from_onehot

from models.dense import DenseModel
from models.observation import ObservationEncoder
from models.action import ActionDecoder
from models.latent import BisimState, get_feat, soft_update_from_to

ModelReturnSpec = namedarraytuple('ModelReturnSpec', ['action', 'state'])


class BisimRLModel(nn.Module):
    def __init__(
            self,
            action_shape,
            latent_size=128,
            action_dist='independent_binary',
            qf_shape=(1,),
            qf_layers=3,
            qf_hidden=256,
            actor_layers=2,
            actor_hidden=256,
            init_temp=0.2,
            dtype=torch.float,
            q_update_tau=0.005,
            encoder_update_tau=0.005,
            **kwargs,
    ):
        super().__init__()
        self.action_shape = action_shape
        output_size = np.prod(action_shape)
        action_size = output_size
        self.action_dist = action_dist

        self.actor_model = ActionDecoder(2 * latent_size, action_size, actor_layers, actor_hidden,
                                         dist=action_dist)
        self.log_alpha = torch.tensor(np.log(init_temp))
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -torch.prod(torch.Tensor(action_shape)).item()

        self.qf1_model = DenseModel(2 * latent_size+action_size, qf_shape, qf_layers, qf_hidden)
        self.qf2_model = DenseModel(2 * latent_size+action_size, qf_shape, qf_layers, qf_hidden)
        self.target_qf1_model = DenseModel(2 * latent_size + action_size, qf_shape, qf_layers, qf_hidden)
        self.target_qf2_model = DenseModel(2 * latent_size + action_size, qf_shape, qf_layers, qf_hidden)

        self.dtype = dtype
        self.q_update_tau = q_update_tau
        self.encoder_update_tau = encoder_update_tau
        self.step_count = 0
        self.init_random_steps = 1000

    def update_target_networks(self):
        soft_update_from_to(target=self.target_qf1_model, source=self.qf1_model, tau=self.q_update_tau)
        soft_update_from_to(target=self.target_qf2_model, source=self.qf2_model, tau=self.q_update_tau)

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
        elif self.action_dist == "independent_binary":
            # action = action_dist.rsample()
            action = action_dist.sample() + action_dist.probs - action_dist.probs.detach()

        return action, action_dist