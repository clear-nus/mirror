import numpy as np
import torch
import torch.nn as nn
from rlpyt.utils.collections import namedarraytuple
from models.dense import DenseModelDist, DenseModelNormal, ModelNormalStd, DenseModel

from models.dense import DenseModel
from models.action import ActionDecoder
from models.latent import BisimState, BisimRepresentation, BisimTransition, BisimRollout, get_feat, soft_update_from_to

ModelReturnSpec = namedarraytuple('ModelReturnSpec', ['action', 'state'])


class PolicyFilter(torch.nn.Module):
    def __init__(self, latent_state_size):
        super(PolicyFilter, self).__init__()
        self.policy_filter = DenseModel(feature_size=latent_state_size,
                                        output_shape=(2,),
                                        layers=1,
                                        hidden_size=16,
                                        dist='normal')

    def forward(self, latent_state):
        action_dist = self.policy_filter(latent_state)
        return action_dist


class PerceptualMask(torch.nn.Module):
    def __init__(self):
        super(PerceptualMask, self).__init__()
        self.obs_mask = torch.nn.Parameter(torch.ones(size=(36,)), requires_grad=True)

    def rsample_obs_mask(self):
        obs_mask_dist = torch.distributions.Bernoulli(logits=self.obs_mask)
        samples = obs_mask_dist.sample()
        obs_mask_rsample = samples + 2.0 * (samples - 0.5) * obs_mask_dist.mean \
                           - 2.0 * (samples - 0.5) * obs_mask_dist.mean.detach()
        return obs_mask_rsample
        # return torch.sigmoid(self.obs_mask)

    def rsample_text_mask(self):
        text_mask_dist = torch.distributions.Bernoulli(logits=self.text_mask)
        text_mask_rsample = text_mask_dist.sample() + text_mask_dist.mean - text_mask_dist.mean.detach()
        return text_mask_rsample


class PerceptualMaskDis(torch.nn.Module):
    def __init__(self):
        super(PerceptualMaskDis, self).__init__()
        self.obs_mask = torch.nn.Parameter((torch.rand(size=(36,)) - 0.5), requires_grad=True)

    def get_obs_mask(self, obs):
        obs_mask = (obs < torch.sigmoid(self.obs_mask)).to(torch.float) \
                   + 1.0 * ((obs < torch.sigmoid(self.obs_mask)).to(torch.float) - 0.0) * torch.sigmoid(self.obs_mask) \
                   - 1.0 * ((obs < torch.sigmoid(self.obs_mask)).to(torch.float) - 0.0) * torch.sigmoid(
            self.obs_mask).detach()
        return obs_mask


class AutoEncoder(nn.Module):
    def __init__(self, latent_state_size=4, lidar_obs_size=4, text_obs_size=4, action_size=2):
        super(AutoEncoder, self).__init__()

        self.encoder = DenseModelNormal(feature_size=2 * lidar_obs_size + 2 * text_obs_size,
                                        output_shape=(latent_state_size,),
                                        hidden_size=128,
                                        layers=3)

        self.encoder_seq = DenseModelNormal(
            feature_size=2 * lidar_obs_size + 2 * text_obs_size + latent_state_size + action_size,
            output_shape=(latent_state_size,),
            hidden_size=128,
            layers=4)

        self.transition = DenseModelNormal(feature_size=latent_state_size + action_size,
                                           output_shape=(latent_state_size,),
                                           layers=3,
                                           hidden_size=128)

        self.decoder_lidar = DenseModel(feature_size=latent_state_size,
                                        output_shape=(lidar_obs_size,),
                                        layers=3,
                                        hidden_size=128,
                                        dist='normal')

        self.decoder_text = DenseModel(feature_size=latent_state_size,
                                       output_shape=(text_obs_size,),
                                       layers=3,
                                       hidden_size=128,
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
            latent_size=128,
            lidar_obs_size=36 * 3 + 5,
            text_obs_size=20 * 6,
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
        self.action_shape = action_shape
        output_size = np.prod(action_shape)
        action_size = output_size
        self.action_dist = action_dist
        feature_size = latent_size

        self.autoencoder = AutoEncoder(latent_state_size=latent_size,
                                       lidar_obs_size=lidar_obs_size,
                                       text_obs_size=text_obs_size,
                                       action_size=action_size)

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

    def update_target_networks(self):
        soft_update_from_to(target=self.target_qf1_model, source=self.qf1_model, tau=self.q_update_tau)
        soft_update_from_to(target=self.target_qf2_model, source=self.qf2_model, tau=self.q_update_tau)

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
            return latent_state_dist.mean
        else:
            return latent_state_dist.mean

    def model_parallel(self):
        self.autoencoder.decoder_lidar.model = nn.DataParallel(self.autoencoder.decoder_lidar.model, device_ids=[0, 1])
        self.autoencoder.decoder_text.model = nn.DataParallel(self.autoencoder.decoder_text.model, device_ids=[0, 1])
        self.autoencoder.transition.model = nn.DataParallel(self.autoencoder.transition.model, device_ids=[0, 1])
        self.autoencoder.encoder.model = nn.DataParallel(self.autoencoder.encoder.model, device_ids=[0, 1])
        self.autoencoder.encoder_seq.model = nn.DataParallel(self.autoencoder.encoder_seq.model, device_ids=[0, 1])

        self.qf1_model.model = nn.DataParallel(self.qf1_model.model, device_ids=[0, 1])
        self.actor_model._action_model = nn.DataParallel(self.actor_model._action_model, device_ids=[0, 1])

    def jit_model(self):
        self.autoencoder.decoder_lidar.model = torch.jit.script(self.autoencoder.decoder_lidar.model)
        self.autoencoder.decoder_text.model = torch.jit.script(self.autoencoder.decoder_text.model)
        self.autoencoder.transition.model = torch.jit.script(self.autoencoder.transition.model)
        self.autoencoder.encoder.model = torch.jit.script(self.autoencoder.encoder.model)
        # self.autoencoder.encoder_seq.model = torch.jit.script(self.autoencoder.encoder_seq.model)


class BCBisimModel(nn.Module):
    def __init__(
            self,
            action_shape,
    ):
        super().__init__()
        self.action_shape = action_shape
        self.basic_model_seq = torch.nn.GRU(input_size=(113 + 120) * 2 + 2, hidden_size=128, num_layers=1, bias=True)
        self.basic_model = torch.nn.GRU(input_size=(113 + 120) * 2, hidden_size=128, num_layers=1, bias=True)
        self.actor = DenseModel(128, (2,), 3, 256)

    def policy(self, state: torch.Tensor):
        action_dist = self.actor(state)
        return action_dist.mean, action_dist

    def get_state_representation(self, lidar_obs, text_obs, action=None, pre_state=None, lidar_mask=None,
                                 text_mask=None):
        size_length = len(lidar_obs.size())
        if len(lidar_obs.size()) == 1:
            _lidar_obs = lidar_obs.unsqueeze(0).unsqueeze(0)
        elif len(lidar_obs.size()) == 2:
            _lidar_obs = lidar_obs.unsqueeze(0)
        else:
            _lidar_obs = lidar_obs

        if len(text_obs.size()) == 1:
            _text_obs = text_obs.unsqueeze(0).unsqueeze(0)
        elif len(text_obs.size()) == 2:
            _text_obs = text_obs.unsqueeze(0)
        else:
            _text_obs = text_obs

        if lidar_mask is not None:
            if len(lidar_mask.size()) == 1:
                _lidar_mask = lidar_mask.unsqueeze(0).unsqueeze(0)
            elif len(lidar_mask.size()) == 2:
                _lidar_mask = lidar_mask.unsqueeze(0)
            else:
                _lidar_mask = lidar_mask

        if text_mask is not None:
            if len(text_mask.size()) == 1:
                _text_mask = text_mask.unsqueeze(0).unsqueeze(0)
            elif len(text_mask.size()) == 2:
                _text_mask = text_mask.unsqueeze(0)
            else:
                _text_mask = text_mask

        if action is not None:
            if len(action.size()) == 1:
                _action = action.unsqueeze(0).unsqueeze(0)
            elif len(action.size()) == 2:
                _action = action.unsqueeze(0)
            else:
                _action = action

        if lidar_mask is None:
            _lidar_mask = torch.ones_like(_lidar_obs).to(lidar_obs.device)
        if text_mask is None:
            _text_mask = torch.ones_like(_text_obs).to(text_obs.device)

        if action is not None:
            input_feature = torch.cat(
                [_lidar_obs * _lidar_mask, _text_obs * _text_mask, _lidar_mask, _text_mask, _action], dim=-1)
        else:
            input_feature = torch.cat([_lidar_obs * _lidar_mask, _text_obs * _text_mask, _lidar_mask, _text_mask],
                                      dim=-1)

        if pre_state is not None:
            if len(pre_state.size()) == 1:
                _pre_state = pre_state.unsqueeze(0).unsqueeze(0)
            elif len(pre_state.size()) == 2:
                _pre_state = pre_state.unsqueeze(0)
            else:
                _pre_state = pre_state

            _pre_state = _pre_state.contiguous()
            latent_state, _ = self.basic_model_seq(input_feature, _pre_state)
        else:
            latent_state, _ = self.basic_model(input_feature)

        if size_length == 2:
            latent_state = latent_state.squeeze(0)
        elif size_length == 1:
            latent_state = latent_state.squeeze(0).squeeze(0)
        return latent_state
