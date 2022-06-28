#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
import torch

from models.cat_model import BisimModel, PerceptualMaskDis, PolicyFilter
from utils.replay_buffer import HumanDemonstrationReplayBuffer
from utils.module import get_parameters


def train_implants(path='./exp_data/participant_32'):
    np.set_printoptions(precision=3)
    device = 'cuda:0'
    max_itr = 100
    batch_size = 500
    batch_t = 50
    basic_model = BisimModel(action_shape=(2,)).to(device)
    basic_model.load_state_dict(torch.load('./current_model/model_latest.pt'))

    replay_buffer = HumanDemonstrationReplayBuffer(lidar_obs_shape=(2 + 4 * 2,),
                                                   text_embed_shape=(4 * 3,),
                                                   human_action_shape=(2 + 4 * 2,),
                                                   capacity=100000,
                                                   batch_size=batch_size,
                                                   device=torch.device('cuda', 0))
    replay_buffer.load(path)

    # define perceptual mask
    perceptual_mask = PerceptualMaskDis().to(device)
    policy_filter = PolicyFilter(basic_model.latent_size).to(device)

    # define optimizer
    optimizer = torch.optim.Adam(get_parameters([perceptual_mask, policy_filter]), lr=0.2)
    avg_loss = 0.0
    # train
    for i in range(max_itr):
        lidar_obs, text_embed, human_action = replay_buffer.sample(batch_t)
        lidar_obs = lidar_obs.transpose(0, 1).to(device).to(torch.float)
        lidar_obs_2d = lidar_obs[..., :-5][..., 1::3]
        text_embed = text_embed.transpose(0, 1).to(device).to(torch.float)
        text_embed = text_embed[..., 0:1].repeat(1, 1, 120)
        human_action = human_action.transpose(0, 1).to(device).to(torch.float)

        lidar_obs_mask = perceptual_mask.get_obs_mask(lidar_obs_2d)
        lidar_obs_mask_rsample_list = []
        for k in range(36):
            lidar_obs_mask_rsample_list += [lidar_obs_mask[:, :, k:k + 1].repeat(1, 1, 3 * 1)]  # 16 17 18]
        lidar_obs_mask_rsample_list += [torch.ones(size=(batch_t, batch_size, 5)).to(device)]

        lidar_obs_mask = torch.cat(lidar_obs_mask_rsample_list, dim=-1)

        latent_state_sample = [[]] * batch_t
        for t in range(batch_t):
            if t == 0:
                latent_state_dist = basic_model.autoencoder.get_latent_state_dist(lidar_obs=lidar_obs[t],
                                                                                  text_obs=text_embed[t],
                                                                                  pre_state=None,
                                                                                  action=None,
                                                                                  lidar_mask=lidar_obs_mask[0],
                                                                                  text_mask=torch.zeros_like(
                                                                                      text_embed[t]).to(torch.float).to(
                                                                                      device))
            else:
                latent_state_dist = basic_model.autoencoder.get_latent_state_dist(
                    lidar_obs=lidar_obs[t],
                    text_obs=text_embed[t],
                    pre_state=latent_state_sample[t - 1],
                    action=human_action[t],
                    lidar_mask=lidar_obs_mask[0],
                    text_mask=torch.zeros_like(text_embed[t]).to(device))
            latent_state_sample[t] = latent_state_dist.sample()
        latent_state_sample = torch.stack(latent_state_sample, dim=0)
        policy_action, policy_action_dist = basic_model.policy(latent_state_sample)
        residule_policy_dist = policy_filter(latent_state_sample)

        optimizer.zero_grad()
        loss = ((torch.tanh(policy_action_dist.mean[:-1, :, 1:2] + 0.1 * residule_policy_dist.mean[:-1, :, 1:2]) - human_action[1:, :, 1:2]).pow(2)).sum(dim=-1).sum(
            dim=0).mean() * 1.0

        loss += ((torch.tanh(policy_action_dist.mean[:-1, :, 0:1] + 0.1 * residule_policy_dist.mean[:-1, :, 0:1]) - human_action[1:, :, 0:1]).pow(2)).sum(dim=-1).sum(
            dim=0).mean() * 0.1
        prior_loss = lidar_obs_mask[..., :].pow(2).sum(dim=-1).mean() * 0.  # - lidar_obs_mask[..., 108-3:108].sum(dim=-1).mean() * 1.0
        regularize_loss = (residule_policy_dist.mean[:-1]).pow(2).mean() * 2
        loss += prior_loss + regularize_loss
        loss.backward()
        optimizer.step()

        avg_loss += loss
        print(
            f'loss:{loss}, {torch.sigmoid(perceptual_mask.obs_mask).detach().cpu().numpy()}')
    print(avg_loss / max_itr)

    if path is None:
        torch.save(perceptual_mask.state_dict(), './current_model/perceptual_mask.pt')
        torch.save(policy_filter.state_dict(), './current_model/policy_filter.pt')
    else:
        torch.save(perceptual_mask.state_dict(), f'{path}/perceptual_mask.pt')
        torch.save(policy_filter.state_dict(), f'{path}/policy_filter.pt')

    return perceptual_mask, policy_filter


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


if __name__ == '__main__':
    train_implants()
