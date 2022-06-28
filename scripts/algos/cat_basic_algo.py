import random

import torch
from rlpyt.algos.base import RlAlgorithm
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args
from tqdm import tqdm

from utils.module import get_parameters, FreezeParameters

from torch.utils.tensorboard import SummaryWriter

torch.autograd.set_detect_anomaly(True)  # used for debugging gradients

loss_info_fields = ['model_loss', 'q1_loss', 'q2_loss', 'actor_loss', 'alpha_loss', 'reward_loss', 'dynamics_loss',
                    "rec_loss", "kl_loss", "align_loss"]
LossInfo = namedarraytuple('LossInfo', loss_info_fields)
OptInfo = namedarraytuple("OptInfo",
                          ['loss', 'grad_norm_model', 'grad_norm_actor', 'grad_norm_alpha',
                           'grad_norm_qf', 'grad_norm_dynamics', 'grad_norm_reward'] + loss_info_fields)


class Bisimulation(RlAlgorithm):
    def __init__(self,
                 model,
                 batch_size=200,
                 batch_length=3,
                 train_every=1000,
                 eval_every=100,
                 train_steps=100,
                 pretrain=1000,
                 model_lr=3e-4,
                 ae_lr=3e-4,
                 dynamics_lr=3e-4,
                 reward_lr=3e-4,
                 qf_lr=3e-4,
                 actor_lr=3e-4,
                 alpha_lr=3e-4,
                 grad_clip=100.0,
                 discount=0.9,
                 OptimCls=torch.optim.Adam,
                 initial_optim_state_dict=None,
                 replay_size=int(1000000),
                 n_step_return=1,
                 updates_per_sync=1,  # For async mode only. (not implemented)
                 target_update_period=1,
                 actor_update_period=1,
                 type=torch.float,
                 device='cuda',
                 save_every=5
                 ):
        super().__init__()
        self._batch_size = batch_size
        del batch_size  # Property.
        save__init__args(locals())

        self.type = type
        self.device = device
        self.writer = SummaryWriter()
        self.itr = 0
        self.beta = 1.0
        self.beta_decay = 0.95

    def optim_initialize(self, model):
        self.model = model
        self.model_modules = [model.autoencoder]
        self.reward_modules = [model.reward_model]
        self.critic_modules = [model.qf1_model,
                               model.qf2_model]
        self.ae_modules = [model.autoencoder]

        self.actor_modules = [model.actor_model]
        self.alpha_modules = [model.log_alpha]

        self.model_optimizer = torch.optim.Adam(get_parameters(self.model_modules),
                                                lr=self.model_lr)
        self.reward_optimizer = torch.optim.Adam(get_parameters(self.reward_modules),
                                                 lr=self.reward_lr)
        self.actor_optimizer = torch.optim.Adam(get_parameters(self.actor_modules),
                                                lr=self.actor_lr)
        self.alpha_optimizer = torch.optim.Adam(self.alpha_modules,
                                                lr=self.alpha_lr)
        self.critic_optimizer = torch.optim.Adam(get_parameters(self.critic_modules),
                                                 lr=self.qf_lr)
        self.ae_optimizer = torch.optim.Adam(get_parameters(self.ae_modules),
                                             lr=self.ae_lr)
        self.opt_info_fields = OptInfo._fields

    def optimize_agent(self, model, replay_buffer, optim_itr):
        for i in tqdm(range(self.train_steps), desc='Imagination'):
            lidar_obs, text_embed, actions, expert_actions, rewards, dones = replay_buffer.sample()
            # er_lidar_obs, er_text_embed, er_actions, er_expert_actions, _, _ = expert_replay_buffer.sample()
            lidar_obs = lidar_obs.transpose(0, 1)
            text_embed = text_embed.transpose(0, 1)
            actions = actions.transpose(0, 1)
            expert_actions = expert_actions.transpose(0, 1)
            rewards = rewards.transpose(0, 1)
            dones = dones.transpose(0, 1)

            batch_t = lidar_obs.shape[0]
            batch_b = lidar_obs.shape[1]

            drop_method = random.choice([0, 1, 2])
            if drop_method == 0:
                lidar_obs_mask = (torch.rand(size=(batch_t, batch_b, 6 + 5)) < 0.35).type(self.type).to(
                    lidar_obs.device)
                text_embed_mask = (torch.rand(size=(batch_t, batch_b, 6)) < 0.35).type(self.type).to(text_embed.device)
                lidar_obs_mask[0, :, :] = 1
                text_embed_mask[0, :, :] = 1
                lidar_obs_mask[:, :, 6:] = 1

                lidar_obs_mask = [lidar_obs_mask[:, :, 0:1].repeat(1, 1, 3 * 1),
                                  lidar_obs_mask[:, :, 1:2].repeat(1, 1, 3 * 8),  # 0 1 2
                                  lidar_obs_mask[:, :, 2:3].repeat(1, 1, 3 * 9),  # 3 4 5
                                  lidar_obs_mask[:, :, 3:4].repeat(1, 1, 3 * 1),  # 6 7 8
                                  lidar_obs_mask[:, :, 4:5].repeat(1, 1, 3 * 9),  # 9
                                  lidar_obs_mask[:, :, 5:6].repeat(1, 1, 3 * 8),
                                  lidar_obs_mask[:, :, 6:]]  # 16 17 18]

                lidar_obs_mask = torch.cat(lidar_obs_mask, dim=-1)  # * 0.0 + 1.0

                text_embed_mask = [text_embed_mask[:, :, 0:1].repeat(1, 1, 20),
                                   text_embed_mask[:, :, 1:2].repeat(1, 1, 20),
                                   text_embed_mask[:, :, 2:3].repeat(1, 1, 20),
                                   text_embed_mask[:, :, 3:4].repeat(1, 1, 20),
                                   text_embed_mask[:, :, 4:5].repeat(1, 1, 20),
                                   text_embed_mask[:, :, 5:6].repeat(1, 1, 20), ]  # 16 17 18]

                text_embed_mask = torch.cat(text_embed_mask, dim=-1)  # * 0.0
            elif drop_method == 1:
                lidar_obs_mask = (torch.rand(size=(batch_t, batch_b, 6 + 5)) < 1.0).type(self.type).to(
                    lidar_obs.device)
                text_embed_mask = (torch.rand(size=(batch_t, batch_b, 6)) < 0.0).type(self.type).to(text_embed.device)
                lidar_obs_mask[0, :, :] = 1
                text_embed_mask[0, :, :] = 1
                lidar_obs_mask[:, :, 6:] = 1

                lidar_obs_mask = [lidar_obs_mask[:, :, 0:1].repeat(1, 1, 3 * 1),
                                  lidar_obs_mask[:, :, 1:2].repeat(1, 1, 3 * 8),  # 0 1 2
                                  lidar_obs_mask[:, :, 2:3].repeat(1, 1, 3 * 9),  # 3 4 5
                                  lidar_obs_mask[:, :, 3:4].repeat(1, 1, 3 * 1),  # 6 7 8
                                  lidar_obs_mask[:, :, 4:5].repeat(1, 1, 3 * 9),  # 9
                                  lidar_obs_mask[:, :, 5:6].repeat(1, 1, 3 * 8),
                                  lidar_obs_mask[:, :, 6:]]  # 16 17 18]

                lidar_obs_mask = torch.cat(lidar_obs_mask, dim=-1)  # * 0.0 + 1.0

                text_embed_mask = [text_embed_mask[:, :, 0:1].repeat(1, 1, 20),
                                   text_embed_mask[:, :, 1:2].repeat(1, 1, 20),
                                   text_embed_mask[:, :, 2:3].repeat(1, 1, 20),
                                   text_embed_mask[:, :, 3:4].repeat(1, 1, 20),
                                   text_embed_mask[:, :, 4:5].repeat(1, 1, 20),
                                   text_embed_mask[:, :, 5:6].repeat(1, 1, 20), ]  # 16 17 18]

                text_embed_mask = torch.cat(text_embed_mask, dim=-1)  # * 0.0
            else:
                lidar_obs_mask = (torch.rand(size=(batch_t, batch_b, 36 * 3 + 5)) < 0.35).type(self.type).to(
                    lidar_obs.device)
                text_embed_mask = (torch.rand(size=(batch_t, batch_b, 6 * 20)) < 0.35).type(self.type).to(
                    text_embed.device)

                lidar_obs_mask[..., -5:] = 1.0
                lidar_obs_mask[0, :, :] = 1
                text_embed_mask[0, :, :] = 1

            # train encoder
            ae_loss, rec_loss, kl_loss, latent_state_samples = self.ae_loss(lidar_obs, text_embed, actions,
                                                                            lidar_obs_mask, text_embed_mask)
            self.model_optimizer.zero_grad()
            (ae_loss).backward()
            self.model_optimizer.step()
            self.writer.add_scalar('Loss/rec_loss', rec_loss, self.itr)

            # # train actor
            if i % self.actor_update_period == 0:
                actor_loss, alpha_loss = self.actor_loss(latent_state_samples, actions, expert_actions)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.writer.add_scalar('Loss/Actor', actor_loss, self.itr)

            # train critics
            qf1_loss, qf2_loss = self.critic_loss(latent_state_samples, actions, rewards, dones)
            self.critic_optimizer.zero_grad()
            (qf1_loss + qf2_loss).backward()
            self.critic_optimizer.step()
            self.writer.add_scalar('Loss/Critic1', qf1_loss, self.itr)

            # train reward
            reward_loss = self.reward_loss(latent_state_samples, actions, rewards, dones)
            self.reward_optimizer.zero_grad()
            reward_loss.backward()
            self.reward_optimizer.step()
            self.writer.add_scalar('Loss/Reward', reward_loss, self.itr)

            self.itr += 1
        # Save model and optimizer
        torch.save(self.model.state_dict(), './current_model/model_latest' + '.pt')
        if optim_itr % self.save_every == 0:
            torch.save(self.model.state_dict(), './current_model/model' + str(optim_itr) + '.pt')
            if optim_itr >= (self.save_every * 5):
                # os.remove('./current_model/model' + str(optim_itr - self.save_every * 5) + '.pt')
                replay_buffer.save()

        self.beta *= self.beta_decay
        if self.beta < 0.1:
            self.beta = 0

        return self.model

    def ae_loss(self, lidar_obs, text_embed, actions, lidar_obs_mask, text_embed_mask):
        # be really careful of the index, (s_t, r_t, a_{t-1})

        lidar_obses = lidar_obs.to(self.device).type(self.type)
        text_embeds = text_embed.to(self.device).type(self.type)
        action = actions.to(self.device).type(self.type)
        batch_t = lidar_obses.shape[0]

        # get latent_state
        latent_state_rsample = [[]] * batch_t
        latent_state_mean = [[]] * batch_t
        latent_state_std = [[]] * batch_t
        for t in range(batch_t):
            if t == 0:
                latent_state_dist = self.model.autoencoder.get_latent_state_dist(lidar_obs=lidar_obses[t],
                                                                                 text_obs=text_embeds[t],
                                                                                 pre_state=None,
                                                                                 action=None,
                                                                                 lidar_mask=lidar_obs_mask[t],
                                                                                 text_mask=text_embed_mask[t])
            else:
                latent_state_dist = self.model.autoencoder.get_latent_state_dist(
                    lidar_obs=lidar_obses[t],
                    text_obs=text_embeds[t],
                    pre_state=latent_state_rsample[t - 1],
                    action=action[t],
                    lidar_mask=lidar_obs_mask[t],
                    text_mask=text_embed_mask[t])
            latent_state_rsample[t] = latent_state_dist.rsample()
            latent_state_mean[t] = latent_state_dist.mean
            latent_state_std[t] = latent_state_dist.stddev

        trans_latent_state_mean = [[]] * batch_t
        trans_latent_state_std = [[]] * batch_t
        for t in range(batch_t):
            if t == 0:
                trans_latent_state_mean[0] = latent_state_mean[0]
                trans_latent_state_std[0] = latent_state_std[0]
            else:
                trans_latent_state_dist = self.model.autoencoder.transition(
                    torch.cat([latent_state_rsample[t - 1], action[t - 1]], dim=-1))
                trans_latent_state_mean[t] = trans_latent_state_dist.mean
                trans_latent_state_std[t] = trans_latent_state_dist.stddev

        latent_state_rsample = torch.stack(latent_state_rsample, dim=0)
        latent_state_mean = torch.stack(latent_state_mean, dim=0)
        latent_state_std = torch.stack(latent_state_std, dim=0)

        trans_latent_state_mean = torch.stack(trans_latent_state_mean, dim=0)
        trans_latent_state_std = torch.stack(trans_latent_state_std, dim=0)

        # reconstruct observations
        lidar_obs_dist = self.model.autoencoder.decoder_lidar(latent_state_rsample)
        text_embed_dist = self.model.autoencoder.decoder_text(latent_state_rsample)

        # compute loss
        rec_loss = -lidar_obs_dist.log_prob(lidar_obses).mean()
        rec_loss -= text_embed_dist.log_prob(text_embeds).mean()

        kl_loss = (trans_latent_state_std.log() - latent_state_std.log()
                   + (latent_state_std.pow(2) + (trans_latent_state_mean - latent_state_mean).pow(2))
                   / (2 * trans_latent_state_std.pow(2) + 1e-5)).mean()

        ae_loss = 1.0 * rec_loss + 1.0 * kl_loss

        # align inferred state and RL state
        return ae_loss, rec_loss, kl_loss, latent_state_rsample.detach()

    def actor_loss(self, latent_state_sample, actions, expert_actions):
        # be really careful of the index, (s_t, r_t, a_{t-1})
        latent_state_samples = latent_state_sample[:-1].to(self.device).type(self.type).detach()
        expert_action = expert_actions[1:].to(self.device)
        action = actions[1:].to(self.device)

        with FreezeParameters(self.model_modules + self.critic_modules):
            feat_sac = latent_state_samples

            # Unbounded policy action and policy dist
            rsample_policy_action, policy_dist = self.model.policy(feat_sac)
            log_pi = policy_dist.log_prob(rsample_policy_action).sum(-1, keepdim=True)

            # Enforcing Action Bound
            log_pi -= torch.log((1 - torch.tanh(rsample_policy_action).pow(2)) + 1e-6).sum(-1, keepdim=True)

            rsample_policy_action = torch.tanh(rsample_policy_action)

            combined_feat = torch.cat([feat_sac, rsample_policy_action], dim=-1)
            q1_target = self.model.qf1_model(combined_feat).mean
            q2_target = self.model.qf2_model(combined_feat).mean
            q_target = torch.min(q1_target, q2_target)

            alpha_loss = (self.model.log_alpha.exp() * (-log_pi - self.model.target_entropy).detach()).mean()

        actor_loss = (self.model.log_alpha.exp().detach() * log_pi - q_target).mean()
        teacher_loss = torch.pow(expert_action.squeeze(0)[..., :] - rsample_policy_action.squeeze(0)[..., :], 2).sum(
            dim=-1).mean()

        actor_loss = (1 - self.beta) * actor_loss + self.beta * teacher_loss

        return actor_loss, alpha_loss

    def critic_loss(self, latent_state_sample, actions, rewards, dones):
        # be really careful of the index, (s_t, r_t, a_{t-1})
        latent_state_samples = latent_state_sample[:-1].to(self.device).type(self.type).detach()
        action = actions[1:].to(self.device).type(self.type)
        reward = rewards[:-1].to(self.device).type(self.type)
        done = dones[:-1].to(self.device).type(self.type)

        feat_sac = latent_state_samples[:-1]
        action_sac = action[:-1].detach()
        reward_sac = reward[:-1].detach()
        terminal = done[:-1].type(torch.int).detach()

        # Update target networks
        self.model.update_target_networks()
        combined_feat = torch.cat([feat_sac, action_sac], dim=-1)
        qf1_pred = self.model.qf1_model(combined_feat).mean
        qf2_pred = self.model.qf2_model(combined_feat).mean

        # Compute qf loss (without gradient)
        with torch.no_grad():
            target_next_feat_sac = latent_state_samples[1:]

            next_rsample_policy_action, next_policy_dist = self.model.policy(target_next_feat_sac)
            next_log_pi = next_policy_dist.log_prob(next_rsample_policy_action).sum(-1, keepdim=True)
            # Enforcing Action Bound
            next_log_pi -= torch.log((1 - torch.tanh(next_rsample_policy_action).pow(2)) + 1e-6).sum(-1, keepdim=True)
            next_rsample_policy_action = torch.tanh(next_rsample_policy_action)

            combined_next_feat = torch.cat([target_next_feat_sac, next_rsample_policy_action], dim=-1)
            target_value = torch.min(self.model.target_qf1_model(combined_next_feat).mean,
                                     self.model.target_qf2_model(combined_next_feat).mean) \
                           - self.model.log_alpha.exp().detach() * next_log_pi
            q_target = reward_sac + (1 - terminal.float()) * self.discount * target_value

        qf1_loss = torch.nn.functional.mse_loss(qf1_pred, q_target.type(self.type))
        qf2_loss = torch.nn.functional.mse_loss(qf2_pred, q_target.type(self.type))

        return qf1_loss, qf2_loss

    def reward_loss(self, latent_state_sample, actions, rewards, dones):
        # be really careful of the index, (s_t, r_t, a_{t-1})
        latent_state_samples = latent_state_sample[:-1].to(self.device).type(self.type).detach()
        action = actions[1:].to(self.device).type(self.type)
        reward = rewards[:-1].to(self.device).type(self.type)
        done = dones[:-1].to(self.device).type(self.type)

        feat_sac = latent_state_samples[:-1]
        reward_sac = reward[:-1].detach()

        # Update target networks
        self.model.update_target_networks()
        reward_pred = self.model.reward_model(feat_sac).mean

        reward_loss = torch.nn.functional.mse_loss(reward_pred, reward_sac.type(self.type))

        return reward_loss
