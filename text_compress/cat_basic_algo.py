import numpy as np
import torch
from rlpyt.algos.base import RlAlgorithm
from rlpyt.replays.sequence.n_step import SamplesFromReplay
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.tensor import infer_leading_dims
from tqdm import tqdm
import os

from models.latent import get_feat, get_dist, stack_states
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
        self.beta_decay = 0.995

    def optim_initialize(self, model):
        self.model = model
        self.model_modules = [model.autoencoder]
        # self.dynamics_modules = [model.transition,
        #                          model.representation,
        #                          model.observation_encoder]
        self.reward_modules = [model.reward_model]
        # model.transition,
        # model.representation,
        # model.observation_encoder]
        self.critic_modules = [model.qf1_model,
                               model.qf2_model]
        # model.representation,
        # model.observation_encoder]
        self.ae_modules = [model.autoencoder]

        self.actor_modules = [model.actor_model]
        self.alpha_modules = [model.log_alpha]

        self.model_optimizer = torch.optim.Adam(get_parameters(self.model_modules),
                                                lr=self.model_lr)
        # self.dynamics_optimizer = torch.optim.Adam(get_parameters(self.dynamics_modules),
        #                                            lr=self.dynamics_lr)
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
        # self.model_optimizer.load_state_dict(torch.load('./current_model/model_optim_latest.pt'))
        # self.reward_optimizer.load_state_dict(torch.load('./current_model/reward_optim_latest.pt'))
        # self.actor_optimizer.load_state_dict(torch.load('./current_model/actor_optim_latest.pt'))
        # self.alpha_optimizer.load_state_dict(torch.load('./current_model/alpha_optim_latest.pt'))
        # self.critic_optimizer.load_state_dict(torch.load('./current_model/critic_optim_latest.pt'))
        # self.ae_optimizer.load_state_dict(torch.load('./current_model/ae_optim_latest.pt'))
        #
        if self.initial_optim_state_dict is not None:
            self.load_optim_state_dict(self.initial_optim_state_dict)
        # must define these fields to for logging purposes. Used by runner.
        self.opt_info_fields = OptInfo._fields

    def optim_state_dict(self):
        """Return the optimizer state dict (e.g. Adam); overwrite if using
                multiple optimizers."""
        return dict(
            model_optimizer_dict=self.model_optimizer.state_dict(),
            actor_optimizer_dict=self.actor_optimizer.state_dict(),
            alpha_optimizer_dict=self.alpha_optimizer.state_dict(),
            value_optimizer_dict=self.critic_optimizer.state_dict(),
            # dynamics_optimizer_dict=self.dynamics_optimizer.state_dict(),
            reward_optimizer_dict=self.reward_optimizer.state_dict(),
            ae_optimizer_dict=self.ae_optimizer.state_dict(),
        )

    def load_optim_state_dict(self, state_dict):
        """Load an optimizer state dict; should expect the format returned
        from ``optim_state_dict().``"""
        self.model_optimizer.load_state_dict(state_dict['model_optimizer_dict'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer_dict'])
        self.alpha_optimizer.load_state_dict(state_dict['alpha_optimizer_dict'])
        self.critic_optimizer.load_state_dict(state_dict['value_optimizer_dict'])
        # self.dynamics_optimizer.load_state_dict(state_dict['dynamics_optimizer_dict'])
        self.reward_optimizer.load_state_dict(state_dict['reward_optimizer_dict'])
        self.ae_optimizer.load_state_dict(state_dict['ae_optimizer_dict'])

    def optimize_agent(self, model, replay_buffer, optim_itr):
        for i in tqdm(range(self.train_steps), desc='Imagination'):
            obses, actions, expert_actions, rewards, dones = replay_buffer.sample()
            obses = obses.transpose(0, 1)
            actions = actions.transpose(0, 1)
            expert_actions = expert_actions.transpose(0, 1)
            rewards = rewards.transpose(0, 1)
            dones = dones.transpose(0, 1)

            description = obses[:-1].to(self.device)
            description = description.type(self.type)

            batch_t = description.shape[0]
            batch_b = description.shape[1]
            mask_blur = torch.randint(low=0, high=2, size=(batch_t, batch_b, 12)).to(description.device)
            mask_blur[0, :, :] = 1
            mask_blur[:, :, 7:] = 1

            mask = [mask_blur[:, :, 0:1], mask_blur[:, :, 0:1], mask_blur[:, :, 0:1],  # 0 1 2
                    mask_blur[:, :, 1:2], mask_blur[:, :, 1:2], mask_blur[:, :, 1:2],  # 3 4 5
                    mask_blur[:, :, 2:3], mask_blur[:, :, 2:3], mask_blur[:, :, 2:3],  # 6 7 8
                    mask_blur[:, :, 3:4],  # 9
                    mask_blur[:, :, 4:5], mask_blur[:, :, 4:5], mask_blur[:, :, 4:5],  # 10 11 12
                    mask_blur[:, :, 5:6], mask_blur[:, :, 5:6], mask_blur[:, :, 5:6],  # 13 14 15
                    mask_blur[:, :, 6:7], mask_blur[:, :, 6:7], mask_blur[:, :, 6:7]]  # 16 17 18]

            mask += [mask_blur[:, :, 7:]]
            mask = torch.cat(mask, dim=-1)

            # train encoder
            ae_loss, rec_loss, kl_loss = self.ae_loss(obses, actions, rewards, mask)
            self.model_optimizer.zero_grad()
            (ae_loss).backward()
            grad_norm_model = torch.nn.utils.clip_grad_norm_(get_parameters(self.model_modules), self.grad_clip)
            self.model_optimizer.step()
            self.writer.add_scalar('Loss/rec_loss', rec_loss, self.itr)

            # train actor
            if i % self.actor_update_period == 0:
                actor_loss, alpha_loss = self.actor_loss(obses, actions, expert_actions, rewards, mask)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                grad_norm_actor = torch.nn.utils.clip_grad_norm_(get_parameters(self.actor_modules), self.grad_clip)
                self.actor_optimizer.step()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                grad_norm_alpha = torch.nn.utils.clip_grad_norm_(self.alpha_modules, self.grad_clip)
                self.alpha_optimizer.step()
                self.writer.add_scalar('Loss/Actor', actor_loss, self.itr)

            # train critics
            qf1_loss, qf2_loss = self.critic_loss(obses, actions, rewards, dones, mask)
            self.critic_optimizer.zero_grad()
            (qf1_loss + qf2_loss).backward()
            grad_norm_qf = torch.nn.utils.clip_grad_norm_(get_parameters(self.critic_modules), self.grad_clip)
            self.critic_optimizer.step()
            self.writer.add_scalar('Loss/Critic1', qf1_loss, self.itr)
            self.writer.add_scalar('Loss/Critic2', qf2_loss, self.itr)

            self.itr += 1
        # Save model and optimizer
        torch.save(self.model.state_dict(), './current_model/model_latest' + '.pt')
        torch.save(self.reward_optimizer.state_dict(), './current_model/reward_optim_latest' + '.pt')
        torch.save(self.actor_optimizer.state_dict(), './current_model/actor_optim_latest' + '.pt')
        torch.save(self.critic_optimizer.state_dict(), './current_model/critic_optim_latest' + '.pt')
        torch.save(self.alpha_optimizer.state_dict(), './current_model/alpha_optim_latest' + '.pt')
        torch.save(self.model_optimizer.state_dict(), './current_model/model_optim_latest' + '.pt')
        torch.save(self.ae_optimizer.state_dict(), './current_model/ae_optim_latest' + '.pt')
        if optim_itr % self.save_every == 0:
            torch.save(self.model.state_dict(), './current_model/model' + str(optim_itr) + '.pt')
            torch.save(self.reward_optimizer.state_dict(), './current_model/reward_optim' + str(optim_itr) + '.pt')
            torch.save(self.actor_optimizer.state_dict(), './current_model/actor_optim' + str(optim_itr) + '.pt')
            torch.save(self.critic_optimizer.state_dict(), './current_model/critic_optim' + str(optim_itr) + '.pt')
            torch.save(self.alpha_optimizer.state_dict(), './current_model/alpha_optim' + str(optim_itr) + '.pt')
            torch.save(self.alpha_optimizer.state_dict(), './current_model/model_optim' + str(optim_itr) + '.pt')
            torch.save(self.alpha_optimizer.state_dict(), './current_model/ae_optim' + str(optim_itr) + '.pt')
            if optim_itr >= (self.save_every * 5):
                os.remove('./current_model/model' + str(optim_itr - self.save_every * 5) + '.pt')
                os.remove('./current_model/reward_optim' + str(optim_itr - self.save_every * 5) + '.pt')
                os.remove('./current_model/actor_optim' + str(optim_itr - self.save_every * 5) + '.pt')
                os.remove('./current_model/critic_optim' + str(optim_itr - self.save_every * 5) + '.pt')
                os.remove('./current_model/alpha_optim' + str(optim_itr - self.save_every * 5) + '.pt')
                os.remove('./current_model/model_optim' + str(optim_itr - self.save_every * 5) + '.pt')
                os.remove('./current_model/ae_optim' + str(optim_itr - self.save_every * 5) + '.pt')
            replay_buffer.save()

        self.beta *= self.beta_decay

        return self.model

    def ae_loss(self, obses, actions, rewards, mask):
        # be really careful of the index, (s_t, r_t, a_{t-1})
        # observation = obses[:, :-1].squeeze(1).unsqueeze(0).to(self.device)
        # description = obses[:, :-1].squeeze(1).unsqueeze(0).to(self.device)
        # observation = observation.type(self.type)
        # description = description.type(self.type)
        # action = actions[:, :-1].squeeze(1).unsqueeze(0).to(self.device)
        # reward = rewards[:, :-1].squeeze(1).unsqueeze(0).to(self.device)

        observation = obses[:-1].to(self.device)
        description = obses[:-1].to(self.device)
        observation = observation.type(self.type)
        description = description.type(self.type)
        action = actions[1:].to(self.device)

        # lead_dim, batch_t, batch_b, description_shape = infer_leading_dims(description, 1)
        batch_t = observation.shape[0]

        # generate irrelevant description
        description_info = description

        # get latent_state
        latent_state_rsample = [[]] * batch_t
        latent_state_mean = [[]] * batch_t
        latent_state_std = [[]] * batch_t
        for t in range(batch_t):
            if t == 0:
                latent_state_dist = self.model.autoencoder.get_latent_state_dist(
                    description=description[t] * mask[t],
                    pre_state=None,
                    action=None,
                    mask=mask[t])
            else:
                latent_state_dist = self.model.autoencoder.get_latent_state_dist(
                    description=description[t] * mask[t],
                    pre_state=latent_state_rsample[t - 1],
                    action=action[t-1],
                    mask=mask[t])
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
                trans_latent_state_dist = self.model.autoencoder.transition(torch.cat([latent_state_rsample[t-1], action[t-1]], dim=-1))
                trans_latent_state_mean[t] = trans_latent_state_dist.mean
                trans_latent_state_std[t] = trans_latent_state_dist.stddev

        latent_state_rsample = torch.stack(latent_state_rsample, dim=0)
        latent_state_mean = torch.stack(latent_state_mean, dim=0)
        latent_state_std = torch.stack(latent_state_std, dim=0)

        trans_latent_state_mean = torch.stack(trans_latent_state_mean, dim=0)
        trans_latent_state_std = torch.stack(trans_latent_state_std, dim=0)

        # reconstruct observations
        normal_obs_dist = self.model.autoencoder.get_description_dist_normal(latent_state_rsample)

        # compute loss
        rec_loss = -normal_obs_dist.log_prob(description_info).mean()

        kl_loss = (trans_latent_state_std.log() - latent_state_std.log()
                   + (latent_state_std.pow(2) + (trans_latent_state_mean - latent_state_mean).pow(2))
                   / (2 * trans_latent_state_std.pow(2) + 1e-5)).mean()

        ae_loss = 1.0 * rec_loss + 1.0 * kl_loss

        # align inferred state and RL state
        return ae_loss, rec_loss, kl_loss

    def actor_loss(self, obses, actions, expert_actions, rewards, mask):
        # be really careful of the index, (s_t, r_t, a_{t-1})
        observation = obses[:-1].to(self.device)
        observation = observation.type(self.type)
        description = obses[:-1].to(self.device)
        description = description.type(self.type)
        # observation = torch.cat((observation[:, :, 0], observation[:, :, 1], observation[:, :, 2]), -1)
        expert_action = expert_actions[1:].to(self.device)
        action = actions[1:].to(self.device)

        # lead_dim, batch_t, batch_b, obs_shape = infer_leading_dims(observation, 1)
        batch_t = observation.shape[0]

        with FreezeParameters(self.model_modules + self.critic_modules):  # + self.dynamics_modules):
            latent_state_rsample = [[]] * batch_t
            for t in range(batch_t):
                if t == 0:
                    latent_state_dist = self.model.autoencoder.get_latent_state_dist(
                        description=description[t] * mask[t],
                        pre_state=None,
                        action=None,
                        mask=mask[t])
                else:
                    latent_state_dist = self.model.autoencoder.get_latent_state_dist(
                        description=description[t] * mask[t],
                        pre_state=latent_state_rsample[t - 1],
                        action=action[t-1],
                        mask=mask[t])
                latent_state_rsample[t] = latent_state_dist.rsample()
            latent_state_rsample = torch.stack(latent_state_rsample, dim=0)
            feat_sac = latent_state_rsample

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
        teacher_loss = torch.pow(expert_action.squeeze(0) - rsample_policy_action.squeeze(0), 2).sum(dim=-1).mean()

        actor_loss = actor_loss + self.beta * teacher_loss

        return actor_loss, alpha_loss

    def critic_loss(self, obses, actions, rewards, dones, mask):
        # be really careful of the index, (s_t, r_t, a_{t-1})
        observation = obses[:-1].to(self.device)
        observation = observation.type(self.type)
        description = obses[:-1].to(self.device)
        description = description.type(self.type)
        action = actions[1:].to(self.device)
        reward = rewards[:-1].to(self.device)
        done = dones[:-1].to(self.device)

        # lead_dim, batch_t, batch_b, obs_shape = infer_leading_dims(observation, 1)
        batch_t = observation.shape[0]

        latent_state_rsample = [[]] * batch_t
        for t in range(batch_t):
            if t == 0:
                latent_state_dist = self.model.autoencoder.get_latent_state_dist(
                    description=description[t] * mask[t],
                    pre_state=None,
                    action=None,
                    mask=mask[t])
            else:
                latent_state_dist = self.model.autoencoder.get_latent_state_dist(
                    description=description[t] * mask[t],
                    pre_state=latent_state_rsample[t - 1],
                    action=action[t-1],
                    mask=mask[t])
            latent_state_rsample[t] = latent_state_dist.rsample()
        latent_state_rsample = torch.stack(latent_state_rsample, dim=0)

        feat_sac = latent_state_rsample[:-1]
        action_sac = action[:-1].detach()
        reward_sac = reward[:-1].detach()
        terminal = done[:-1].type(torch.int).detach()

        # Update target networks
        self.model.update_target_networks()
        self.model.update_encoder()

        combined_feat = torch.cat([feat_sac, action_sac], dim=-1)
        qf1_pred = self.model.qf1_model(combined_feat).mean
        qf2_pred = self.model.qf2_model(combined_feat).mean

        # Compute qf loss (without gradient)
        with torch.no_grad():
            target_next_feat_sac = latent_state_rsample[1:]

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
