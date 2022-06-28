import torch
import numpy as np
import random

import torch
import numpy as np
import random


class ReplayBuffer(object):
    """Buffer to store and replay environment transitions."""

    def __init__(self, obs_shape, action_shape, reward_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        # Initialize all the buffers
        self.obs_buffer = np.empty(shape=(capacity, *obs_shape), dtype=np.float32)
        self.action_buffer = np.empty(shape=(capacity, *action_shape), dtype=np.float32)
        self.expert_buffer = np.empty(shape=(capacity, *action_shape), dtype=np.float32)
        self.reward_buffer = np.empty(shape=(capacity, *reward_shape), dtype=np.float32)
        self.done_buffer = np.empty(shape=(capacity, *reward_shape), dtype=np.float32)
        self.idx = 0

    def add(self, obs, action, expert_action, reward, done):
        if self.idx < self.capacity:
            self.obs_buffer[self.idx] = obs
            self.action_buffer[self.idx] = action
            self.expert_buffer[self.idx] = expert_action
            self.reward_buffer[self.idx] = reward
            self.done_buffer[self.idx] = done
            self.idx += 1
        else:
            self.obs_buffer = self.obs_buffer[1:]
            self.obs_buffer = np.append(self.obs_buffer,
                                        obs.reshape((1, obs.shape[0])),
                                        axis=0)
            self.action_buffer = self.action_buffer[1:]
            self.action_buffer = np.append(self.action_buffer,
                                           action.reshape((1, action.shape[0])),
                                           axis=0)
            self.expert_buffer = self.expert_buffer[1:]
            self.expert_buffer = np.append(self.expert_buffer,
                                           expert_action.reshape((1, expert_action.shape[0])),
                                           axis=0)
            self.reward_buffer = self.reward_buffer[1:]
            self.reward_buffer = np.append(self.reward_buffer,
                                           reward.reshape((1, reward.shape[0])),
                                           axis=0)
            self.done_buffer = self.done_buffer[1:]
            self.done_buffer = np.append(self.done_buffer,
                                         done.reshape((1, done.shape[0])),
                                         axis=0)

    def sample(self, time=30):
        idxs = np.random.randint(
            0, self.capacity - time + 1 if self.idx == self.capacity else self.idx - time + 1, size=self.batch_size)
        obses = torch.as_tensor(self.obs_buffer[idxs], device=self.device).unsqueeze(1)
        actions = torch.as_tensor(self.action_buffer[idxs], device=self.device).unsqueeze(1)
        expert_actions = torch.as_tensor(self.expert_buffer[idxs], device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(self.reward_buffer[idxs], device=self.device).unsqueeze(1)
        dones = torch.as_tensor(self.done_buffer[idxs], device=self.device).unsqueeze(1)

        for i in range(1, time):
            next_obses = torch.as_tensor(self.obs_buffer[idxs + i], device=self.device).unsqueeze(1)
            next_actions = torch.as_tensor(self.action_buffer[idxs + i], device=self.device).unsqueeze(1)
            next_expert_actions = torch.as_tensor(self.expert_buffer[idxs + i], device=self.device).unsqueeze(1)
            next_rewards = torch.as_tensor(self.reward_buffer[idxs + i], device=self.device).unsqueeze(1)
            next_dones = torch.as_tensor(self.done_buffer[idxs + i], device=self.device).unsqueeze(1)
            obses = torch.cat((obses, next_obses), 1)
            actions = torch.cat((actions, next_actions), 1)
            expert_actions = torch.cat((expert_actions, next_expert_actions), 1)
            rewards = torch.cat((rewards, next_rewards), 1)
            dones = torch.cat((dones, next_dones), 1)

        return obses, actions, expert_actions, rewards, dones

    def save(self):
        np.save('./current_model/obs_buffer.npy', self.obs_buffer)
        np.save('./current_model/action_buffer.npy', self.action_buffer)
        np.save('./current_model/expert_buffer.npy', self.expert_buffer)
        np.save('./current_model/reward_buffer.npy', self.reward_buffer)
        np.save('./current_model/done_buffer.npy', self.done_buffer)

    def load(self, dir):
        self.obs_buffer = np.load(dir + 'obs_buffer.npy')
        self.action_buffer = np.load(dir + 'action_buffer.npy')
        self.expert_buffer = np.load(dir + 'expert_buffer.npy')
        self.reward_buffer = np.load(dir + 'reward_buffer.npy')
        self.done_buffer = np.load(dir + 'done_buffer.npy')


class ReplayBufferText(object):
    """Buffer to store and replay environment transitions."""

    def __init__(self, obs_shape, text_shape, action_shape, reward_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        # Initialize all the buffers
        self.obs_buffer = np.empty(shape=(capacity, *obs_shape), dtype=np.float32)
        self.text_buffer = np.empty(shape=(capacity, *text_shape), dtype=np.float32)
        self.action_buffer = np.empty(shape=(capacity, *action_shape), dtype=np.float32)
        self.expert_buffer = np.empty(shape=(capacity, *action_shape), dtype=np.float32)
        self.reward_buffer = np.empty(shape=(capacity, *reward_shape), dtype=np.float32)
        self.done_buffer = np.empty(shape=(capacity, *reward_shape), dtype=np.float32)
        self.idx = 0

    def add(self, obs, text, action, expert_action, reward, done):
        if self.idx < self.capacity:
            self.obs_buffer[self.idx] = obs
            self.text_buffer[self.idx] = text
            self.action_buffer[self.idx] = action
            self.expert_buffer[self.idx] = expert_action
            self.reward_buffer[self.idx] = reward
            self.done_buffer[self.idx] = done
            self.idx += 1
        else:
            self.obs_buffer = self.obs_buffer[1:]
            self.obs_buffer = np.append(self.obs_buffer,
                                        obs.reshape((1, obs.shape[0])),
                                        axis=0)
            self.text_buffer = self.text_buffer[1:]
            self.text_buffer = np.append(self.text_buffer,
                                         text.reshape((1, text.shape[0])),
                                         axis=0)
            self.action_buffer = self.action_buffer[1:]
            self.action_buffer = np.append(self.action_buffer,
                                           action.reshape((1, action.shape[0])),
                                           axis=0)
            self.expert_buffer = self.expert_buffer[1:]
            self.expert_buffer = np.append(self.expert_buffer,
                                           expert_action.reshape((1, expert_action.shape[0])),
                                           axis=0)
            self.reward_buffer = self.reward_buffer[1:]
            self.reward_buffer = np.append(self.reward_buffer,
                                           reward.reshape((1, reward.shape[0])),
                                           axis=0)
            self.done_buffer = self.done_buffer[1:]
            self.done_buffer = np.append(self.done_buffer,
                                         done.reshape((1, done.shape[0])),
                                         axis=0)

    def sample(self, time=40):
        idxs = np.random.randint(
            0, self.capacity - time + 1 if self.idx == self.capacity else self.idx - time + 1, size=self.batch_size)
        obses = torch.as_tensor(self.obs_buffer[idxs], device=self.device).unsqueeze(1)
        texts = torch.as_tensor(self.text_buffer[idxs], device=self.device).unsqueeze(1)
        actions = torch.as_tensor(self.action_buffer[idxs], device=self.device).unsqueeze(1)
        expert_actions = torch.as_tensor(self.expert_buffer[idxs], device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(self.reward_buffer[idxs], device=self.device).unsqueeze(1)
        dones = torch.as_tensor(self.done_buffer[idxs], device=self.device).unsqueeze(1)

        for i in range(1, time):
            next_obses = torch.as_tensor(self.obs_buffer[idxs + i], device=self.device).unsqueeze(1)
            next_texts = torch.as_tensor(self.text_buffer[idxs + i], device=self.device).unsqueeze(1)
            next_actions = torch.as_tensor(self.action_buffer[idxs + i], device=self.device).unsqueeze(1)
            next_expert_actions = torch.as_tensor(self.expert_buffer[idxs + i], device=self.device).unsqueeze(1)
            next_rewards = torch.as_tensor(self.reward_buffer[idxs + i], device=self.device).unsqueeze(1)
            next_dones = torch.as_tensor(self.done_buffer[idxs + i], device=self.device).unsqueeze(1)
            obses = torch.cat((obses, next_obses), 1)
            texts = torch.cat((texts, next_texts), 1)
            actions = torch.cat((actions, next_actions), 1)
            expert_actions = torch.cat((expert_actions, next_expert_actions), 1)
            rewards = torch.cat((rewards, next_rewards), 1)
            dones = torch.cat((dones, next_dones), 1)

        return obses, texts, actions, expert_actions, rewards, dones

    def save(self):
        np.save('./current_model/obs_buffer.npy', self.obs_buffer)
        np.save('./current_model/text_buffer.npy', self.text_buffer)
        np.save('./current_model/action_buffer.npy', self.action_buffer)
        np.save('./current_model/expert_buffer.npy', self.expert_buffer)
        np.save('./current_model/reward_buffer.npy', self.reward_buffer)
        np.save('./current_model/done_buffer.npy', self.done_buffer)

    def load(self, dir):
        self.obs_buffer = np.load(dir + 'obs_buffer.npy')
        self.text_buffer = np.load(dir + 'text_buffer.npy')
        self.action_buffer = np.load(dir + 'action_buffer.npy')
        self.expert_buffer = np.load(dir + 'expert_buffer.npy')
        self.reward_buffer = np.load(dir + 'reward_buffer.npy')
        self.done_buffer = np.load(dir + 'done_buffer.npy')


class HumanDemonstrationReplayBuffer(object):
    """Buffer to store and replay environment transitions."""

    def __init__(self, lidar_obs_shape, text_embed_shape, human_action_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        # Initialize all the buffers
        self.lidar_obs_buffer = np.empty(shape=(capacity, *lidar_obs_shape), dtype=np.float32)
        self.text_embed_buffer = np.empty(shape=(capacity, *text_embed_shape), dtype=np.float32)
        self.human_action_buffer = np.empty(shape=(capacity, *human_action_shape), dtype=np.float32)
        self.idx = 0

    def add(self, lidar_obs, text_embed, human_action):
        if self.idx < self.capacity:
            self.lidar_obs_buffer[self.idx] = lidar_obs
            self.text_embed_buffer[self.idx] = text_embed
            self.human_action_buffer[self.idx] = human_action
            self.idx += 1
        else:
            self.lidar_obs_buffer = self.lidar_obs_buffer[1:]
            self.lidar_obs_buffer = np.append(self.lidar_obs_buffer,
                                              lidar_obs.reshape((1, lidar_obs.shape[0])),
                                              axis=0)
            self.text_embed_buffer = self.text_embed_buffer[1:]
            self.text_embed_buffer = np.append(self.text_embed_buffer,
                                               text_embed.reshape((1, text_embed.shape[0])),
                                               axis=0)

    def sample(self, time=30):
        idxs = np.random.randint(
            0, self.capacity - time + 1 if self.idx == self.capacity else self.idx - time + 1,
            size=self.batch_size)
        lidar_obs = torch.as_tensor(self.lidar_obs_buffer[idxs], device=self.device).unsqueeze(1)
        text_embed = torch.as_tensor(self.text_embed_buffer[idxs], device=self.device).unsqueeze(1)
        human_action = torch.as_tensor(self.human_action_buffer[idxs], device=self.device).unsqueeze(1)

        for i in range(1, time):
            next_lidar_obs = torch.as_tensor(self.lidar_obs_buffer[idxs + i], device=self.device).unsqueeze(1)
            next_text_embed = torch.as_tensor(self.text_embed_buffer[idxs + i], device=self.device).unsqueeze(1)
            next_humna_action = torch.as_tensor(self.human_action_buffer[idxs + i],
                                                device=self.device).unsqueeze(1)

            lidar_obs = torch.cat((lidar_obs, next_lidar_obs), 1)
            text_embed = torch.cat((text_embed, next_text_embed), 1)
            human_action = torch.cat((human_action, next_humna_action), 1)

        return lidar_obs, text_embed, human_action

    def save(self):
        raise NotImplementedError

    def load(self, path=None):
        if path is None:
            self.lidar_obs_buffer = np.load(f'./lidar_obs.npy')[:]
            self.text_embed_buffer = np.load(f'./text_embed.npy')[:]
            self.human_action_buffer = np.load(f'./human_actions.npy')[:]
        else:
            self.lidar_obs_buffer = np.load(f'{path}/lidar_obs.npy')[:]
            self.text_embed_buffer = np.load(f'{path}/text_embed.npy')[:]
            self.human_action_buffer = np.load(f'{path}/human_actions.npy')[:]

            # self.lidar_obs_buffer = np.concatenate([np.load(f'{path}/lidar_obs_0.npy'), np.load(f'{path}/lidar_obs_1.npy')], axis=0)
            # self.text_embed_buffer = np.concatenate([np.load(f'{path}/text_embed_0.npy'), np.load(f'{path}/text_embed_1.npy')], axis=0)
            # self.human_action_buffer = np.concatenate([np.load(f'{path}/human_actions_0.npy'), np.load(f'{path}/human_actions_1.npy')], axis=0)

        self.idx = self.lidar_obs_buffer.shape[0]
