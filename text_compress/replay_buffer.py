import torch
import numpy as np
import random


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

    def sample(self, time=30):
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
