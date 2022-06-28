import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as tf
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.buffer import buffer_method

BisimState = namedarraytuple('BisimState', ['sample', 'mean', 'std'])


def stack_states(bisim_states: list, dim):
    return BisimState(
        torch.stack([state.sample for state in bisim_states], dim=dim),
        torch.stack([state.mean for state in bisim_states], dim=dim),
        torch.stack([state.std for state in bisim_states], dim=dim),
    )

def get_feat(bisim_state: BisimState):
    return bisim_state.mean

def get_dist(bisim_state: BisimState):
    return td.independent.Independent(td.Normal(bisim_state.mean, bisim_state.std), 1)

def get_dist_detach(bisim_state: BisimState):
    return td.independent.Independent(td.Normal(bisim_state.mean.detach(), bisim_state.std.detach()), 1)

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


class BisimTransition(nn.Module):
    def __init__(self, action_size, latent_size=30, hidden_size=200, activation=nn.ReLU,
                 distribution=td.Normal):
        super().__init__()
        self._action_size = action_size
        self._latent_size = latent_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._transition_model = self._build_transition_model()
        self._dist = distribution

    def _build_transition_model(self, layers=2):
        transition_model = [nn.Linear(self._action_size + self._latent_size, self._hidden_size)]
        transition_model += [self._activation()]
        for i in range(layers):
            transition_model += [nn.Linear(self._hidden_size, self._hidden_size)]
            transition_model += [self._activation()]
        transition_model += [nn.Linear(self._hidden_size, 2 * self._latent_size)]
        return nn.Sequential(*transition_model)

    def forward(self, prev_action: torch.Tensor, prev_state: BisimState):
        transition_input = torch.cat([get_feat(prev_state), prev_action], dim=-1)
        next_state = self._transition_model(transition_input)
        mean, std = torch.chunk(next_state, 2, dim=-1)
        std = tf.softplus(std) + 0.1
        dist = self._dist(mean, std)
        sample = dist.rsample()
        return BisimState(sample, mean, std)


class BisimRepresentation(nn.Module):
    def __init__(self, obs_embed_size, latent_size=30, hidden_size=200, activation=nn.ReLU, distribution=td.Normal):
        super().__init__()
        self._obs_embed_size = obs_embed_size
        self._latent_size = latent_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._dist = distribution
        self._latent_embed_model = self._build_latent_embed_model()

    def _build_latent_embed_model(self, layers=2):
        latent_embed_model = [nn.Linear(self._obs_embed_size, self._hidden_size)]
        latent_embed_model += [self._activation()]
        for i in range(layers):
            latent_embed_model += [nn.Linear(self._hidden_size, self._hidden_size)]
            latent_embed_model += [self._activation()]
        latent_embed_model += [nn.Linear(self._hidden_size, 2 * self._latent_size)]
        return nn.Sequential(*latent_embed_model)

    def forward(self, obs_embed: torch.Tensor, prev_action: torch.Tensor = None, prev_state: BisimState = None):
        latent_state = self._latent_embed_model(obs_embed)
        mean, std = torch.chunk(latent_state, 2, dim=-1)
        std = tf.softplus(std) + 0.1
        dist = self._dist(mean, std)
        sample = dist.rsample()
        return BisimState(sample, mean, std)


class BisimRollout(nn.Module):
    def __init__(self, representation_model: BisimRepresentation,
                 target_representation_model: BisimRepresentation,
                 transition_model: BisimTransition):
        super().__init__()
        self.representation_model = representation_model
        self.target_representation_model = target_representation_model
        self.transition_model = transition_model

    def forward(self, steps: int, obs_embed: torch.Tensor, action: torch.Tensor, prev_state: BisimState):
        return self.rollout_representation(steps, obs_embed, action, prev_state)

    def rollout_representation(self, steps: int, obs_embed: torch.Tensor, action: torch.Tensor,
                               prev_state: BisimState, is_target=False):
        """
        Roll out the model with actions and observations from data.
        :param steps: number of steps to roll out
        :param obs_embed: size(time_steps, batch_size, embedding_size)
        :param action: size(time_steps, batch_size, action_size)
        :param prev_state: RSSM state, size(batch_size, state_size)
        :return: prior, posterior states. size(time_steps, batch_size, state_size)
        """
        posteriors = []
        for t in range(steps):
            if is_target:
                latent_state = self.target_representation_model(obs_embed[t], None, None)
            else:
                latent_state = self.representation_model(obs_embed[t], None, None)
            posteriors.append(latent_state)
            prev_state = latent_state
        return stack_states(posteriors, dim=0)

    def rollout_transition(self, steps: int, action: torch.Tensor, prev_state: BisimState):
        """
        Roll out the model with actions from data.
        :param steps: number of steps to roll out
        :param action: size(time_steps, batch_size, action_size)
        :param prev_state: RSSM state, size(batch_size, state_size)
        :return: prior states. size(time_steps, batch_size, state_size)
        """
        priors = []
        for t in range(steps):
            latent_state = self.transition_model(action[t], prev_state)
            priors.append(latent_state)
            prev_state = latent_state
        return stack_states(priors, dim=0)

    def rollout_policy(self, steps: int, policy, prev_state: BisimState):
        """
        Roll out the model with a policy function.
        :param steps: number of steps to roll out
        :param policy: RSSMState -> action
        :param prev_state: RSSM state, size(batch_size, state_size)
        :return: next states size(time_steps, batch_size, state_size),
                 actions size(time_steps, batch_size, action_size)
        """
        latent_state = prev_state
        next_latent_states = []
        actions = []
        latent_state = buffer_method(latent_state, 'detach')
        for t in range(steps):
            action, _ = policy(buffer_method(latent_state, 'detach'))
            latent_state = self.transition_model(action, latent_state)
            next_latent_states.append(latent_state)
            actions.append(action)
        next_latent_states = stack_states(next_latent_states, dim=0)
        actions = torch.stack(actions, dim=0)
        return next_latent_states, actions
