'''
Author: Sihong He
Date: 2019-01-25 19:25:30
LastEditTime: 2021-02-24 16:21:11
LastEditors: Sihong He
Description: In User Settings Edit
FilePath: /Robust-MARL/maddpg/trainer/replay_buffer.py
'''
import numpy as np
import random
from scipy.stats import truncnorm  

# to-do: generate noise following truncated normal distribution
def truncated_normal(mean = 0.0, std = 1.0, threshold = 1.0):
    lower, upper = -threshold, threshold
    mu, sigma = mean, std
    X = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    return X

class ReplayBuffer(object):
    def __init__(self, size, noise_type = 0, noise_std = 1.0):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0
        self._noise_type = noise_type
        self._noise_std = noise_std
        self.X = truncated_normal(std = self._noise_std)

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done):
        # add noise
        if self._noise_type == 1:#reward
            reward = reward + self.X.rvs(1)[0]
        elif self._noise_type == 2:#action
            temp = [act + self.X.rvs(1)[0] for act in action]
            action = temp
        elif self._noise_type == 3:#observation
            temp = [obs + self.X.rvs(1)[0] for obs in obs_t]
            obs_t = temp
                
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)
