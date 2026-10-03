"""Masked recurrent PPO (P3.3): sb3-contrib RecurrentPPO + invalid-action masking.

sb3-contrib 2.9 has MaskablePPO and RecurrentPPO but no masked recurrent variant, so we add
masking ourselves (our code, not a library feature):
  * MaskChannel (gym wrapper) appends a 2nd observation channel whose first 70 cells hold the
    legal-action mask -> the mask travels with the observation through rollout buffer and
    minibatches, so it is available both when acting and in evaluate_actions;
  * MaskedEncoderExtractor feeds only channel 0 (the frame) to nets.Encoder;
  * MaskedRecurrentPolicy reads channel 1 in forward / get_distribution / evaluate_actions and
    sets illegal logits to -1e8 in _get_action_dist_from_latent.
The LSTM (sb3-contrib default, 1 layer, 256 units) sits between the encoder's 256-d global vector
and the linear actor/critic heads.
"""
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from nets import Encoder, N_ACTIONS

OBS2_SPACE = spaces.Box(0, 15, (2, 64, 64), np.uint8)


def add_mask_channel(obs, mask):
    ch = np.zeros((1, 64, 64), np.uint8)
    ch.reshape(-1)[:N_ACTIONS] = mask.astype(np.uint8)
    return np.concatenate([obs, ch], 0)


class MaskChannel(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = OBS2_SPACE

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return add_mask_channel(obs, self.env.action_masks()), info

    def step(self, a):
        obs, r, term, trunc, info = self.env.step(a)
        return add_mask_channel(obs, self.env.action_masks()), r, term, trunc, info

    def action_masks(self):
        return self.env.action_masks()


class MaskedEncoderExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, gdim=256):
        super().__init__(observation_space, features_dim=gdim)
        self.enc = Encoder(gdim=gdim)

    def forward(self, obs):
        return self.enc(obs[:, :1])[1]


def mask_from_obs(obs):
    return obs[:, 1].reshape(obs.shape[0], -1)[:, :N_ACTIONS] > 0.5


class MaskedRecurrentPolicy(RecurrentActorCriticPolicy):
    _mask = None

    def _get_action_dist_from_latent(self, latent_pi):
        logits = self.action_net(latent_pi)
        if self._mask is not None and self._mask.shape[0] == logits.shape[0]:
            logits = logits.masked_fill(~self._mask, -1e8)
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(self, obs, lstm_states, episode_starts, deterministic=False):
        self._mask = mask_from_obs(obs)
        return super().forward(obs, lstm_states, episode_starts, deterministic)

    def get_distribution(self, obs, lstm_states, episode_starts):
        self._mask = mask_from_obs(obs)
        return super().get_distribution(obs, lstm_states, episode_starts)

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
        self._mask = mask_from_obs(obs)
        return super().evaluate_actions(obs, actions, lstm_states, episode_starts)


RPOLICY_KWARGS = dict(features_extractor_class=MaskedEncoderExtractor,
                      features_extractor_kwargs=dict(gdim=256), normalize_images=False,
                      net_arch=dict(pi=[], vf=[]), lstm_hidden_size=256, n_lstm_layers=1)
