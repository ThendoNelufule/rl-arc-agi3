"""SB3 glue: the shared nets.Encoder as a features extractor + the PPO policy kwargs.
Kept separate from train_ppo.py so the Kaggle bundle can rebuild the policy without training code."""
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from nets import Encoder, N_ACTIONS


class EncoderExtractor(BaseFeaturesExtractor):
    """nets.Encoder -> 256-d global vector (uint8 colour ids in, one-hot inside)."""

    def __init__(self, observation_space, gdim=256):
        super().__init__(observation_space, features_dim=gdim)
        self.enc = Encoder(gdim=gdim)

    def forward(self, obs):
        return self.enc(obs)[1]


POLICY_KWARGS = dict(features_extractor_class=EncoderExtractor,
                     features_extractor_kwargs=dict(gdim=256),
                     normalize_images=False, net_arch=dict(pi=[], vf=[]))

OBS_SPACE = spaces.Box(0, 15, (1, 64, 64), np.uint8)
ACT_SPACE = spaces.Discrete(N_ACTIONS)
