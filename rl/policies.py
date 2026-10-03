"""Policy interface used by the eval harness. Every agent (random, DQN, PPO) implements this.

Persistence (declared for the report): one Policy object lives for one seed's
whole run across all games; reset() is called at the start of each game.
observe() is the online-adaptation hook (no-op unless an agent learns at eval time).
"""
import numpy as np


class Policy:
    name = "base"

    def __init__(self, ckpt=None):
        self.ckpt = ckpt

    def reset(self, game_id: str, seed: int):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, mask, info) -> int:
        raise NotImplementedError

    def observe(self, obs, action, reward, next_obs, terminated, info):
        pass


class RandomPolicy(Policy):
    """Sanity-check reference: uniform over currently legal actions."""
    name = "random"

    def act(self, obs, mask, info) -> int:
        return int(self.rng.choice(np.flatnonzero(mask)))


POLICIES = {"random": RandomPolicy}


def make_policy(name, **kw):
    if name not in POLICIES:
        raise SystemExit(f"unknown policy {name!r}; known: {sorted(POLICIES)}")
    return POLICIES[name](**kw)
