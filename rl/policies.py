"""Policy interface used by the eval harness. Every agent (random, DQN, PPO) implements this.

Persistence (declared for the report): one Policy object lives for one seed's
whole run across all games; reset() is called at the start of each game.
observe() is the online-adaptation hook (no-op unless an agent learns at eval time).
"""
import json
from pathlib import Path

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


class DQNPolicy(Policy):
    """Pretrained Q-network (rl/train_dqn.py), epsilon-greedy over legal actions.
    ckpt = path to model.pt; model.json next to it holds arch + eval config.
    Frozen: weights fixed during evaluation."""
    name = "dqn"
    online = False
    # test-time TD learning (dqn-online only); weights restored to pretrained at every game
    ONLINE = dict(lr=1e-4, batch=32, start=32, gamma=0.99, target_every=100, buffer=1000)

    def __init__(self, ckpt=None, **kw):
        super().__init__(ckpt)
        import torch
        import nets
        if ckpt is None:
            raise SystemExit("dqn policies need --ckpt path/to/model.pt")
        p = Path(ckpt)
        cfg_path = p.with_suffix(".json")
        self.cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        self.eps = float(self.cfg.get("eval_eps", 0.05))
        self.torch = torch
        self.sd = torch.load(p, weights_only=True, map_location="cpu")
        self.Net = getattr(nets, self.cfg.get("arch", "QNet"))
        self.net = self.Net(**self.cfg.get("arch_kwargs", {}))
        self.net.load_state_dict(self.sd)
        self.tau = float(self.cfg.get("effect_tau", 0.0))   # P2.3 effect masking (QEffNet only)
        self.use_eff = self.tau > 0 and hasattr(self.net, "both")
        self.net.eval()

    def reset(self, game_id, seed):
        super().reset(game_id, seed)
        self.torch.manual_seed(seed)
        self.noop = set()             # per-game exact memory: (frame hash, action) seen to do nothing
        if self.online:
            self.net.load_state_dict(self.sd)
            self.tgt = self.Net(**self.cfg.get("arch_kwargs", {}))
            self.tgt.load_state_dict(self.sd)
            self.opt = self.torch.optim.Adam(self.net.parameters(), lr=self.ONLINE["lr"])
            self.buf, self.n_obs = [], 0
        self.counts = {}            # per-game visit counts for the novelty bonus (P2.2)
        self.beta = float(self.cfg.get("novelty_beta", 0.0))

    def q_values(self, obs):
        with self.torch.no_grad():
            return self.net(self.torch.from_numpy(np.ascontiguousarray(obs[None])))[0].numpy()

    def effect_mask(self, obs, mask):
        """Legal actions minus those predicted (p < tau) or remembered to leave the frame unchanged."""
        h = obs.tobytes()
        keep = mask.copy()
        for a in np.flatnonzero(keep):
            if (h, a) in self.noop:
                keep[a] = False
        with self.torch.no_grad():
            q, e = self.net.both(self.torch.from_numpy(np.ascontiguousarray(obs[None])))
        keep &= self.torch.sigmoid(e[0]).numpy() >= self.tau
        return (keep if keep.any() else mask), q[0].numpy()

    def act(self, obs, mask, info) -> int:
        if self.use_eff:
            mask, q = self.effect_mask(obs, mask)
        legal = np.flatnonzero(mask)
        if self.rng.random() < self.eps:
            return int(self.rng.choice(legal))
        if not self.use_eff:
            q = self.q_values(obs)
        return int(legal[np.argmax(q[legal])])

    def observe(self, obs, action, reward, next_obs, terminated, info):
        changed = not np.array_equal(obs, next_obs)
        if self.use_eff and not changed:
            self.noop.add((obs.tobytes(), int(action)))
        if not self.online:
            return
        import torch.nn.functional as F
        torch, o = self.torch, self.ONLINE
        if self.beta > 0:             # same bonus as in training, counts over the whole game
            h = next_obs.tobytes()
            c = self.counts[h] = self.counts.get(h, 0) + 1
            reward = reward + self.beta / np.sqrt(c)
        self.buf.append((obs, action, reward, next_obs, float(terminated), float(changed)))
        if len(self.buf) > o["buffer"]:
            self.buf.pop(0)
        self.n_obs += 1
        if len(self.buf) < o["start"]:
            return
        idx = self.rng.integers(0, len(self.buf), o["batch"])
        b = [self.buf[i] for i in idx]
        ob = torch.from_numpy(np.stack([x[0] for x in b]))
        a = torch.tensor([x[1] for x in b])
        r = torch.tensor([x[2] for x in b], dtype=torch.float32)
        nob = torch.from_numpy(np.stack([x[3] for x in b]))
        d = torch.tensor([x[4] for x in b])
        self.net.train()
        if self.use_eff:
            q_all, e_all = self.net.both(ob)
            q = q_all.gather(1, a[:, None])[:, 0]
            chg = torch.tensor([x[5] for x in b])
            eff_loss = F.binary_cross_entropy_with_logits(e_all.gather(1, a[:, None])[:, 0], chg)
        else:
            q = self.net(ob).gather(1, a[:, None])[:, 0]
        with torch.no_grad():
            y = r + o["gamma"] * (1 - d) * self.tgt(nob).max(1).values
        loss = F.smooth_l1_loss(q, y)
        if self.use_eff:
            loss = loss + eff_loss
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()
        self.net.eval()
        if self.n_obs % o["target_every"] == 0:
            self.tgt.load_state_dict(self.net.state_dict())


class DQNOnlinePolicy(DQNPolicy):
    """Same as dqn, plus test-time TD learning on the game being played (observe() hook)."""
    name = "dqn-online"
    online = True


class PPOPolicy(Policy):
    """Pretrained MaskablePPO actor (rl/train_ppo.py), sampling from the masked policy.
    ckpt = model.pt (policy state_dict); rebuilt from sb3_nets.POLICY_KWARGS, no pickles."""
    name = "ppo"

    def __init__(self, ckpt=None, **kw):
        super().__init__(ckpt)
        import torch
        from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
        from sb3_nets import ACT_SPACE, OBS_SPACE, POLICY_KWARGS
        if ckpt is None:
            raise SystemExit("ppo policies need --ckpt path/to/model.pt")
        p = Path(ckpt)
        cfg_path = p.with_suffix(".json")
        self.cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        self.torch = torch
        self.pi = MaskableActorCriticPolicy(OBS_SPACE, ACT_SPACE, lambda _: 0.0, **POLICY_KWARGS)
        self.pi.load_state_dict(torch.load(p, weights_only=True, map_location="cpu"))
        self.pi.set_training_mode(False)

    def reset(self, game_id, seed):
        super().reset(game_id, seed)
        self.torch.manual_seed(seed)

    def act(self, obs, mask, info) -> int:
        with self.torch.no_grad():
            o = self.torch.from_numpy(np.ascontiguousarray(obs[None]))
            dist = self.pi.get_distribution(o, action_masks=mask[None])
            a = dist.mode() if self.cfg.get("deterministic") else dist.sample()
        return int(a[0])


POLICIES = {"random": RandomPolicy, "dqn": DQNPolicy, "dqn-online": DQNOnlinePolicy,
            "ppo": PPOPolicy}


def make_policy(name, **kw):
    if name not in POLICIES:
        raise SystemExit(f"unknown policy {name!r}; known: {sorted(POLICIES)}")
    return POLICIES[name](**kw)
