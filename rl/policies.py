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


class PPOLSTMPolicy(Policy):
    """Pretrained masked RecurrentPPO (rl/recurrent.py). Nothing adapts in the weights at eval;
    the LSTM state carries memory within an episode (reset at game start and after GAME_OVER,
    as in training)."""
    name = "ppo-lstm"

    def __init__(self, ckpt=None, **kw):
        super().__init__(ckpt)
        import torch
        from recurrent import MaskedRecurrentPolicy, OBS2_SPACE, RPOLICY_KWARGS
        from sb3_nets import ACT_SPACE
        if ckpt is None:
            raise SystemExit("ppo-lstm needs --ckpt path/to/model.pt")
        self.torch = torch
        self.pi = MaskedRecurrentPolicy(OBS2_SPACE, ACT_SPACE, lambda _: 0.0, **RPOLICY_KWARGS)
        self.pi.load_state_dict(torch.load(Path(ckpt), weights_only=True, map_location="cpu"))
        self.pi.set_training_mode(False)

    def reset(self, game_id, seed):
        super().reset(game_id, seed)
        self.torch.manual_seed(seed)
        self.state, self.start = None, True

    def act(self, obs, mask, info) -> int:
        from recurrent import add_mask_channel
        o = add_mask_channel(obs, mask)[None]
        a, self.state = self.pi.predict(o, state=self.state, episode_start=np.array([self.start]),
                                        deterministic=False)
        self.start = False
        return int(a[0])

    def observe(self, obs, action, reward, next_obs, terminated, info):
        if terminated:
            self.start = True


class GraphExplorePolicy(Policy):
    """Model-based graph exploration (extra improvement, P3+/P4 track).

    Idea credit: the action-effect predictor (does action a change the frame in state s?) follows
    the ARC-AGI-3 preview winner "Stochastic Goose" (Dries Smit / Tufa Labs, 2025), which trained a
    CNN online to predict frame-changing actions and sampled from it. What WE implemented:
      * exact state graph: node = hash of the 64x64 frame; edges (s, a) -> s' learned online;
        actions that left the frame unchanged or led to GAME_OVER are marked and never re-tried;
      * frontier search: in the current state try untried legal actions first, ranked by the
        effect head's p(change) (pretrained in TEAM-12, fine-tuned online with BCE each step);
        if the current state has none left, BFS over known changing edges to the nearest state
        that still has untried actions and follow the path (undo/ACTION7 is just another action;
        the harness resets on GAME_OVER, there is no voluntary RESET in the frozen action space);
      * per-level memory: graph cleared when levels_completed increases; effect head carries over.
    ckpt = QEffNet model.pt (effect head). Uses only the observed frame + legal-action mask.
    """
    name = "graph"
    USE_EFFECT = True
    FINETUNE = dict(lr=1e-4, batch=32, start=32, buffer=1000)

    def __init__(self, ckpt=None, **kw):
        super().__init__(ckpt)
        import torch
        import nets
        self.torch = torch
        self.net = None
        if self.USE_EFFECT:
            if ckpt is None:
                raise SystemExit("graph policy needs --ckpt (QEffNet model.pt)")
            p = Path(ckpt)
            cfg = json.loads(p.with_suffix(".json").read_text()) if p.with_suffix(".json").exists() else {}
            if cfg.get("arch") != "QEffNet":
                raise SystemExit("graph policy needs a QEffNet checkpoint (effect head)")
            self.sd = torch.load(p, weights_only=True, map_location="cpu")
            self.net = nets.QEffNet()

    def reset(self, game_id, seed):
        super().reset(game_id, seed)
        self.torch.manual_seed(seed)
        self.clear_graph()
        self.levels = None
        self.prev = None                    # (hash, obs, action)
        if self.net is not None:
            self.net.load_state_dict(self.sd)
            self.net.eval()
            self.opt = self.torch.optim.Adam(self.net.parameters(), lr=self.FINETUNE["lr"])
            self.ebuf = []

    def clear_graph(self):
        self.tried = {}                     # hash -> set(actions tried)
        self.edges = {}                     # hash -> {action: next_hash} for frame-changing actions
        self.legal = {}                     # hash -> legal action indices
        self.bad = set()                    # (hash, action) no-op or deadly

    @staticmethod
    def h(obs):
        import hashlib
        return hashlib.blake2b(obs.tobytes(), digest_size=12).digest()

    def p_change(self, obs):
        if self.net is None:
            return np.full(70, 0.5)
        with self.torch.no_grad():
            _, e = self.net.both(self.torch.from_numpy(np.ascontiguousarray(obs[None])))
        return self.torch.sigmoid(e[0]).numpy()

    def untried(self, hs):
        return [a for a in self.legal.get(hs, []) if a not in self.tried.get(hs, set())]

    def path_to_frontier(self, start):
        from collections import deque
        q, first = deque([start]), {start: None}
        while q:
            u = q.popleft()
            if u != start and self.untried(u):
                while first[u][0] != start:
                    u = first[u][0]
                return first[u][1]
            for a, v in self.edges.get(u, {}).items():
                if v not in first and (u, a) not in self.bad:
                    first[v] = (u, a)
                    q.append(v)
        return None

    def act(self, obs, mask, info) -> int:
        lv = info.get("levels_completed", 0)
        if self.levels is not None and lv > self.levels:
            self.clear_graph()
        self.levels = lv
        hs = self.h(obs)
        self.legal[hs] = [int(a) for a in np.flatnonzero(mask)]
        cand = self.untried(hs)
        if cand:
            pc = self.p_change(obs)
            noise = self.rng.random(len(cand)) * 1e-3          # random tie-break
            return int(cand[int(np.argmax(pc[cand] + noise))])
        step = self.path_to_frontier(hs)
        if step is not None:
            return int(step)
        ok = [a for a in self.legal[hs] if (hs, a) not in self.bad] or self.legal[hs]
        return int(self.rng.choice(ok))

    def observe(self, obs, action, reward, next_obs, terminated, info):
        hs, hn = self.h(obs), self.h(next_obs)
        self.tried.setdefault(hs, set()).add(int(action))
        changed = hs != hn
        if terminated and info.get("state") == "GAME_OVER":
            self.bad.add((hs, int(action)))
        elif changed:
            self.edges.setdefault(hs, {})[int(action)] = hn
        else:
            self.bad.add((hs, int(action)))
        if self.net is None:
            return
        self.ebuf.append((obs, int(action), float(changed)))
        if len(self.ebuf) > self.FINETUNE["buffer"]:
            self.ebuf.pop(0)
        if len(self.ebuf) < self.FINETUNE["start"]:
            return
        import torch.nn.functional as F
        torch = self.torch
        idx = self.rng.integers(0, len(self.ebuf), self.FINETUNE["batch"])
        b = [self.ebuf[i] for i in idx]
        ob = torch.from_numpy(np.stack([x[0] for x in b]))
        a = torch.tensor([x[1] for x in b])
        y = torch.tensor([x[2] for x in b], dtype=torch.float32)
        self.net.train()
        _, e = self.net.both(ob)
        loss = F.binary_cross_entropy_with_logits(e.gather(1, a[:, None])[:, 0], y)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.net.eval()


class GraphStatsPolicy(GraphExplorePolicy):
    """Graph v2: same state-graph frontier exploration, but untried actions are ranked by a
    per-game ONLINE effect model instead of the pretrained effect head (which measured chance-level
    AUC ~0.45-0.58 on dev games). Beta(1,1) posteriors over change rates, reset every game,
    actions ranked by a Thompson sample:
      simple action i : P(change | action i)           (counts over this game so far)
      click cell k    : P(change | colour under the click point of k)
    Optional prior (subclass): a pretrained PPO policy's pi(a|s) breaks ties between equally
    promising actions. Only the observed frame is used."""
    name = "graph-stats"
    USE_EFFECT = False
    PRIOR_W = 0.0

    def __init__(self, ckpt=None, **kw):
        super().__init__(None)
        self.ckpt = ckpt
        self.prior = None
        if self.PRIOR_W > 0:
            if ckpt is None:
                raise SystemExit(f"{self.name} needs --ckpt (PPO model.pt) for its policy prior")
            self.prior = PPOPolicy(ckpt)
        from arc_gym import decode_action
        self.click_xy = [decode_action(a)[1] for a in range(70)]

    def reset(self, game_id, seed):
        super().reset(game_id, seed)
        self.stat = {}                      # key -> [changed, tried]

    def key(self, obs, a):
        xy = self.click_xy[a]
        return ("s", a) if xy is None else ("c", int(obs[0, xy["y"], xy["x"]]))

    def p_change(self, obs):
        # Thompson sampling from Beta(changed+1, unchanged+1): a posterior MEAN made the most-tried
        # movement action win every new state (ls20 smoke test: 95% one action); sampling keeps
        # equally effective actions in rotation while still demoting reliably ineffective ones.
        p = np.empty(70)
        for a in range(70):
            c, n = self.stat.get(self.key(obs, a), (0, 0))
            p[a] = self.rng.beta(c + 1.0, n - c + 1.0)
        if self.prior is not None:
            with self.torch.no_grad():
                o = self.torch.from_numpy(np.ascontiguousarray(obs[None]))
                pi = self.prior.pi.get_distribution(o).distribution.probs[0].numpy()
            p = p + self.PRIOR_W * pi / max(pi.max(), 1e-8)
        return p

    def changed(self, obs, next_obs):
        return not np.array_equal(obs, next_obs)

    def observe(self, obs, action, reward, next_obs, terminated, info):
        k = self.key(obs, int(action))
        c = self.stat.setdefault(k, [0, 0])
        c[0] += int(self.changed(obs, next_obs)); c[1] += 1
        super().observe(obs, action, reward, next_obs, terminated, info)


class GraphStatsPPOPolicy(GraphStatsPolicy):
    """Graph v2 + pretrained PPO (TEAM-21) policy prior for ordering untried actions."""
    name = "graph-stats-ppo"
    PRIOR_W = 0.3


class GraphClockPolicy(GraphStatsPPOPolicy):
    """TEAM-32: TEAM-31 + online "ticker row" abstraction. Evidence: every dev move/mixed game has a
    step-counter bar that advances 1 pixel per action (ls20 rows 61-62, dc22/ka59 row 63, lf52 row
    0), so every frame was unique (graph never recognised revisits) and in lf52 every action looked
    effective. Learned per game from observed frames only: after >= 10 frame-changing steps, a row is
    a ticker if it changed in >= 50% of changing steps, with median <= 2 changed pixels in that row,
    caused by >= 3 distinct actions. Ticker rows are zeroed before hashing and change detection.
    The ticker set only grows (an earlier flapping version kept wiping the graph: dc22 no-ops 0.15 ->
    0.40 in a smoke test); each addition rebuilds graph + effect statistics (early in a game)."""
    name = "graph-clock"
    MIN_STEPS, MIN_FRAC, MAX_PX, MIN_ACTS = 10, 0.5, 2, 3

    def reset(self, game_id, seed):
        super().reset(game_id, seed)
        self.r_cnt = np.zeros(64); self.r_px = [[] for _ in range(64)]
        self.r_acts = [set() for _ in range(64)]; self.n_chg = 0
        self.ticker = np.zeros(64, bool)
        self.rebuilds = 0

    def masked(self, obs):
        if not self.ticker.any():
            return obs
        o = obs.copy()
        o[0, self.ticker] = 0
        return o

    def h(self, obs):
        return GraphExplorePolicy.h(self.masked(obs))

    def changed(self, obs, next_obs):
        return not np.array_equal(self.masked(obs), self.masked(next_obs))

    def update_ticker(self, obs, action, next_obs):
        d = obs[0] != next_obs[0]
        if not d.any():
            return
        self.n_chg += 1
        rows = np.flatnonzero(d.any(1))
        for r in rows:
            self.r_cnt[r] += 1; self.r_px[r].append(int(d[r].sum())); self.r_acts[r].add(int(action))
        if self.n_chg < self.MIN_STEPS:
            return
        new = np.array([self.r_cnt[r] / self.n_chg >= self.MIN_FRAC and len(self.r_px[r]) > 0 and
                        np.median(self.r_px[r]) <= self.MAX_PX and len(self.r_acts[r]) >= self.MIN_ACTS
                        for r in range(64)])
        new |= self.ticker                  # monotone: a ticker row stays a ticker (no flapping)
        if (new != self.ticker).any():
            self.ticker = new
            self.rebuilds += 1
            self.clear_graph(); self.stat = {}

    def observe(self, obs, action, reward, next_obs, terminated, info):
        self.update_ticker(obs, action, next_obs)
        super().observe(obs, action, reward, next_obs, terminated, info)


class GraphClockPrunePolicy(GraphClockPolicy):
    """TEAM-33: TEAM-32 + pruning of confidently ineffective actions. Evidence (TEAM-32 seed-0
    diagnostic): once ticker rows are masked, states genuinely recur and the frontier search tries
    ALL ~70 legal actions in every recurring state (dc22 raw no-ops 0.15 -> 0.40). An untried action
    is skipped when this game's statistics for its key (simple action / clicked colour) show
    >= PRUNE_N tries with zero effect; if nothing promising is left, the agent moves to the frontier."""
    name = "graph-clock-prune"
    PRUNE_N = 3

    def untried(self, hs):
        out = []
        obs = self.obs_of.get(hs)
        for a in super().untried(hs):
            if obs is not None:
                c, n = self.stat.get(self.key(obs, a), (0, 0))
                if n >= self.PRUNE_N and c == 0:
                    continue
            out.append(a)
        return out

    def act(self, obs, mask, info):
        self.obs_of[self.h(obs)] = obs
        return super().act(obs, mask, info)

    def clear_graph(self):
        super().clear_graph()
        self.obs_of = {}


class GraphTimerPolicy(GraphClockPrunePolicy):
    """TEAM-34: TEAM-33 + timer-aware death handling. Evidence: on every dev move/mixed game GAME_OVER
    comes from a fixed per-life step budget (lives of ~129/125/100/65 steps on ls20/dc22/ka59/lf52
    under ANY policy; identical death counts for random, PPO and graph agents), so banning the last
    (state, action) before each death removed arbitrary, often useful edges. Now a (state, action)
    is banned as deadly only after it has led to GAME_OVER twice (real hazards repeat, timer deaths
    don't); a single death edge is kept as a normal (non-changing) edge."""
    name = "graph-timer"

    def clear_graph(self):
        super().clear_graph()
        self.deaths = {}

    def observe(self, obs, action, reward, next_obs, terminated, info):
        if terminated and info.get("state") == "GAME_OVER":
            k = (self.h(obs), int(action))
            self.deaths[k] = self.deaths.get(k, 0) + 1
            if self.deaths[k] < 2:
                info = dict(info, state="TIMEOUT_SUSPECTED")     # not banned by the base class
        GraphClockPrunePolicy.observe(self, obs, action, reward, next_obs, terminated, info)


class GraphNoEffectPolicy(GraphExplorePolicy):
    """Ablation: same graph exploration, untried actions in random order (no effect head)."""
    name = "graph-noeff"
    USE_EFFECT = False


POLICIES = {"graph": GraphExplorePolicy, "graph-noeff": GraphNoEffectPolicy,
            "graph-stats": GraphStatsPolicy, "graph-stats-ppo": GraphStatsPPOPolicy,
            "graph-clock": GraphClockPolicy, "graph-clock-prune": GraphClockPrunePolicy,
            "graph-timer": GraphTimerPolicy, "random": RandomPolicy, "dqn": DQNPolicy, "dqn-online": DQNOnlinePolicy,
            "ppo": PPOPolicy, "ppo-lstm": PPOLSTMPolicy}


def make_policy(name, **kw):
    if name not in POLICIES:
        raise SystemExit(f"unknown policy {name!r}; known: {sorted(POLICIES)}")
    return POLICIES[name](**kw)
