"""Action-effect model trained on a PPO agent's own training experience (graph-exploration track).

EffectVecEnv wraps the training VecEnv, sees each step's actions (step_async) and resulting
frames (step_wait) and trains a separate nets.QEffNet's effect head online with BCE on
"did action a change the frame in state s" (terminal transitions use info["terminal_observation"]).
It never touches the PPO policy or its rewards, so PPO training is unchanged (matched to TEAM-21).
Only the effect head is trained; the Q heads of the QEffNet are unused (kept so the saved file
loads directly into the graph-explorer policy).
"""
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.vec_env import VecEnvWrapper

from nets import QEffNet


class EffectVecEnv(VecEnvWrapper):
    def __init__(self, venv, lr=1e-4, batch=64, buffer=20000, device="cpu", seed=0):
        super().__init__(venv)
        torch.manual_seed(seed + 777)
        self.dev = torch.device(device)
        self.net = QEffNet().to(self.dev)
        self.opt = torch.optim.Adam([p for n, p in self.net.named_parameters()
                                     if n.startswith(("enc.", "esimple.", "eclick."))], lr=lr)
        self.rng = np.random.default_rng(seed)
        self.bo = np.zeros((buffer, 1, 64, 64), np.uint8)
        self.ba = np.zeros(buffer, np.int64)
        self.by = np.zeros(buffer, np.float32)
        self.n, self.i, self.cap, self.batch = 0, 0, buffer, batch
        self.prev_obs, self.actions = None, None
        self.loss_sum, self.loss_n, self.pos_sum = 0.0, 0, 0.0

    def reset(self):
        obs = self.venv.reset()
        self.prev_obs = obs[:, :1].copy()
        return obs

    def step_async(self, actions):
        self.actions = np.asarray(actions).copy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rew, dones, infos = self.venv.step_wait()
        nxt = obs[:, :1].copy()
        for k in range(len(nxt)):
            after = infos[k].get("terminal_observation") if dones[k] else None
            after = nxt[k] if after is None else np.asarray(after)[:1]
            changed = float(not np.array_equal(after, self.prev_obs[k]))
            self.bo[self.i], self.ba[self.i], self.by[self.i] = self.prev_obs[k], self.actions[k], changed
            self.i = (self.i + 1) % self.cap
            self.n = min(self.n + 1, self.cap)
            self.pos_sum += changed
        self.prev_obs = nxt
        if self.n >= self.batch:
            idx = self.rng.integers(0, self.n, self.batch)
            o = torch.from_numpy(self.bo[idx]).to(self.dev)
            a = torch.from_numpy(self.ba[idx]).to(self.dev)
            y = torch.from_numpy(self.by[idx]).to(self.dev)
            _, e = self.net.both(o)
            loss = F.binary_cross_entropy_with_logits(e.gather(1, a[:, None])[:, 0], y)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            self.loss_sum += loss.item(); self.loss_n += 1
        return obs, rew, dones, infos

    def save(self, path_pt, path_json, tag):
        import json
        torch.save({k: v.cpu() for k, v in self.net.state_dict().items()}, path_pt)
        open(path_json, "w").write(json.dumps(dict(algo="effect_model", arch="QEffNet", arch_kwargs={},
                                                   tag=tag, note="effect head only (Q heads untrained)"),
                                              indent=2))
