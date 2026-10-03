"""Random Network Distillation (Burda et al. 2019) for SB3 PPO (P3.2).

RNDVecEnv wraps the training VecEnv: for every new observation it computes the predictor's error
against a fixed random target network (both nets.Encoder -> 128-d; target orthogonally
initialised and its features standardised per dimension with running stats), adds
  r = r_ext + coef * err / ema_mean(err)
to the reward, and trains the predictor online on the same observations (MSE, Adam).
The extrinsic reward is kept in info["r_ext"] so logging counts levels correctly.

Simplification (documented): one combined reward stream and one value head (Burda et al. use
separate intrinsic/extrinsic value heads with non-episodic intrinsic returns).
RND is training-only; the evaluated PPO policy does not use it (pre-registered PPO convention).
"""
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import VecEnvWrapper

from nets import Encoder


class RNDNet(nn.Module):
    def __init__(self, out=128, head=False):
        super().__init__()
        self.enc = Encoder(gdim=256)
        layers = [nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, out)] if head else [nn.Linear(256, out)]
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(self.enc(x)[1])


class EMAMean:
    """Exponential-moving mean of the RND error, used as the normaliser: r_int = coef * err / ema.
    Keeps the AVERAGE bonus at ~coef per step however fast the predictor's error falls (an
    all-history or EMA std normaliser let the bonus collapse to ~1e-4 within 4k steps in smoke
    tests), while states more novel than average still get proportionally more."""

    def __init__(self, alpha=0.01):
        self.alpha, self.v = alpha, None

    def update(self, x):
        m = float(np.mean(x))
        self.v = m if self.v is None else self.v + self.alpha * (m - self.v)

    @property
    def scale(self):
        return self.v + 1e-8


class RNDVecEnv(VecEnvWrapper):
    def __init__(self, venv, coef=0.05, lr=1e-4, device="cpu", seed=0):
        super().__init__(venv)
        torch.manual_seed(seed + 12345)
        self.dev = torch.device(device)
        self.target = RNDNet(head=False).to(self.dev).eval()
        for m in self.target.modules():          # Burda et al.: orthogonal init, gain sqrt(2)
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2)); nn.init.zeros_(m.bias)
        for p in self.target.parameters():
            p.requires_grad_(False)
        # running per-dim standardisation of target features: ARC frames are mostly identical
        # background, so raw random features barely vary and the predictor matched them within
        # ~90 steps (error 2e-3 -> 1e-6, seen in debugging); standardising restores contrast.
        self.t_mu = torch.zeros(128, device=self.dev)
        self.t_var = torch.ones(128, device=self.dev)
        self.t_init = False
        self.pred = RNDNet(head=True).to(self.dev)
        self.opt = torch.optim.Adam(self.pred.parameters(), lr=lr)
        self.coef, self.rs = coef, EMAMean()
        self.int_sum, self.int_n = 0.0, 0

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rew, dones, infos = self.venv.step_wait()
        x = torch.from_numpy(np.ascontiguousarray(obs)).to(self.dev)
        with torch.no_grad():
            raw = self.target(x)
            if not self.t_init:
                self.t_mu, self.t_var, self.t_init = raw.mean(0), raw.var(0) + 1e-6, True
            else:
                self.t_mu += 0.01 * (raw.mean(0) - self.t_mu)
                self.t_var += 0.01 * (((raw - self.t_mu) ** 2).mean(0) - self.t_var)
            tgt = (raw - self.t_mu) / (self.t_var.sqrt() + 1e-4)
        err = (self.pred(x) - tgt).pow(2).mean(1)
        loss = err.mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        e = err.detach().cpu().numpy()
        self.rs.update(e)
        r_int = self.coef * e / self.rs.scale
        for info, r in zip(infos, rew):
            info["r_ext"] = float(r)
        self.int_sum += float(r_int.sum()); self.int_n += len(r_int)
        return obs, (rew + r_int).astype(np.float32), dones, infos
