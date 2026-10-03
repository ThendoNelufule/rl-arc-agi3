"""PPO training on the 15 train games (P3.x), SB3 2.9 MaskablePPO (sb3-contrib).

TEAM-20-ppo-base: MaskablePPO with the shared nets.Encoder as features extractor (same encoder
as DQN; SB3's standard linear actor/critic heads on the 256-d global vector), extrinsic reward
only, invalid-action masking from ArcGymEnv.action_masks(). 15 envs (one per train game) in a
DummyVecEnv; matched to DQN at 400k env steps.

Outputs (experiments/train/<tag>/): config.json, metrics.csv, train_summary.json,
model.pt (policy state_dict only, no pickles), model.json (arch + eval config).

Usage: .venv/bin/python rl/train_ppo.py --tag TEAM-20-ppo-base --steps 400000
"""
import argparse, csv, json, logging, platform, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
logging.disable(logging.INFO)
from sb3_contrib import MaskablePPO, RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from arc_gym import ArcGymEnv, get_arcade
from recurrent import MaskChannel, MaskedRecurrentPolicy, RPOLICY_KWARGS
from effect import EffectVecEnv
from rnd import RNDVecEnv
from sb3_nets import POLICY_KWARGS
from train_dqn import hardware

ROOT = Path(__file__).resolve().parents[1]


class LogCallback(BaseCallback):
    def __init__(self, path, every):
        super().__init__()
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["env_steps", "levels_total", "episodes", "frame_change_rate",
                         "entropy", "value_loss", "approx_kl", "sps", "wall_s", "intrinsic_per_step"])
        self.every, self.next, self.levels, self.eps = every, every, 0, 0
        self.chg, self.n, self.t0, self.prev = 0, 0, time.time(), None
        self.per_game = {}

    def _intrinsic(self):
        env = self.model.get_env()
        if not isinstance(env, RNDVecEnv):
            return 0.0
        v = env.int_sum / max(env.int_n, 1)
        env.int_sum, env.int_n = 0.0, 0
        return round(v, 5)

    def _on_step(self):
        obs = self.locals["new_obs"]
        rew, dones, infos = self.locals["rewards"], self.locals["dones"], self.locals["infos"]
        obs = obs[:, :1]                         # frame channel only (recurrent obs has a mask channel)
        if self.prev is not None:
            ch = ~np.all(obs.reshape(len(obs), -1) == self.prev.reshape(len(obs), -1), axis=1)
            self.chg += int(ch[~dones].sum()); self.n += int((~dones).sum())
        self.prev = obs.copy()
        for r, d, info in zip(rew, dones, infos):
            r = info.get("r_ext", r)            # extrinsic only (RND adds intrinsic to rewards)
            g = info.get("game_id", "?")
            pg = self.per_game.setdefault(g, dict(levels=0, episodes=0))
            if r > 0:
                self.levels += int(r); pg["levels"] += int(r)
            if d:
                self.eps += 1; pg["episodes"] += 1
        if self.num_timesteps >= self.next:
            self.next += self.every
            lg = self.model.logger.name_to_value
            wall = time.time() - self.t0
            row = [self.num_timesteps, self.levels, self.eps, round(self.chg / max(self.n, 1), 4),
                   round(-lg.get("train/entropy_loss", 0.0), 4), round(lg.get("train/value_loss", 0.0), 6),
                   round(lg.get("train/approx_kl", 0.0), 5), round(self.num_timesteps / wall, 1),
                   round(wall, 1), self._intrinsic()]
            self.w.writerow(row); self.f.flush()
            self.chg = self.n = 0
            print("t={} levels={} eps={} chg={} ent={} vloss={} kl={} sps={} wall={} int={}".format(*row),
                  flush=True)
        return True


def make_env_fns(games, ep_len, arc, mask_channel=False):
    if mask_channel:
        return [(lambda g=g: MaskChannel(ArcGymEnv(g, max_steps=ep_len, arcade=arc))) for g in games]
    return [(lambda g=g: ArcGymEnv(g, max_steps=ep_len, arcade=arc)) for g in games]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--steps", type=int, default=400_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ep-len", type=int, default=500)
    p.add_argument("--n-steps", type=int, default=128, help="rollout length per env")
    p.add_argument("--batch", type=int, default=240)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--log-every", type=int, default=10_000)
    p.add_argument("--rnd-coef", type=float, default=0.0,
                   help="P3.2 RND intrinsic reward coefficient (0 = off)")
    p.add_argument("--effect-model", action="store_true",
                   help="also train a separate action-effect model on the agent's experience")
    p.add_argument("--recurrent", action="store_true",
                   help="P3.3 masked RecurrentPPO (LSTM 256) instead of MaskablePPO")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out = ROOT / "experiments" / "train" / args.tag
    if out.exists() and not args.overwrite:
        sys.exit(f"{out} exists; pick a new --tag (or --overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    games = json.loads((ROOT / "splits.json").read_text())["train"]
    venv = DummyVecEnv(make_env_fns(games, args.ep_len, get_arcade(), mask_channel=args.recurrent))
    venv.seed(args.seed)
    eff = None
    if args.effect_model:
        venv = eff = EffectVecEnv(venv, device=args.device, seed=args.seed)
    if args.rnd_coef > 0:
        venv = RNDVecEnv(venv, coef=args.rnd_coef, device=args.device, seed=args.seed)
    algo_cls, pol, kw = ((RecurrentPPO, MaskedRecurrentPolicy, RPOLICY_KWARGS) if args.recurrent
                         else (MaskablePPO, "CnnPolicy", POLICY_KWARGS))
    model = algo_cls(pol, venv, n_steps=args.n_steps, batch_size=args.batch,
                     n_epochs=args.epochs, learning_rate=args.lr, gamma=args.gamma,
                     gae_lambda=args.gae, clip_range=args.clip, ent_coef=args.ent,
                     policy_kwargs=kw, seed=args.seed, device=args.device, verbose=0)
    cb = LogCallback(out / "metrics.csv", args.log_every)
    t0 = time.time()
    model.learn(args.steps, callback=cb)
    train_s = time.time() - t0

    torch.save(model.policy.state_dict(), out / "model.pt")
    if eff is not None:
        eff.save(out / "effect.pt", out / "effect.json", args.tag)
        print(f"effect model: BCE {eff.loss_sum / max(eff.loss_n, 1):.4f} (last-window avg n/a), "
              f"positive rate {eff.pos_sum / max(eff.n, 1):.3f}")
    (out / "model.json").write_text(json.dumps(dict(
        algo="recurrent_ppo" if args.recurrent else "maskable_ppo", tag=args.tag, gdim=256, deterministic=False,
        rnd_coef=args.rnd_coef), indent=2))
    with open(out / "per_game.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["game", "levels", "episodes"])
        for g, s in sorted(cb.per_game.items()):
            w.writerow([g, s["levels"], s["episodes"]])
    (out / "train_summary.json").write_text(json.dumps(dict(
        tag=args.tag, env_steps=int(model.num_timesteps), train_seconds=round(train_s, 1),
        train_hours=round(train_s / 3600, 2), levels_total=cb.levels,
        hardware=hardware(args.device),
        versions=dict(python=platform.python_version(), torch=torch.__version__,
                      numpy=np.__version__)), indent=2))
    print(f"done: {model.num_timesteps} env steps, {train_s/3600:.2f} h -> {out}")


if __name__ == "__main__":
    main()
