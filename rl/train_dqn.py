"""DQN training on the 15 train games (P2.x).

TEAM-10-dqn-base = vanilla DQN (Mnih et al. 2015): replay, hard target network, Huber loss,
epsilon-greedy over legal actions, extrinsic reward only (level delta). One env per train game
runs in lock-step (15 parallel envs, batched action selection), so every game gets equal data.

Outputs (experiments/train/<tag>/): config.json, metrics.csv, per_game.csv,
train_summary.json (train time + hardware, kept separate from eval), model.pt (state_dict),
model.json (arch + eval config read by DQNPolicy).

Usage: .venv/bin/python rl/train_dqn.py --tag TEAM-10-dqn-base --steps 400000
"""
import argparse, csv, hashlib, json, logging, platform, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
logging.disable(logging.INFO)
from arc_gym import ArcGymEnv, get_arcade
from nets import QEffNet, QNet, N_ACTIONS
from replay import Replay

ROOT = Path(__file__).resolve().parents[1]


def cpu_name():
    try:
        if platform.system() == "Linux":
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        out = subprocess.run(["powershell", "-c", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return out or platform.processor()
    except Exception:
        return platform.processor()


def hardware(device):
    """Hardware record for train_summary.json (report: compute per run)."""
    import os
    hw = dict(cpu=cpu_name(), logical_cpus=os.cpu_count(), torch_threads=torch.get_num_threads(),
              device=str(device), os=platform.platform(),
              host="kaggle" if os.path.exists("/kaggle") else "local")
    try:
        import psutil
        hw["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    if str(device).startswith("cuda"):
        hw["gpu"] = torch.cuda.get_device_name(0)
    return hw


def masked_argmax(q, mask):
    return q.masked_fill(~mask, -1e9).argmax(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--steps", type=int, default=400_000, help="total env steps (all envs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ep-len", type=int, default=500)
    p.add_argument("--buffer", type=int, default=100_000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--train-freq", type=int, default=4, help="env steps per gradient step")
    p.add_argument("--learning-starts", type=int, default=20_000)
    p.add_argument("--target-update", type=int, default=10_000, help="env steps between syncs")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-frac", type=float, default=0.5)
    p.add_argument("--eval-eps", type=float, default=0.05)
    p.add_argument("--novelty-beta", type=float, default=0.0,
                   help="P2.2 count-based bonus beta/sqrt(N(frame)), counts reset each episode (0=off)")
    p.add_argument("--log-every", type=int, default=10_000)
    p.add_argument("--effect-coef", type=float, default=0.0,
                   help="P2.3 action-effect head: BCE weight (0 = off, plain QNet)")
    p.add_argument("--effect-tau", type=float, default=0.1,
                   help="P2.3 mask actions with predicted p(change) < tau (training + eval)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    dev = torch.device(args.device)

    out = ROOT / "experiments" / "train" / args.tag
    if out.exists() and not args.overwrite:
        sys.exit(f"{out} exists; pick a new --tag (or --overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    games = json.loads((ROOT / "splits.json").read_text())["train"]
    arc = get_arcade()
    envs = [ArcGymEnv(g, max_steps=args.ep_len, arcade=arc) for g in games]
    obs = np.stack([e.reset(seed=args.seed)[0] for e in envs])
    masks = np.stack([e.action_masks() for e in envs])

    use_eff = args.effect_coef > 0
    Net = QEffNet if use_eff else QNet
    net, tgt = Net().to(dev), Net().to(dev)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    buf = Replay(args.buffer, N_ACTIONS)

    G = len(games)
    stats = {g: dict(levels=0, episodes=0, deaths=0, steps=0, changed=0, distinct=[]) for g in games}
    ep_seen = [set() for _ in games]
    ep_count = [dict() for _ in games]      # episodic visit counts (novelty bonus)
    int_sum = 0.0
    mf = open(out / "metrics.csv", "w", newline="")
    mw = csv.writer(mf)
    mw.writerow(["env_steps", "updates", "eps", "loss", "q_mean", "levels_total",
                 "episodes", "frame_change_rate", "sps", "wall_s", "intrinsic_per_step"])
    t0, t = time.time(), 0
    updates, losses, qs, next_log, next_sync = 0, [], [], args.log_every, args.target_update
    win_change, win_steps = 0, 0

    while t < args.steps:
        eps = max(args.eps_end, args.eps_start - (args.eps_start - args.eps_end) * t /
                  (args.eps_frac * args.steps))
        with torch.no_grad():
            m_t = torch.from_numpy(masks).to(dev)
            if use_eff and t >= args.learning_starts:
                q_all, e_all = net.both(torch.from_numpy(obs).to(dev))
                keep = m_t & (torch.sigmoid(e_all) >= args.effect_tau)
                keep[keep.sum(1) == 0] = m_t[keep.sum(1) == 0]     # never mask everything
                act_masks = keep.cpu().numpy()
                greedy = masked_argmax(q_all, keep).cpu().numpy()
            else:
                act_masks = masks
                greedy = masked_argmax(net(torch.from_numpy(obs).to(dev)), m_t).cpu().numpy()
        for i, e in enumerate(envs):
            g = games[i]
            a = int(rng.choice(np.flatnonzero(act_masks[i]))) if rng.random() < eps else int(greedy[i])
            nobs, r, term, trunc, info = e.step(a)
            nmask = e.action_masks()
            changed = not np.array_equal(nobs, obs[i])
            r_train = r
            if args.novelty_beta > 0:
                h = hashlib.blake2b(nobs.tobytes(), digest_size=8).digest()
                c = ep_count[i][h] = ep_count[i].get(h, 0) + 1
                bonus = args.novelty_beta / np.sqrt(c)
                r_train += bonus; int_sum += bonus
            buf.add(obs[i], a, r_train, nobs, float(term), nmask, float(changed))
            st = stats[g]
            st["steps"] += 1
            st["changed"] += changed; win_change += changed; win_steps += 1
            ep_seen[i].add(hashlib.blake2b(nobs.tobytes(), digest_size=8).digest())
            if r > 0:
                st["levels"] += int(r)
            if term or trunc:
                st["episodes"] += 1
                st["deaths"] += info["state"] == "GAME_OVER"
                st["distinct"].append(len(ep_seen[i])); ep_seen[i] = set(); ep_count[i] = {}
                nobs, _ = e.reset()
                nmask = e.action_masks()
            obs[i], masks[i] = nobs, nmask
        t += G

        if t >= args.learning_starts:
            for _ in range(max(1, G // args.train_freq)):
                o, a, r, no, d, nm, chg = buf.sample(args.batch, rng)
                o, no = torch.from_numpy(o).to(dev), torch.from_numpy(no).to(dev)
                a, r, d, nm = (torch.from_numpy(a).to(dev), torch.from_numpy(r).to(dev),
                               torch.from_numpy(d).to(dev), torch.from_numpy(nm).to(dev))
                if use_eff:
                    q_all, e_all = net.both(o)
                    q = q_all.gather(1, a[:, None])[:, 0]
                    eff_loss = F.binary_cross_entropy_with_logits(
                        e_all.gather(1, a[:, None])[:, 0], torch.from_numpy(chg).to(dev))
                else:
                    q = net(o).gather(1, a[:, None])[:, 0]
                with torch.no_grad():
                    qn = tgt(no).masked_fill(~nm, -1e9).max(1).values
                    y = r + args.gamma * (1 - d) * qn
                loss = F.smooth_l1_loss(q, y)
                if use_eff:
                    loss = loss + args.effect_coef * eff_loss
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step()
                updates += 1; losses.append(loss.item()); qs.append(q.mean().item())
        if t >= next_sync:
            tgt.load_state_dict(net.state_dict()); next_sync += args.target_update

        if t >= next_log or t >= args.steps:
            next_log += args.log_every
            wall = time.time() - t0
            row = [t, updates, round(eps, 4), round(float(np.mean(losses)) if losses else 0, 6),
                   round(float(np.mean(qs)) if qs else 0, 5),
                   sum(s["levels"] for s in stats.values()),
                   sum(s["episodes"] for s in stats.values()),
                   round(win_change / max(win_steps, 1), 4), round(t / wall, 1), round(wall, 1),
                   round(int_sum / max(win_steps, 1), 5)]
            mw.writerow(row); mf.flush()
            print("t={} upd={} eps={} loss={} q={} levels={} eps_done={} chg={} sps={} wall={} int={}".format(*row),
                  flush=True)
            losses, qs, win_change, win_steps, int_sum = [], [], 0, 0, 0.0
            torch.save(net.state_dict(), out / "model.pt")

    train_s = time.time() - t0
    sd = {k: v.cpu() for k, v in net.state_dict().items()}
    torch.save(sd, out / "model.pt")
    (out / "model.json").write_text(json.dumps(dict(
        algo="dqn", arch="QEffNet" if use_eff else "QNet", arch_kwargs={}, eval_eps=args.eval_eps,
        tag=args.tag, novelty_beta=args.novelty_beta,
        effect_tau=args.effect_tau if use_eff else 0.0), indent=2))
    with open(out / "per_game.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game", "levels", "episodes", "deaths", "steps", "frame_change_rate",
                    "mean_distinct_states_per_episode"])
        for g, s in stats.items():
            w.writerow([g, s["levels"], s["episodes"], s["deaths"], s["steps"],
                        round(s["changed"] / max(s["steps"], 1), 4),
                        round(float(np.mean(s["distinct"])) if s["distinct"] else 0, 1)])
    (out / "train_summary.json").write_text(json.dumps(dict(
        tag=args.tag, env_steps=t, updates=updates, train_seconds=round(train_s, 1),
        train_hours=round(train_s / 3600, 2), levels_total=sum(s["levels"] for s in stats.values()),
        hardware=hardware(dev),
        versions=dict(python=platform.python_version(), torch=torch.__version__,
                      numpy=np.__version__)), indent=2))
    print(f"done: {t} env steps, {updates} updates, {train_s/3600:.2f} h -> {out}")


if __name__ == "__main__":
    main()
