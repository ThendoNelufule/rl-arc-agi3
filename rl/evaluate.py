"""P1.3: evaluation harness (the report's evidence factory).

Protocol (fixed for every agent in the project):
  - Each game gets a fixed action BUDGET (default 500), counted across
    GAME_OVER resets; play stops early only on WIN.
  - N independent seeds, each with a FRESH Arcade, i.e. a fresh official scorecard.
  - Logged for every run, successful or not: levels, actions, deaths,
    wall time, level-vs-action curve, official scorecard score.
  - Held-out games are refused unless --final is passed (leakage guard).

Usage:
  .venv/bin/python rl/evaluate.py --policy random --split dev --tag NAME [--record]
"""
import argparse, csv, json, logging, os, platform, sys, time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
logging.disable(logging.INFO)
import arc_agi
from arc_agi import OperationMode
from arc_gym import ArcGymEnv
from policies import make_policy

ROOT = Path(__file__).resolve().parents[1]


def play_game(arc, game, policy, seed, budget, rec_dir=None):
    env = ArcGymEnv(game, max_steps=budget + 1, arcade=arc)
    policy.reset(game, seed)
    obs, info = env.reset(seed=seed)
    best, win_levels = info["levels_completed"], info["win_levels"]
    curve, deaths, won, t = [(0, best)], 0, False, 0
    frames, acts = [obs[0]], []
    t0 = time.time()
    for t in range(1, budget + 1):
        a = policy.act(obs, env.action_masks(), info)
        nobs, r, term, trunc, info = env.step(a)
        policy.observe(obs, a, r, nobs, term, info)
        if rec_dir is not None:
            frames.append(nobs[0]); acts.append(a)
        if info["levels_completed"] > best:
            best = info["levels_completed"]; curve.append((t, best))
        obs = nobs
        if info["state"] == "WIN":
            won = True
            break
        if term:                      # GAME_OVER: reset, keep spending budget
            deaths += 1
            obs, info = env.reset()
    if rec_dir is not None:
        np.savez_compressed(rec_dir / f"{game}_s{seed}.npz",
                            frames=np.stack(frames),
                            actions=np.array(acts, dtype=np.int16))
    row = dict(game=game, seed=seed, levels=best, win_levels=win_levels,
               frac=round(best / max(win_levels, 1), 4), actions=t, won=won,
               deaths=deaths, seconds=round(time.time() - t0, 2))
    return row, curve


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def plot_curves(curves, games, seeds, budget, path, title):
    xs = np.arange(budget + 1)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    totals = []
    for s in range(seeds):
        tot = np.zeros(budget + 1)
        for g in games:
            y = np.zeros(budget + 1)
            for c in sorted((c for c in curves if c["seed"] == s and c["game"] == g),
                            key=lambda c: c["action"]):
                y[c["action"]:] = c["levels"]
            tot += y
        totals.append(tot)
        ax.step(xs, tot, where="post", alpha=0.4, lw=1, label=f"seed {s}")
    ax.step(xs, np.mean(totals, 0), where="post", lw=2.2, color="k", label="mean")
    ax.set_xlabel("actions per game")
    ax.set_ylabel("levels completed (sum over games)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pkg_version(name):
    try:
        return version(name)
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="random")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--split", default="dev", choices=["train", "dev", "heldout", "all"])
    p.add_argument("--games", default=None, help="comma list; overrides --split")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--budget", type=int, default=500)
    p.add_argument("--tag", required=True)
    p.add_argument("--record", action="store_true")
    p.add_argument("--final", action="store_true", help="allow held-out games")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    splits = json.loads((ROOT / "splits.json").read_text())
    if args.games:
        games = [g.strip() for g in args.games.split(",")]
    elif args.split == "all":
        games = sorted(splits["train"] + splits["dev"] + splits["heldout"])
    else:
        games = splits[args.split]
    if set(games) & set(splits["heldout"]) and not args.final:
        sys.exit("Refusing: held-out games requested without --final (leakage guard).")

    out = ROOT / "experiments" / "runs" / args.tag
    if out.exists() and not args.overwrite:
        sys.exit(f"{out} exists; pick a new --tag (or pass --overwrite).")
    out.mkdir(parents=True, exist_ok=True)
    rec = out / "recordings" if args.record else None
    if rec:
        rec.mkdir(exist_ok=True)

    rows, curves, seed_scores = [], [], {}
    t_all = time.time()
    for seed in range(args.seeds):
        # ARC3_ARCADE_MODE=offline on internet-less Kaggle eval kernels (scorecard is local)
        arc = arc_agi.Arcade(operation_mode=OperationMode(os.environ.get("ARC3_ARCADE_MODE", "normal")))
        policy = make_policy(args.policy, ckpt=args.ckpt)
        for g in games:
            row, curve = play_game(arc, g, policy, seed, args.budget, rec)
            rows.append(row)
            curves += [dict(seed=seed, game=g, action=a, levels=l) for a, l in curve]
            print(f"  seed {seed} {g}: levels {row['levels']}/{row['win_levels']}"
                  f"  actions {row['actions']}  deaths {row['deaths']}  {row['seconds']}s")
        sc = arc.get_scorecard()
        seed_scores[seed] = float(getattr(sc, "score", sc) or 0.0)
        print(f"seed {seed}: official local score = {seed_scores[seed]:.4f}")

    write_csv(out / "results.csv", rows)
    write_csv(out / "curves.csv", curves)
    plot_curves(curves, games, args.seeds, args.budget, out / "curves.png",
                f"{args.tag}: {args.policy} on {args.split} ({args.seeds} seeds)")
    vals = list(seed_scores.values())
    summary = dict(
        tag=args.tag, policy=args.policy, ckpt=args.ckpt, split=args.split,
        games=games, seeds=args.seeds, budget=args.budget,
        official_score_per_seed=seed_scores,
        official_score_mean=float(np.mean(vals)), official_score_std=float(np.std(vals)),
        levels_per_seed={s: sum(r["levels"] for r in rows if r["seed"] == s) for s in seed_scores},
        mean_level_frac=float(np.mean([r["frac"] for r in rows])),
        eval_seconds=round(time.time() - t_all, 1),
        versions=dict(python=platform.python_version(), arc_agi=pkg_version("arc-agi"),
                      arcengine=pkg_version("arcengine"), numpy=np.__version__,
                      gymnasium=pkg_version("gymnasium"), platform=platform.platform()),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nofficial score: {summary['official_score_mean']:.4f} "
          f"± {summary['official_score_std']:.4f} over {args.seeds} seeds")
    print(f"levels per seed: {summary['levels_per_seed']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
