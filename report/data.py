"""Single source of numbers for the report: reads experiments/ (runs, train, log) only.

Every figure/table in report/ is built from these loaders, so no number is typed by hand.
"""
import csv, json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
TRAIN = ROOT / "experiments" / "train"

# (key, label, family, run-tag prefix on dev, how the dev runs are organised)
# family: random | dqn | ppo | graph ; "seeds" = 3 training seeds (prefix-s{0,1,2}-dev), "single" = one dir
VERSIONS = [
    ("random", "Random", "random", "REF-random-dev", "single"),
    ("T10", "TEAM-10 DQN", "dqn", "TEAM-10-dqn-base", "seeds"),
    ("T10f", "TEAM-10f DQN frozen", "dqn", "TEAM-10f-dqn-frozen", "seeds"),
    ("T11", "TEAM-11 +novelty", "dqn", "TEAM-11-dqn-novelty", "seeds"),
    ("T11f", "TEAM-11f frozen", "dqn", "TEAM-11f-dqn-novelty-frozen", "seeds"),
    ("T12", "TEAM-12 +effect mask", "dqn", "TEAM-12-dqn-novelty-mask", "seeds"),
    ("T12f", "TEAM-12f frozen", "dqn", "TEAM-12f-dqn-novelty-mask-frozen", "seeds"),
    ("T20", "TEAM-20 PPO", "ppo", "TEAM-20-ppo-base", "seeds"),
    ("T21", "TEAM-21 +RND", "ppo", "TEAM-21-ppo-rnd", "seeds"),
    ("T22", "TEAM-22 +LSTM", "ppo", "TEAM-22-ppo-rnd-lstm", "seeds"),
    ("G1", "GRAPH-v1 (DQN effect head)", "graph", "X-graph-team12", "seeds"),
    ("T30", "TEAM-30 graph+online stats", "graph", "TEAM-30-graph-stats-dev", "single"),
    ("T31", "TEAM-31 +PPO prior", "graph", "TEAM-31-graph-stats-ppo", "seeds"),
    ("T32", "TEAM-32 +ticker mask", "graph", "TEAM-32-graph-clock", "seeds"),
    ("T33", "TEAM-33 +prune", "graph", "TEAM-33-graph-clock-prune", "seeds"),
    ("T34", "TEAM-34 +timer deaths", "graph", "TEAM-34-graph-timer", "seeds"),
    ("T35", "TEAM-35 +click equivalence", "graph", "TEAM-35-graph-patch", "seeds"),
    ("T35n", "TEAM-35n no prior", "graph", "TEAM-35n-graph-patch-noprior-dev", "single"),
    ("T36", "TEAM-36 prior on clicks", "graph", "TEAM-36-graph-patch-clickprior", "seeds"),
    ("T37", "TEAM-37 +inverse actions", "graph", "TEAM-37-graph-inverse", "seeds"),
    ("T38", "TEAM-38 patch stats", "graph", "TEAM-38-graph-patchstats", "seeds"),
    ("T39", "TEAM-39 value frontier", "graph", "TEAM-39-graph-value", "seeds"),
    ("T40", "TEAM-40 relax when stuck", "graph", "TEAM-40-graph-relax", "seeds"),
    ("T41", "TEAM-41 +object clicks", "graph", "TEAM-41-graph-object", "seeds"),
    ("T42", "TEAM-42 small objects", "graph", "TEAM-42-graph-smallobj", "seeds"),
    ("T43", "TEAM-43 objects as fallback", "graph", "TEAM-43-graph-objfallback", "seeds"),
    ("T44", "TEAM-44 object promotion", "graph", "TEAM-44-graph-objpromote", "seeds"),
    ("T45", "TEAM-45 component clicks", "graph", "TEAM-45-graph-components", "seeds"),
    ("T45n", "TEAM-45n no prior", "graph", "TEAM-45n-graph-components-noprior-dev", "single"),
    ("T46", "TEAM-46 centres fallback", "graph", "TEAM-46-graph-components-fb", "seeds"),
    ("T47", "TEAM-47 size prior", "graph", "TEAM-47-graph-components-size", "seeds"),
]
FAMILY = {v[0]: v[2] for v in VERSIONS}
LABEL = {v[0]: v[1] for v in VERSIONS}


def run_dirs(prefix, kind, split="dev", ho=False):
    pre = "HO-" if ho else ""
    if kind == "single":
        d = RUNS / (pre + prefix.replace("-dev", f"-{split}") if ho else prefix)
        return [d] if d.exists() else []
    return [d for d in (RUNS / f"{pre}{prefix}-s{s}-{split}" for s in range(3)) if d.exists()]


def load_run(d):
    s = json.load(open(d / "summary.json"))
    rows = list(csv.DictReader(open(d / "results.csv")))
    curves = list(csv.DictReader(open(d / "curves.csv"))) if (d / "curves.csv").exists() else []
    return s, rows, curves


def aggregate(dirs):
    """Score over runs. With training seeds: mean/std over train-seed means (population std, as in the
    notes); every individual (train seed x eval seed) run is kept for plotting."""
    if not dirs:
        return None
    per_dir, runs, rows_all, secs = [], [], [], []
    for d in dirs:
        s, rows, _ = load_run(d)
        per_dir.append(s["official_score_mean"])
        runs += list(s["official_score_per_seed"].values())
        rows_all += rows
        secs.append(s.get("eval_seconds", np.nan))
    multi = len(dirs) > 1
    lv = sum(int(r["levels"]) for r in rows_all)
    poss = sum(int(r["win_levels"]) for r in rows_all)
    return dict(
        mean=float(np.mean(per_dir) if multi else np.mean(runs)),
        std=float(np.std(per_dir) if multi else np.std(runs)),
        std_runs=float(np.std(runs)), runs=runs, per_train_seed=per_dir if multi else None,
        n_runs=len(runs), levels=lv, levels_possible=poss,
        games_with_level=sorted({r["game"] for r in rows_all if int(r["levels"]) > 0}),
        actions=float(np.mean([int(r["actions"]) for r in rows_all])),
        deaths=float(np.mean([int(r["deaths"]) for r in rows_all])),
        eval_seconds=float(np.nansum(secs)), rows=rows_all)


def version_results(key, split="dev"):
    v = next(x for x in VERSIONS if x[0] == key)
    if split == "dev":
        return aggregate(run_dirs(v[3], v[4]))
    if split == "heldout":
        tag = v[3] if v[4] == "seeds" else v[3].replace("-dev", "")
        if key == "random":
            return aggregate([RUNS / "HO-random-heldout"] if (RUNS / "HO-random-heldout").exists() else [])
        if v[4] == "single":
            d = RUNS / f"HO-{tag}-heldout"
            return aggregate([d] if d.exists() else [])
        return aggregate(run_dirs(v[3], "seeds", "heldout", ho=True))
    if split == "train":
        names = {"random": ["SUPP-random-train"], "T30": ["SUPP-graph-stats-train"],
                 "T35n": ["SUPP-graph-patch-noprior-train"], "T45n": ["SUPP-graph-components-noprior-train"],
                 "T12": ["SUPP-TEAM-12-dqn-novelty-mask-s0-train"], "T21": ["SUPP-TEAM-21-ppo-rnd-s0-train"],
                 "T37": ["SUPP-TEAM-37-graph-inverse-s0-train"], "T45": ["SUPP-TEAM-45-graph-components-s0-train"]}
        return aggregate([RUNS / n for n in names.get(key, []) if (RUNS / n).exists()])


def level_curve(dirs, budget=500):
    """Total levels (summed over the split's games) reached by action t, one curve per (dir, eval seed)."""
    out = []
    for d in dirs:
        _, _, curves = load_run(d)
        by = {}
        for r in curves:
            by.setdefault(int(r["seed"]), {}).setdefault(r["game"], []).append((int(r["action"]), int(r["levels"])))
        for seed, games in by.items():
            tot = np.zeros(budget + 1)
            for pts in games.values():
                pts.sort()
                c = np.zeros(budget + 1)
                for a, l in pts:
                    c[min(a, budget):] = l
                tot += c
            out.append(tot)
    return np.array(out)


def train_metrics(prefix):
    out = []
    for s in range(3):
        f = TRAIN / f"{prefix}-s{s}" / "metrics.csv"
        if f.exists():
            out.append(list(csv.DictReader(open(f))))
    return out


def train_summary(prefix):
    return [json.load(open(TRAIN / f"{prefix}-s{s}" / "train_summary.json"))
            for s in range(3) if (TRAIN / f"{prefix}-s{s}" / "train_summary.json").exists()]


def kaggle_submissions():
    return list(csv.DictReader(open(ROOT / "experiments" / "kaggle_submissions.csv")))


def proxy(name):
    b = json.load(open(ROOT / "kaggle_runs" / name / "benchmark.json"))
    return {g["game_id"].split("-")[0]: dict(score=g["final_score"], levels=g["levels_completed"],
                                             n=g["number_of_levels"]) for g in b["game_runs"]}


if __name__ == "__main__":
    for v in VERSIONS:
        r = version_results(v[0])
        if r:
            print(f"{v[0]:6s} {r['mean']:.4f} +- {r['std']:.4f} (runs sd {r['std_runs']:.4f}, n={r['n_runs']}) "
                  f"levels {r['levels']}/{r['levels_possible']} games {r['games_with_level']} act {r['actions']:.0f}")
        else:
            print(v[0], "MISSING")
