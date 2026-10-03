"""Aggregate eval runs of one version across TRAINING seeds (report numbers).

Each eval run tag = one training seed evaluated with 3 eval seeds (rl/evaluate.py summary.json).
Reports: per training seed mean official score; across training seeds mean +- std (n = #train
seeds, the unit of independence for a learning method) and mean +- std over all (train x eval)
runs; total levels.

Usage: .venv/bin/python rl/aggregate.py TAG_s0 TAG_s1 TAG_s2 [--name LABEL]
"""
import argparse, json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(tag):
    return json.loads((ROOT / "experiments" / "runs" / tag / "summary.json").read_text())


def aggregate(tags, name=None):
    sums = [load(t) for t in tags]
    per_train = [s["official_score_mean"] for s in sums]
    allruns = [v for s in sums for v in s["official_score_per_seed"].values()]
    levels = [sum(s["levels_per_seed"].values()) for s in sums]
    res = dict(name=name or tags[0], tags=tags, n_train_seeds=len(tags),
               score_mean=float(np.mean(per_train)), score_std_train_seeds=float(np.std(per_train)),
               score_mean_all=float(np.mean(allruns)), score_std_all=float(np.std(allruns)),
               n_runs=len(allruns), per_train_seed=per_train, levels_per_train_seed=levels,
               levels_total=int(sum(levels)))
    print(f"{res['name']}: official {res['score_mean']:.4f} +- {res['score_std_train_seeds']:.4f} "
          f"(train seeds, n={len(tags)}) | all {res['n_runs']} runs {res['score_mean_all']:.4f} "
          f"+- {res['score_std_all']:.4f} | levels {levels} total {res['levels_total']}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--name")
    a = ap.parse_args()
    aggregate(a.tags, a.name)
