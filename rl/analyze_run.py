"""Behaviour diagnostics from an eval run recorded with --record (evidence for the next stage).

Per game x seed, from recordings/<game>_s<seed>.npz (frames[0] = start, frames[i+1] = after action i):
  noop_rate      share of actions after which the frame did not change at all
  distinct_frac  distinct frames visited / actions (state-space coverage)
  top_action     share of the single most frequent action (policy collapse indicator)
  action_entropy entropy of the action histogram in bits (log2(70) = 6.13 is uniform)
Writes experiments/runs/<tag>/behaviour.csv and prints per-game means + overall means.

Usage: .venv/bin/python rl/analyze_run.py TAG [TAG ...]
"""
import csv, hashlib, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def analyze(tag):
    rec = ROOT / "experiments" / "runs" / tag / "recordings"
    rows = []
    for f in sorted(rec.glob("*.npz")):
        game, seed = f.stem.rsplit("_s", 1)
        z = np.load(f)
        fr, ac = z["frames"], z["actions"]
        n = len(ac)
        if n == 0:
            continue
        same = np.all(fr[1:n + 1] == fr[:n], axis=(1, 2))
        distinct = len({hashlib.blake2b(x.tobytes(), digest_size=8).digest() for x in fr})
        counts = np.bincount(ac, minlength=70).astype(float)
        p = counts[counts > 0] / n
        rows.append(dict(game=game, seed=int(seed), actions=n,
                         noop_rate=round(float(same.mean()), 4),
                         distinct_frac=round(distinct / n, 4),
                         top_action=round(float(counts.max() / n), 4),
                         action_entropy=round(float(-(p * np.log2(p)).sum()), 3)))
    out = rec.parent / "behaviour.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    keys = ["noop_rate", "distinct_frac", "top_action", "action_entropy"]
    print(f"== {tag}")
    for g in sorted({r["game"] for r in rows}):
        rs = [r for r in rows if r["game"] == g]
        print(f"  {g:6s} " + "  ".join(f"{k}={np.mean([r[k] for r in rs]):.3f}" for k in keys))
    print("  ALL    " + "  ".join(f"{k}={np.mean([r[k] for r in rows]):.3f}" for k in keys))
    return rows


if __name__ == "__main__":
    for t in sys.argv[1:]:
        analyze(t)
