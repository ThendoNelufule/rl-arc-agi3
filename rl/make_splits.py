"""P1.2: profile every public game and freeze a stratified train/dev/held-out split.

Writes:
  splits.json                    frozen split (refuses to overwrite without --force)
  experiments/game_profile.csv   per-game action type + level count (report material)
"""
import csv, json, random, sys
from pathlib import Path

sys.path.insert(0, "rl")
from arc_gym import ArcGymEnv, N_SIMPLE, get_arcade

SEED = 2026
OUT = Path("splits.json")

if OUT.exists() and "--force" not in sys.argv:
    sys.exit("splits.json already exists and is FROZEN. Not overwriting.")

arc = get_arcade()
games = sorted(e.game_id.split("-")[0] for e in arc.get_environments())

rows = []
for g in games:
    env = ArcGymEnv(g, arcade=arc)
    env.reset(seed=0)
    m = env.action_masks()
    simple = bool(m[:N_SIMPLE].any())
    click = bool(m[N_SIMPLE:].any())
    kind = "mixed" if simple and click else ("click" if click else "move")
    win = env._info(env._last)["win_levels"]
    rows.append({"game": g, "kind": kind, "win_levels": win,
                 "simple_actions": int(m[:N_SIMPLE].sum()), "click": click})
    print(f"  {g}: {kind:5}  win_levels={win}")

by_kind = {}
for r in rows:
    by_kind.setdefault(r["kind"], []).append(r["game"])

rng = random.Random(SEED)
split = {"train": [], "dev": [], "heldout": []}
for kind in sorted(by_kind):
    gs = sorted(by_kind[kind])
    rng.shuffle(gs)
    n_held = max(1, round(0.2 * len(gs))) if len(gs) >= 3 else 0
    n_dev = max(1, round(0.2 * len(gs))) if len(gs) >= 3 else 0
    split["heldout"] += gs[:n_held]
    split["dev"] += gs[n_held:n_held + n_dev]
    split["train"] += gs[n_held + n_dev:]

for k in split:
    split[k] = sorted(split[k])
for r in rows:
    r["split"] = next(k for k in split if r["game"] in split[k])

OUT.write_text(json.dumps({"seed": SEED, **split}, indent=2))
Path("experiments").mkdir(exist_ok=True)
with open("experiments/game_profile.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print("\nkinds:", {k: len(v) for k, v in by_kind.items()})
for k in ("train", "dev", "heldout"):
    print(f"{k:8} ({len(split[k])}): {split[k]}")
