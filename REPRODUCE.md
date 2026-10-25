# Reproducing "Where ARC-AGI-3 Agents Stall" (team Vladmir_Putin)

Musala Ndouvhada and Thendo Nelufule, University of the Witwatersrand (COMS4061A/7071A, 2026).
All scores are on the official ARC-AGI-3 0–100 scale. The report is `report/main.pdf`; every number in it is read
from the logs in `experiments/` by `report/data.py`.

## Layout

| path | what |
|---|---|
| `rl/arc_gym.py` | Gymnasium adapter: obs (1,64,64) colour ids, `Discrete(70)` actions, reward = new levels, `action_masks()` |
| `rl/evaluate.py` | **the** evaluation protocol: 500 actions/game across deaths, N seeds, fresh scorecard per seed; held-out refused without `--final` |
| `rl/train_dqn.py`, `rl/train_ppo.py`, `rl/rnd.py`, `rl/recurrent.py`, `rl/nets.py`, `rl/replay.py`, `rl/effect.py`, `rl/sb3_nets.py` | DQN chain (T10–T12, T12a), PPO chain (T20–T22), RND, recurrent PPO, shared encoder |
| `rl/policies.py` | all evaluated agents (`dqn`, `dqn-online`, `ppo`, `ppo-lstm`, `graph-*`, `random`) |
| `rl/analyze_run.py`, `rl/noop_repeats.py`, `rl/aggregate.py` | behaviour diagnostics (no-op rate, repeated known no-ops, collapse) and aggregation |
| `splits.json` | the frozen train/dev/held-out split (seed 2026), never regenerated |
| `scripts/kaggle_train.py` | runs training/evaluation jobs on Kaggle kernels (T4 / CPU, internet off) and pulls results back |
| `kaggle_runs/*/jobs.json` | the exact job lines of every training and evaluation kernel we ran |
| `experiments/train/*/config.json`, `model.json`, `metrics.csv`, `train_summary.json` | training configs, curves, time and hardware per run |
| `experiments/runs/*/` | every evaluation run: `summary.json`, `results.csv`, `curves.csv`, recordings |
| `experiments/log.csv`, `experiments/kaggle_submissions.csv` | the run log, and all 16 Kaggle submissions with IDs |
| `scripts/build_m2.py` | builds our forks of Franzen's M2 notebook (base, MOD-1, MOD-2, MOD-F, proxies, `-sub` packaging) from the *pulled public notebooks* |
| `recon/analysis/franzen_v3_failure_modes.py` | failure-mode decomposition of Franzen's 100 public trajectories (data extracted from skarin's public audit notebook) |
| `report/` | `data.py` (numbers), `figures.py`, `tables.py`, `main.tex` (RLC 2026 template), `build.sh` |

## Setup

- **Python.** The local venv matches the Kaggle submission image: Python 3.12, `requirements-local.txt` (torch 2.10 CPU, numpy 2.0.2, gymnasium 1.2.0, SB3/sb3-contrib 2.9.0), plus `arc-agi`.
  ```
  py -3.12 -m venv .venv && .venv/bin/python -m pip install -r requirements-local.txt arc-agi kaggle
  ```
- **Kaggle training versions.** Training and evaluation for the report ran on Kaggle (Python 3.13.15, torch 2.11.0+cu128, numpy 2.1.3, arc-agi 0.9.8, arcengine 0.9.3).
- **Credentials.** Put your *own* Kaggle API token in `.kaggle/access_token`. It is gitignored and never committed. The Makefile and the scripts read it as `KAGGLE_API_TOKEN`.
- **Game files.** These come from the competition (`/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files`), or locally from `arc-agi`. They are not redistributed here.

## Reproduce the RL and graph results

Training (Kaggle T4, the 3 seeds of a version in parallel in one kernel; 400k steps each). For example, the PPO+RND baseline T21:
```
.venv/bin/python scripts/kaggle_train.py push team21 --accel t4 \
  --job "train_ppo.py --tag TEAM-21-ppo-rnd-s0 --seed 0 --steps 400000 --device cuda --rnd-coef 0.05" \
  --job "train_ppo.py --tag TEAM-21-ppo-rnd-s1 --seed 1 --steps 400000 --device cuda --rnd-coef 0.05" \
  --job "train_ppo.py --tag TEAM-21-ppo-rnd-s2 --seed 2 --steps 400000 --device cuda --rnd-coef 0.05"
.venv/bin/python scripts/kaggle_train.py pull team21
```
The exact job lines for every version are in `kaggle_runs/<name>/jobs.json` (e.g. `kaggle_runs/team12a/jobs.json` for the mask-only ablation).
Checkpoints are copied to `kaggle_dataset/weights/ckpt/` and published as the Kaggle dataset `arc3-rl-assets` (`make dataset-push`).

Evaluation (Kaggle CPU, 1 thread per job; same protocol for every agent):
```
.venv/bin/python scripts/kaggle_train.py push eval-x --accel cpu --maxpar 4 --threads 1 \
  --job "evaluate.py --policy ppo --ckpt {ASSETS}/weights/ckpt/TEAM-21-ppo-rnd-s0.pt --split dev --seeds 3 --record --tag TEAM-21-ppo-rnd-s0-dev"
```
Locally, a single run is `.venv/bin/python rl/evaluate.py --policy random --split dev --seeds 3 --tag REF-random-dev`.

- **Graph explorer.** Training-free: `--policy graph-components` with `--ckpt` set to the T21 checkpoint (prior), or `--policy graph-components-noprior`.
- **Held-out.** Needs `--final`. It was run exactly once, after development (kernels `heldout-rl`, `heldout-graph`).
- **Diagnostics.** `rl/analyze_run.py TAG` and `rl/noop_repeats.py TAG`.

## Reproduce the LLM-platform work (Franzen M2 fork)

The third-party notebooks are **not redistributed** here. Pull them yourself and build our variants:
```
kaggle kernels pull dfranzen/arc-agi-3-milestone-2-solution -p recon/nb/dfranzen__arc-agi-3-milestone-2-solution -m
kaggle kernels pull heliosli/try-merging-it-you-might-get-lucky-and-win-gold -p recon/nb/heliosli__try-merging-it-you-might-get-lucky-and-win-gold -m
kaggle kernels pull shiiin9/affectify-arc-31-54-in-a-single-sub -p recon/nb/shiiin9__affectify-arc-31-54-in-a-single-sub -m
.venv/bin/python scripts/build_m2.py      # writes kaggle_sub/m2-*/ (MOD-2 also needs the Fable F001 notebook in recon/fable/)
```
The nominated notebook is `kaggle_sub/m2-modf-sub` (pushed as `thendonelufule/arc3-m2-modf-sub` v1). All changes and experiments are listed in
`submission/MODIFICATIONS.md`. The agent itself is Franzen's work, on Tufa Labs' Duck harness.

## Rebuild the report

```
sh report/build.sh path/to/tectonic      # figures + tables from experiments/, then LaTeX -> report/main.pdf
```
