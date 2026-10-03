# Progress
Current phase: P2.2 TEAM-11-dqn-novelty
- 2026-09-18 P0 done: pipeline verified, Kaggle 0.02 (random)
- 2026-10-03 P1.1 adapter, P1.2 frozen splits, P1.3 eval harness done
- 2026-10-03 P1.4 done: private dataset thendonelufule/arc3-rl-assets (SB3 2.9.0 + sb3-contrib
  2.9.0 wheels, weights/ + MANIFEST.json). Verified internet-OFF twice: arc3-offline-verify
  (latest image) and submission-kernel commit v3 (pinned image). Notebook globs the mount path
  and exports ARC3_ASSETS. `make dataset-push MSG=...` versions the dataset.
  Submission image = Python 3.12.13, torch 2.10.0+cpu, numpy 2.0.2, gymnasium 1.2.0, 2 threads;
  local venv pinned to match (requirements-local.txt). Weights format: state_dict .pt + JSON.
  Autonomous working mode adopted (CLAUDE.md).
- 2026-10-03 P2.1 done: TEAM-10-dqn-base (dqn-online) dev 0.0000 +- 0.0000 (3x3, 0 levels);
  frozen variant 0.0000 and collapses to one action. Training moved to Kaggle T4 (4.5x faster);
  infra: rl/nets.py, replay.py, train_dqn.py, train_ppo.py, policies (dqn, dqn-online, ppo),
  analyze_run.py, aggregate.py, scripts/kaggle_train.py, agent wrapper + 9h wall-clock guard.
- 2026-10-03 Submissions: one notebook per version (arc3-<slug>, image pinned), weights = seed 0.
  arc3-team-10-dqn-base v1 pushed + smoke OK; awaiting manual "Submit to Competition".
