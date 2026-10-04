# Progress
Current phase: P2.3 TEAM-12-dqn-novelty-mask (training) + P3.1 TEAM-20-ppo-base (training)
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
- 2026-10-03 P2.2 done: TEAM-11-dqn-novelty dev 0.0000 +- 0.0000 (3x3, 0 levels); 27-33% of actions
  repeat known no-ops -> motivates P2.3. Submission notebook arc3-team-11-dqn-novelty v1 pushed.
  TEAM-12 (novelty + effect masking) and TEAM-20 (PPO base) training in parallel on T4s.
- 2026-10-03 P3.1 done: TEAM-20-ppo-base dev 0.0004 +- 0.0006 (3x3, 1 level) = random level.
  Notebook arc3-team-20-ppo-base v1 pushed. TEAM-21 (PPO+RND) training; TEAM-12 training.

- 2026-10-04 00:20 P3.2 done: TEAM-21-ppo-rnd dev 0.0247 +- 0.0317 (3x3, 5 levels) = best so far.
  Eval moved to Kaggle CPU (OOM locally). TEAM-22 (RND+LSTM) and TEAM-23 (RND + effect model for
  graph exploration on the PPO chain) training on T4s.

- 2026-10-04 00:40 P2 + P3 v1 complete. DQN champion TEAM-12 (0.0044); PPO champion TEAM-21 (0.0247);
  TEAM-22 LSTM failed (0.0003). Graph track: TEAM-30 0.3895, TEAM-31 0.4611 (best).

- 2026-10-04 01:00 graph track: TEAM-32 0.4615, TEAM-33 0.4842, TEAM-34 0.4842 (null), TEAM-35 0.4929
  (best; all within noise). TEAM-23 effect model: negative diagnostic. Plateau = lp85 L1 only.

- 2026-10-04 01:45 graph track continued: TEAM-36 0.4943, TEAM-37 0.5041 (BEST, 18 levels), TEAM-38 0.1237 (failed),
  TEAM-39 0.4971, TEAM-40 0.5032 (no gain). Budget 1000/2000 = no change. Remaining blocker = click
  action space (design lock) -> asked team. FINAL AGENT if nothing changes by 08:00: TEAM-37.
- 2026-10-04 02:10 object clicks approved: TEAM-41 0.4698 (26 lv), TEAM-42 0.3789 (28 lv), TEAM-43 0.5445 (27 lv) = BEST.
  FINAL AGENT if nothing changes by 08:00: TEAM-43.
- 2026-10-04 02:20 TEAM-44 (adaptive object promotion) 0.5682 = BEST. FINAL if nothing changes: TEAM-44.

## Submission queue (ranked by dev official score; Kaggle = 1 submission/day, never used for tuning)
| rank | notebook (version) | dev official (3x3) | dev levels | Kaggle public |
| 1 | arc3-team-44-graph-objpromote (v1) | 0.5682 +- 0.0627 | 27 | pending |
| 1b | arc3-team-43-graph-objfallback (v1) | 0.5445 +- 0.0787 | 27 | pending |
| 2 | arc3-team-37-graph-inverse (v1) | 0.5041 +- 0.0445 | 18 | pending |
| 3 | arc3-team-36-graph-patch-clickprior (v1) | 0.4943 +- 0.0445 | 15 | pending |
| 4 | arc3-team-35-graph-patch (v1) | 0.4929 +- 0.0445 | 9 | pending |
| 2 | arc3-team-33-graph-clock-prune (v1) | 0.4842 +- 0.0366 | 9 | pending |
| 3 | arc3-team-31-graph-stats-ppo (v1) | 0.4611 +- 0.0379 | 9 | pending |
| 4 | arc3-team-21-ppo-rnd (v1) | 0.0247 +- 0.0317 | 5 | pending |
| 5 | arc3-team-20-ppo-base (v1) | 0.0004 +- 0.0006 | 1 | pending |
| 6 | arc3-team-11-dqn-novelty (v1) | 0.0000 | 0 | pending |
| - | arc3-team-10-dqn-base (v1) | 0.0000 | 0 | 0.00 |
Note: graph versions are within noise on score; TEAM-37 has the best score AND the most levels (18).
