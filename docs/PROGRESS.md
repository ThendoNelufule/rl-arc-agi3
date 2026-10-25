# Progress
Current phase: P5 REPORT (final revision) + P6 SHIP; P4.2 held-out done (once)
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
- 2026-10-04 02:30 TEAM-45 (one click per component) 0.7913 = BEST. FINAL if nothing changes: TEAM-45.

- 2026-10-16 PLAN CHANGE: final-submission push based on the strongest public approach (team decision;
  our TEAM-37 scored 0.07 on the public LB; corrected 2026-10-25 from the submissions API, TEAM-45 was never submitted). Step 1 done: TEAM-12 notebook pushed; MAX_ACTIONS=500
  confirmed effective. Step 2 (recon of public notebooks/rules) in progress.

## Submission queue (ranked by dev official score; Kaggle = 1 submission/day, never used for tuning)
| rank | notebook (version) | dev official (3x3) | dev levels | Kaggle public |
| 0a | **arc3-m2-base-sub (v1) = attributed dfranzen M2 base; SUBMIT 2026-10-16**  | n/a (LLM agent; public LB of original 27.9, copies 25.8 +- 3.9) | - | pending |
| 0b | **arc3-m2-mod1-sub (v1) = base + stagnation restart + D' (reused, credited); SUBMIT 2026-10-17 (team, after 02:00)** | n/a | - | pending |
| 1 | arc3-team-45-graph-components (v1) | 0.7913 +- 0.0874 | 27 | pending |
| - | arc3-team-12-dqn-novelty-mask (v1, submit later; DQN champion) | 0.0044 (3x3) | 3 | pending |
| 2 | arc3-team-44-graph-objpromote (v1) | 0.5682 +- 0.0627 | 27 | pending |
| 3 | arc3-team-43-graph-objfallback (v1) | 0.5445 +- 0.0787 | 27 | pending |
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

## Submission plan (team): alternate base-sub and best MOD-sub daily until 2026-10-27; nominate highest score.
- 2026-10-17 base LB 28.83. New goal: MOD-2 toward 40 (stretch); proxy = 25 public games before any submission.
- 2026-10-17 MOD-2 proxy A/B: MOD-1 47.39 vs MOD-2 42.83 (-4.56) -> MOD-2 FAILED, no submission version.
  MOD candidate stays arc3-m2-mod1-sub. Proxy noise (per-game diff SD ~26) means small mods can't be detected
  with one pass.
- 2026-10-17 MOD-1 LB 24.88 (< base 28.83). TEAM DECISION: submit arc3-m2-base-sub daily until 2026-10-27,
  nominate the best score. No MOD-3 unless credible +10. Scores: base 28.83 (10-16), mod1 24.88 (10-17).
- 2026-10-17 MOD-F (base + OUR fixes: stub dead reasoning, stale-state message + prompt line, cap 525 min) smoke OK.
  TEAM DECISION: daily submission = arc3-m2-modf-sub (v1) from 2026-10-18 02:00 until 2026-10-27; every roll must be a
  modified version (brief req. 6); fallback = arc3-m2-mod1-sub. Nominate the best-scoring modified version.
- 2026-10-25 Kaggle rolls 18-25 Oct (attribution verified from the submissions API `url` field):
  MOD-F (arc3-m2-modf-sub v1, scriptVersionId 359469291) x6: 28.40, 22.10, 0.00, 29.15, 0.00, 0.00;
  MOD-1 21.58 (2nd roll; 1st 24.88); TEAM-12 0.00. MOD-F mean 13.28 (non-zero runs 26.55).
  All three MOD-F zeros produced a byte-identical 3411 B parquet (scoring runs 3777-3823 B; TEAM-12's 0.00 = 3505 B),
  i.e. no game scored at all = whole-run failure, not a weak run. Investigation (code + smoke log, no submissions):
  no MOD-F-specific defect found (see REPORT_NOTES "MOD-F zeros"). TEAM DECISION: MOD-F stopped; remaining rolls
  (25 Oct night, 26 Oct) = arc3-m2-mod1-sub v1.
- 2026-10-25 P4.3 NOMINATION: MOD-F arc3-m2-modf-sub v1, submission 57556604, public 29.15, 2026-10-23 09:42:37 UTC
  (our best score; contains our own modifications). Switch only if a MOD-1 roll beats 29.15 before the deadline.
- 2026-10-25 P4.2 held-out DONE (--final, run exactly once, team-approved; Kaggle CPU kernels heldout-rl + heldout-graph):
  random 0, T10 0, T12 0, T20 0, T21 0.0005 (1 lv tu93), T22 0, T37 0.1334 (22 lv), T45 0.8848 (34 lv; ~3/4 from
  tn36 L1 in 22-45 actions, human 32), T45n 0.7667 (28 lv). bp35/sk48 never solved.
  Train split (kernel train-key, training seed 0): T12 0.0002, T21 0.2395 (fits its own training games), T37 0.0305,
  T45 0.0203 (11 lv), T45n 0.0811 (earlier run).
- 2026-10-25 Ablation P4.1: T12a = T12 without novelty (mask only), 3 seeds training on T4 (kernel arc3-train-team12a).
- 2026-10-25 Report rewritten as one research story (RL -> graph -> state of the art -> LLM platform); authors filled;
  MODIFICATIONS.md + EVIDENCE.md + CONTRIBUTIONS.md in submission/; Paper Track text report/paper_track.md (1,371 words);
  Moodle zip via scripts/make_moodle_zip.sh. Daily rolls 25/26 Oct: arc3-m2-mod1-sub v1 (team clicks).
