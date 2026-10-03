# Report notes (one entry per evaluated version)
Template: ## <kernel tag> | limitation observed | hypothesis | intervention | config & hyperparams
| compute (train/eval time, hardware) | results (run tag, official score mean+-std, levels, curves) | conclusion | failure cases

## TEAM-00-random (sanity check)
- dev, 3 seeds, budget 500: official 0.0004 +- 0.0006; 1 level total (lp85, seed 1). Kaggle public: 0.02.
- Observation: deaths identical across seeds on dc22/ka59/lf52 (per-life step limit).

## Infrastructure (P1.4): reproducibility notes for the report
- Kaggle eval has no internet: SB3/sb3-contrib wheels + weights ship in a private dataset; installed
  with `pip --no-index`. Submission image: Python 3.12.13, torch 2.10.0 (CPU), numpy 2.0.2,
  gymnasium 1.2.0, 2 CPU threads. Local training venv pinned to identical library versions.
- Weights stored as tensor-only state_dicts (torch.load weights_only=True) + JSON config, which avoids
  pickle incompatibilities across Python/library versions.
- Gotcha worth one sentence: kernel images are pinned at first save; dataset mount paths differ
  across images (/kaggle/input/<slug> vs /kaggle/input/datasets/<user>/<slug>).

## Protocol decisions for P2/P3 (pre-registered 2026-10-03, BEFORE any TEAM-10 result was seen)
- THE DQN baseline TEAM-10 = DQN pretrained on the 15 train games + test-time TD learning on the
  game being played (policy `dqn-online`). What adapts at eval time: all Q-network weights, via
  online TD updates from the agent's own transitions in that game (reward = level delta, visible
  in the frame data); weights are restored to the pretrained ones at the start of every game, so
  nothing carries over between eval games. Rationale: eval games are unseen, and online adaptation
  is the only way a value-based learner can use the eval game's own reward signal.
  The frozen network (`dqn`, no eval-time updates) is reported as a VARIANT/ablation of TEAM-10,
  not as a second baseline. The same convention applies to every later DQN version.
- Seeds: 3 independent TRAINING seeds (0,1,2) per version x 3 eval seeds each. Seed 0 of each
  version runs first (drives the evidence chain); seeds 1-2 run in the background. Any version
  that ends up single-seed is flagged as such in the report with the reason (compute).
- Matched budgets: every DQN and PPO version gets 400k env steps on the same 15 train games,
  episode cap 500, same encoder (nets.Encoder). Wall-clock compute differs by algorithm and is
  logged separately (train_summary.json) from eval time (summary.json).
- Kaggle constraints (competition pages, read 2026-10-03): 110 private games (55 public LB /
  55 private LB), notebook run-time <= 9 h (CPU or GPU), internet off; Swarm plays all games in
  parallel threads. At 500 actions/game: 55k actions in 32.4k s -> <= 0.59 s/action overall.

## Compute: training hardware benchmark (2026-10-03, DQN, learning phase = after learning_starts)
| hardware | env steps/s per run | 400k-step run |
| local Ryzen 5 5500U, 6 torch threads, CPU | ~75 | ~85 min |
| Kaggle CPU kernel (4 vCPU Xeon 2.2 GHz) | ~61 | ~110 min |
| Kaggle T4 (+4 vCPU Xeon 2.0 GHz), 1 job | ~330 | ~21 min |
| Kaggle T4, 2 jobs in parallel | ~335 each | ~21 min for both |
Decision: TEAM-10 seed 0 trained locally (already running); every other training run on Kaggle T4,
all seeds of a version as parallel jobs in one kernel (scripts/kaggle_train.py), weights pulled back,
ALL evaluation local (same machine for every agent). Hardware + train time per run are in each
run's train_summary.json (host/cpu/gpu fields). Benchmark logs: kaggle_runs/bench-*/.

## Kaggle runtime estimate for test-time learning (2026-10-03)
Measured locally at 1 torch thread (= per-game thread on Kaggle): frozen dqn 1.7 ms/action,
dqn-online 89.9 ms/action (one batch-32 TD update per action). 110 games x 500 actions x 90 ms
= ~1.4 CPU-h (~2 CPU-h on Kaggle's slower Xeon), spread over parallel game threads -> well under
the 9 h limit. Safety: agent stops learning after 6.5 h and stops playing after 8 h (wall clock).
Unmeasured: gateway latency per action (check in the first real submission log).

## TEAM-10-dqn-base (P2.1) | 2026-10-03
- Limitation observed (previous stage = random): sparse reward (random: ~1 level / 7.5k actions on
  train, 1 level / 7.5k on dev); 35% of random actions on dev change nothing (lp85: 96.5%).
- Hypothesis: a value-based learner pretrained on the 15 train games, plus test-time TD learning
  on the eval game, can pick up level-completing behaviour where reward appears.
- Intervention: vanilla DQN (Mnih 2015): replay 100k (4-bit packed), hard target sync 10k env
  steps, Huber, Adam 1e-4, gamma 0.99, batch 32, train_freq 4, learning_starts 20k, eps 1.0->0.05
  over 50% of 400k steps; masked argmax/targets over legal actions. Encoder nets.Encoder
  (k4s4 -> k2s2 -> 3x3 conv to an 8x8 map aligned with click cells), spatial 1x1-conv click head
  + MLP simple-action head (1.19M params). 15 envs in lock-step (one per train game).
  Eval (pre-registered baseline `dqn-online`): eval eps 0.05, one TD update (batch 32, lr 1e-4,
  target sync 100) per action on the game being played; weights reset to pretrained per game.
- Compute: train 3 seeds x 400k env steps on Kaggle T4 (+4 vCPU Xeon 2.0 GHz), 0.30-0.32 h each
  (s1,s2 parallel in one kernel; s0 separately). Aborted local attempt (OOM at 360k, Ryzen 5 5500U)
  kept in experiments/train/ABORTED-*. Eval local Ryzen 5 5500U: ~8 min per train seed online
  (~0.13 s/action incl. env), ~1 min frozen.
- Training signal: 7 / 4 / 8 level completions in 400k steps (s0/s1/s2), from ar25, cn04, ft09,
  m0r0, r11l, sp80, tr87; Q-values flat ~0.05, loss ~1e-5 throughout -> nothing to discriminate.
- Results (dev, 3 train seeds x 3 eval seeds, budget 500): official 0.0000 +- 0.0000, 0 levels
  (random: 0.0004, 1 level). Frozen variant TEAM-10f: 0.0000 +- 0.0000, 0 levels.
  Runs: TEAM-10-dqn-base-s{0,1,2}-dev, TEAM-10f-dqn-frozen-s{0,1,2}-dev.
- Behaviour (rl/analyze_run.py, mean over dev games x eval seeds; random in brackets):
  online: noop 0.36-0.54 [0.35], distinct-state frac 0.30-0.42 [0.44], top-action share
  0.34-0.40 [0.08], action entropy 2.5-2.7 bits [5.2].
  frozen: top-action share 0.77-0.93, entropy 0.6-1.1 bits -> collapses onto ONE action.
- Conclusion: with ~6 rewards per 400k steps the Q-function stays flat, so greedy action selection
  degenerates into repeating an arbitrary action (frozen) and explores LESS than random even with
  test-time learning (online). The limiting factor is the absence of any learning signal, not
  capacity. -> P2.2: dense, task-agnostic exploration signal (count-based novelty bonus).
- Failure cases: lp85 (click game) 99.9% no-op actions; dc22 ~49% no-ops; ft09 during training
  0.4-3% of greedy clicks change the frame (feeds P2.3 action-effect masking).
- Kaggle submission (2026-10-03): notebook thendonelufule/arc3-team-10-dqn-base v1, policy
  dqn-online, weights = training seed 0 (rule fixed before any Kaggle score: always seed 0),
  pinned image py3.12.13/torch 2.10. Commit-run smoke test OK. Public LB score: PENDING.

## PPO chain eval-time convention (pre-registered 2026-10-03, BEFORE any TEAM-20 result)
- TEAM-20-ppo-base = MaskablePPO (sb3-contrib 2.9.0) pretrained on the 15 train games, policy
  `ppo`: actions SAMPLED from the masked policy; NOTHING adapts at eval time (no weight updates).
  On-policy PPO updates within a 500-action budget would be a few tiny batches, so test-time
  weight updates are not part of this chain; adaptation for PPO is instead studied through
  memory (P3.3 recurrent policy: the LSTM state is what adapts within a game).
- Same 15 train games, 400k env steps, episode cap 500, same nets.Encoder (SB3 linear actor and
  critic heads on the 256-d global vector), n_steps 128 x 15 envs, batch 240, 4 epochs, lr 2.5e-4,
  gamma 0.99, GAE 0.95, clip 0.2, ent 0.01. Training seeds 0,1,2 on Kaggle T4 (parallel jobs).

## TEAM-11-dqn-novelty (P2.2) | 2026-10-03
- Limitation observed (TEAM-10): ~6 rewards per 400k training steps -> flat Q; frozen policy
  collapses onto one action (top-action share 0.77-0.93), online policy explores less than random.
- Hypothesis: a dense, task-agnostic count-based novelty signal gives the Q-function something to
  learn on every game, breaking the collapse and raising state coverage -> more levels found.
- Intervention (stacked on TEAM-10, everything else identical): r_train = r_level + beta/sqrt(N(s')),
  beta = 0.05, N = exact-frame visit count; training: counts reset every episode (episodic novelty);
  eval (dqn-online): counts over the whole game, same bonus fed to the test-time TD updates.
- Compute: 3 seeds x 400k on one Kaggle T4 kernel (3 parallel jobs), 0.43 h each. Eval local,
  3 train seeds evaluated in parallel (2 threads each): ~21.6 min per seed online, ~1.2 min frozen.
- Training: frame-change rate 0.67-0.68 (TEAM-10: 0.61); Q ~1.2 (bonus-driven); levels 5/5/6
  (TEAM-10: 7/4/8) -> novelty changed behaviour in training but did not find more levels.
- Results (dev, 3x3): official 0.0000 +- 0.0000, 0 levels. Frozen variant TEAM-11f: 0.0000, 0 levels.
- Behaviour (online, s0/s1/s2): noop 0.39/0.45/0.41, distinct frac 0.36/0.33/0.33, top-action
  0.50/0.58/0.54 (TEAM-10 0.34-0.40), entropy 1.8-2.3 -> collapse NOT fixed (slightly worse).
- Key diagnostic (new metric, recordings): share of ALL actions that repeat a (state, action) pair
  already observed to leave the frame unchanged: TEAM-11 0.27/0.33/0.27 (= 66-72% of its no-ops),
  TEAM-10 0.22-0.39, frozen 0.32-0.37, random 0.07. Count bonus only penalises such repeats
  slowly via TD, so a third of the 500-action budget is burnt on known no-ops.
- Conclusion: novelty is not enough because the budget is dominated by repeated ineffective
  actions; the agent needs to stop choosing actions it knows (or can predict) do nothing.
  -> P2.3: action-effect masking (exact per-game no-op memory + learned effect predictor).
- Failure cases / limitation: lp85 99.9% no-ops for EVERY agent incl. random (96.5%): none of the
  64 click-cell centres hits an effective target -> 8x8 click resolution (design lock) is the
  bottleneck there, not exploration.
- Kaggle: notebook arc3-team-11-dqn-novelty v1 (seed 0 weights), smoke OK; public LB PENDING.

## TEAM-20-ppo-base (P3.1) | 2026-10-03
- Limitation observed (random reference): sparse reward, ~35% no-op actions; DQN chain collapses
  onto repeated actions (TEAM-10/11).
- Hypothesis: an on-policy, entropy-regularised actor-critic (MaskablePPO) trained on the same 15
  games learns transferable behaviour without the value-collapse seen in DQN.
- Intervention: MaskablePPO, shared nets.Encoder, hyperparams pre-registered above; 400k steps.
  Eval `ppo`: frozen policy, actions sampled from the masked distribution.
- Compute: 3 seeds x 400k on one Kaggle T4 kernel (3 parallel jobs), 0.37 h each.
  Eval local: ~80 s per train seed (3 eval seeds), 3 seeds in parallel.
- Training: levels 3/5/5 in 400k steps; policy entropy 3.49 -> 2.87-3.04 nats (max over legal
  ~3.5-4.2) -> the policy barely moved from uniform; frame-change rate 0.63-0.65.
- Results (dev, 3x3): official 0.0004 +- 0.0006 (train seeds), all 9 runs 0.0004 +- 0.0011;
  1 level total (lp85, train seed 0, eval seed 2). Random: 0.0004, 1 level -> indistinguishable.
- Behaviour: noop 0.36, distinct frac 0.37-0.44, top-action 0.15, entropy 4.5-4.7 bits
  (random 0.35 / 0.44 / 0.08 / 5.2) -> near-random behaviour, no collapse.
- Conclusion: PPO does not collapse like DQN but learns essentially nothing: with ~4 extrinsic
  rewards per 400k steps the only gradient is the entropy bonus. -> P3.2: intrinsic motivation
  (RND), a learned novelty signal that, unlike TEAM-11's exact counts, generalises across
  near-identical frames.
- Kaggle: notebook arc3-team-20-ppo-base v1 (seed 0), smoke OK; public LB PENDING.

## RND implementation notes (P3.2 development, before any TEAM-21 result)
- First smoke tests: RND bonus collapsed to ~1e-4 within 4k steps. Debugging showed the random
  target's features barely vary across ARC frames (mostly identical background): predictor error
  fell 2e-3 -> 1e-6 in ~90 updates. Fixes: orthogonal init (gain sqrt 2) for the target, running
  per-dimension standardisation of target features, bonus normalised by an EMA of the mean error
  (alpha 0.01) so the average bonus stays ~coef per step. After the fix the raw error plateaus at
  ~0.1 and the bonus at 0.02-0.05/step. coef = 0.05 (same scale as TEAM-11's beta).

## PROTOCOL CHANGE: evaluation moved to Kaggle CPU kernels (approved by team, 2026-10-04 00:00)
- Why: the local laptop (7.3 GB RAM, ~6 GB held by other apps) OOM-killed a batch of 6 parallel
  local evals (TEAM-12 online + graph explorer; none finished, partial outputs deleted).
- What changed: WHERE evaluation runs, not HOW. rl/evaluate.py protocol is unchanged (500-action
  budget across deaths, 3 eval seeds, fresh scorecard per seed, everything logged, held-out guarded).
  Runs on Kaggle CPU script kernels (4 vCPU Xeon, ~30 GB RAM, no internet), up to 4 eval jobs in
  parallel per kernel at 1 OMP thread each, arcade in OFFLINE mode (local scorecard; verified
  identical to NORMAL mode locally: same seeds -> same score 0.0016 and levels on lp85,ls20).
  Checkpoints for every training seed are in the dataset (arc3-rl-assets/weights/ckpt/).
- Consistency: TEAM-10, TEAM-11, TEAM-20 and the random reference were RE-RUN on Kaggle CPU, so
  every reported version is evaluated on identical hardware. The earlier local results are kept in
  experiments/runs/_local_superseded/ (TEAM-10/11 0.0000, TEAM-20 0.0004 locally) and are only
  used as a cross-check. Eval wall-clock times from now on refer to Kaggle CPU (1 thread per job).
- Every run folder pulled from Kaggle carries evaluated_on.txt naming its kernel.

## TEAM-20 re-run on Kaggle CPU (consistency check)
- Kaggle CPU result identical to the local one: 0.0004 +- 0.0006, 1 level (lp85, train s0, eval
  seed 2) -> evaluation is deterministic given seeds and hardware change did not alter results.
  Eval time: ~116 s per train seed (3 eval seeds), 1 thread.

## TEAM-21-ppo-rnd (P3.2) | 2026-10-04
- Limitation observed (TEAM-20): ~4 extrinsic rewards per 400k steps -> PPO barely moves from
  uniform (entropy 3.49 -> ~3.0 nats); dev behaviour = random.
- Hypothesis: a learned novelty signal (RND) that generalises across near-identical frames gives
  PPO a dense gradient towards state-changing, novel behaviour that transfers to unseen games.
- Intervention (stacked on TEAM-20): RND intrinsic reward, coef 0.05, single combined reward
  stream (simplification vs Burda et al.'s two value heads), target orthogonal init + per-dim
  standardised features, bonus / EMA(mean error). RND only during training; eval = frozen `ppo`.
- Compute: 3 seeds x 400k, one Kaggle T4 kernel (3 parallel jobs), 0.45-0.46 h each.
  Eval Kaggle CPU 1 thread: ~52-118 s per train seed.
- Training: levels 5/7/6 (TEAM-20: 3/5/5); intrinsic bonus held at ~0.049/step; entropy ~3.0.
- Results (dev, 3x3): official 0.0247 +- 0.0317 (train seeds), all 9 runs 0.0247 +- 0.0651;
  5 levels (train seeds 1/2/2). TEAM-20: 0.0004, 1 level.
  Levels: lp85 in 4/9 runs (after 124-489 actions), ls20 once (41 actions -> that single eval
  run scores 0.2085 and dominates the mean: efficiency is squared). High variance, n=3.
- Behaviour: noop 0.355, distinct frac 0.40-0.45, top-action 0.11-0.13, entropy 4.84-4.94 bits;
  repeated-known-no-op share 0.084-0.100 of actions (24-28% of no-ops; TEAM-20 0.07-0.13).
- Conclusion: RND is the first intervention that moves dev performance (5x levels, 60x score),
  mostly through lp85 and one efficient ls20 solve; the policy is still memoryless and still
  re-tries known no-ops ~9% of the time -> P3.3 adds memory (masked LSTM, TEAM-22).
- Kaggle: notebook arc3-team-21-ppo-rnd v1 (seed 0 weights), smoke OK; public LB PENDING.
- Note on TEAM-22 timing: TEAM-22 training was launched before this result (to keep the GPU busy);
  its adoption as P3.3 is decided by the evidence above (memoryless repeats), as pre-planned.
