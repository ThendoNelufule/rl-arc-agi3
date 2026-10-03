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
  pinned image py3.12.13/torch 2.10. Commit-run smoke test OK. Public LB score: 0.00 (reported by
  team 2026-10-04; consistent with dev 0.0000; random baseline on LB was 0.02).

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


## Kaggle CPU re-runs of TEAM-10 / TEAM-11 (consistency) | 2026-10-04
- TEAM-10 0.0000 (0 levels), TEAM-10f 0.0000, TEAM-11 0.0000, TEAM-11f 0.0000: identical to the
  local results. Eval time on Kaggle CPU (1 thread): dqn-online ~1050-1190 s per train seed,
  frozen ~50-60 s.
- TEAM-10 Kaggle public LB: 0.00 (logged; not used for any decision).

## GRAPH-v1 (exploratory, cross-chain): graph exploration + TEAM-12 pretrained effect head
- Run before the "stronger algorithm" was known; uses DQN TEAM-12's effect head (3 train seeds).
- Results (dev, 3x3): 0.0013 +- 0.0010, 4 levels (all lp85). Ablation without effect head
  (graph-noeff, 3 eval seeds): 0.0002, 1 level. Random (Kaggle CPU): 0.0004, 1 level.
- Behaviour: noop 0.26-0.31 (random 0.35), repeated known no-ops 0-4% (TEAM-12 head: graph
  never re-tries a recorded no-op unless the state is exhausted). Eval ~860-920 s/seed (online BCE).
- KEY DIAGNOSTIC: AUC of the pretrained TEAM-12 effect head on dev random transitions:
  within-game 0.42-0.58 (chance; pooled 0.19-0.32 because base rates differ by game) for all
  3 seeds. The pretrained effect predictor does NOT transfer to unseen games; GRAPH-v1's gain
  comes from the exact graph memory + online fine-tuning, not from pretraining.
  -> effect knowledge must be learned online per game, fast (graph v2 = TEAM-30/31).

## TEAM-22-ppo-rnd-lstm (P3.3) | 2026-10-04 -- FAILED to improve
- Limitation observed (TEAM-21): memoryless policy re-tries known no-ops (8-10% of actions).
- Hypothesis: a recurrent policy can remember what it already tried within an episode.
- Intervention (stacked on TEAM-21): masked RecurrentPPO (our masking via a mask observation
  channel, rl/recurrent.py), LSTM 256, same RND (0.05) and PPO hyperparameters, 400k steps.
  Eval `ppo-lstm`: frozen weights, LSTM state carried within an episode (reset at game start
  and after GAME_OVER, as in training).
- Compute: 3 seeds x 400k on one T4 kernel, 0.93-0.94 h each (2x TEAM-21). Eval Kaggle CPU
  114-116 s per train seed.
- Training: levels 5/5/8; entropy 3.45 -> 2.72-2.78 (TEAM-21 ~3.0).
- Results (dev, 3x3): 0.0003 +- 0.0004, 1 level (lp85) vs TEAM-21 0.0247, 5 levels.
- Behaviour: repeated known no-ops 0.059/0.094/0.138 of actions (TEAM-21 0.084-0.100): no
  reduction -> the LSTM did not learn to use memory for "don't repeat what failed" from 400k steps
  on 15 games; slightly more peaked policy, fewer lucky lp85 solves.
- Conclusion: learned implicit memory fails here; explicit memory (state graph) is needed.
  TEAM-21 remains the PPO-chain champion.
- Launch-timing note: trained speculatively before TEAM-21's result to keep the GPU busy; the
  adoption decision used TEAM-21's evidence (memoryless repeats) as pre-planned.

## TEAM-12-dqn-novelty-mask (P2.3) | 2026-10-04
- Limitation observed (TEAM-11): 27-33% of all actions repeat a (state, action) already seen to do
  nothing; novelty alone cannot stop it fast enough.
- Hypothesis: predicting and masking ineffective actions removes that waste.
- Intervention (stacked on TEAM-11): QEffNet = QNet + action-effect head (same encoder; MLP for
  simple actions, spatial 1x1 conv for clicks) trained with BCE (weight 1.0) on "frame changed";
  actions with p(change) < 0.1 masked in training and eval (never all); eval also keeps an exact
  per-game no-op memory and fine-tunes the effect head online with the TD update.
- Compute: 3 seeds x 400k on one T4 kernel, 0.49-0.50 h each. Eval Kaggle CPU 1 thread:
  ~1334-1382 s per train seed online, ~55 s frozen.
- Training: frame-change rate 0.77-0.82 (TEAM-11 0.67, TEAM-10 0.61); levels 4/6/4.
- Results (dev, 3x3): official 0.0044 +- 0.0063, 3 levels (TEAM-11: 0.0000, 0 levels).
  Frozen variant TEAM-12f: 0.0036 +- 0.0033, 6 levels.
- Behaviour (online): repeated known no-ops 0.169/0.012/0.042 (TEAM-11 0.27-0.33) -> reduced
  strongly in 2 of 3 seeds; noop 0.19-0.36.
- Conclusion: masking ineffective actions is the first DQN-chain change that produces levels on
  dev; but the pretrained effect head itself is chance-level on unseen games (GRAPH-v1 note), so
  the gain comes from the exact no-op memory + online fine-tuning. DQN chain champion = TEAM-12.
- Kaggle: notebook not pushed yet (DQN chain far below graph track; queued if requested).

## EXTRA IMPROVEMENT: model-based graph exploration on the stronger algorithm (PPO)
Credit: the action-effect idea (predict which actions change the frame, prefer those) follows the
ARC-AGI-3 preview winner "Stochastic Goose" (Dries Smit / Tufa Labs, 2025), which trained a CNN
online to predict frame-changing actions. Implemented by us (rl/policies.py GraphExplorePolicy,
GraphStatsPolicy): exact frame-hash state graph, recorded (s,a)->s' edges, never re-trying
recorded no-ops/deadly actions, BFS to the nearest frontier state with untried actions, graph
cleared per level, ranking of untried actions by an effect model; Thompson-sampled per-game
online effect statistics keyed by action (simple) / colour under the click point (clicks);
optional pretrained-PPO prior. No undo/reset beyond the frozen action space (ACTION7 is just an
action; harness resets on GAME_OVER).

## TEAM-30-graph-stats | 2026-10-04
- Limitation observed (GRAPH-v1 + TEAM-12 diagnostic): pretrained effect heads are chance-level
  (AUC 0.42-0.58) on unseen games; learned memory (TEAM-22) failed -> knowledge must be explicit and
  acquired online per game.
- Hypothesis: explicit state-graph memory + a fast, per-game, online effect model (Beta-Thompson
  over action / clicked colour) explores far more efficiently than any trained policy here.
- Intervention: graph exploration (above) with online stats only; no trained weights at all.
- Compute: no training. Eval Kaggle CPU 1 thread: 96 s for 9 eval seeds (~14 ms/action incl. env).
- Results (dev, 9 eval seeds; no training seeds exist): official 0.3895 +- 0.1488 (over 9 runs),
  9 levels: lp85 level 1 in EVERY run after 16-72 actions (median 20).
- Behaviour: noop 0.25 overall; dc22 0.15 (random 0.46), ka59 0.11 (0.34); lp85 0.986:
  after L1 all 64 click-cell centres are exhausted with no effect, and the agent spends the rest
  of the budget re-trying them (repeated-known-no-op share 0.164, almost all on lp85).
- Smoke-test note: posterior MEAN ranking made one movement action win every new state (ls20 95%
  one action); switched to Thompson sampling before the dev eval.

## TEAM-31-graph-stats-ppo | 2026-10-04  -- CURRENT BEST
- Limitation observed (TEAM-30): ties among untried actions broken randomly; on movement games
  the agent wanders.
- Hypothesis: the best trained policy (TEAM-21 PPO+RND) carries weak but useful priors over which
  actions matter; adding pi(a|s) to the online effect sample orders the frontier better.
- Intervention: TEAM-30 + 0.3 * pi(a|s)/max pi from TEAM-21 (each train seed's own weights).
- Compute: training = TEAM-21 (0.45 h/seed T4). Eval Kaggle CPU 108-112 s per train seed.
- Results (dev, 3 train seeds x 3 eval seeds): official 0.4611 +- 0.0379 (train seeds),
  all 9 runs 0.4611 +- 0.0942; 9 levels (lp85 L1 every run after 12-24 actions).
- Behaviour: ls20/dc22 become directed (top-action 0.89, distinct 0.92-0.93) but still 0 levels.
- Conclusion: best agent so far by a wide margin. Remaining blockers: (1) click games stall once
  the 64 click centres are exhausted (design lock: 8x8 click grid) -> asked the team;
  (2) move games: no level within 500 actions despite high coverage.
- Kaggle: notebook arc3-team-31-graph-stats-ppo v1 (TEAM-21 seed-0 prior), smoke OK, ~14 ms/action.

## TEAM-32-graph-clock | 2026-10-04
- Limitation observed (TEAM-31 recordings): every dev move/mixed game shows a step-counter bar that
  advances 1 pixel per action (ls20 rows 61-62 changed in 99-100% of steps, dc22 row 63 77-83%,
  ka59 row 63 64%, lf52 row 0 100% and NOTHING else changes in lf52) -> every frame is unique, the
  graph never recognises revisited positions, and in lf52 every action looks "effective".
- Hypothesis: masking these rows restores state identity -> real graph search on move games.
- Intervention (on TEAM-31): online per-game "ticker row" detection from observed frames only
  (>=10 changing steps; row changed in >=50% of them; median <=2 changed px in that row; caused
  by >=3 distinct actions); ticker rows zeroed before hashing and change detection; ticker set is
  monotone (a flapping first version kept wiping the graph). Detected exactly the bar rows on all
  4 games, none on lp85, one rebuild per game.
- Results (dev, 3x3): 0.4615 +- 0.0377, 11 levels: lp85 L1 every run + ls20 L1 twice (after
  413/416 actions -> tiny score). Eval 2-3 min per train seed (Kaggle CPU).
- Behaviour: noop 0.33 (TEAM-31 0.23): with real state identity the frontier search now tries
  all ~70 legal actions in every recurring state (dc22 seed-0 raw no-ops 0.15 -> 0.40).

## TEAM-33-graph-clock-prune | 2026-10-04
- Limitation observed (TEAM-32 seed-0 diagnostic, confirmed by its full eval): exhaustive trying of
  every legal action in recurring states.
- Intervention (on TEAM-32): an untried action is skipped if this game's statistics for its key
  (simple action / clicked colour) show >= 3 tries with zero effect.
- Results (dev, 3x3): 0.4842 +- 0.0366, 9 levels (lp85 L1 every run, 12-20 actions).
- Behaviour: noop 0.30, distinct frac 0.58-0.63 (TEAM-32 0.55-0.57).
- Honest reading: TEAM-31/32/33 (0.461/0.462/0.484, std ~0.037 over train seeds) are within
  noise of each other; differences come from the number of clicks lp85 L1 takes (12-24).
- Diagnostic for next step: deaths on all move/mixed dev games come from a FIXED per-life step
  budget (lives ~129/125/100/65 steps on ls20/dc22/ka59/lf52 under random, PPO and graph agents;
  identical death counts across agents) -> banning the last (s,a) before each death is wrong.
- Kaggle: notebook arc3-team-33-graph-clock-prune v1 pushed (TEAM-21 seed-0 prior).

## TEAM-34-graph-timer | 2026-10-04 -- null result
- Limitation observed (TEAM-33 diagnostic): GAME_OVER on all dev move/mixed games is a fixed
  per-life step budget, yet the graph banned the last (s,a) before every death.
- Intervention (on TEAM-33): ban a (s,a) as deadly only after it caused GAME_OVER twice.
- Results (dev, 3x3): 0.4842 +- 0.0366, 9 levels -- identical to TEAM-33 (same lp85 solves;
  no new levels on move games). Behaviour essentially unchanged (noop 0.30, distinct 0.59-0.63).
- Conclusion: correct but not the bottleneck; kept (it is strictly more sensible for timer games).

## TEAM-23 effect model (PPO-experience action-effect predictor) -- negative diagnostic
- Trained alongside TEAM-21's PPO config (3 seeds, T4 0.48-0.49 h each) to give the PPO chain an
  effect model for graph exploration. Within-game AUC on dev random transitions: dc22 0.48-0.50,
  ka59 0.46-0.54, lp85 0.39-0.51 (chance), ls20 0.36-1.00 (trivial: movement changes the frame).
  Same conclusion as for the DQN head: pretrained effect prediction does not transfer to unseen
  games under either algorithm. Not evaluated as an agent (superseded by online statistics).

## TEAM-35-graph-patch | 2026-10-04  -- BEST ON DEV (within noise of TEAM-31..34)
- Limitation observed (TEAM-34 recordings): mixed games burn the budget on clicks (dc22 398/500,
  lf52 477/500 actions) while finding almost no new states (dc22 22 states; ~1 new per 100 steps
  after step 100): all 64 click cells are tried separately in every state.
- Intervention (on TEAM-34): visual click equivalence within a state: click cells with a pixel-
  identical 8x8 patch (ticker rows masked) form one group; one representative per untried group;
  a group counts as tried once any member was tried in that state.
- Results (dev, 3x3): 0.4929 +- 0.0445 (train seeds), all 9 runs 0.4929 +- 0.1000; 9 levels
  (lp85 L1 every run, 15-25 actions). Behaviour: noop 0.30, distinct 0.62-0.63 (TEAM-34 0.59-0.63).
  Seed-0 diagnostic: dc22 distinct states 22 -> 30, lf52 9 -> 18; clicks still dominate.
- Plateau analysis: ALL graph-track score comes from lp85 level 1. Remaining blockers:
  (1) click resolution (design lock, 8x8 cell centres): lp85 L2+ has zero effective centres; dc22
      and lf52 spend 78-95% of actions on clicks with almost no discoveries;
  (2) ls20: ~110-120 distinct states in 500 actions (4 timer-limited lives of ~129 steps).
- Kaggle: notebook arc3-team-35-graph-patch v1 (TEAM-21 seed-0 prior).

## Supplementary generalisation check on the 15 TRAIN games (training-free agents only)
- Purpose: the dev set has only 5 games; training-free graph agents never saw the train games,
  so the train split is an unbiased extra test bed FOR THEM (not used for the PPO-prior agents,
  whose prior was trained there). Supplementary evidence only; decisions remain dev-based.
- Results (train split, 3 eval seeds): random 0.0111 (4 levels: r11l x3, ft09); graph-stats
  (TEAM-30 logic) 0.0578 (4 levels: g50t, m0r0 x2, ft09); graph-patch-noprior (TEAM-35 logic,
  no prior) 0.1209 (7 levels on 6 games: sp80, m0r0 x3, r11l, cn04, ft09).
- Reading: the ticker/prune/click-equivalence fixes generalise beyond dev (2x graph-stats, 11x
  random on unseen games).

## TEAM-35n-graph-patch-noprior (ablation of TEAM-35's PPO prior) | 2026-10-04
- Results (dev, 9 eval seeds; training-free): 0.4713 +- 0.0729 (over runs), 15 levels:
  lp85 L1 every run (15-21 actions) + ls20 L1 in 6/9 runs (297-473 actions).
- vs TEAM-35 (with prior): 0.4929, 9 levels, ls20 0/9 -> the PPO prior slightly speeds up lp85
  clicks but biases movement and kills ls20 exploration.

## TEAM-36-graph-patch-clickprior | 2026-10-04  -- BEST ON DEV (score and levels)
- Limitation observed (TEAM-35 vs TEAM-35n): the PPO prior helps click ordering but hurts
  movement exploration.
- Intervention (on TEAM-35): prior added to click actions only; simple actions ranked by the
  online effect statistics alone.
- Results (dev, 3x3): 0.4943 +- 0.0445 (train seeds), all 9 runs 0.4943 +- 0.1000; 15 levels
  (5 per train seed): lp85 L1 every run (15-25 actions), ls20 L1 in 6/9 runs (367/473 actions).
- Conclusion: dominates TEAM-35 (score) and TEAM-35n (score) with the most levels.

## TEAM-37-graph-inverse | 2026-10-04  -- BEST ON DEV
- Limitation observed (TEAM-36 ls20 instrumentation, 3 seeds): ~370/500 steps spent trying untried
  moves, ~130 travelling to the frontier; ls20 L1 only after 367-473 actions or not at all. In every
  newly reached maze state one of the ~4 moves predictably undoes the move that led there.
- Hypothesis: learning inverse action pairs online and not re-trying the undo move cuts ~1/4 of the
  exploration cost in navigation games.
- Intervention (on TEAM-36): per game, count s -a-> s' -b-> s (undo) vs counter-examples for simple
  actions; inverse(a) = b once undo >= 2 with zero counter-examples; on reaching s' via a, (s', b)
  -> s is recorded as an assumed edge (not re-tried, usable by path search).
  Learned pairs on dev: up<->down and left<->right (ls20, dc22, ka59); none on lp85 (clicks only).
- Results (dev, 3x3): 0.5041 +- 0.0445 (train seeds), all 9 runs 0.5041 +- 0.1018; 18 levels:
  lp85 L1 every run (15-25 actions) + ls20 L1 in 9/9 runs after 128/205/287 actions
  (TEAM-36: 6/9 runs after 367/473).
- Conclusion: biggest efficiency gain on navigation so far; still no level on dc22/ka59/lf52 and no
  level 2 anywhere.
