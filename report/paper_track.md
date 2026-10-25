# Exploration, not value estimation, limits learning agents on ARC-AGI-3

**Team Vladmir_Putin** (University of the Witwatersrand, COMS4061A/7071A Reinforcement Learning). [NAME 1], [NAME 2].
**Linked code submission:** `thendonelufule/arc3-m2-modf-sub` (script version 359469291), submission 57556604, public score 29.15 (2026-10-23 09:42:37 UTC).
**Code and data:** https://github.com/ThendoNelufule/rl-arc-agi3. **Full report:** see the attached PDF.

> **Attribution first.** Our scored entry is an attributed fork of **Daniel Franzen's** public *ARC-AGI-3 Milestone 2 Solution* ([notebook](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution), [code](https://github.com/da-fr/arc-agi-3-solution), Apache-2.0). That solution is built on **Tufa Labs' Duck harness** ([code](https://github.com/Tufalabs/duck-harness), Jeroen Cottaar et al.) and serves Qwen3.8-Flash-Next (Intel AutoRound W4A16, Albucino's MTP drafter) through Pennyroyal, John Pezzulli's SGLang fork, with patches by Mamy Ratsimbazafy and Gabriel Olympie. The agent, model, serving and scheduler are theirs. Our own contribution to the entry is three small fixes, listed below. Our own research is the RL and graph-exploration study.

## 1. Question

Can a learning agent trained offline on the 25 public games learn exploration behaviour that completes levels in unseen ARC-AGI-3 games? If not, what limits it? We answered this under one frozen protocol before switching to an LLM agent for the leaderboard.

## 2. Setup

- **Adapter.** Gymnasium environment. Observation: 64×64 colour ids. Actions: `Discrete(70)` = ACTION1–5, ACTION7, and 64 clicks at the centres of an 8×8 grid. Reward = newly completed levels (training only).
- **Split.** The 25 public games were split once, stratified by move/click/mixed, and frozen: 15 train, 5 dev (all decisions), 5 held-out (evaluated once at the end).
- **Protocol.** 500 actions per game, counted across deaths. Fresh scorecard per seed. 3 training seeds × 3 evaluation seeds. We report the official scorecard score.

## 3. Learning agents: both baselines are indistinguishable from random

- **Two chains, all runs reported.** DQN (Mnih et al., 2015; test-time TD updates on the eval game) and MaskablePPO (Schulman et al., 2017) were each trained for 400k steps on the 15 training games. Each got two stacked improvements, and each improvement is motivated by a diagnostic from the previous stage.

| Version | Dev score (mean ± SD over training seeds) | Levels / runs |
|---|---|---|
| Random | 0.0004 | 1 / 3 |
| DQN base | 0.0000 | 0 / 9 |
| + count novelty | 0.0000 | 0 / 9 |
| + action-effect masking | 0.0044 ± 0.0063 | 3 / 9 |
| PPO base | 0.0004 ± 0.0006 | 1 / 9 |
| + RND | **0.0247 ± 0.0317** | 5 / 9 |
| + LSTM | 0.0003 ± 0.0004 | 1 / 9 |

- **No learning signal.** In 400k steps each training run sees only 3–8 level completions, and DQN's Q-function stays flat.
  - The frozen DQN puts 77–93% of its actions on a single action.
  - PPO's entropy barely moves from uniform.
- **Repeated no-ops.** Count novelty changed behaviour but not results: 27–33% of all actions repeated a (state, action) pair already seen to change nothing.
- **Effect masking helped, but not via pretraining.** The pretrained effect head is at chance on unseen games (within-game AUC 0.42–0.58). The gain came from exact per-game no-op memory.
- **RND was the largest learned gain.** It needed fixes before it worked: target-feature standardisation and EMA-normalised bonuses, because ARC frames are mostly identical background.
- **Recurrence failed.** The LSTM did not learn to avoid repeats.

**Lesson:** the binding constraint is exploration inside an unseen game. The knowledge that helps (what does nothing, where the agent has been) is game-specific, so it must be explicit and acquired online.

## 4. A training-free graph explorer

- **Starting point.** Following *Stochastic Goose* (Smit, 2025), we built an explorer with:
  - an exact frame-hash state graph per level;
  - no re-tries of recorded no-op or deadly actions;
  - BFS to the nearest frontier state;
  - Beta-Thompson online estimates of which actions or clicked colours have effects.
- **Each step was diagnosed from recordings:**
  - online detection and masking of step-counter "ticker" rows, which otherwise make every frame unique;
  - learned inverse action pairs: ls20 level 1 in 9/9 runs after 128–287 actions, instead of 6/9 after 367–473;
  - one click per connected component, after we showed that the 8×8 click grid had exhausted the reachable state space.
- **Results.** On dev it went from 0.389 (online statistics only) to **0.791 ± 0.087** (component clicks). Level 1 of lp85 and lf52 is now solved at human speed.
- **Held-out (evaluated once).** The component-click version scores 0.885: tu93 two levels, tn36 and cd82 level 1. The earlier inverse-action version drops from 0.504 on dev to 0.133. Bp35 and sk48 stay unsolved.
- **Kaggle.** The inverse-action version scored **0.07** on the hidden games. Gains measured on 5 dev games are fragile, and no version ever finished a game or got past early levels, which RHAE weights most.
- **PPO-prior ablation.** Removing the trained PPO prior costs score through ls20 but changes the levels reached little. The structure does the work, not the learning.

## 5. Our leaderboard entry and our three fixes (MOD-F)

We audited Franzen's agent on a one-pass, 25-game proxy (2,341 model replies) and made three changes.

1. **Truncated reasoning.** Reasoning-only replies cut off at the 12,288-token cap were kept verbatim: 13 replies re-sent about 13k tokens of broken reasoning into later requests. We switch on the harness's existing stub for them (`ARC3_STUB_DEAD_REASONING=1`).
2. **False advice.** The stale-state refusal told the model the error "can be caught … to keep acting", which is false: every later action in that snippet is refused too. 92 actions were silently refused in 14 snippets that followed this advice. We corrected the message and added one system-prompt line describing the no-op rules.
3. **Time-limit safety.** Proxy timings projected 32,524 s against the 32,400 s limit, so we cut the per-game cap from 532 to 525 minutes.

We also checked a published lever: OpenAI's "retained reasoning" setting. It is already active in this agent. Thinking is re-sent every turn (prompt growth per generated token: median 1.83; prefix-cache hit rate 93%), so we built nothing for it.

## 6. What a single Kaggle score can and cannot show

| Version | Public scores (every run) |
|---|---|
| Base (unchanged fork) | 28.83 |
| MOD-1 (two reused public add-ons) | 24.88, 21.58 |
| **MOD-F (ours)** | 28.40, 22.10, 0.00, **29.15**, 0.00, 0.00 |

- **Variance.** Public copies of the same notebook score 25.77 ± 3.93. A leaderboard entry is a team's best submission, so the expected best of N rises with N, to about 35.6 at N = 100 under that spread. Franzen's best is 38.85 from 106 entries, consistent with this plus genuine improvements.
- **No claimed effect.** All our non-zero runs lie inside the spread, so **we do not claim that our fixes change the expected score.**
- **Proxy noise.** A matched proxy A/B of MOD-2 (Fable facts plus our own scheduler decay) against MOD-1 differed by −4.56 points, with a 95% CI of −15.2 to +5.4 and a per-game SD of about 26. One pass resolves only effects of about ±10.
- **MOD-2 failed.** Our decay starved seven games of GPU time for most of the run. Five of them explain about 7.8 points of the drop.
- **Three zero runs.** Three MOD-F runs produced byte-identical outputs in which no game scored, i.e. whole-run failures. Our changes are deterministic, the same version scored three times, and the inputs did not change. We found no code cause, and rerun logs are not visible to participants.

## 7. Takeaways for other teams

1. Report every submission, not the best one. On this leaderboard, best-of-N alone can add about 10 points.
2. Learned exploration from 15–25 games does not transfer. Explicit, per-game, online memory does, but hand-designed orderings overfit a small dev set.
3. Both in our explorer and in Franzen's trajectories (43.9 of the lost points come from later-level stalls, but only 2.6 from inefficiency), points are lost because agents run out of hypotheses about the goal, not because they are slow on levels they solve.

*Generative-AI use:* we used Claude and Claude Code (Anthropic) for planning, code, analysis and drafting. We verified all numbers against our logs.
