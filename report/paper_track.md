# Where ARC-AGI-3 agents stall: from DQN to a pretrained LLM agent

**All scores are on the official 0–100 scale:** 0.79 means 0.79 out of 100, not 79%.

**Team Vladmir_Putin**: Musala Ndouvhada and Thendo Nelufule (University of the Witwatersrand, COMS4061A/7071A).
**Linked submission:** notebook `thendonelufule/arc3-m2-modf-sub` v1 (scriptVersionId 359469291), submission 57556604, public score 29.15 (2026-10-23 09:42:37 UTC).
**Code, logs, full report and MODIFICATIONS.md:** https://github.com/ThendoNelufule/rl-arc-agi3

> **Attribution.** The scored agent is **Daniel Franzen's** public *ARC-AGI-3 Milestone 2 Solution* ([notebook](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution), [code](https://github.com/da-fr/arc-agi-3-solution), Apache-2.0). It is built on **Tufa Labs' Duck harness** ([code](https://github.com/Tufalabs/duck-harness), Jeroen Cottaar et al.) and serves Qwen3.8-Flash-Next (Intel AutoRound W4A16, Albucino's MTP drafter) through Pennyroyal (John Pezzulli's SGLang fork; patches by Mamy Ratsimbazafy and Gabriel Olympie). **The agent is not our work.** Ours are the research question, the RL and graph agents, the analysis, and three small fixes, listed with their evidence in MODIFICATIONS.md.

## 1. Question

Can agents trained with reinforcement learning learn to explore and solve unseen ARC-AGI-3 games? If not, what limits them? We followed the question from DQN and PPO, through a graph explorer, to the strongest public LLM agent. Every step was motivated by the previous result.

## 2. Learning agents: the bottleneck is exploration inside an unseen game

**Setup.** A Gymnasium adapter (64×64 colour ids; `Discrete(70)` = ACTION1–5, ACTION7, 64 grid clicks; reward = new levels, training only). The 25 public games are split once: 15 train, 5 dev, 5 held-out (evaluated once). Protocol: 500 actions per game, 3 training × 3 evaluation seeds.

| version (local official score) | dev | held-out | Kaggle |
|---|---|---|---|
| random | 0.0004 | 0 | 0.02 (starter) |
| DQN with test-time TD (T10) | 0 | 0 | 0.00 |
| + count novelty (T11) | 0 | n/a | 0.00 |
| + action-effect masking (T12) | 0.0044 | 0 | 0.00 |
| PPO (T20) | 0.0004 | 0 | 0.00 |
| + RND (T21) | **0.0247** | 0.0005 | 0.05 |
| + LSTM (T22) | 0.0003 | 0 | n/a |

- **No learning signal.** Each training run sees only 3–8 level completions in 400k steps. DQN's Q-function stays flat, so the frozen DQN spends 77–93% of its actions on one action. PPO barely moves from uniform.
- **Novelty changes behaviour, not results.** 27–33% of all actions repeat a (state, action) pair already seen to change nothing.
- **Effect masking helps, but not through pretraining.** Masking those actions completes the first DQN levels. The pretrained effect head is at chance on unseen games (AUC 0.42–0.58); the gain comes from exact per-game memory.
- **RND fits training games.** It is the largest learned gain, yet it scores 0.24 on its own training games and 0.0005 on held-out games: it fits its training games rather than learning to explore.
- **Recurrence fails.** The LSTM does not learn to avoid repeats.

**Lesson:** the useful knowledge (what does nothing, where the agent has been) is specific to the game, so it must be explicit and acquired online.

## 3. Making it explicit: a graph explorer, and dev overfitting

- **Design.** Following *Stochastic Goose* (Smit, 2025), our training-free explorer keeps an exact state graph per level, never re-tries known no-ops, walks to the nearest frontier state, and ranks untried actions by per-game Thompson-sampled effect estimates.
- **Fixes from recordings:** masking step-counter rows (which make every frame unique), learned inverse action pairs, and one click per connected component once the 8×8 grid had exhausted the reachable states.
- **Dev result.** The score rose to **0.791 out of 100**, with lp85 and lf52 level 1 at human speed.
- **Transfer is narrow.**
  - An earlier version falls from 0.504 on dev to 0.133 on held-out and **0.07 on Kaggle**.
  - The final version scores 0.885 on held-out, about three quarters of it from one game (tn36, solved at human speed), but only 0.020 on 15 other public games.
  - No version finishes a game. It stalls when its frontier is exhausted, because it has no hypothesis about the goal.
- **Gap to LLM agents.** Averaged over all 25 public games it scores about 0.35 out of 100. Franzen's LLM agent scores 46.5 on the same games, two orders of magnitude more.

## 4. Research phase: the state of the art

- **What we studied:** the milestone winners (Stochastic Goose, Duck, Franzen M2), 26 public notebooks (each diffed against its parent), OpenAI's post on retained reasoning, and the leaderboard.
- **Excluded** (integrity): a byte-identical copy with credits removed, notebooks with hard-coded game IDs or opaque encoded patches, and a solver that reads game source and hidden state.
- **Choice.** Nearly every strong public notebook derives from Franzen's M2, and the two leaders publish no code. We adopted Franzen's M2 as a **research platform**, to test whether our findings also limit a strong pretrained agent.

## 5. Our work on the platform

**Reproducible fork and evaluation proxy (ours).**
- **Fork.** A pinned Docker image (the 1 Oct image update broke the bundled wheels), with every variant generated by a script. In the submission packaging, heavy cells run only in the competition rerun; development versions have a 3-game smoke test.
- **Reproduction.** Our unchanged fork scores 28.83 on Kaggle (Franzen: 27.89).
- **Proxy.** It plays all 25 public games for one pass with the competition's per-game compute share (~2 GPU-hours).

**Where the points go (ours).**
- **Franzen's trajectories.** On his 100 public trajectories (reproducing every stored score), later-level stalls cost **43.9 points**, an unsolved level 1 costs 7.0, and inefficiency on solved levels only **2.6**.
- **Our logs.** Our proxy logs (2,341 replies) agree: 50.2 points lost to unfinished levels vs 2.46 to excess actions.
- **Loss searches.** Eight independent, LLM-assisted read-only loss searches over logs and code, each claim re-checked by us, found no single large leak.

The LLM agent, like ours, is efficient where it knows what to do and stalls where it runs out of hypotheses.

**Hypothesis tests.**
- **Retained reasoning** (OpenAI's lever) is already active. Thinking is re-sent every turn (prompt growth per generated token: median 1.83; prefix-cache hit 93%), so we built nothing.
- **MOD-1** (two reused add-ons: heliosli's stagnation restart and shiiin9's D′ scheduler) scored 24.88 and 21.58 on Kaggle: no gain.
- **MOD-2** adds **our reinvestment rule**, which decays a game's GPU priority after 130k tokens on one level, because late tokens return about half the average points. Against a pre-registered +2 threshold in a matched A/B test, it scored **−4.56** (95% CI −15.2 to +5.4). Post-mortem: our rule starved seven games of GPU time. We did not submit it.
- **MOD-F**, our three fixes from the audit:
  1. Reasoning-only replies cut at the 12,288-token cap were re-sent verbatim (13 replies, about 13k tokens each). We enable the harness's existing stub.
  2. A refusal message falsely said a stale-state error "can be caught", and 92 actions were silently refused as a result. We corrected it and added one prompt line.
  3. Proxy timings projected 32,524 s against a 32,400 s limit, so we cut the per-game cap from 532 to 525 minutes.

## 6. What a Kaggle score can show

| version | public scores (every run) |
|---|---|
| base | 28.83 |
| MOD-1 | 24.88, 21.58 |
| **MOD-F** | 28.40, 22.10, 0.00, **29.15**, 0.00, 0.00 |

- **Variance.** Public copies of this notebook score 25.77 ± 3.93. A leaderboard entry is a team's best submission, so the expected best of N grows to about 35.6 at N = 100. Franzen's 38.85 from 106 entries is consistent with that plus genuine gains.
- **No claimed effect.** All our non-zero runs lie inside the spread, so **we do not claim that any modification changes the expected score**. One proxy pass resolves only ±10-point effects.
- **Three zero runs.** Three MOD-F runs produced byte-identical outputs in which no game scored. Our changes are deterministic, the same version scored three times, and the inputs were unchanged. We found no code cause; rerun logs are not visible to participants.

## 7. Conclusion

RL agents trained on public games did not learn to explore unseen ones. Explicit per-game memory helped, but only where its mechanisms applied. The LLM agent shares the bottleneck: it is efficient on levels it understands and stalls on later ones. Inferring goals, not covering states faster, is where both must improve. For other teams: report every submission, not the best one, because best-of-N alone can add about 10 points.

*Generative-AI use:* we used Claude and Claude Code (Anthropic) for planning, code, analysis (including the loss searches) and drafting. We verified all numbers against our logs.
