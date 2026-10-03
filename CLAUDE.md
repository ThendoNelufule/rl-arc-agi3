# CLAUDE.md: ARC-AGI-3 RL assignment (team Vladmir_Putin)

## Who / what
Two-person Wits Honours team (COMS4061A/7071A). Assignment: two learning agents for ARC-AGI-3
(DQN = course algorithm, PPO = non-course algorithm), each baseline + 2 substantive stacked
improvements, evaluated rigorously; RLC-2026-template report (8 pages); Kaggle code submission
(ARC Prize 2026 - ARC-AGI-3) + Paper Track write-up. Moodle due 27 Oct 2026 23:59 SAST.
Goal: 100%, incl. top of the class leaderboard (official Kaggle public score).

## How to work with us (MANDATORY)
- Start AND end every session by stating the current phase, e.g. "We are at P2.1 (step 3/5)".
- AUTONOMOUS MODE (since 2026-10-03): Claude runs every step itself, sub-phase after
  sub-phase (P1.4 -> P2 -> P3), without waiting for us between steps.
- Only stop and ask us when: (a) something needs the Kaggle website ("Submit to Competition"
  click, accepting rules); (b) changing a design lock below; (c) touching held-out games
  (--final); (d) something fails and 2 serious fix attempts didn't work.
- Train locally on CPU unless clearly too slow; then propose Kaggle GPU (ask first).
- Log training time/hardware separately from eval time for every run.
- Each improvement must be motivated by the previous stage's evidence.
- After EVERY sub-phase: update docs/PROGRESS.md, docs/REPORT_NOTES.md, experiments/log.csv,
  commit AND push. When a kernel is ready to submit, push it and tell us exactly which
  kernel/version to click "Submit to Competition" on.
- End each sub-phase with a 3-line status: phase | result | what's next.
- Terminal for everything (Git Bash). Casual, direct, concise.
- When we say "handover", print a paste-ready summary: phase status, new results (tags +
  numbers), decisions, open problems. We paste it to our planning chat.

## Phase tracker
P0 IGNITION: done (team on Kaggle leaderboard, random = 0.02).
P1 SPINE: P1.1 adapter done | P1.2 splits done | P1.3 eval harness done | P1.4 Kaggle weights dataset: done
P2 DQN: P2.1 TEAM-10-dqn-base | P2.2 TEAM-11-dqn-novelty (count-based intrinsic reward)
        | P2.3 TEAM-12-dqn-novelty-mask (action-effect masking)
P3 PPO: P3.1 TEAM-20-ppo-base (SB3) | P3.2 TEAM-21-ppo-rnd (RND) | P3.3 TEAM-22-ppo-rnd-lstm (recurrent)
P4 SCIENCE: P4.1 ablations (dev only) | P4.2 held-out eval (--final, ONCE) | P4.3 champion nomination
P5 REPORT (done in planning chat from REPORT_NOTES.md) | P6 SHIP: P6.1 Paper Track | P6.2 clean notebook | P6.3 Moodle zip
Each improvement must be motivated by the previous stage's evidence. Not hyperparameter tweaks.

## Design locks (do not change without asking us)
- rl/arc_gym.py: obs (1,64,64) uint8 colour ids (one-hot inside the net); Discrete(70):
  0-4 ACTION1-5, 5 ACTION7, 6-69 ACTION6 clicks on 8x8 cell centres; reward = level delta
  (training only); terminated on WIN/GAME_OVER; illegal action = no-op; action_masks() available.
- splits.json is FROZEN: train 15 / dev 5 / heldout 5 (stratified by move/click/mixed). Never regenerate.
- rl/evaluate.py is THE protocol: budget 500 actions/game across deaths, 3 seeds, fresh scorecard
  per seed, all runs logged, held-out refused without --final. Agents implement rl/policies.py.
- Report the official scorecard score unshaped. Never present shaped training return as the score.
- No game-specific scripts or hidden state (integrity rule). Kaggle eval has NO internet:
  weights + any extra pip wheels must come from attached Kaggle Datasets.
- Fair comparisons: same games, matched interaction/compute budgets, seeds recorded.

## Landmines (Windows machine)
- NEVER run `make setup`: a stray MinGW Python on PATH hijacks it. Venv was built with
  `py -3.12 -m venv .venv`; .venv/bin is a junction to .venv/Scripts. Use `.venv/bin/python`.
- If scripts/slim_framework.py is ever rerun, fix the bad byte:
  replace b'\x97' with b'-' in vendor/ARC-AGI-3-Agents/agents/__init__.py.
- `2>/dev/null` hides sys.exit messages; use `2>&1 | tail` to see them.
- Kaggle handle: thendonelufule; notebook builds via `make submit`, status via `make status`;
  the final "Submit to Competition" click is manual on the website (5 submissions/day).
- Env speed: ~400-750 steps/sec locally.

## Known facts so far
- 25 public games: 13 mixed, 6 click, 6 move. Some games kill on a fixed per-life budget
  (dc22, ka59, lf52 die identically across seeds).
- Random on dev (p13-random-dev): official 0.0004 +- 0.0006, 1 level total across 3 seeds.
- 8x8 click grid may be too coarse for small-target click games (possible improvement).
