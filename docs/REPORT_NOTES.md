# Report notes (one entry per evaluated version)
Template: ## <kernel tag> | limitation observed | hypothesis | intervention | config & hyperparams
| compute (train/eval time, hardware) | results (run tag, official score mean+-std, levels, curves) | conclusion | failure cases

## TEAM-00-random (sanity check)
- dev, 3 seeds, budget 500: official 0.0004 +- 0.0006; 1 level total (lp85, seed 1). Kaggle public: 0.02.
- Observation: deaths identical across seeds on dc22/ka59/lf52 (per-life step limit).
