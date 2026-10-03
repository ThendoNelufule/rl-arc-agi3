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
