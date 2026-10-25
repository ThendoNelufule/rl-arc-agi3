# Manual steps (only the team can do these), in order

Deadlines: Moodle 27 Oct 2026 23:59 SAST. Paper Track (Kaggle competition `arc-prize-2026-paper-track`): 9 Nov 2026 23:59 UTC.
We plan to submit the Paper Track on 27 Oct.

## 0. Tonight and tomorrow: daily rolls
- Click **Submit to Competition** on `arc3-m2-mod1-sub` **Version 1**.
- **If a MOD-1 roll beats 29.15**, the nomination changes. Tell Claude, which then updates `submission/EVIDENCE.md`,
  the report (abstract + §7), `report/paper_track.md` and the ZIP. Then do steps 1–2 for the new submission instead.

## 1. Screenshot of the nominated submission
1. Open https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/submissions
2. Find submission **57556604**: `arc3-m2-modf-sub - Version 1`, public score **29.15**, 2026-10-23.
3. Screenshot the row, including the score and the Select checkbox from step 2.
4. Save it as `submission/submission_57556604.png`. The ZIP builder uses it in place of the placeholder.

## 2. Select it for final scoring
- On the same page, tick **Select** on submission 57556604.

## 3. Paper Track (on 27 Oct)
1. Both members open https://www.kaggle.com/competitions/arc-prize-2026-paper-track, click **Join Competition** and accept the rules.
2. **The team must match the ARC-AGI-3 team.** In the Paper Track's **Team** tab, the team leader creates/renames the team
   `Vladmir_Putin` and invites the other member, who accepts. Each team gets **one** submission.
3. Go to https://www.kaggle.com/competitions/arc-prize-2026-paper-track/projects and click **New Writeup**.
4. Fill in:
   - **Title:** `Where ARC-AGI-3 Agents Stall: From DQN to a Pretrained LLM Agent`
   - **Subtitle:** `RL agents, a graph explorer and a fork of Franzen's M2 all lose their points to stalls on later levels, not to inefficiency`
   - **Track:** ARC-AGI-3 (required to submit)
   - **Body:** paste `evidence/paper_track/writeup.md` from the ZIP (= `report/paper_track.md`), *without* its first line (the `#` title). It is under 1,500 words (1,432 raw, including table syntax).
   - **Media Gallery:** upload `cover.png` as the **cover image** (required); optionally add `ladder.png`, `generalisation.png`,
     `kaggle.png`, `trajectories.png`, `training.png` (all in `evidence/paper_track/`).
   - **Project Links:** attach the notebook `thendonelufule/arc3-m2-modf-sub`. It may stay private: Kaggle makes attached private
     resources public after the deadline. The page shows its latest version (v2, header text only); the code is byte-identical to the
     scored v1.
   - *(Optional)* public project link: the GitHub repo (only after step 4) or a PDF of the report.
5. **Save**, check the preview, then click **Submit** (top right). Unsubmitted drafts are not judged.
6. Copy the Writeup URL into `submission/PAPER_TRACK_LINK.txt` (line "Writeup URL:").

## 4. Make the GitHub repo public (scan passed)
- **Before flipping.** Note that commit metadata contains Thendo's work email (`Thendo.Nelufule@brilliware.com`, 40+ commits).
  This is not a secret, but it becomes public with the history.
- **Flip.** https://github.com/ThendoNelufule/rl-arc-agi3 → Settings → General → Danger Zone → **Change visibility** → Public.
  The default branch is already `dev`, which holds all our work.

## 5. Rebuild and check the ZIP
```
sh scripts/make_moodle_zip.sh        # -> dist/Vladmir_Putin.zip (refuses to build if it finds a credential or binary)
.venv/bin/python -m zipfile -l dist/Vladmir_Putin.zip
```
Check that `evidence/` now holds `submission_57556604.png` (not the placeholder) and that `evidence/paper_track/PAPER_TRACK_LINK.txt`
holds the URL.

## 6. Moodle
- **One member** uploads `dist/Vladmir_Putin.zip` before **27 Oct 23:59 SAST**.
