#!/bin/sh
# Build dist/Vladmir_Putin.zip for Moodle:
#   report.pdf | notebook/ (scored v1) | code/ (our code + configs + README) | MODIFICATIONS.md | evidence/
# Guards: no credentials, no notebook outputs, no large binaries.
# Usage: sh scripts/make_moodle_zip.sh   (run from the repo root)
set -e
PY=.venv/bin/python
OUT=dist/Vladmir_Putin
rm -rf "$OUT" "$OUT.zip"
mkdir -p "$OUT/notebook" "$OUT/code" "$OUT/evidence/paper_track"

# 1. report
cp report/main.pdf "$OUT/report.pdf"

# 2. the scored notebook: v1 (scriptVersionId 359469291), pulled back from Kaggle (cell sources identical to what we pushed)
cp kaggle_sub/m2-modf-sub-v1/arc3-m2-modf-sub.ipynb kaggle_sub/m2-modf-sub-v1/kernel-metadata.json "$OUT/notebook/"

# 3. our code, configs and a reproduction README (no weights, no game files, no third-party notebooks)
C="$OUT/code"
cp REPRODUCE.md "$C/README.md"
mkdir -p "$C/rl" "$C/scripts" "$C/agent" "$C/report" "$C/recon/analysis" "$C/configs/train" "$C/configs/kaggle_jobs"
cp rl/*.py "$C/rl/"
cp scripts/*.py scripts/*.sh "$C/scripts/"
cp agent/my_agent.py "$C/agent/"
cp report/data.py report/figures.py report/tables.py report/build.sh report/main.tex report/refs.bib report/rlj.sty report/rlj.bst "$C/report/"
cp recon/analysis/franzen_v3_failure_modes.py recon/analysis/franzen_v3_trajectories.json "$C/recon/analysis/"
cp splits.json requirements-local.txt Makefile "$C/"
cp kaggle_dataset/requirements-kaggle.txt "$C/configs/"
for d in experiments/train/TEAM-*; do
  t=$(basename "$d"); mkdir -p "$C/configs/train/$t"
  for f in config.json model.json train_summary.json; do [ -f "$d/$f" ] && cp "$d/$f" "$C/configs/train/$t/"; done
done
for j in kaggle_runs/*/jobs.json; do
  n=$(basename "$(dirname "$j")"); mkdir -p "$C/configs/kaggle_jobs/$n"; cp "$j" "$C/configs/kaggle_jobs/$n/"
done

# 4. modifications
cp submission/MODIFICATIONS.md "$OUT/"

# 5. evidence
E="$OUT/evidence"
cp experiments/kaggle_submissions.csv "$E/"
cp submission/EVIDENCE.md "$E/NOMINATED_ENTRY.md"
cp submission/CONTRIBUTIONS.md "$E/"
if [ -f submission/submission_57556604.png ]; then cp submission/submission_57556604.png "$E/"; else cp submission/SCREENSHOT_PLACEHOLDER.txt "$E/"; fi
cp report/paper_track.md "$E/paper_track/writeup.md"
cp submission/PAPER_TRACK_LINK.txt "$E/paper_track/"
cp submission/paper_track_media/*.png "$E/paper_track/"

# guards
TOKEN=$(tr -d '\r\n' < .kaggle/access_token 2>/dev/null || true)
if [ -n "$TOKEN" ] && grep -r -l -F "$TOKEN" "$OUT" >/dev/null 2>&1; then echo "Kaggle token found in package" >&2; exit 1; fi
if grep -r -l -E "KGAT_[A-WYZa-wyz0-9]{8,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-ant-[A-Za-z0-9_-]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY" "$OUT" >/dev/null 2>&1; then
  echo "credential-like string found in package" >&2; exit 1; fi
if find "$OUT" -name "*.pt" -o -name "*.npz" -o -name "*.zip" -o -name "access_token" -o -name ".kaggle" | grep -q .; then
  echo "binary/credential file in package" >&2; exit 1; fi
big=$(find "$OUT" -type f -size +1500k ! -name "arc3-m2-modf-sub.ipynb")
if [ -n "$big" ]; then echo "large file(s): $big" >&2; exit 1; fi
$PY -c "import json,sys; nb=json.load(open('$OUT/notebook/arc3-m2-modf-sub.ipynb',encoding='utf-8')); n=sum(len(c.get('outputs',[])) for c in nb['cells']); sys.exit(f'notebook has {n} outputs' if n else 0)"

(cd dist && ../$PY -m zipfile -c Vladmir_Putin.zip Vladmir_Putin)
ls -la "$OUT.zip"
