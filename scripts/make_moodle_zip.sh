#!/bin/sh
# Build Kaggle-Vladmir_Putin.zip for Moodle: report PDF, nominated notebook (no outputs, no credentials),
# submission evidence, Paper Track text, member contributions.
# Usage: sh scripts/make_moodle_zip.sh   (run from the repo root)
set -e
OUT=dist/Kaggle-Vladmir_Putin
rm -rf "$OUT" "$OUT.zip"
mkdir -p "$OUT/notebook"
cp report/main.pdf "$OUT/report.pdf"
cp kaggle_sub/m2-modf-sub/arc3-m2-modf-sub.ipynb kaggle_sub/m2-modf-sub/kernel-metadata.json "$OUT/notebook/"
cp submission/EVIDENCE.md submission/CONTRIBUTIONS.md experiments/kaggle_submissions.csv "$OUT/"
cp report/paper_track.md "$OUT/paper_track.md"
[ -f submission/PAPER_TRACK_LINK.txt ] && cp submission/PAPER_TRACK_LINK.txt "$OUT/"
# guard: never ship credentials or notebook outputs
if grep -r -l -E "KGAT_|ghp_|access_token" "$OUT" >/dev/null 2>&1; then echo "credential-like string found" >&2; exit 1; fi
.venv/bin/python -c "import json,sys; nb=json.load(open('$OUT/notebook/arc3-m2-modf-sub.ipynb',encoding='utf-8')); n=sum(len(c.get('outputs',[])) for c in nb['cells']); sys.exit(f'notebook has {n} outputs' if n else 0)"
(cd dist && .venv/bin/python -m zipfile -c Kaggle-Vladmir_Putin.zip Kaggle-Vladmir_Putin 2>/dev/null || ../.venv/bin/python -m zipfile -c Kaggle-Vladmir_Putin.zip Kaggle-Vladmir_Putin)
ls -la "$OUT.zip"
