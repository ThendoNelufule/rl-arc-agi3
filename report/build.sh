#!/bin/sh
# Build the report PDF: figures + tables from logged data, then LaTeX (tectonic).
# Usage: sh report/build.sh [path/to/tectonic]
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
$PY figures.py >/dev/null
$PY tables.py >/dev/null
${1:-tectonic} main.tex
