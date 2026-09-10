#!/usr/bin/env bash
set -euo pipefail

# Rebuild paper-facing result artifacts from the saved retrieval outputs that are
# distributed with the repository, then compile the manuscript PDF. This script
# is intentionally conservative: it does not launch API-backed agent runs or GPU
# embedding jobs. Use scripts/10_run_baselines.py and scripts/11_run_agent.py to
# regenerate retrieval outputs before invoking this script when a full rerun is
# required.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for split in dev test; do
  if compgen -G "results/baselines/*_${split}.jsonl" > /dev/null; then
    python scripts/12_score_retrieval.py --split "$split" --exclude-abstention
    python scripts/14_make_tables.py --split "$split"
    python scripts/15_make_figures.py --split "$split"
  else
    echo "Skipping ${split}: no saved retrieval outputs matching results/baselines/*_${split}.jsonl" >&2
  fi
done

cd paper
rm -f neurips_2026.aux neurips_2026.bbl neurips_2026.blg neurips_2026.log neurips_2026.out neurips_2026.pdf
pdflatex -interaction=nonstopmode -halt-on-error neurips_2026.tex
bibtex neurips_2026
pdflatex -interaction=nonstopmode -halt-on-error neurips_2026.tex
pdflatex -interaction=nonstopmode -halt-on-error neurips_2026.tex

echo "Built $ROOT/paper/neurips_2026.pdf"
