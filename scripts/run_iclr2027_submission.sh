#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER="$ROOT/paper/iclr2027_submission"

cd "$ROOT"
python3 scripts/45_submission_readiness.py --require-ready
python scripts/31_submission_audit.py
python scripts/32_make_submission_figure.py

cd "$PAPER"
rm -f cerbench_iclr2027.aux cerbench_iclr2027.bbl cerbench_iclr2027.blg \
      cerbench_iclr2027.log cerbench_iclr2027.out
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error cerbench_iclr2027.tex
bibtex cerbench_iclr2027
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error cerbench_iclr2027.tex
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error cerbench_iclr2027.tex
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error cerbench_iclr2027.tex

if rg -n "undefined citations|undefined references|Citation .* undefined|Reference .* undefined|multiply defined|LaTeX Error" cerbench_iclr2027.log; then
  echo "Fatal LaTeX reference/citation error" >&2
  exit 1
fi

main_end_page="$(sed -n 's/.*newlabel{page:mainend}{{[^}]*}{\([0-9][0-9]*\)}.*/\1/p' cerbench_iclr2027.aux | head -1)"
if [[ -z "$main_end_page" || "$main_end_page" -gt 9 ]]; then
  echo "Initial-submission main text exceeds 9 pages (marker page: ${main_end_page:-missing})" >&2
  exit 1
fi

pdfinfo cerbench_iclr2027.pdf | rg "^(Pages|Page size):"
if pdffonts cerbench_iclr2027.pdf | tail -n +3 | awk '$6 != "yes" {bad=1} END {exit bad}'; then
  echo "All PDF fonts are embedded."
else
  echo "At least one PDF font is not embedded." >&2
  exit 1
fi

echo "ICLR 2027 build and compliance checks passed; main text ends on page $main_end_page."
