#!/usr/bin/env bash
# Build both release variants with pdflatex + bibtex (4 passes).
# Usage: ./build.sh            # builds arxiv and openreview
#        ./build.sh arxiv      # builds one variant
set -euo pipefail
cd "$(dirname "$0")"
variants=("$@")
if [ ${#variants[@]} -eq 0 ]; then variants=(arxiv openreview); fi
for v in "${variants[@]}"; do
  main="cerbench_${v}"
  echo "== building ${main}.pdf =="
  pdflatex -interaction=nonstopmode -halt-on-error "${main}.tex" >/dev/null
  bibtex "${main}" >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error "${main}.tex" >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error "${main}.tex" >/dev/null
  grep -E "Warning: (Citation|Reference)" "${main}.log" && { echo "unresolved refs in ${main}"; exit 1; } || true
  echo "ok: ${main}.pdf"
done
