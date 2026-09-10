# Release manuscript: two variants from one body

**Title:** Fixed Rankings, Moving Leaders: A Relevance-Label and Source-Identity Audit of Scientific Retrieval Evaluation

| File | Variant | Front matter | Artifact links |
|---|---|---|---|
| `cerbench_arxiv.pdf` | arXiv preprint | Aayam Bansal, Ishaan Gangwani (Synthetic Sciences); header "Preprint. Under review." | GitHub + Hugging Face URLs in the Reproducibility Statement and Appendix L |
| `cerbench_openreview.pdf` | ICLR 2027 double-blind | Anonymous authors; official review header and line numbers | none; says artifacts accompany the submission and will be released |

Both PDFs are compiled from the same `cerbench_body.tex`; the wrappers (`cerbench_arxiv.tex`, `cerbench_openreview.tex`) only set `\ifanon`, the author block, PDF metadata, and the header. Main text ends on page 9 in both (ICLR limit: 9 pages); references and Appendices A–N bring each document to 22–23 pages.

What changed relative to `../iclr2027_final/cerbench_final.tex`:

- New title and a five-sentence abstract that leads with the result rather than the method.
- Contribution list in the introduction; cross-references from the main text to the new appendices.
- Appendix expanded from four to fourteen sections. New material, all taken from retained artifacts (no reruns): historical families/splits/system configurations (C); complete endpoint means, all nine reversed pairs, the missing-pair rationals, and the forensic cluster inference (D); the full disclosure grid — per-fraction summary, per-system means, top-1 votes, and average ranks (E); the identity-repair candidate and component splits (F); the restoration funnel, chunking policy, and dataset schema (G); the BM25 scoring contract, probe queries, and measurements (H); the seven candidate-budget conditions and legacy-shard audit (I); per-family structural success rules and selective decisions (J); the designed-but-unexecuted human validation kit and its sizing (K); release layout (L); test-suite breakdown (M); reproduction commands (N).
- Ethics statement now states why article text is not redistributed.

No number in the manuscript comes from a new computation; every quantity maps to a file listed in `../iclr2027_final/CLAIMS_TO_ARTIFACTS.md` or to the docs under `docs/`.

## Build

```sh
./build.sh              # both variants, pdflatex + bibtex
./build.sh arxiv        # one variant
```

Requires a TeX distribution with `pdflatex` and `bibtex`. The build fails on unresolved citations or references. The last build had zero overfull boxes.

## Submission bundles

- `arxiv_bundle.zip` / `arxiv_bundle.tar.gz` — the same flat arXiv upload in both accepted formats (files at the archive root, no wrapper directory): `cerbench_arxiv.tex`, `cerbench_body.tex`, `cerbench_arxiv.bbl`, style files, `figures/`. arXiv compiles `cerbench_arxiv.tex` (the only file with `\documentclass`); both archives were test-compiled standalone.
- `OPENREVIEW_FORM.md` — title, keywords, TL;DR, abstract (TeX-safe), primary area, AI-assistance boxes.
- `supplementary_material.zip` — anonymized supplement for OpenReview (code, task files, saved rankings, judgments, traces, audits, paper source); built by `make_supplement.py` from the committed repository tree with names and absolute paths scrubbed. Not committed to git.
