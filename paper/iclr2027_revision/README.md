# CERBench working revision — blocked, not submission-ready

`cerbench_revision.tex` is a new narrative draft emphasizing relevance-construction sensitivity, explicitly separating historical string-label arithmetic from biomedical evidence. It does not replace or modify the old manuscript/PDF/archives. All unrun confirmatory experiments are identified as unrun; there are no invented human, structural or matched-budget results.

**Do not submit this draft.** Its historical inputs fail identity and qrel validation. The current authorization permits local work only and no annotators are available. See `../../SUBMISSION_READINESS.md` for the evidence matrix and requirements to unblock.

LaTeX compilation was not performed: `pdflatex`, `bibtex`, `pdfinfo`, and `pdffonts` were not available in the tested environment. Once a local TeX installation exists, a draft-only syntax build from this directory can use the existing style files:

```sh
TEXINPUTS="../iclr2027_submission//:" pdflatex -interaction=nonstopmode -halt-on-error cerbench_revision.tex
bibtex cerbench_revision
TEXINPUTS="../iclr2027_submission//:" pdflatex -interaction=nonstopmode -halt-on-error cerbench_revision.tex
TEXINPUTS="../iclr2027_submission//:" pdflatex -interaction=nonstopmode -halt-on-error cerbench_revision.tex
```

A successful draft build is not scientific clearance. Recheck page count, references, embedded fonts, metadata/anonymity, licensing, author approvals and claim-to-artifact alignment after completed experiments. Do not include local evidence sheets, absolute-path manifests or coordinator provenance in a public supplement without explicit sanitization and licensing review.
