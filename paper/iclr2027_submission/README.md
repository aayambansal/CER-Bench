# CER-Bench — ICLR 2027 LaTeX source

**Historical snapshot — NOT CLEARED FOR SUBMISSION (local revision audit).** This directory contains the earlier anonymous manuscript and its existing archives. The September revision found conflicting document identities, 58 corpus-to-local-XML PMID disagreements, and a missing relevant pair in stored pooled labels. The previous formatting/numerical checks do not validate the scientific evidence. Do not upload the existing `dist/` files as the repaired submission.

The historical `.tex`, PDF and ZIP files are preserved. Current status and deliverables are described in `../../SUBMISSION_READINESS.md`; the new working narrative is in `../iclr2027_revision/`. The build and packaging scripts now fail closed at `scripts/45_submission_readiness.py` until a separately reviewed versioned release replaces this blocked snapshot.

## Build

From the benchmark root (`analysis/synthetic-science-search`), the historical build command is:

```sh
bash scripts/run_iclr2027_submission.sh
```

This command currently exits at the scientific-readiness gate **before** changing old scores or build files. Its downstream historical workflow recomputes the paper-facing audit, regenerates the main figure, and checks LaTeX formatting; those steps are not a substitute for source identity, human relevance or experimental validation.

For a LaTeX-only rebuild from this directory:

```sh
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error cerbench_iclr2027.tex
bibtex cerbench_iclr2027
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error cerbench_iclr2027.tex
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error cerbench_iclr2027.tex
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error cerbench_iclr2027.tex
```

The output is `cerbench_iclr2027.pdf`. In the audited build, counted main text ends on page 7, within the **9-page initial-submission limit**. The required AI-use statement and optional ethics and reproducibility statements follow the counted main text and do not count toward the limit. References and appendices follow.

The `\iclrfinalcopy` command remains commented out, as required for double-blind review.

The superseded pre-audit source is retained as `cerbench_iclr2027_pre_revision.tex` for provenance and must not be included in the OpenReview source archive.

## ICLR 2027 deadlines and policy checks

- Genuine abstract due: **September 18, 2026, 11:59 PM AoE**.
- Full paper and supplement due: **September 25, 2026, AoE**.
- Initial main-text limit: **9 pages**; references and appendices are excluded.
- Review is double-blind; the paper and supplement must remain anonymous.
- The AI-use statement is mandatory.
- All authors need OpenReview profiles, and authors cannot be added after the abstract deadline.

Official authority: <https://iclr.cc/Conferences/2027/AuthorGuidelines>.

## Official template provenance

The style files were downloaded on 2026-08-20 from the official ICLR 2027 author-guidelines link:

<https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip>

Archive SHA-256:

```text
0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e
```

The bundled style files match this hash and were not modified.
