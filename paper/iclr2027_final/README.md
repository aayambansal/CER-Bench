# CER-Bench final-deliverable manuscript

Read the completed paper: **`cerbench_final.pdf`**. Editable source: `cerbench_final.tex`.

Title: **Relevance Labels and Source Identity Can Change Scientific Retrieval Conclusions: A CER-Bench Audit**.

This is the requested complete final-deliverable revision, not submission clearance, acceptance assurance, or a validated corrected-corpus benchmark. It replaces a benchmark-winner narrative with a joint source-identity and relevance-label audit. It contains the abstract, main argument, methods and results, limitations, reproducibility and ethics/AI-use statements, references, and technical appendices. No historical manuscript, PDF, archive, corpus, or source implementation was changed by this writing phase.

## Files and evidence

- `cerbench_final.tex`: complete anonymous ICLR-format source.
- `cerbench_final.pdf`: compiled complete paper, **13 pages total; main text ends on page 8**.
- `refs.bib`: unchanged copy of the existing submission bibliography; the manuscript cites 14 entries, with unused entries retained rather than rewritten. BibTeX prints the cited subset.
- `iclr2027_conference.sty`, `iclr2027_conference.bst`, `natbib.sty`, `fancyhdr.sty`: unchanged copies from `paper/iclr2027_submission/`.
- `figures/qrel_disclosure.pdf`: unchanged vector experiment figure. SHA-256: `155ef70e796a739f58f9963a706f1c42ea12a78a125f7861daf1e0803b4bff09`.
- `figures/qrel_disclosure_readable.pdf`: active, print-legible layout generated from the same saved statistics by `make_paper_figure.py`; its source/output hashes are recorded separately. It does not change results.
- `CLAIMS_TO_ARTIFACTS.md`: evidence ledger and limits on interpretation.
- `verify_static.py`: local, provider-free checks for citation keys, cross-references, environments, immutable copies, selected source counts, and prohibited manuscript placeholders/private paths.
- `static_checks.json`: written by a successful static-check invocation; it is not a compilation or scientific-validation report.
- `pdf_checks.json`: successful compilation, pagination, font, private-text and TeX-log checks. `pdf_review/` retains rendered pages/contact sheet for visual inspection.

The minimal `CERBench_ICLR2027_final_source.zip` contains everything needed to
compile the paper, not the complete benchmark or raw-text evidence. Plotting,
static-check and PDF-inspection helpers remain in the project folder; reproducing
their data-dependent checks requires the project artifacts named in the ledger.

## Build and review

From this directory, use an available TeX environment:

```sh
tectonic cerbench_final.tex
```

The final paper was compiled with SHA-verified Tectonic 0.17.0 and inspected with PyMuPDF 1.26.7. Main text ends on page 8, within the nine-page limit; references and appendices bring the complete document to 13 pages. All fonts are embedded, with no unresolved references/citations or overfull boxes. The lead inspected the page contact sheet, title page and enlarged figure page; the figure was reformatted for legibility and an unbreakable appendix path was fixed. The official style was not changed. Its anonymous review-mode header is a template convention, not evidence that an OpenReview submission occurred.

Alternatively build with `pdflatex`, `bibtex`, and two further `pdflatex` passes. Tectonic may need cached packages or network access to its bundle. `build_final.py --compiler /path/to/tectonic` reproduces compilation and PDF checks with PyMuPDF/Pillow installed; this is not scientific clearance.

Run static checks from the analysis root with an interpreter containing the standard library:

```sh
python paper/iclr2027_final/verify_static.py
```

## Empirical boundary

Usable provider credentials remain unavailable and **no new model experiments have run**. The new empirical computations are authoritative metadata comparison, conservative local full-text restoration, historical-token Monte Carlo disclosure, and actual unscored local BM25 checks. The 56 native adapter plans are dry plans; its 24 local checks are not 24 live LLM trials. Structural and selective metric definitions are tested software, not biomedical scores.

Human validation was explicitly waived and not performed. The corrected 4,936-document/10,313-chunk dataset has no fresh source partition, newly generated benchmark tasks, human relevance labels, or corrected-corpus leaderboard. Historical tasks/qrels must not be silently reused. The 238-task post hoc repair (94/49/95) is already exposed and is not a fresh held-out benchmark. Source ownership is not semantic fidelity, clinical validity, or redistribution approval.

An intermediate adapter suite exposed stale historical provenance. The old failed/filtered receipts remain preserved. A new versioned native-backend integration kept strict hash checks and reran the complete configured repository suite: **311 passed, zero failures/errors/skips, no deselection**. Authoritative evidence is `results/readiness/verified_bm25/real_corpus_v2/{all_tests.xml,verification.json}` relative to the benchmark root. Old build timings and equation checks remain explicitly historical measurements, not newly rerun measurements.

## Remaining scientific and author actions

The paper is a complete deliverable, but not the requested fully rerun automated benchmark. Secure provider credentials never became accessible to the execution tools, so no fresh task generation or cross-model experiments were performed. Source-disjoint fresh tasks, completed model comparisons and appropriate automated-label audits remain necessary for stronger empirical claims. Human relevance remains absent by the user's explicit decision. Authors must still review contribution/scope, ethics/licensing, authorship/OpenReview/reciprocal-review requirements and the supplement before any submission. Neither the PDF nor passing tests guarantee acceptance.
