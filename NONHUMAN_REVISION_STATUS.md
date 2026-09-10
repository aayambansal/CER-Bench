# CERBench non-human revision and completed paper

> **Current manuscript (September 10, 2026):** the audit paper was retitled
> *Fixed Rankings, Moving Leaders: A Relevance-Label and Source-Identity Audit of
> Scientific Retrieval Evaluation*, given an expanded fourteen-section appendix, and
> split into a named arXiv preprint and an anonymized OpenReview copy under
> [`paper/release/`](paper/release/README.md). No numbers changed; nothing was rerun.
> The status below describes the evidence base both variants rest on.

## Deliverable

**[Read the complete revised paper](paper/iclr2027_final/cerbench_final.pdf)** (previous draft; current variants in `paper/release/`)

Title (previous draft): **Relevance Labels and Source Identity Can Change Scientific Retrieval
Conclusions: A CER-Bench Audit**.

This is a complete evidence-grounded manuscript, not a guarantee of ICLR
acceptance or a completed rerun of the original benchmark. It deliberately
replaces unsupported winner claims with the source/label audit supported by the
available evidence. It has no invented human, model, or capability-score results.

- Official anonymous ICLR format; original style files unchanged.
- Main text ends on **page 8**; **13 pages total** with references and appendices.
- All fonts embedded; no unresolved references/citations or overfull boxes.
- Page contact sheet, title and enlarged figure visually checked.
- Editable source and claim-to-artifact ledger are in `paper/iclr2027_final/`.
- Final configured software suite: **311 tests passed**, no failures/errors/skips
  and no deselection.

## New completed work since the local-only report

### Authoritative source repair

All **4,936 intended PMIDs** were re-fetched from NCBI PubMed and compared. The
root cause was broad `.//ArticleId` traversal under `PubmedData`, which includes
the reference list. Correct direct-child parsing changed **247 PMC associations**;
226 changed historical PMC IDs occur in cited references. Three changed old DOIs
also occur in cited references. There were **815 metadata-changed records** apart
from canonical-ID renaming, but that broader total is not all proven bug damage:
current metadata and parser policies also differ.

All intended records were recovered. Request-transcription mistakes and their
supplemental recovery are retained in the source log. The historical corpus,
rankings and manuscript archives were not silently overwritten.

Sources: `docs/AUTHORITATIVE_CORPUS.md`,
`results/readiness/authoritative_corpus/`.

### Source-owned full text and real retrieval

Conservative JATS extraction restored **525 own-article bodies**. The resulting
dataset contains **4,936 documents and 10,313 chunks**, including **21 title-only
documents**. It does not consist of 4,936 complete full-text papers. Source blocks,
omissions, Unicode-codepoint offsets and raw-file hashes are recorded. Licensed
text was not declared blanket-redistributable; machine-readable licenses are not
human licensing adjudication.

The rebuilt BM25 index has **63,109 vocabulary terms and 2,010,185 postings**.
Eight fixed, unscored probes completed under three non-LLM controls. Repeating
the original query with the same total candidate budget produced the same final
selection as original-query top-24. These are actual retrieval checks, not
benchmark accuracy or evidence of capability satisfaction. Native provider
integration now uses this exact verified index rather than an incompatible
portable scoring formula.

Sources: `docs/VERIFIED_FULLTEXT.md`, `docs/VERIFIED_RETRIEVAL.md`,
`data/processed/authoritative_fulltext_v1/`,
`data/processed/authoritative_fulltext_v1_bm25/`,
`results/readiness/verified_bm25/real_corpus_v2/`.

### Controlled label-disclosure experiment

A new experiment ran **1,000 paired nested disclosures** of the 30 existing
pooled judgments per supported query. It retains seed positives, includes both
positive and negative judgments in each disclosure permutation, holds all saved
rankings fixed, and treats historical IDs only as strings.

- Mean-R@20 leader: agent at fraction 0.2; BGE at 0.3.
- This is a tested-grid bracket, not a continuous threshold estimate.
- Paired empirical difference intervals include zero at both fractions.
- BGE receives 56.7% of top-1 votes at 0.3 and 92.0% at 0.5.
- Full expansion gives nine strict pairwise reversals, Kendall tau-b 0.5843,
  and Spearman rho 0.7356 relative to seed labels.

These results are conditional on the existing biased automated pool. They do
not determine a correct biomedical ranking, model missing-not-at-random evidence
outside the pool, or replace human relevance judgments.

Sources: `docs/QREL_DISCLOSURE.md`,
`results/readiness/qrel_disclosure/v2/`, and the final paper's figure and table.

### Secure model-execution software

Native OpenAI, Anthropic and Google clients, secret redaction, physical-line
JSONL parsing, native verified-BM25 execution, pending-outcome protection,
campaign-wide reservation accounting, and all seven budget conditions are
implemented and tested. Keys are read only from the runtime environment, never
hard-coded or copied from the conversation.

**No new model calls were completed.** Presence-only checks after configuration
updates still found no usable provider variables in the execution environment.
Modal was connected, but its exposed configuration continued to report outbound
networking disabled. No token CLI command was executed and no credential values
were copied to repository files, logs, configuration, or the manuscript.

The user authorized non-human experiments and accepted a conservative $50 pilot
cap before scaling. That authorization is not proof of credential availability
or evidence that money was spent. The final paper clearly states that fresh
cross-model runs and a corrected-corpus leaderboard are absent.

## Remaining gaps—explicitly not hidden by the final PDF

1. Human relevance validation is absent by explicit user instruction. No
   automated output is represented as independent human work.
2. Fresh source/component-disjoint task construction on the corrected corpus
   has not been run; historical questions and qrels were not promoted to new gold.
3. No live cross-model candidate-budget experiment, new trained controller,
   corrected-corpus system leaderboard, real structural score table or selective
   risk–coverage result exists.
4. The original full-text-heavy benchmark ambition is not fully met; most
   documents still lack restored bodies, and some extraction is lossy.
5. Authorship, OpenReview requirements, reciprocal reviewing, ethics/licensing,
   and actual submission remain author actions. The review-mode header is the
   template's convention, not an assertion that the paper has been submitted.

To continue model experiments, configure **rotated** provider keys through a
secure mechanism that actually injects them into the execution runtime; do not
paste them into chat. Then verify model/pricing/corpus/task bindings and the
campaign cap before running. That future work may change the paper's conclusions
and must be reported as a new version, not backfilled into historical results.

## Final checks and provenance

- Complete test suite: `results/readiness/verified_bm25/real_corpus_v2/all_tests.xml`.
- Native integration verification: same directory, `verification.json`.
- PDF checks: `paper/iclr2027_final/pdf_checks.json`.
- Claim ledger: `paper/iclr2027_final/CLAIMS_TO_ARTIFACTS.md`.
- Static source checks: `paper/iclr2027_final/static_checks.json`.
- Reproducible PDF/figure scripts: `build_final.py`, `make_paper_figure.py` in
  the final paper directory.

Earlier failed, filtered, interrupted-build and stale-provenance attempts are
retained. Final success is based on the separate completed verification, not on
discarding those attempts or summing overlapping test suites.
