# CERBench ICLR 2027 revision: local work completed, submission BLOCKED

> **Historical local-only revision report.** Subsequent non-human work verified all
> 4,936 PubMed records, repaired cited-ID contamination, restored 525 own-article
> bodies, rebuilt retrieval, ran a 1,000-replicate label-disclosure experiment, and
> produced a complete compiled paper. See [current status](NONHUMAN_REVISION_STATUS.md)
> and [final PDF](paper/iclr2027_final/cerbench_final.pdf). The earlier candidate,
> blocked experiment state and verification counts below remain a historical
> snapshot rather than being silently rewritten as current results.

**Do not submit the old archives or describe this revision as acceptance-ready.**
The local implementation is substantially stronger, but the historical data has
source-identity defects in addition to the review's ground-truth and budget
confounds. Acceptance cannot be guaranteed; software correctness cannot replace
independent biomedical validation.

## Current authority and deadlines

The user explicitly selected **local work only** and **no human annotators
available**. No paid APIs, remote compute, new model experiments, human judgments,
or submission uploads were performed in this revision. Local provider fixtures
and synthetic test annotations are not empirical results.

Official ICLR 2027 guidance was retrieved during this session:
<https://iclr.cc/Conferences/2027/AuthorGuidelines>.

- Genuine abstract and complete author list: **September 18, 2026, 23:59 AoE**.
- Full paper and supplement: **September 25, 2026, 23:59 AoE**.
- Main-text limit: **9 pages**, not 9 pages including references and appendices.
- Double-blind submission, AI-use statement, OpenReview profiles, reciprocal
  reviewing or documented exemption, and dual-submission compliance require
  author attention. Do not submit a placeholder abstract or claim unrun results.

## New decisive findings

1. **Source identity and text attachment fail.** The corpus contains 4,936
   records but 4,923 distinct document IDs. Ten colliding IDs account for 13
   excess rows with conflicting PMIDs/titles. The chunks have 17,902 rows but
   17,770 IDs. Independent local XML inspection found **58 corpus records whose
   PMID disagrees with the attached article-front PMID**. This is not fixed by
   adding a namespace prefix or retaining the last dictionary entry.
2. **The defect touches evaluated evidence.** Colliding IDs appear at **174
   ranked positions across 12 saved methods**, affecting **44 test tasks**.
   The broader XML mismatch impact is not exhausted by that duplicate-ID count.
3. **Stored expanded qrels omit a relevant judgment.** For `multihop_0189`,
   document `37307965` has a `RELEVANT` verdict with trailing explanation in the
   raw judgments but is absent from `expanded_gold.json`. Reconstructing all
   normalized judgments plus seeds yields **1,370 pairs, not 1,369**. Raw/stored
   historical files were not silently rewritten.
4. **Controller records remain incomplete and unbound.** The latest legacy
   eight-shard run covers **62/125** queries. Fourteen rows have invalid final
   selections according to the new selection contract. Older groups cannot be
   mixed in as independent or provenance-matched completions.

Evidence: [identity repair](docs/IDENTITY_REPAIR.md),
[strict historical failure](results/readiness/qrel_sensitivity/v1_final/REPORT.md),
[budget audit](results/readiness/budget/IMPLEMENTATION_REPORT.md).

## Completed local work

| Review requirement | Implemented and checked | Not yet established |
| --- | --- | --- |
| Trustworthy source records | Canonical-ID audit, raw-XML checks, quarantine, loss accounting, hash-linked candidate | Adjudicated identities/full text; regenerated evidence and indices |
| Clean splits | Deterministic component split, updated row metadata, overlap assertions before atomic publication | Genuinely new unexposed test queries; semantic cluster independence |
| Capability-native metrics | Explicit comparison/pair/path/constraint/time/study-value/negative evidence rules; invalid/missing annotations fail or remain unscored | Real adjudicated structural labels and model results |
| Fair iteration controls | Seven 24-document protocols, common final selector, feedback/upfront/latest-round ablations, provenance and budget tests | Live runner and complete matched-budget multi-family model experiments |
| Human qrels | Blinded independent sheets, seeds + top-20 pools + random out-of-pool checks, exact span validation, uncertainty/adjudication rules | Any completed human judgments or final adjudicated export |
| Human-written queries | 80 blank collection slots and independent-writing protocol | Any actual human-written queries |
| Ranking instability | Tie-aware token rank correlations, exact reversals, stored/reconstructed comparison, component uncertainty diagnostics | Independently validated biological ordering; completeness curves on clean evidence |
| Selective retrieval | ANSWER/ABSTAIN/UNCERTAIN distinction, external-loss risk, tie-block curves, dev-only empirical calibration | Calibrated real-data baselines; conformal guarantees |
| Full-text robustness | Detection and quarantine of misattachments | Safe restored full-text corpus and full-text-heavy evaluation |
| Paper | New narrative source, explicit claim boundaries, historical package warnings, build/package gate | Completed empirical paper, newly compiled PDF, final compliance approval |

### Conservative data candidate—not the new benchmark release

`data/processed/identity_repair_v1/` retains 4,936 canonical PMID metadata
records, **4,856 abstract chunks**, and **238 tasks**, with all full text stripped.
It quarantines 13,046 chunks and 66 tasks rather than guessing ambiguous
associations. Eighty metadata records lack retained chunks; retained tasks do not
depend on those uncovered records. Synthetic answers and scientific relevance
were not adjudicated.

`data/benchmark/v1_2_repaired_components/` contains **94 train / 49 dev / 95
test** tasks. The 186 components have zero checked cross-split overlap. Actual
tasks have no explicit semantic-cluster annotations, so passing explicit-cluster
checks is not evidence of semantic independence. All tasks were already exposed.
The older `v1_1_component_disjoint` version and its stale row metadata are kept
as historical data, not silently replaced. Use only explicit versioned paths.

### What the historical rank calculation does—and does not—show

The separate forensic mode uses document IDs **only as opaque strings**, never
dereferencing conflicted records. Across the ten original pool systems, seed to
expanded R@20 ordering has:

- **Kendall tau-b: 0.5843065475**.
- **Spearman rho: 0.7355657079**.
- **9 strict reversals**, 35 concordant pairs, and one seed tie.
- The missing pooled label changes scores but not the ten-system ordering.

These are reproducible properties of saved rankings/label sets, **not evidence of
the correct biomedical ranking**. Strict real-data validation still fails.
Component sign-flip/bootstrap diagnostics remain conditional historical-token
analyses, not causal evidence of reasoning benefits.

Exact results and hashes:
[forensic report](results/readiness/qrel_sensitivity/v1_forensic/REPORT.md),
[verification](results/readiness/qrel_sensitivity/FORENSIC_VERIFICATION.json).

### Human kit is provisional and is not ready to distribute

The independently prepared kit selects 98 supported queries before identity
filtering, retains 93 plus all 17 abstention cases, and has **12,657 pairs per
annotator**. All labels are blank. Its broader 15-system union has partial system
coverage and does not constitute a completed top-20 judgment pool.

This kit uses a different, less conservative identity filter than the abstract-only
repair candidate. **Do not mix it with that candidate or begin annotation now.**
Rebuild a single kit against the adjudicated corpus, agreed primary systems and
locked split. It currently serves as a workflow/template validation artifact.

Two annotators imply 25,314 pair judgments, plus task-level reviews and
adjudication. The documented 2–5 minutes/pair planning assumption yields roughly
844–2,110 person-hours before adjudication; this is not measured time. Run a
small, separately identified calibration pilot before choosing a feasible final
pool scope. Do not silently cap or cherry-pick candidates to fit a desired budget.

See [human protocol](docs/HUMAN_VALIDATION_PROTOCOL.md). The current validators
do not enable final-qrel export or completed task-level reconciliation. Independent
adjudicated import/export remains a future integration task, not a claimed feature.

## Verification and rerun entry points

**Final integrated verification: 166 tests passed, zero failures/errors/skips.**
The driver parsed 63 Python files, verified both synthetic evaluator modes,
revalidated the blank human kit, exercised the build/package rejection paths,
and confirmed 50 tracked historical corpus/ranking/manuscript/archive files
remained byte-identical during that integration run. Earlier broader identity
verification is documented separately; the 50-file count is not a claim that
every file in the connected project was unchanged.

- [Final verification receipt](results/readiness/integration_final_v2/revision_verification.json)
- [Machine-readable blocked status](results/readiness/integration_final_v2/submission_status.json)
- [166-test JUnit report](results/readiness/integration_final_v2/final_tests.xml)
- [Reproducible integration driver](results/readiness/verify_revision.py)

Two earlier integration attempts exposed a flat-versus-nested split-manifest
interface mismatch and the annotation validator's report-directory restriction.
Both were corrected; their logs remain preserved rather than represented as
passing runs. The final versioned attempt above is authoritative. Draft checks
verified referenced BibTeX keys and balanced environments only, not compilation.

From `analysis/synthetic-science-search`, using Python 3.11 with NumPy, SciPy
and pytest (no model dependencies needed for the new evaluator tests):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
python3 scripts/45_submission_readiness.py --require-ready
```

The test suite should pass. **The readiness command should exit 2 with BLOCKED.**
An intentional validation failure is not a successful experiment; it is the
correct guardrail response. The historical analysis default also exits 1 on its
input contract violations. Use `--help` on the new scripts for explicit versioned
input/output options; generated artifacts refuse overwrite.

- `scripts/39_audit_repair_identity.py`: conservative candidate and quarantine.
- `scripts/38_make_component_disjoint_splits.py`: new candidate component splits.
- `scripts/40_evaluate_validated_runs.py`: strict retrieval/structural/selective evaluation.
- `scripts/35_generate_equal_budget_queries.py`: offline plans, not a live API runner.
- `scripts/41_audit_budget_runs.py`: unbound/incomplete legacy shard audit.
- `scripts/42_prepare_validation_kit.py`: provisional blank human materials.
- `scripts/43_validate_human_labels.py`: strict sheet/span/provenance checks.
- `scripts/44_analyze_historical_qrel_sensitivity.py`: strict failure by default;
  separate explicit forensic-token mode, never release clearance.
- `scripts/45_submission_readiness.py`: current fail-closed submission gate.

Test and verification reports live under `results/readiness/` and
`annotations/human_qrel_v2/`. Existing raw data, original task splits, historical
rankings/scores, and old PDFs/ZIPs were preserved. Some historical scripts were
hardened or disabled; old operational behavior is not guaranteed. There is no Git
repository or commit available to claim an immutable historical code revision.

## Paper and claim boundaries

Working source: [cerbench_revision.tex](paper/iclr2027_revision/cerbench_revision.tex).
It reframes the paper around relevance construction, removes the causal
``iteration is the mechanism'' interpretation, and separates tested evaluation
contracts from unrun experiments. It is visibly marked **not for submission**.

No new PDF was produced: `pdflatex`, `bibtex`, `pdfinfo`, and `pdffonts` were not
available in the tested environment. The earlier PDF and source archives are
historical. Their previous page/font checks do not apply to the new draft. The
full-build and review-packaging entry points now stop before overwriting them.

## What is needed to unblock a defensible submission

In dependency order—not in order of cosmetic ease:

1. **Correct source acquisition/identity.** Independently resolve PMID/PMCID
   associations, restore correctly linked full text, validate representative
   methods/results/tables, and review per-record licensing. New external retrieval
   needs separate authorization; do not treat old source claims as verification.
2. **Lock a fresh evaluation design.** Fix corpus/evidence components before
   new question generation; keep held-out labels and method-selection data
   separated. Existing exposed questions are diagnostic, not fresh confirmation.
3. **Arrange independent biomedical people.** Two annotators and an adjudicator;
   independently authored queries; feasible workload/compensation and any
   applicable institutional review. AI judges cannot substitute for this.
4. **Authorize and execute new experiments.** Select available unrelated model
   families only after capability/budget checks; implement the live trace runner;
   run clean baselines and seven candidate controls, with all attempts retained.
   The current instruction authorizes none of that external spending.
5. **Finalize labels, structural export and statistics.** Reconcile task
   requirements, export provenance-checked adjudicated qrels, test importer
   alignment, calibrate on dev, lock methods and score test once. Include
   pool-completeness sensitivity, fulltext and query-origin strata.
6. **Revise only to supported claims, compile and approve.** Align every table
   with a frozen producing run; resolve anonymity/licensing, AI disclosure,
   author/OpenReview requirements and final PDF checks. Submission requires
   explicit author action, not merely a successful automated build.

If these resources cannot be obtained before the deadline, the defensible option
is to narrow/defer the submission rather than represent this blocked snapshot
as human-validated or acceptance-ready. The local tools and preserved evidence
remain useful for a subsequent rigorous release.
