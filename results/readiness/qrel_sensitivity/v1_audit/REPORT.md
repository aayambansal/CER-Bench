# Historical qrel sensitivity: input validation FAILED

**release_blocked: true**

release_blocked: true. Historical diagnostic only: all primary results are compromised by corpus identity failures. No repaired benchmark, publication result, validated gold, fulltext advantage, or causal agent claim. Artificial label subsampling is conditional on the original biased pool and fixed final queries/rankings; it cannot establish completeness outside that pool.

## Blocking evidence

Historical corpus: 4936 rows, 4923 distinct IDs, 10 duplicated IDs, 13 excess rows. Duplicate records disagree on PMID/title and other metadata. No first/last-record selection or deduplication was performed.

| Ambiguous doc_id | Physical corpus rows | PMIDs |
|---|---|---|
| 2949280 | 2828, 2861 | 39546202, 39982678 |
| 30620402 | 574, 1693 | 33566254, 33259039 |
| 4343198 | 84, 181 | 30610645, 33597771 |
| 5564292 | 703, 749 | 39542905, 40465127 |
| 5706658 | 50, 57, 92 | 39085660, 39158822, 39012436 |
| 5906799 | 1463, 3455 | 41028170, 37311817 |
| 6286148 | 3, 43, 339 | 38308006, 30610625, 40828286 |
| 7433347 | 1483, 1493 | 40562951, 39548270 |
| 7581537 | 3476, 3573 | 35534554, 37679571 |
| 8386155 | 3704, 3766, 3905 | 41203968, 41258061, 36376393 |

## Independent inventory checks (not retrieval results)

Test universe: 125; supported: 108; empty seed qrels: 17. All 12 primary saved runs were checked against every test ID, including empty-support tasks. See INPUT_AUDIT.json for any additional errors and all affected task/document/rank references.

Raw verdicts: 3240; positive: 1106; negative: 2134. Seed qrel pairs: 264; ID-level reconstructed expanded pairs: 1370. Stored-expanded set mismatches: 1. Matching strings do not establish valid document identity or scientific relevance.

Affected-input counts: {"distinct_test_tasks_with_run_duplicate_identity_references": 44, "judgment_duplicate_identity_reference_count": 34, "run_duplicate_identity_references": 174, "seed_or_negative_reference_count_all_splits": 6}

## Not executed

Retrieval metrics, stored-score comparisons, correlations/reversals, bootstrap/sign-flip/Holm, completeness curves, and fulltext score strata were deliberately not produced. The fail-closed input contract takes precedence over filling result tables. The analysis implementation and formula tests are present, but the end-to-end numerical branch is unverified on these invalid inputs.

## Identity scope and limitations

Candidate identity report independently records 58 PMID/XML mismatches. It was read only for counts/scope, never to remap old results. All historical primary results remain compromised. No repaired benchmark or publication claim is made. Artificial label thinning, if later run under an authorized valid input contract, would still be conditional on the original biased pool. Historical source flags cannot support valid fulltext claims.

## Sources and next action

`manifest.json` hashes all original input and analysis code paths; `INPUT_AUDIT.json` lists exact affected IDs and physical rows. Corpus source: `data/processed/corpus.jsonl`; test runs and judgments: `results/baselines`; tasks: `data/benchmark/{train,dev,test}.jsonl`; identity scope: `data/processed/identity_repair_v1/report.json`. Methods: `docs/QREL_SENSITIVITY_METHODS.md`.

Resolve the historical input contract before attempting numerical analysis; do not silently choose between conflicting rows or map saved results into the candidate corpus. Paper and overall readiness gate are unchanged. Local only: no humans, network/API, install, paid jobs, or git actions.
