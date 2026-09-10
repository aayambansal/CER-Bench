# Offline budget-protocol implementation handoff

## Outcome

Implemented and tested the pure seven-condition matched-candidate-budget protocol,
safe offline request generation, structural run auditing and matched-subset scorer
safeguards. **No LLM experiments or publication results were generated.** All
execution was local; no provider requests, network retrieval, package installs or
paid jobs were used.

## Verification

- Final regression suite: **28 passed in 0.18 seconds**, zero failures.
  JUnit evidence: `results/readiness/budget/tests.xml`.
- Full test split: **125 tasks × 7 conditions = 875 unique dry-run plans**,
  zero executed search/model calls. Every plan's status and provenance binding
  were checked. Final source/index hashes match the manifest.
- Final plan manifest ID:
  `d0bd2113995c25db6f81adf30de41365fa5f602c150f14340ad253fa95d189c4`.
- Test task-file SHA-256:
  `33158e70ebd3e6438530553b2686354f6d2fa07e60373e9fd4ae63bd52796d76`.
- All **13 legacy query files** retained their initial-audit SHA-256 hashes at
  final verification. No existing baseline/query JSONL or scored output was
  opened for writing.

Tests exercise all branches, unique-doc budgeting and underfill, original-selector
and RRF consistency, upfront query generation before search, no-feedback message
contents, accumulated versus latest-round evidence, clipping/token accounting,
invalid responses/timeouts without fallback, manifest/resume mismatches,
duplicate/missing/conflicting shards, socket-blocked default dry-run/live gate,
matched seed/pooled task subsets and immutable scorer output.

## Legacy audit findings

| Separate filename group | Coverage | Unbound rows | Invalid final selections |
|---|---:|---:|---:|
| `equal_budget_llm_queries_run8` | 62/125 | 62 | 14 |
| `equal_budget_llm_queries` | 14/125 | 14 | 1 |
| `equal_budget_llm_queries_smoketest` | 1/125 | 1 | 0 |

These groups were **not merged**. All eight run8 shard files exist, but 63 task
identities are missing. Per-shard coverage is `5/16, 15/16, 5/16, 6/16, 4/16,
6/15, 15/15, 6/15` for shards 0–7. The four-shard group is `4/32, 4/31, 3/31,
3/31`, leaving 111 tasks missing. The smoketest leaves 124 tasks missing.

No parsing errors, within-group duplicate task identities, candidate count/dedup
violations, or modulo-task-order shard misassignments were observed. Invalid final
selections contain out-of-candidate and/or duplicate string IDs; exact selected
and candidate IDs are retained in the final audit. There were no observed mixed
model values within the filename groups, but missing historical hashes/provider
metadata make **all 77 rows unbound**, not provenance-verified. The audit does not
establish primary-selector ranking correctness or successful historical calls.

## Changes and outputs

Changed only the assigned source/document/config files:

- `src/evaluation/budget_protocol.py`: seven injectable conditions, cap24 unique
  candidates, common original-query selector, RRF diagnostic, raw per-call traces,
  observation/retrieval accounting, manifest hashes and strict resume validation.
- `scripts/35_generate_equal_budget_queries.py`: replaced implicit provider
  execution with default offline dry plans; explicit task/split/repeat/index
  inputs, provider/model metadata, write-once manifest/output, hard live gate.
- `scripts/41_audit_budget_runs.py`: read-only exact shard/task coverage,
  duplicate/count/selection/provenance audit, no cross-repeat merge.
- `scripts/34_equal_budget_controls.py`: shared matched subset for incomplete
  conditions, seed/pooled/paired metrics, corrected decision metrics, strict
  single-run binding and non-overwriting outputs.
- `tests/test_budget_protocol.py`: deterministic offline regression suite.
- `configs/equal_budget_protocol.json`: explicit protocol settings.
- `docs/EQUAL_BUDGET_PROTOCOL.md`: condition semantics, commands, accounting,
  provenance rules and limitations.

Authoritative final readiness outputs:

- `results/readiness/budget/legacy_audit_final.json`
- `results/readiness/budget/test_request_plans_final.jsonl`
- `results/readiness/budget/test_request_plans_final.jsonl.manifest.json`
- `results/readiness/budget/tests.xml`
- This report.

Earlier `legacy_audit.json` and `test_request_plans.jsonl[.manifest.json]` are
retained intermediate diagnostics/plans. Use the `_final` outputs for handoff;
earlier plans bind earlier source hashes and are not resumable with final sources.

## Limitations and next action

The available local Python lacks `rank_bm25` and YAML. The new protocol/audit/plans
need neither; the existing BM25 pickle still requires `rank_bm25`, so real-corpus
BM25 scoring was **not run**. Scoring integration was verified with deterministic
fixtures, not reported as empirical retrieval quality. No provider adapter or
live partial-run/retry scheduler was implemented; `--live` deliberately fails.

The no-accumulation ablation means **latest-round evidence only, retaining query
history**. The heuristic is keyword windows, explicitly **not RM3**. Matching 24
unique candidates does not match compute, observed chunks, token use or costs.
`selective_accuracy` remains null without answer-correctness labels; UNCERTAIN
abstains from answering without predicting corpus absence.

Keep legacy work quarantined as unbound. Any future live work needs separately
authorized provider support, actual served-model metadata and raw-response
persistence, a reviewed index adapter, and a fresh provenance-bound repeat across
the full condition/task grid. Do not resume or merge the old shards. No such
network or package work is authorized or needed for this offline handoff.

## Commands executed for final verification

From the analysis directory, using
`/Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python`:

```bash
"$PY" -m pytest tests/test_budget_protocol.py -q -p no:cacheprovider \
  --junitxml=results/readiness/budget/tests.xml

"$PY" scripts/41_audit_budget_runs.py \
  --tasks data/benchmark/test.jsonl --split test \
  --shards results/baselines/equal_budget_llm_queries*.jsonl \
  --expected-shards equal_budget_llm_queries_run8=8 \
  --expected-shards equal_budget_llm_queries=4 \
  --output results/readiness/budget/legacy_audit_final.json

"$PY" scripts/35_generate_equal_budget_queries.py \
  --tasks data/benchmark/test.jsonl --split test \
  --corpus data/processed/corpus.jsonl \
  --index-file data/processed/indices/bm25/bm25_index.pkl \
  --index-file data/processed/indices/bm25/chunk_ids.json \
  --index-file data/processed/chunks.jsonl --repeat offline-plan-v2-final \
  --output results/readiness/budget/test_request_plans_final.jsonl
```

Choose fresh audit/plan output names when rerunning: exclusive creation is
intentional. Additional read-only verification checked all 13 legacy file hashes,
all final-plan source/index hashes, 875 unique task/condition identities and zero
executed-call counters.
