# Verified BM25 implementation handoff

## Outcome

Implemented standalone stdlib/NumPy inverted chunk BM25 and structurally labelled
programmatic-fiction fixtures. No network/model calls, secrets, installations or Git
operations. No historical corpus/pickle loading or existing candidate diagnostic run.

## Changes / outputs

All paths below are relative to `analysis/synthetic-science-search`:

* `src/retrieval/verified_bm25.py`: strict identity/hash-bound JSON/NPZ index,
  BM25Okapi equations, stable tokenizer/order, callable full chunk search, <=24 unique
  candidate prefix, original-query max-chunk document ranking.
* `scripts/49_index_and_run_verified_bm25.py`: tested `synthetic`, `build`, `run`,
  `benchmark` CLI subcommands; required dataset manifest and explicit complete
  tasks/qrels universe; exact search and selector traces, hashes and timings.
* `tests/test_verified_bm25.py`: 35 tests including equation reference, tampering,
  identity rejection, structural labels, complete universe and adapter injection.
* `docs/VERIFIED_RETRIEVAL.md`: schemas, equations, API, exact executable commands,
  integrity threat model, benchmark assumptions and integration limitations.
* `results/readiness/verified_bm25/synthetic/`: 48 documents, 144 chunks, four tasks,
  generated-factor annotations, non-gold qrels, expected IDs, manifest, safe index,
  `run/runs.jsonl`, and `run/run_manifest.json`.
* `results/readiness/verified_bm25/benchmark/benchmark.json`: raw bounded timings,
  seed/input/source hashes and explicitly unmeasured projection.
* `results/readiness/verified_bm25/tests.xml`: complete JUnit test evidence.

## Evidence

Executable: `/Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python`.

```bash
/Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python -m pytest tests/test_verified_bm25.py tests/test_budget_protocol.py -q --junitxml=results/readiness/verified_bm25/tests.xml
```

**63 passed, no skips** (35 new + 28 existing), 0.59 s. All four synthetic baseline
rows completed with 24 unique candidates. Saved run hash independently verified:
`2e2c6ca82cf2a8c23789b234442f4255536fd35ebeacfa934d4612f5ae39ab6e`.

The inspected budget protocol uses callable search, with no named SearchProtocol
class. `VerifiedBM25` implements that callable plus explicit `search` and `rank`.
Integration with `48_run_live_budget.py` exercised its injected `search_factory`
in dry mode and actual local retrieval, with provider construction forbidden.
It did not execute a live model request or replace that worker's default backend.

At 6,000 synthetic chunks (468,000 tokens, vocabulary 23), build was 0.168890 s,
verified load 0.107216 s and median full search 0.001985 s across 15 searches.
One-million-chunk extrapolation: build 28.15 s, load 17.87 s, full search 0.525 s,
assuming equivalent token lengths/posting density and linear / N log N scaling.
These estimates are NOT measured real-corpus performance. Full hit serialization
and peak memory were not benchmarked; the fixture is deliberately unrealistic.

## Limitations

The new authoritative dataset has not been indexed or audited here. Its worker must
supply/adapt explicit `article_id` and `corpus_id` fields and the documented hash/status
manifest. Identity attestations do not prove source-text authenticity. This work never
upgrades labels to human gold and sets `publication_claim=false` on all run reports.
No RM3/PRF was implemented; existing heuristic keyword expansion is not RM3.

Verified loading intentionally rebuilds postings once to establish semantic input
binding. Full ranking/trace output can dominate memory/I/O on large corpora.
The live worker's default portable search is a DIFFERENT scoring convention and
does not accept this NPZ index automatically. Its injected factory must explicitly
call `VerifiedBM25.load` with all original paths and hash-bind both JSON and NPZ.

## Next action

Lead: reconcile the authoritative worker's manifest to the documented strict schema,
build a fresh index after that identity audit, then run an explicit expected task
universe with separately bound qrels. For live use, wire the verified factory rather
than the portable default and retain the same original-query selector. Rebenchmark
the real corpus locally before scheduling any credentialled/provider experiment.
