# Provider48 / native verified BM25 integration v2

**No new model runs. No provider/API/network calls. No credential injection,
secret files/tool arguments, Modal SDK, or human validation.**

The existing eight predeclared probes were used unchanged and remain unscored:
their family names are intentions, not verified task properties or gold labels.
Original corpus/chunks JSONL loaded directly; no array transport copy was used.
v1 live-adapter and the retriever worker's historical outputs were not overwritten.

## Observed result

`verified_originals_summary.json` was produced by `verify_dry.py` with assertions
against current source/input/output hashes:

| Check | Observed |
|---|---:|
| Corpus documents | 4,936 |
| Chunks | 10,313 |
| Original probe tasks | 8 |
| Zero-model condition plans | 56 |
| Local deterministic retrieval checks | 24 |
| Completed checks with exactly 24 unique candidates | 24/24 |
| Both final selectors return 20 documents | 24/24 |
| Provider/model calls | 0 |
| Local search invocations, including cache hits | 136 |
| Uncached local searches | 28 |
| Largest serialized check trace | 88,567 bytes |
| Protocol per-row trace bound | 8,388,608 bytes |
| Bound project Python source files checked | 77 |

Both original `index.json` and `postings.npz`, the dataset manifest, original
corpus/chunks, original probe envelope and source code are provenance-bound.
`synthetic:false` denotes the native verified path, not model execution or
scientific validation. Generation rows remain `status:dry_run`; local checks
are separately `execution_kind:offline_unscored_retrieval_check`.

Provenance ID:
`f0b56a4d87a6cf78eadd8c30ad7ddba3030f882633d9b30408371512dd0e97a7`.

## Exact tested commands

Run from `analysis/synthetic-science-search`:

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python scripts/48_run_live_budget.py --dry-run --dry-retrieval-check --backend verified_bm25 --config configs/live_pilot.json --corpus data/processed/authoritative_fulltext_v1/corpus.jsonl --chunks data/processed/authoritative_fulltext_v1/chunks.jsonl --dataset-manifest data/processed/authoritative_fulltext_v1/dataset_manifest.json --index data/processed/authoritative_fulltext_v1_bm25/index.json --tasks results/readiness/verified_bm25/real_corpus/probes.json --split dev --stage dev-smoke --repeat 1 --output results/readiness/live_adapter/v2/verified_originals_dry

PYTHONDONTWRITEBYTECODE=1 /Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python results/readiness/live_adapter/v2/verify_dry.py

PYTHONDONTWRITEBYTECODE=1 /Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python -m pytest -q -p no:cacheprovider tests/test_live_provider.py tests/test_budget_protocol.py tests/test_verified_bm25.py tests/test_evaluation_integration.py -k 'not test_actual_smoke_provenance' --junitxml=results/readiness/live_adapter/v2/offline_tests.xml
```

These outputs now exist. Choose new output/report paths to rerun without changing
evidence. The verifier summary is exclusive-create.

Final filtered suite: **113 passed, 1 deselected in 1.62 seconds**. Unit tests
block socket connections. Coverage includes literal Unicode separators in both
JSONL and ledger, native VerifiedBM25 loading/hashes/non-synthetic provenance,
injected-factory synthetic marking, selector equivalence, 24-document budgets,
8/16 versus 8/8 observation schedules, trace-size guard, shared campaign cap,
unknown cross-ledger outcomes and campaign lock contention, plus existing safety
and retriever/protocol/evaluation tests.

The initial unfiltered cross-suite produced **110 passed, 1 failed**. Its report
is preserved as `initial_cross_suite.xml`. Failure:
`tests/test_verified_bm25.py::test_actual_smoke_provenance`; that historical smoke
report asserts the prior script48 source hash. We neither edited the other
worker's test/report nor relabeled it passing. New v2 provenance was verified
against the updated code. Lead should keep the historical run immutable and
treat its source snapshot as historical, or explicitly version its revalidation.

## Execution gates / limits

* Native live execution requires one approved nonsecret `--campaign-budget` JSON.
  All campaign directories must share its canonical path, lock and ledger.
  Both campaign and run cap remain <= $50; every attempt reserves against both
  before dispatch. No aggregate budget file was created or approved here.
* This is not provider-account-wide enforcement against independent campaigns
  or other tools. Lead must authorize only one campaign for this pilot.
* Full eight-task/seven-condition plan needs **48 logical model calls**, before
  retries. The current template permits only **42 attempts** and is unapproved.
  It cannot finish that full live plan; lead must explicitly approve a suitable
  plan/cap rather than silently tuning or increasing it.
* Exact model, prices, reserve evidence, secure credentials and live input-hash
  approvals still require lead action. None were guessed or retrieved here.
* Accumulated feedback still observes 8 then 16 documents; no-accumulated sees
  8 then 8. This preserves existing protocol conditions. Actual dry API-input
  observations are zero, because no API requests occurred.
* Trace-size guard is post-row; input token reserve is checked before each API
  call. Socket timeout is not a hard wall-clock limit. Source/data/campaign
  approval files must remain immutable during a run.

Changed only `scripts/48_run_live_budget.py`, `src/agents/provider_clients.py`,
`tests/test_live_provider.py`, `docs/LIVE_EXPERIMENTS.md`, and new outputs in this
v2 directory. Existing configs, retriever/index/corpus, budgets and other scripts
were not modified. No commit or network experiment was performed.
