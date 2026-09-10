# Live generation adapter: blocked until approved inputs exist

No model/API calls, credential discovery, Modal SDK, or human validation were
performed. Unit tests use synthetic fixtures. The v2 integration also loads the
real authoritative corpus/index for **unscored, local-only retrieval**.
`configs/live_pilot.json` is deliberately **unapproved**, with null model/prices
and evidence. It cannot authorize a paid run. $50 is a conservative **per-run
ceiling**, not permission to run multiple $50 pilots. Native live execution now
also requires a shared approved `--campaign-budget` file. Every run directory
in one pilot must use that same file and aggregate lock/ledger (below).

## Inputs and authority required

Use an approved nonsecret JSON config, with the template's exact fields:

* `provider`: `openai`, `anthropic`, or `google`; exact `model` ID (no aliases
  inferred by this implementation). Positive input/output USD per million token
  prices are mandatory; missing prices cannot become zero.
* `approved: true`; `model_evidence`, `pricing_evidence`, and
  `token_reserve_evidence` each contain `path` and `sha256` for a reviewed,
  **nonsecret local evidence file**. Evidence must establish model availability,
  compatibility with the exact native endpoint/JSON/temperature/output settings,
  current billing (including reasoning/cache tokens), and safe maximum token
  accounting. Hash equality verifies the supplied attestation, not its truth.
  Model IDs and prices in tests are invented and must never authorize live use.
* `input_verification` contains `approved: true` and `tasks_sha256`,
  `corpus_sha256`, `chunks_sha256`, `index_sha256`; verified BM25 additionally
  requires `postings_sha256` and `dataset_manifest_sha256`. Review provenance, split,
  identity repair, and corpus/task/index alignment before approving these hashes.
* Positive integer `input_token_reserve`, `output_token_reserve` (at least 400),
  positive timeout, `max_retries` 0–2, positive `max_attempts`, and
  `0 < max_usd <= 50`. The input guard requires request UTF-8 bytes + 4096 <=
  input reserve; this is conservative but **not a universal tokenizer proof**.
  Approved reserve evidence must cover provider overhead and hidden/reasoning
  output. The adapter halts if reported usage exceeds either bound.

Canonical runtime environment keys are `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and
`GOOGLE_API_KEY`; corresponding `SHARED_` aliases are supported. No CLI key flags,
dotenv/config secrets, dotfile access, credential printing, or secret URLs exist.
Do not pass keys through tool arguments or write them to files. Wait for secure
runtime injection; absent credentials fail before dispatch, including baseline-only
live runs. No SDK dependencies are needed.

### Native verified BM25 (v2 production default)

`--backend verified_bm25` (the CLI default) lazily imports and calls
`VerifiedBM25.load` directly, **without an injected retriever factory**. `--index`
accepts the original index directory or its `index.json`; adjacent `postings.npz`
and `--dataset-manifest` are mandatory. All three files are hashed separately.
The loader verifies identities, metadata, NPZ arrays, source/config hashes and
recomputed postings. No array copy/conversion of corpus or chunks is needed.

JSONL now uses physical file iteration, not `str.splitlines()`, for inputs and
attempt ledgers. Literal U+2028/U+2029/U+0085 inside strings survive; duplicate
keys and nonfinite input JSON still fail. Original corpus/chunks are hashed.

Native verified plans are `synthetic:false` unless a chat/search factory is
injected. This denotes the native path, **not scientific/human validation or
model execution**. The explicit `verified-bm25-unscored-probes-v1` envelope is
accepted with its original file hash and without inventing qrels. Selected
probe records are saved solely to declare the execution universe.

Full stable chunk rankings preserve the 24-document budget, inspected chunks
and original-question **max-chunk** document ranking. No top-24 chunk shortcut
or silent selector change is introduced. Serialized protocol traces have an
8 MiB per-row bound (fail closed, no silent truncation). This is a post-execution
trace guard, not a pre-request memory limit. Displayed snippets remain 420
characters; the native client checks input/token reserves before every request.

### Explicit portable backend (opt-in legacy fixtures)

`--backend portable` deliberately does not unpickle historical indices or silently fall
back to another search backend. Corpus is a JSON array of unique document IDs or
records with `doc_id`. Chunks are JSON/JSONL records with unique `chunk_id`, corpus
`doc_id`, and `text`. Tasks require unique `task_id`, `question`, `task_family`,
and matching explicit `split`. Preserve all other annotations locally; only
question/history/chunk evidence enter model requests.

The supported, explicitly selected index is JSON:

```json
{"schema":"cerbench.portable-bm25.v1","rows":[{"chunk_id":"example","tokens":["example","text"]}]}
```

Rows must exactly match chunk order and lowercase `[a-z0-9]+` text tokenization.
BM25 uses k1=1.5, b=0.75 and log(1+(N-df+0.5)/(df+0.5)), chunk-ID tie breaking,
and full chunk rankings. This is a new portable substrate, **not claimed identical
to legacy retrieval**. A verified export/approval is still needed for real data.
Programmatic `execute(..., client_factory=..., search_factory=...)` injection
supports offline tests; any replacement marks manifest and rows `synthetic:true`.
For backwards compatibility only, programmatic args lacking `backend` use portable
and record it explicitly; the CLI never silently chooses portable. There is no
fake-provider CLI flag that could accidentally produce empirical runs.

## CLI and frozen plan

From `analysis/synthetic-science-search`, the network-free invocation shape is:

```bash
/Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python scripts/48_run_live_budget.py \
  --dry-run --backend verified_bm25 --config configs/live_pilot.json \
  --tasks VERIFIED_TASKS.json --corpus VERIFIED_CORPUS.json \
  --chunks VERIFIED_CHUNKS.jsonl --index VERIFIED_INDEX/index.json \
  --dataset-manifest VERIFIED_DATASET_MANIFEST.json \
  --split dev --stage dev-smoke --repeat 1 --output NEW_PLAN_DIRECTORY
```

Paths above are placeholders, not verified execution inputs. Dry-run is default
and performs zero model/search calls, but validates the explicit local index.
`--dry-retrieval-check` additionally runs only the three deterministic conditions
per task and saves separately labeled `retrieval_checks.json`. Model-assisted
conditions remain zero-call plans. This option cannot accompany live execution.
Only after review and secure credential visibility, replace `--dry-run` with
`--live` and supply the approved config and `--campaign-budget`. Do not launch it yet.

`dev-smoke` selects lexicographically first task ID per family on dev, repeat 1.
`frozen` uses every supplied task and requires repeat 1, 2, or 3; run all three in
separate outputs **only under separately approved aggregate funding**. Freeze the
same config/source/input hashes for all three; repeat does not promise provider
seed determinism. The runner does not tune, select winners, calibrate, or claim
human validity. Cross-directory freeze and selecting the correct shared campaign
remain operator gates; the campaign ledger enforces the supplied aggregate cap.

All seven existing `equal-budget-v2` conditions are executed unchanged:

| Condition | Model calls | Candidate rounds |
|---|---:|---|
| original_top24 | 0 | 24 |
| repeated_original_3x8 | 0 | 8+8+8 |
| heuristic_keywords_3x8 | 0 | 8+8+8 |
| one_shot_rewrite | 1 | 24 |
| upfront_three_queries | 1 | 8+8+8 |
| evidence_feedback | 2 | 8+8+8 |
| no_accumulated_evidence | 2 | 8+8+8 |

Each completed condition admits exactly 24 unique documents. Common final
selection is original-question rank restricted to candidates, top 20;
RRF k=60 is a separate diagnostic top 20, not a replacement primary selector.
Evidence observations, snippets (420 characters), inspected chunks, query history,
cache hits and search/selector invocations are retained by the existing protocol.
Token/search compute is **not equalized**. Six logical model calls per task over
all conditions; retries add actual attempts to the ledger.

Observation schedules are unchanged: `evidence_feedback` displays 8 then 16
documents; `no_accumulated_evidence` displays 8 then 8. Both admit 8 new documents
per round. Claiming both observe only 8 would misdescribe/alter the protocol.

## Safety, accounting, resume

Only fixed OpenAI chat-completions, Anthropic messages, and Google
generateContent HTTPS endpoints are available. No proxy/redirect following,
arbitrary endpoint override, URL keys, search-web calls, or Modal operations.
Transport timeout is a socket-operation timeout, **not a hard wall-clock deadline**.
Response bodies are bounded to 4 MiB. Error bodies and redirect locations are
never printed or persisted. Successful raw model content (redacted), provider
usage, body and allowlisted header request IDs, parsed JSON and finish reason
are retained; authentication headers are not. Known runtime key values and JSON
escaped spellings are redacted recursively. This is not a general DLP detector
for unknown or arbitrarily transformed credentials; never put secrets in tasks.

An exclusive advisory output-directory lock is held throughout execution.
Native live runs also require a nonsecret campaign JSON with exactly:

```json
{"schema":"cerbench.campaign-budget.v1","approved":true,"campaign_id":"LEAD_ASSIGNED_ID","max_usd":50,"max_attempts":48}
```

This is a schema example, **not a supplied approval**. The lead must choose the
identity, attempts and budget. The resolved file path has sibling `.lock`,
`.binding.json`, and `.ledger.jsonl` files. Its exclusive lock is held for the
entire live run, preventing parallel spend across participating directories.
The approval hash is bound on first use; changing approval cannot reset spend.
Each attempt reserves aggregate and per-run USD before transport. Aggregate
events record run provenance and attempt IDs. Partial cross-ledger writes block
resume as pending unknown outcomes, even if transport never started.

All directories in one pilot **must use the same campaign file**. A different
approved file is new authority, not a budget extension. Enforcement is not
account-wide across independent campaigns or other software: the lead must not
approve parallel campaigns for this pilot. Keep campaign approval immutable
throughout execution, just like other bound inputs.

Before each attempt a pending USD reservation is append+fsync persisted. All
reservations remain charged against the cap even when reported cost is smaller
or an HTTP failure occurs. Missing usage has `actual_usd:null`, never zero.
429/500/502/503/504 may retry at most twice with bounded backoff, subject to total
request and reservation caps before every retry. Timeout/transport/malformed
unknown responses are not retried. Failed attempts remain in `ledger.jsonl` and
row `provider_attempt_events`; protocol logical-call totals are not retry totals.

Manifest binds corpus, chunks, tasks, index, all project Python source files,
loaded stdlib/runtime dependency files, Python executable/version, config,
evidence hashes, condition set, stage, selected task IDs and repeat. Changes fail
resume. Hashes establish file identity, not upstream scientific verification.
Files must remain immutable throughout a run (hashing is not a filesystem snapshot).
Crash-pending rows or unknown ledger outcomes require manual reconciliation;
the runner never silently skips/reissues them. Incomplete rows stop the run.
Do not delete a pending file to retry a possibly billed request. There is no
automatic reconciliation command. CLI failures use generic sanitized stderr.

## Explicit output adapters and verification

`runs.json` retains full protocol rows; `manifest.json` binds them.
`budget_runs.jsonl` and `budget_runs.jsonl.manifest.json` explicitly adapt those
rows for the existing budget auditor (`scripts/41_audit_budget_runs.py`).
`CONDITION.common_selector_docs.eval.json` and `CONDITION.rrf_docs.eval.json`
map **only** task ID and chosen selector to strict `retrieved_doc_ids` rows.
Pass the matching `selected_tasks.json` universe to
`scripts/40_evaluate_validated_runs.py`; evaluate conditions/repeats/selectors
separately. No invented confidence, decisions, adjudication, or gold labels.
Any future/new budget schema must receive a reviewed explicit mapping; compatibility
with an unavailable schema is not asserted.

Offline regression command (tested; no network):

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python -m pytest -q -p no:cacheprovider tests/test_live_provider.py tests/test_budget_protocol.py tests/test_verified_bm25.py tests/test_evaluation_integration.py -k 'not test_actual_smoke_provenance' --junitxml=results/readiness/live_adapter/v2/offline_tests.xml
```

v1 outputs remain untouched and synthetic. v2 contains real-corpus unscored local
retrieval checks, zero-model plans and software verification. The excluded
historical provenance test asserts the old script48 hash; its preserved report
is stale after this requested change, not silently regenerated. Initial combined
suite: 110 passed, 1 failed on that hash. Current filtered suite: 113 passed,
1 deselected. New v2 integration binds the current 77 Python source files.

The paper must state **no new model runs** unless credentials are securely
injected and a separately authorized live execution actually succeeds.
