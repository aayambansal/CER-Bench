# Offline equal-candidate-budget protocol (v2)

## Status and interpretation

This implementation provides a pure, injectable protocol, offline request planning,
structural shard auditing, and matched-subset scoring logic. **No new LLM
experiments have been run.** Unit-test responses and search rankings are
deterministic fixtures, not empirical retrieval results. Request plans have
`status: dry_run`, contain no retrieved candidates, and cannot count as completed
work or be resumed as completed experiments.

The shared budget is a **cap of 24 unique candidate documents**, not equality of
compute, ranked chunks, model-visible evidence, prompt tokens, latency, or cost.
An underfilled corpus produces `underfilled`, not padding or duplicated documents.
Primary selection returns at most 20 candidates using the original question's
retrieval ranking, regardless of the candidate-generation condition. With the
existing BM25 adapter this is descending best-chunk original-query BM25 score,
stable index-order ties. RRF with k=60 is diagnostic only, with document-ID ties.
RRF ranks each unique document once per query, not once per chunk.

## Conditions

| Identifier | Candidate generation | Model calls |
|---|---|---:|
| `original_top24` | Original question, 24 new documents | 0 |
| `repeated_original_3x8` | Original question repeated, excluding prior documents | 0 |
| `heuristic_keywords_3x8` | Question, first 24 unique tokens, last 24 unique tokens | 0 |
| `one_shot_rewrite` | One question-only rewrite, 24 new documents | 1 |
| `upfront_three_queries` | Exactly three question-only queries fixed before any retrieval, 8 new documents each | 1 |
| `evidence_feedback` | Original question, then two refinements seeing all previous admitted evidence | 2 |
| `no_accumulated_evidence` | Same iteration, but each refinement sees only the latest round's evidence | 2 |

The no-accumulation ablation retains query history; it removes older **evidence**,
not query history. The heuristic is token-window reformulation, **not RM3**, not a
relevance model, and not pseudorelevance feedback. The original top24 and repeated
original conditions are identical in candidates for a stable adequate index, but
not in logical retrieval requests. No final LLM decision call is added: the
primary selector is the same non-LLM selector for every condition.

## Offline API and accounting

`src/evaluation/budget_protocol.py` has no provider/network dependencies.
`run_protocol(task, condition, search, client, ...)` requires injected functions:

- `search(query)` returns a stable ranked iterable of chunk dictionaries with
  explicit string `doc_id`, `chunk_id`, and optional `text`. A BM25 adapter must
  supply stable descending chunk scores. Doc-ID inference from chunk-ID strings
  is deliberately excluded from the pure protocol.
- `client(request)` returns the complete JSON-compatible envelope, with `parsed`
  holding the query object and optional `usage` and `finish_reason`. A future
  adapter must preserve original provider response fields too; `parsed` is an
  added interpretation, not a replacement for raw response text. An adapter
  should return invalid provider envelopes rather than discard their raw text.
- Task annotations never enter a model request. Only question, permitted query
  history, and permitted evidence do. Upfront/rewrite prompts contain question
  only. Snippet clipping defaults to 420 characters, not a token-budget claim.

Rows retain each logical search's purpose, query, cache hit, total ranked chunk
rows/unique chunks/unique documents, candidate-generation inspected chunk IDs,
admitted chunk rows, shortfalls, and primary/diagnostic selections. Full-ranking
search counts are computational retrieval output, **not model observations**.
Within-run query caching is explicit. Each admitted document contributes one
representative chunk; duplicate chunks/documents cannot consume extra budget.

Each model call retains raw request, raw response, exact displayed observation,
observed unique documents/chunks, clipping characters, usage, finish reason,
latency, and typed error. Invalid responses, length termination, and timeouts
produce `error`; no retries, question fallback, or hidden completion credit is
applied. Earlier candidates/calls survive a later failure. Reported token totals
sum known usage only; missing-token-count calls are separately counted. No token
usage is inferred from characters. Model-visible chunk exposures may exceed
unique observed chunks because evidence can be shown repeatedly.

## Planning and provenance

`scripts/35_generate_equal_budget_queries.py` defaults to **dry run**, even when
API credentials are present. It has no HTTP client import. `--live` fails before
opening task files or constructing a provider. `--dry-run` is optional. Live
support is intentionally unavailable rather than pretending an experiment ran.

Required choices are task path, explicit split, corpus file, index files, and
repeat identifier. Model defaults to the legacy `openai/gpt-5.4-mini`; provider
defaults to `openrouter`. These are requested metadata, not confirmation of a
served model version. A future live adapter must capture actual served model and
provider response metadata and review credentials/network policy separately.

Each plan has a JSON sidecar manifest binding task-file bytes, selected task IDs,
split, corpus bytes, all supplied index files, generation/protocol/config source
bytes, prompts, configuration, requested model/provider, and repeat via SHA-256.
Include chunks and index-to-ID mapping as index inputs: they determine evidence.
The manifest ID is a canonical hash of its content, excluding the ID itself.
Paths are part of provenance, so moving inputs intentionally invalidates resume.
Artifacts are content-bound, not cryptographically authenticated run attestations.

`validate_resume` accepts only identical valid manifests and completed, unique
task/condition rows bound to that manifest, model/provider and condition set;
budgets and selections are structurally checked. Legacy/unbound, conflicting,
duplicate, underfilled, error, and dry-run rows are rejected. CLI dry plans are
write-once and are never silently resumed. There is no partial live-run scheduler
or provider retry persistence in this offline implementation.

Example from the analysis directory (all operations local):

```bash
PY=/Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python
"$PY" scripts/35_generate_equal_budget_queries.py \
  --tasks data/benchmark/test.jsonl --split test \
  --corpus data/processed/corpus.jsonl \
  --index-file data/processed/indices/bm25/bm25_index.pkl \
  --index-file data/processed/indices/bm25/chunk_ids.json \
  --index-file data/processed/chunks.jsonl \
  --repeat offline-plan-v2 \
  --output results/readiness/budget/test_request_plans.jsonl
```

This hashes files but neither unpickles the index nor performs retrieval. Existing
output/sidecar paths cause an error rather than overwrite. Choose fresh names for
subsequent plans with changed sources.

## Auditing legacy shards

```bash
"$PY" scripts/41_audit_budget_runs.py \
  --tasks data/benchmark/test.jsonl --split test \
  --shards results/baselines/equal_budget_llm_queries*.jsonl \
  --expected-shards equal_budget_llm_queries_run8=8 \
  --expected-shards equal_budget_llm_queries=4 \
  --output results/readiness/budget/legacy_audit.json
```

The audit groups filenames after removing `_shardN`; this is a **declared naming
convention**, not recovered repeat provenance. Different groups remain separate.
It reports exact missing/unexpected task IDs, missing shard files, expected
modulo-task-order shard membership, duplicate task/condition identities, parsing
errors, query/round counts, candidate deduplication, final-selection membership,
model/provider conflicts and manifest binding. The four-shard declaration comes
from the stored shard0–3 naming pattern; absence of historical manifests prevents
independent confirmation of original scheduling parameters.

Legacy rows without hashes are **unbound**, even when structurally valid. A
uniform missing provider or provenance value is not proof of homogeneity. Neither
legacy chunk observations, failed attempts, truncation, one-shot retrieval
candidates, nor original selector order can be reconstructed from those rows.
Publication readiness is always false in this structural audit. Byte hashes are
recorded for every audited shard and task input; no source file is modified.

## Scoring safeguards

`scripts/34_equal_budget_controls.py` requires explicit `--tasks` and `--split`.
Without `--llm-queries` it scores only non-LLM controls. With a single input run,
duplicate task/condition or mixed model/provider/repeat/provenance rows fail.
Bound v2 rows require matching task/split/corpus/index hashes, complete status,
24 unique candidates, and agreement with the original-query primary selector.
Legacy rows require `--allow-unbound-exploratory` and cannot become bound by
scoring. This is explicitly nonpublication diagnostics, not permission to merge
different legacy repeats. No multi-shard merge is performed by the scorer.

All available conditions are restricted to **one shared task-ID intersection**
before condition outputs, seed and pooled metrics, paired bootstrap deltas, and
legacy decision metrics. Gold-empty tasks are excluded consistently within each
qrels family. Seed and pooled qrels need not have equal n, but conditions within
each family do. Available counts and exact matched task IDs are reported. A
declared v2 condition with no completed rows forces an empty intersection. Empty
aggregates and bootstrap deltas produce nulls, not NaN or a crash.

`selective_accuracy` is null because no generated scientific answers are scored.
The old all-task binary classification accuracy is now `absence_label_accuracy`.
`answered_answerability_proxy_accuracy` is limited to ANSWER rows and remains a
synthetic-label proxy, not answer correctness. UNCERTAIN reduces answer coverage
and increases nonanswer rate but **does not predict corpus absence**. ABSTAIN alone
is the positive absence prediction. Usage for legacy input is labeled full-input
reported usage, not matched-subset compute.

Default scoring output is under `results/readiness/budget/scoring`; an existing
output directory is rejected and files use exclusive creation. Original scored
outputs under `results/baselines/equal_budget` are preserved. This v2 heuristic
differs from the earlier clause heuristic; old outputs are not relabeled as v2.

## Verification and remaining limits

```bash
"$PY" -m pytest tests/test_budget_protocol.py -q -p no:cacheprovider \
  --junitxml=results/readiness/budget/tests.xml
```

Tests block socket construction, DNS, and connections, exercise every branch,
dedup/underfill, identical selector/RRF, upfront no-feedback ordering, ablation
observations, clipping, invalid/timeout traces, hashes/resume mismatches, shard
coverage/duplicates/conflicts, default dry-run/live gate, immutable outputs and
matched-subset scorer integration using local deterministic fixtures.

The available local Python has NumPy/SciPy/pytest, but not `rank_bm25` or YAML.
No installs are needed for the new protocol, audit, plans or tests. **Real-corpus
BM25 scoring was not run**: the legacy pickle depends on absent `rank_bm25`.
No provider adapter, LLM experiments, compute-equality evidence, human-verified
absence evaluation, or publication result is delivered here.
