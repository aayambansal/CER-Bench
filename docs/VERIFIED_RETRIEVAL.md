# Verified local BM25 and synthetic contracts

**Latest integration receipt: `results/readiness/verified_bm25/real_corpus_v2/`.**
After legitimate provider48 updates, native verified-backend integration and the full
configured repository suite passed: **311 tests, zero failures/errors/skips**. The v1
receipts below remain immutable historical evidence, not attestations to current
provider-worker source. See the final section for current native wiring/results.

This is an **offline implementation/integration diagnostic, not a publication result**.
It uses Python stdlib and NumPy only. It never imports `search_api.py`, opens a pickle,
constructs a provider, reads credentials, downloads data, or uses an installed BM25 package.
The initial synthetic verification was followed by the real-corpus smoke documented
below. Existing candidate artifacts are untouched; diagnostic mode does not upgrade
candidate labels to gold.

## Scoring and ranking contract

The sole tokenizer lowercases Unicode text, then extracts regex `[a-z0-9]+` matches.
Single-character terms, numeric terms, and repetitions are retained. Punctuation,
hyphens, underscores and non-ASCII letters are boundaries; there is no stemming,
stop list, synonym expansion, title concatenation, metadata boosting or BPE.
Only each chunk's `text` is indexed. This intentionally differs from the legacy tokenizer.

For N chunks, chunk length L, corpus mean chunk length avgL, term frequency f and
document frequency df (here **chunk** frequency), the implementation uses:

```text
raw_idf(t) = ln((N - df(t) + 0.5) / (df(t) + 0.5))
average_idf = arithmetic mean of raw_idf over every vocabulary term
idf(t) = 0.25 * average_idf if raw_idf(t) < 0 else raw_idf(t)
score(q,c) = sum over query token occurrences t of
             idf(t) * f(t,c) * (1.5 + 1)
             / (f(t,c) + 1.5 * (1 - 0.75 + 0.75 * L(c)/avgL))
```

This is the `rank_bm25.BM25Okapi` scoring convention, tested against an independent
direct implementation of its equations rather than installing/importing that package.
Repeated query terms contribute repeatedly. The replacement IDF can itself be negative:
it is **not** clipped to zero and does not use `log(1 + ratio)`. OOV terms contribute zero.
Empty chunks are allowed within a nonempty-token corpus; a wholly token-empty index is rejected.

All chunk scores, including zero/negative scores, participate in descending ranking.
Ties use stable **input chunk order**, never chunk-ID parsing or lexical chunk-ID order.
Document scoring is the maximum original-query score across **all** chunks belonging to
that discovered document. Ties between documents use the order of their first winning
chunk. "Relevant chunk" here means a lexical scoring contribution, **not** a qrel judgment.
The retriever accepts no task annotation or oracle label. The runner passes only the
task ID/question to the protocol; explicit qrels are used afterward for diagnostic recall.

## Input and identity contract

Inputs are JSONL objects with these required fields:

* Corpus: `doc_id`, `article_id`, `corpus_id` (all explicit strings).
* Chunks: `chunk_id`, `doc_id`, `article_id`, `corpus_id`, `text`.

`doc_id` and `article_id` must each be unique across corpus rows, with nonempty,
untrimmed-equal IDs. `chunk_id` must be unique. Every chunk references an existing
document, every document has a chunk, and chunk/article and all corpus identities must
agree exactly. A chunk ID is opaque. Duplicate JSON keys/nonfinite values are rejected.
These checks detect inconsistent mappings, not falsely attested source content: an
upstream authoritative identity audit must establish that `article_id` is canonical
and that text actually belongs to that article. Alias resolution is deliberately not
guessed here. A new authoritative worker may need an explicit schema conversion.

The build **requires** an explicit dataset manifest with this schema (hashes are exact
SHA-256 of input file bytes, not normalized JSON):

```json
{
  "schema": "verified-retrieval-dataset-v1",
  "status": "authoritative_validated",
  "corpus_id": "YOUR_EXPLICIT_CORPUS_ID",
  "identity_validated": true,
  "canonical_ids_unique": true,
  "document_count": 123,
  "chunk_count": 456,
  "corpus_sha256": "EXACT_CORPUS_FILE_SHA256",
  "chunks_sha256": "EXACT_CHUNKS_FILE_SHA256"
}
```

Do not merely relabel old data to satisfy this gate. The flags are an upstream
attestation, not an audit performed by BM25. The other explicitly selected statuses
are `synthetic_structural` and `candidate_diagnostic`; the default never accepts them.

## Safe index and API

`build_index(corpus, chunks, manifest, output, mode=...)` writes only:

* `postings.npz`: numeric term-major postings (`indptr`, `indices`, `tf`), `lengths`, `idf`.
* `index.json`: exact ordered vocabulary, document IDs, chunk IDs, chunk-to-document
  relation, configuration, corpus identity/status, input/config/retriever-code hashes,
  per-array dtype/shape/content hashes, NPZ hash and JSON integrity hash, build timing.

`VerifiedBM25.load(index_dir, corpus=..., chunks=..., manifest=..., mode=...)` requires
all original inputs again. It validates every stored hash, ID ordering and configuration,
loads NPZ with `allow_pickle=False`, and recomputes postings/IDFs once to verify semantic
binding to current input text. This deliberately makes verified loading O(input size),
not a cheap mmap open. Source edits invalidate existing indexes; rebuild explicitly.
Hashes detect corruption/substitution relative to supplied inputs, not authenticity
against an attacker who can rewrite both source data and attestations.

The inspected `src/evaluation/budget_protocol.py` exposes a callable search contract,
not a named `SearchProtocol` class. This implementation supports:

```python
hits = index(query)  # equivalent to search(query): complete ranked chunk dictionaries
hits = index.search(query, top_k=100)  # chunk cap, not a document cap
hits = index.search(query, max_unique_docs=24, exclude_doc_ids=already_seen)
ranked_docs = index.rank(original_query, discovered_doc_ids, top_k=20)
```

Search rows contain `doc_id`, `chunk_id`, `text`, and exact float `score`.
`rank` rows contain `doc_id`, winning `chunk_id`, and `score`; candidates must be
at most 24 unique known documents. The unique-document search prefix includes duplicate
chunks until the final new document is encountered, then stops immediately. For three
chunks per document and tied scores, 24 unique documents require 70 chunk rows, not 24
or 72. Prefix budgets bound admission, **not** scoring computation or trace size.
The budget protocol receives the full callable ranking, manages its own 24-candidate
admission, and agrees exactly with this max-chunk selector in integration tests.

The independent worker's `scripts/48_run_live_budget.py` was tested through its injected
`search_factory(chunks, index_json)` seam in **dry mode**, with provider construction
explicitly forbidden. Actual verified retrieval occurs inside the injected factory.
Its default `portable_search` uses a different scoring equation and is **not this index**.
The lead must explicitly wire a verified factory with corpus/chunks/manifest/NPZ binding
before any real live run; this patch does not alter files owned by that worker.

## Executable CLI

Run from `analysis/synthetic-science-search`. All output destinations must be fresh;
there is no implicit overwrite/resume of a partial index or run.

```bash
PY=/Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python
CLI=scripts/49_index_and_run_verified_bm25.py

# After the authoritative dataset has passed its independent identity audit:
"$PY" "$CLI" build --corpus /path/corpus.jsonl --chunks /path/chunks.jsonl \
  --manifest /path/dataset_manifest.json --index /path/new-index

"$PY" "$CLI" run --corpus /path/corpus.jsonl --chunks /path/chunks.jsonl \
  --manifest /path/dataset_manifest.json --index /path/new-index \
  --tasks /path/tasks.jsonl --qrels /path/qrels.json \
  --expected-task-ids /path/expected_task_ids.json --output /path/new-run
```

The run command requires all three evaluation inputs explicitly:

* Tasks JSONL: unique `task_id`, nonempty `question`; annotations are never sent to retrieval.
* Expected IDs: nonempty JSON list of the **complete intended** task universe; tasks and
  qrels must match it exactly. Correct selection of this universe remains the lead's responsibility.
* Qrels JSON: `schema="verified-retrieval-qrels-v1"`, matching `corpus_sha256` and
  `tasks_sha256`, `label_status`, and `qrels={task_id: {doc_id: nonnegative_integer_grade}}`.
  Empty judgments are explicit and produce null recall, not a fabricated zero.
  Unlisted documents remain unjudged; the diagnostic does not assert they are negative.

Exact mode/label pairings:

| Dataset mode | Required qrel label |
|---|---|
| `authoritative_validated` | `explicit_qrels_not_human_validated` |
| `synthetic_structural` | `synthetic_structural_not_gold` |
| `candidate_diagnostic` | `candidate_not_gold` |

Authoritative corpus identity is not human relevance validation. This runner deliberately
makes no main publication claim, even for authoritative inputs. It does not replace the
project's human-validation/evaluation gate. CLI conditions are `original_top24` (default),
`repeated_original_3x8`, and `heuristic_keywords_3x8`. The latter is lexical keyword
windowing from the existing protocol, **not PRF or RM3**. No PRF/RM3 is implemented.

`runs.jsonl` retains every exact executed search ranking (chunk hits/text/scores/timing),
the original selector trace, all 24 candidate document scores/winning chunks, top-20
selection, existing budget traces, label status and diagnostic recall. Rows stream to
disk task by task. `run_manifest.json` binds all evaluation/index/source inputs and
output hashes, task IDs, timing and completeness. Interrupted runs without a final
manifest are partial and cannot silently resume. Large full-ranking traces can dominate I/O.

## Synthetic structural fixture and measured scaling

The tested fixture uses 48 fictional documents, 144 chunks and four tasks, seed 1729.
Text is programmatically generated from system/process/context factors plus sampled
filler. No head/title metadata or label objects enter chunk text. The content naturally
contains the factors being queried; this is intentionally easy, not a realism claim.
Qrels are derived from exact equality of **all three generated factors**, not from the
retrieval ranking. Each task's separate `synthetic_contract` records constraints,
required evidence roles, supporting IDs and non-gold label status. The standalone
`structural_labels.json` records the generator factors for every document. These are
structural labels only, not expert annotations, causal evidence or scientific truth.

The commands below were executed successfully using the Python executable above:

```bash
OUT=results/readiness/verified_bm25
"$PY" "$CLI" synthetic --output "$OUT/synthetic"
"$PY" "$CLI" build --corpus "$OUT/synthetic/corpus.jsonl" \
  --chunks "$OUT/synthetic/chunks.jsonl" --manifest "$OUT/synthetic/dataset_manifest.json" \
  --index "$OUT/synthetic/index" --mode synthetic_structural
"$PY" "$CLI" run --corpus "$OUT/synthetic/corpus.jsonl" \
  --chunks "$OUT/synthetic/chunks.jsonl" --manifest "$OUT/synthetic/dataset_manifest.json" \
  --index "$OUT/synthetic/index" --mode synthetic_structural \
  --tasks "$OUT/synthetic/tasks.jsonl" --qrels "$OUT/synthetic/qrels.json" \
  --expected-task-ids "$OUT/synthetic/expected_task_ids.json" --output "$OUT/synthetic/run"
"$PY" "$CLI" benchmark --output "$OUT/benchmark"
"$PY" -m pytest tests/test_verified_bm25.py tests/test_budget_protocol.py -q
```

Benchmark output records raw timings for 15 queries at 600 and 6,000 chunks (78 tokens
per chunk, only 23 vocabulary terms), plus input/source hashes and seed. Indexing is
O(tokens + vocabulary sorting); numeric query scoring is O(N + query postings),
ranking O(N log N), and full hit construction O(N). Memory retains corpus text,
postings and score/order arrays; verification temporarily retains rebuilt postings.

`benchmark/benchmark.json` contains the actual measurements and an explicitly unmeasured
one-million-chunk projection using linear build/load and N log N full-search scaling
from the largest point. These warm-cache, tiny-vocabulary fictional measurements are
**not measured real-corpus performance**. No peak RSS or full trace serialization timing
is available. Measure the new authoritative corpus separately before a large run.

Final local verification: **63 passed** (35 new retrieval/contract tests plus 28 existing
budget-protocol tests), no skips, in 0.59 seconds. `results/readiness/verified_bm25/tests.xml`
preserves individual test results. All four synthetic baseline tasks completed with
exactly 24 unique candidates, and the saved run's SHA-256 was rechecked successfully.

| Synthetic chunks | Build seconds | Verified load seconds | Median full search seconds |
|---:|---:|---:|---:|
| 600 | 0.01515 | 0.01261 | 0.0002043 |
| 6,000 | 0.16889 | 0.10722 | 0.0019850 |

The stated scaling model projects approximately 28.15 seconds to build, 17.87 seconds
to verify/load and 0.525 seconds per full search at one million equally short chunks.
These are **unmeasured extrapolations**, not capacity commitments or real-corpus results.

## Actual authoritative-fulltext index: completed unscored smoke

The subsequent real-data run is separate from the preserved synthetic results.
It indexed `data/processed/authoritative_fulltext_v1/{corpus.jsonl,chunks.jsonl}`
against dataset manifest SHA-256
`59d7751daeb67925903629ed6c542c1e98874d17b0499e43f3a387cc1168b5ae`.
The input contains 4,936 documents, 10,313 chunks, and 4,243,853 indexed tokens.
The upstream restoration summary reports 525 restored-body documents and 21 title-only
documents; this is not 4,936 full-body articles. Identity validation is metadata-based;
human and biomedical relevance validation remain false.

**Final index files:**

* `data/processed/authoritative_fulltext_v1_bm25/index.json` — 1,641,922 bytes;
  SHA-256 `1656d63a13830b62fa856586efb52424fda9c60527c81e2a4295ee87ebf67e35`.
* `data/processed/authoritative_fulltext_v1_bm25/postings.npz` — 4,804,331 bytes;
  SHA-256 `167bd9d7aa4d644b24e197387095a3200ea7c444048d4a8d176942847513070e`.

The real index contains 63,109 vocabulary terms and 2,010,185 term/chunk postings.
Build measured **3.16048 s**; full verified load measured **2.28054 s**. Process-lifetime
RSS high water was **538.53 MiB after build**, **668.50 MiB after load**, and
**1,969.48 MiB after the entire integration**, including duplicated adapter transport
and verification state. These are cumulative `resource.getrusage` values on Darwin,
not isolated per-phase memory requirements or allocation deltas. Timing was measured
without tracemalloc; these are actual observations, not the earlier extrapolation.

Eight queries were declared in `results/readiness/verified_bm25/real_corpus/probes.json`
before indexing/search: constraint, comparative, contradiction, abstention, multihop,
temporal, aggregation and negative capability **intentions**. No capability was scored,
and no abstention correctness, contradiction discovery or biomedical relevance is claimed.
Every query admitted 24 unique candidate documents. Prefixes required 24–48 chunk rows;
only the first 24 chunk IDs/document IDs/scores per query were saved, with no article text.
Measured prefix-search latency ranged from **0.923 to 1.524 ms**, median **1.182 ms**
(eight single measurements, warm loaded index; not an accuracy evaluation or stable latency SLA).

All **24 actual non-LLM protocol runs** (eight queries × three conditions) completed.
The selector was independently checked by reducing the unsorted original-query score
array to each candidate's best chunk. Repeated-original 3×8 and original-top24 agreed
on ordered candidates, original-query selection and RRF selection for every probe.
Independent source-text re-tokenization and naive BM25 equations yielded **290 numerical
comparisons across 29 sampled chunks and ten queries**, maximum absolute error
**7.105427357601002e-15**. Tests: **68 passed, no skips** (40 retrieval/contract tests
including real-output checks, plus 28 existing budget-protocol tests).

Receipts and bounded diagnostics are under `results/readiness/verified_bm25/real_corpus/`:
`summary.json`, `verification.json`, `tests.xml`, `build_load_resources.json`,
`query_diagnostics.json`, `protocol_diagnostics.json`, `independent_formula_checks.json`,
`adapter_integration.json`, and `adapter_dry/`. The verification receipt checked all
75 adapter-bound source hashes, 213 runtime dependency hashes, both index files and
all saved diagnostic hashes. Eleven pre-existing synthetic files were unchanged.

### Adapter integration and discovered transport limitation

The provider worker's JSONL reader uses `str.splitlines()`. Embedded U+2028/U+2029
characters in valid corpus JSON strings caused its original-input dry run to fail
with `JSONDecodeError: Unterminated string starting at: line 1 column 14 (char 13)`.
That partial attempt and index are retained under `real_corpus/attempt1/`; it is not
the final successful run. No provider-worker source was edited.

The repaired integration passes semantically identical whole-JSON arrays, generated
from the exact original JSONL records, to the adapter's supported `.json` loader:
`real_corpus/adapter_transport/{corpus.json,chunks.json}`. These local transport copies
retain all input text; they are not bounded hit logs and are **not authorized for public
redistribution**. Their hashes are bound by both adapter and final smoke manifests.
The verified factory still loads and validates the original JSONL/hash manifest and
both original index files; it rejects any chunk/metadata disagreement.

The helper `verified_factory(index_dir, inputs)` lives in the reproducible smoke script,
leaving the retriever and script 49 unchanged so the synthetic index remains loadable.
The lead can load the helper and inject it without editing worker files:

```python
import importlib.util
from pathlib import Path

path = Path("results/readiness/verified_bm25/real_corpus/run_smoke.py")
spec = importlib.util.spec_from_file_location("local_smoke", path)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)
inputs = {name: Path("data/processed/authoritative_fulltext_v1") / filename
          for name, filename in {"corpus": "corpus.jsonl", "chunks": "chunks.jsonl",
                                 "manifest": "dataset_manifest.json"}.items()}
search_factory = smoke.verified_factory(
    Path("data/processed/authoritative_fulltext_v1_bm25"), inputs)
# With the independently loaded adapter module and fresh output:
# args.corpus/chunks = paths to adapter_transport/*.json
# args.index = path to authoritative_fulltext_v1_bm25/index.json
# args.live = False
# adapter.execute(args, search_factory=search_factory, client_factory=forbid_provider)
```

The completed dry adapter integration produced 24 **plans**, not live executions.
Eight actual local retrieval calls occurred inside the instrumented injected factory;
the dry plans themselves executed no search or generation. The adapter's
`synthetic=true` flag for injected dependencies was retained verbatim, not relabelled
as provider empirical evidence. No feedback generation, historical tasks or qrels
were used. A future provider run must retain explicit NPZ binding and independently
satisfy its authorization/evidence gates; this dry run does not authorize it.

### Reproduction

The following command was executed successfully after the transport repair. It refuses
existing index/final-plan destinations; choose fresh destinations for another run:

```bash
PY=/Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python
OUT=results/readiness/verified_bm25/real_corpus
"$PY" "$OUT/run_smoke.py" \
  --corpus data/processed/authoritative_fulltext_v1/corpus.jsonl \
  --chunks data/processed/authoritative_fulltext_v1/chunks.jsonl \
  --manifest data/processed/authoritative_fulltext_v1/dataset_manifest.json \
  --index data/processed/authoritative_fulltext_v1_bm25 \
  --probes "$OUT/probes.json" --output "$OUT"
"$PY" -m pytest tests/test_verified_bm25.py tests/test_budget_protocol.py \
  -q --junitxml="$OUT/tests.xml"
"$PY" "$OUT/verify_outputs.py"
```

Final provenance ID: `480873185f5b1d549d03f8415f391f3758a21cdd8e332803bb5489275196b5b8`.

## Native48 integration v2: current source bindings

Provider48 now supports the native `verified_bm25` backend, binds `index.json`,
`postings.npz` and the dataset manifest, and correctly parses original JSONL containing
embedded U+2028/U+2029. The v1 transport workaround and injected factory remain a
description of the historical test only; **neither is needed in the current native path**.

`results/readiness/verified_bm25/real_corpus_v2/run_integration.py` executed
`adapter.execute(args)` with native defaults, `backend="verified_bm25"`, `live=False`,
and `dry_retrieval_check=True`, using the exact original authoritative inputs and
unchanged index. There was no index regeneration, provider construction or model call.
It checked all 24 actual non-LLM retrieval runs, all original-query max-chunk selectors,
and repeated-original equivalence across the same eight predeclared intentions.
The adapter also produced 24 dry plans, explicitly separate from the local retrieval
checks. Flags are `synthetic=false` (native, not injected), `live=false`,
`human_validation=false`, and `label_status="unscored_retrieval_smoke_no_qrels"`.
Native execution does not imply human validation, relevance accuracy or feedback generation.

Current v2 measurements:

* Explicit verified index load: **1.711876 s**.
* Native adapter invocation: **3.152019 s**, including its own verified load, local
  protocol checks, provenance collection and trace writes—not just search latency.
* Independent selector/equivalence-check loop: **0.183194 s**.

The old build time and 290 naive-formula comparisons are referenced under
`historical_evidence`, labelled `copied_historical_v1_not_rerun`; they are not new
measurements. The original index/config/retriever hashes are unchanged. Forty-three
historical files, including the v1 receipt and synthetic outputs, were verified unchanged.

The `actual_smoke` test fixture explicitly targets the completed **v2** summary.
Provenance assertions remain strict for current source/output/index and native
runtime bindings. A separate regression test confirms that applying the old v1
source bindings to changed current code raises `Stale source binding`, while the
old summary's own integrity digest remains valid. No test was removed or deselected.

All code and test revisions preceded the v2 manifest freeze. The summary does not
hash itself or a not-yet-written pytest output. `verify_integration.py` creates a
separate post-suite receipt binding the frozen summary, native manifest, JUnit XML
and verification code; no execution manifest is rewritten after testing.

Executed from the configured repository root `analysis/synthetic-science-search`:

```bash
PY=/Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python
OUT=results/readiness/verified_bm25/real_corpus_v2
"$PY" "$OUT/run_integration.py"
"$PY" -m pytest -q --junitxml="$OUT/all_tests.xml"
"$PY" "$OUT/verify_integration.py"
```

The full repository-configured suite completed with **311 passed in 6.36 s; zero
failures, errors or skips**. Post-suite verification checked 79 frozen current source
bindings, 213 native runtime bindings, all diagnostic/index hashes and historical
file preservation. Receipts: `summary.json`, `native48/manifest.json`,
`all_tests.xml`, `verification.json`. No manuscript was edited.

V2 provenance ID: `d833bada0b973b033da0b1402d10b2a5592c532e8dd7eb567f0db4e6ac7b3749`.
