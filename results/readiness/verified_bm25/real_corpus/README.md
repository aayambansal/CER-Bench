# Real-corpus index: unscored retrieval smoke

**Complete. No qrels, accuracy claims, human validation or provider calls.**

Index: `data/processed/authoritative_fulltext_v1_bm25/{index.json,postings.npz}`,
relative to `analysis/synthetic-science-search`. Original inputs:
`data/processed/authoritative_fulltext_v1/{corpus.jsonl,chunks.jsonl,dataset_manifest.json}`.
Manifest SHA-256: `59d7751daeb67925903629ed6c542c1e98874d17b0499e43f3a387cc1168b5ae`.

4,936 documents / 10,313 chunks / 4,243,853 tokens / 63,109 vocabulary terms /
2,010,185 postings. Upstream summary: 525 restored fulltext bodies and 21 title-only
documents. Metadata identity validation is not human or biomedical relevance validation.

## Actual measurements and checks

* Build: 3.16048 s; verified load: 2.28054 s.
* Process-lifetime high-water RSS: 538.53 MiB after build, 668.50 MiB after load,
  1,969.48 MiB after full integration including adapter transport/verification.
  Not per-phase allocations or isolated index memory.
* Eight predeclared capability-intention probes, no labels. All admitted 24 unique
  candidates. 24–48 inspected prefix chunk rows; only first 24 saved, no hit text.
* Prefix-search time: 0.923–1.524 ms, median 1.182 ms (one loaded-index measurement
  per query; not accuracy or a latency SLA).
* All three non-LLM conditions × eight probes: 24 complete actual local runs.
* Exact original-query max-chunk selector and repeated-original/original-top24
  equivalence checked for every probe.
* 290 independent naive-equation comparisons, max error 7.105427357601002e-15.
* **68 tests passed, no skips**. See `tests.xml` and `verification.json`.
* 75 adapter-bound source hashes and 213 runtime dependency hashes verified;
  eleven previous synthetic files unchanged.

## Adapter result and limitation

Eight real local searches occurred inside the injected factory. The worker executor
produced 24 offline **dry plans**, not real generation/feedback runs. Its injected-
dependency `synthetic=true` tag was preserved. All provider/model calls were zero.

The initial adapter attempt failed because its JSONL `str.splitlines()` reader splits
embedded Unicode U+2028/U+2029. Partial evidence is retained in `attempt1/`. The
successful run used the adapter's whole-JSON-array path with identical records in
`adapter_transport/*.json`, bound by hashes. The factory verifies against the original
JSONL inputs, original manifest and both index files. Provider-worker files are untouched.
Transport copies are local source data, not retrieval hit dumps; no public redistribution
is authorized. The immutable handoff bundle omits these raw-text transport copies.

See `docs/VERIFIED_RETRIEVAL.md` for runnable commands and a `verified_factory` injection
example. `run_smoke.py` and `verify_outputs.py` preserve the executed procedure.
`summary.json` binds the index, inputs, code and diagnostic artifacts. Its provenance ID:
`480873185f5b1d549d03f8415f391f3758a21cdd8e332803bb5489275196b5b8`.

No historical tasks/qrels were migrated. This establishes local index/integration
readiness only. It does not establish the eight capabilities, biomedical correctness,
retrieval accuracy, abstention validity or any model-feedback effect.
