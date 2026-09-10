# Current native48 integration v2

**Complete: 311 tests passed in 6.36 s, zero failures/errors/skips.** No deselection.
The fixture now targets this new receipt; all source/output/index/runtime assertions
remain strict. Historical v1 receipts are unchanged and retain their original hashes.

## Scope and flags

The unchanged real BM25 index was reused against the original authoritative JSONL
inputs and pinned manifest. No index build, transport copies, injected dependencies,
historical task migration, API calls, secrets, installations or Git operations.

The current provider48 native backend correctly reads the embedded Unicode separators
and binds both index files plus the dataset manifest. It executed 24 offline non-LLM
retrieval checks across the eight predeclared topical intentions, and produced 24
separate dry plans. All max-original-query chunk selectors and same-query repeated
budget equivalences passed.

* `synthetic=false`: native backend, not a dependency-injection fixture.
* `live=false`, `human_validation=false`, `executed_model_calls=0`.
* `feedback_generation_executed=false`, `publication_claim=false`.
* `label_status=unscored_retrieval_smoke_no_qrels`.

These are local integration checks, not capability accuracy or model-feedback evidence.

## Measurements and provenance

Current verified load: **1.711876 s**. Native adapter execution: **3.152019 s**, including
its own load, retrieval checks, provenance hashing and output writes. Selector-check
loop: **0.183194 s**. RSS fields are cumulative process high water.

Build measurements and numerical formula checks were **not rerun**. The summary links
v1 evidence under `historical_evidence`, with the explicit origin
`copied_historical_v1_not_rerun`. No copied historic timing is presented as current.

Final source revisions, including the tests, preceded manifest freeze. The post-suite
receipt is separate, avoiding cyclic summary/JUnit hashes or mutable test-source
revisions after freezing. Verification checked **79 current source bindings**, **213
runtime bindings**, unchanged index bytes, and **43 unchanged historical files**.

## Outputs

* `summary.json`: current frozen integration receipt.
* `plan.json`: frozen inputs, sources and preserved historical-file hashes.
* `native48/manifest.json`: native backend's own provenance, including both index files.
* `native48/retrieval_checks.json`: original native traces retained locally.
* `native48/runs.json`: dry plans, not live model outputs.
* `protocol_diagnostics.json`: bounded text-free candidate scores and selections.
* `all_tests.xml`: complete configured repository suite evidence.
* `verification.json`: post-suite integrity receipt.
* `run_integration.py`, `verify_integration.py`: exact reusable procedure.

The immutable diagnostic bundle omits native article-text traces; their hashes and
local paths remain in the manifests. Index files remain at
`data/processed/authoritative_fulltext_v1_bm25/{index.json,postings.npz}`.
The current README and `docs/VERIFIED_RETRIEVAL.md` are documentation added after
execution; neither changes executable dependencies. No manuscript was edited.

Provenance ID: `d833bada0b973b033da0b1402d10b2a5592c532e8dd7eb567f0db4e6ac7b3749`.

```bash
/Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python -m pytest -q --junitxml=results/readiness/verified_bm25/real_corpus_v2/all_tests.xml
```

The lead may now remove the obsolete stale-source test-failure footnote: the full
suite passed against a fresh, strictly source-bound current integration receipt.
