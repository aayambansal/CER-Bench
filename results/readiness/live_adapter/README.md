# Offline adapter handoff — no empirical execution

Observed final verification: **60 tests passed in 0.74 s**. Test report:
`offline_tests.xml`. Tests disable socket connection creation and inject fake
provider transports. No real API keys were inspected, no API/model calls were
made, no Modal SDK invoked, and no human validation conducted.

`SYNTHETIC_integration/` contains invented inputs and all seven condition outputs.
All complete with 24 candidates and both 20-document selectors. Logical calls
are 0,0,0,1,1,2,2; synthetic provider attempts total 6. This verifies plumbing,
not retrieval quality or live API compatibility. The budget audit reports no
issues, no unbound rows, and no provenance conflict; publication_ready remains
false. Strict evaluator accepted the explicit common-selector export with
status completed and no warnings. No scientific judgments were invented.

`SYNTHETIC_dry_plan/` contains seven zero-call plans using the unapproved template.
Exact tested command from the project analysis directory:

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python scripts/48_run_live_budget.py --dry-run --config configs/live_pilot.json --tasks results/readiness/live_adapter/SYNTHETIC_integration/tasks.json --corpus results/readiness/live_adapter/SYNTHETIC_integration/corpus.json --chunks results/readiness/live_adapter/SYNTHETIC_integration/chunks.json --index results/readiness/live_adapter/SYNTHETIC_integration/index.json --split dev --stage dev-smoke --repeat 1 --output results/readiness/live_adapter/SYNTHETIC_dry_plan
```

The output now exists; choose a fresh output directory to repeat the dry CLI.
Full regression command and production input contract: `docs/LIVE_EXPERIMENTS.md`.

Implemented files:

* `src/agents/provider_clients.py`
* `scripts/48_run_live_budget.py`
* `tests/test_live_provider.py`
* `configs/live_pilot.json`
* `docs/LIVE_EXPERIMENTS.md`

No existing budget modules, script 47, or script 49 were edited. The working
directory has no Git metadata; `git status` returned `fatal: not a git repository
(or any of the parent directories): .git`. No commit was attempted.

## Remaining authority/input gates

1. Secure runtime credential availability (no dotfiles, tool-argument keys or
   file-based secrets).
2. Approved exact model ID, native-endpoint compatibility, current per-million
   input/output prices, and conservative token-bound evidence as nonsecret
   hashed files. Template model/prices remain null and approval false.
3. Verified corpus, chunks, split-tagged tasks, and explicitly approved portable
   BM25 JSON index/export with matching hashes. Legacy pickle equivalence is not
   assumed. No real-data portable export was produced in this phase.
4. One approved aggregate pilot budget no greater than $50. The implemented cap
   and lock are per output/run, not global across separate directories.
5. Review mapping if the lead's newer strict budget schema differs from existing
   `equal-budget-v2`/script 41. Existing budget and strict evaluation exports were
   checked; acceptance by an unavailable new schema is not claimed.

Timeout is per socket operation rather than a hard wall deadline. Inputs/source
must remain immutable while running; manifests are hashes, not snapshots.
Cross-repeat freeze and aggregate funding require lead coordination. Crashed
pending rows/unknown billed outcomes block automatic resume and need explicit
manual reconciliation. No live execution should be launched from these fixtures.
