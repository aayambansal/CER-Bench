# Decision Log

All scope and design decisions are recorded here. Each entry is immutable once made.
New information leads to new entries, not edits to old ones.

---

## D001: Submission track
- **Date**: 2026-04-11
- **Decision**: Target NeurIPS 2026 E&D track by default.
- **Rationale**: Benchmark-first strategy is safest for the timeline. E&D does not
  require a new model, only meaningful evaluation advancement. Same deadlines as main
  track (abstract May 4, paper May 6).
- **Pivot condition**: Switch to main track only if agent shows >5pt Recall@10 gain
  over best non-agentic baseline by CP3 (Apr 26).

## D002: Vertical
- **Date**: 2026-04-11
- **Decision**: Biomedical / life-science literature.
- **Rationale**: Best open-access full text (PMC OA), richest metadata (MeSH, organisms,
  assays), largest scholarly APIs, easiest reviewer comprehension.
- **What this excludes**: Chemistry, materials science, synthetic biology, protocols,
  patents. These are future work.

## D003: No large-model training before baseline results
- **Date**: 2026-04-11
- **Decision**: The agentic controller will be prompt-based (using a frontier LLM as
  backbone) until at least CP2. Fine-tuning a small controller is only permitted after
  CP2 if zero-shot performance is clearly close but insufficient.
- **Rationale**: Training risk is too high for the timeline. The Context-1 blueprint
  shows transfer outside training distribution, so zero-shot is a credible first attempt.

## D004: Corpus size
- **Date**: 2026-04-11
- **Decision**: Pilot corpus: 10-50K papers. Full corpus: 50-250K papers.
- **Rationale**: Larger than needed for retrieval benchmarking hurts iteration speed.
  Quality of benchmark tasks matters more than corpus scale.

## D005: Benchmark size
- **Date**: 2026-04-11
- **Decision**: Pilot: 50-100 tasks. Submission target: 300-800 tasks.
- **Rationale**: Enough for statistically meaningful comparisons if tasks are genuinely
  difficult and well-audited. Quality over quantity.

## D006: Task families
- **Date**: 2026-04-11
- **Decision**: Four mandatory families: constraint-satisfaction, comparative,
  contradiction/conditionality, abstention.
- **Rationale**: These represent the hardest real scientific retrieval patterns and are
  exactly where existing benchmarks are weakest.

## D007: Chunking strategy
- **Date**: 2026-04-11
- **Decision**: Chunk by scientific structure (abstract, intro, methods subsections,
  results subsections, discussion, figure captions, table text). No fixed-token sliding
  windows.
- **Rationale**: Scientific retrieval quality depends on structural context. A methods
  paragraph and a results paragraph about the same topic serve different retrieval needs.

## D008: API provider for bulk LLM calls
- **Date**: 2026-04-11
- **Decision**: Anthropic Claude Opus 4 (`claude-opus-4-6`) via user's own API key.
- **Rationale**: User provided Anthropic API key. Opus 4 is the strongest available
  model for structured scientific extraction and task verification.
- **Constraint**: Must use user's own API key, not shared keys.

## D009: Retrieval backbone
- **Date**: 2026-04-11
- **Decision**: SPECTER2 for dense scientific retrieval. BM25 via Pyserini or rank_bm25.
  Cross-encoder reranker (ms-marco-MiniLM or similar).
- **Rationale**: SPECTER2 is the standard scientific embedding model with retrieval
  adapters trained across 23 fields. No reason to deviate from the established baseline.

## D010: Context-1 usage
- **Date**: 2026-04-11
- **Decision**: Use Context-1 as a design blueprint, not as the deployed model.
  Build a prompt-based controller that copies its loop (explore -> generate -> verify ->
  distract -> chain) with added metadata filters and citation graph hops.
- **Rationale**: Deploying the 20B model requires A100-80GB and significant inference
  budget. A prompt-based controller is faster to iterate on and easier to ablate.
  Zero-shot Context-1 model evaluation is optional and secondary.

## D011: Compute infrastructure
- **Date**: 2026-04-11
- **Decision**: Lambda Labs 1x A100-SXM4-**40GB** (not 80GB) at `ubuntu@129.213.26.40`.
  SSH via `Lambda Cloud MacBook Pro.pem`.
- **Rationale**: Already provisioned. 40GB is sufficient for SPECTER2 embedding,
  cross-encoder reranking, and OpenScholar 8B inference. Only limits deployment of
  the actual Context-1 20B model in FP16, which we are not planning to do.
- **Specs**: 216GB RAM, 472GB free disk, NVIDIA driver 580.105.08.

## D012: Task families expanded to 8
- **Date**: 2026-04-11
- **Decision**: Expand from 4 to 8 task families:
  A) Constraint-satisfaction, B) Comparative, C) Contradiction/conditionality,
  D) Abstention, E) Multi-hop evidence chains, F) Temporal evolution,
  G) Aggregation/quantitative collection, H) Negative/null result retrieval.
- **Rationale**: Each new family tests a genuinely different retrieval capability:
  - Multi-hop tests cross-paper entity chaining (no single paper answers the query)
  - Temporal tests time-distributed retrieval (must span eras, not just recency)
  - Aggregation tests exhaustive recall (find ALL reported values, not just one)
  - Negative tests overcoming positive-result bias (find null/negative findings)
  These widen the benchmark's coverage and differentiation from EpiBench and existing
  benchmarks that focus on single-answer retrieval.
- **Task distribution**: constraint 15%, comparative 13%, contradiction 13%,
  abstention 15%, multihop 12%, temporal 10%, aggregation 12%, negative 10%.
- **Target**: Still 300-800 tasks total. ~40-100 per family at submission.
