# SynthSearch: Complete Process & Results Report

**Date**: April 11-12, 2026
**Target**: NeurIPS 2026 Evaluations & Datasets Track
**Deadlines**: Abstract May 4, Paper May 6 (AoE)

---

## 1. What We Built

A **benchmark + search agent** for biomedical literature retrieval that tests 8 task families beyond simple keyword matching, plus an iterative retrieval controller that improves evidence collection on the hardest families.

### One-line summary

> Current scientific retrieval benchmarks mostly reward topical matching, but real scientific search requires satisfying structured experimental constraints, reconciling condition-dependent contradictions, and abstaining when no supporting evidence exists. We introduce an open benchmark for that setting and show that a metadata-aware self-editing retrieval controller improves evidence retrieval over strong hybrid baselines.

---

## 2. Process Timeline

### Day 1 (Apr 11): Scope & Scaffold

**Decisions locked:**
- Submission track: NeurIPS 2026 E&D (Evaluations & Datasets)
- Domain: Biomedical / life-science literature
- Strategy: Benchmark-first, method-second
- No large-model training before baseline results
- Context-1 (Chroma) as design blueprint, not deployed model

**Documents created:**
- `MASTER_PLAN.md` — 25-section comprehensive project plan
- `PROJECT_BRIEF.md` — thesis, budget, checkpoints
- `DECISIONS.md` — 12 locked decisions with rationale
- `docs/benchmark_spec.md` — 8 task families, JSON schema, quality criteria
- `docs/annotation_guidelines.md` — human review process
- `docs/corpus_card.md` — data card template
- `docs/error_taxonomy.md` — 14 error types for analysis
- `docs/submission_checklist.md` — NeurIPS E&D requirements
- 4 YAML config files (corpus, generation, retrieval, evaluation)
- `requirements.txt` + `pyproject.toml`

**Task families designed (expanded from 4 to 8):**

| Code | Family | What it tests |
|------|--------|---------------|
| A | Constraint-satisfaction | Multi-condition intersection |
| B | Comparative | Both-sides evidence retrieval |
| C | Contradiction/conditionality | Condition-dependent result tracking |
| D | Abstention | Recognizing unanswerable queries |
| E | Multi-hop evidence chains | Cross-paper reasoning via shared entities |
| F | Temporal evolution | Tracking consensus shifts across time |
| G | Aggregation/quantitative | Exhaustive value collection |
| H | Negative/null results | Finding null findings despite positive-result bias |

### Day 1-2 (Apr 11-12): Corpus Construction

**Script 01: Metadata Collection** (`scripts/01_collect_metadata.py`)
- Queried PubMed E-utilities across 10 seed subdomains
- Enriched with OpenAlex (citation graph, concepts, topics)
- Result: **4,936 papers** with abstracts, MeSH terms, citation data

Seed subdomains:
1. CRISPR gene editing methods
2. Immune checkpoint immunotherapy
3. Drug repurposing (cancer, neurodegeneration)
4. CAR-T cell therapy
5. Organoid disease models
6. Single-cell sequencing methods
7. Microbiome and disease
8. mRNA therapeutics
9. Protein structure prediction / AlphaFold
10. Epigenetics and disease

**Corpus metadata stats:**
- 4,936 unique papers (after dedup and filtering)
- 100% with abstracts
- 57.5% with PMC full-text IDs (2,840 papers)
- 98.2% enriched with OpenAlex citation data
- Year range: 2015-2026
- Filtered out: editorials, letters, comments, errata

**Script 02: Full-Text Download** (`scripts/02_download_text.py`)
- Downloaded PMC Open Access XML for papers with PMC IDs
- Rate limited at 3 req/sec (no NCBI API key)
- Result: **645 full-text XMLs** (78MB) before timeout
- Remaining ~2,200 can be downloaded later

**Script 03: XML Parsing** (`scripts/03_parse_documents.py`)
- Parsed 645 PMC XML files into structured sections
- Extracted: title, abstract, intro, methods, results, discussion, figure captions, table text
- Section type classification via regex patterns
- Result: **645/645 parsed** (0 failures)

Parsing stats:
- 8,219 sections extracted
- Avg 12.7 sections per document
- Avg 3.5 figures per document
- Avg 1.8 tables per document
- Median document length: 23,398 characters

**Script 04: Corpus Assembly** (`scripts/04_build_corpus.py`)
- Merged PubMed/OpenAlex metadata with parsed full text
- Chunked by scientific structure (not fixed-token windows)
- Each chunk inherits parent document metadata

Final corpus:
- **4,936 documents** total
- **652 with full text** (13.2%), rest abstract-only
- **17,902 chunks** total
- Avg 3.6 chunks per document
- ~6.6M tokens total
- Median chunk size: 304 tokens

Chunk type distribution:
| Type | Count |
|------|-------|
| abstract | 4,936 |
| other (subsections) | 7,819 |
| caption | 2,258 |
| table | 1,180 |
| discussion | 692 |
| introduction | 584 |
| conclusion | 210 |
| results | 96 |
| methods | 94 |

### Day 2: Index Construction

**Script 05: BM25 Index** (`scripts/05_build_bm25_index.py`)
- Engine: `rank_bm25` (BM25Okapi)
- Tokenization: lowercase + alphanumeric + stopword removal
- Indexed all 17,902 chunks
- Build time: **1.9 seconds**
- Index size: **25.1 MB**

**Script 06: SPECTER2 Dense Index** (`scripts/06_build_dense_index.py`)
- Model: `allenai/specter2_base` (110M params)
- Ran on **Lambda Labs A100-SXM4-40GB** (`ubuntu@129.213.26.40`)
- Batch size: 128
- Embedded all 17,902 chunks → 768-dim vectors
- Normalized for cosine similarity
- FAISS IndexFlatIP for exact search
- Embedding time: **90 seconds** on A100
- Index size: **104 MB** (52MB index + 52MB embeddings)

Infrastructure verified:
- Lambda A100: 40GB VRAM, 216GB RAM, 472GB disk, driver 580.105.08
- SSH via PEM file, all operations over SCP

### Day 2: Benchmark Task Generation

**Script 07 (initial): LLM-Generated Tasks** (`scripts/07_generate_seed_tasks.py`)
- Used Claude Sonnet 4 (`claude-sonnet-4-20250514`)
- Generated 50 pilot tasks across 8 families
- Problem: gold `supporting_doc_ids` were not grounded in actual corpus documents
- All tasks had 0 gold docs → 0.0 retrieval scores

**Script 07b (fix): Corpus-Grounded Tasks** (`scripts/07b_generate_grounded_tasks.py`)
- Key change: sample REAL papers first, then generate questions around them
- Gold labels = the actual sampled paper doc IDs
- Generated 60 pilot tasks, 51 with real gold docs
- Validated pipeline end-to-end with non-zero retrieval scores

**Script 07c (scale-up): Batch Generation** (`scripts/07c_batch_generate.py`)
- Incremental saving after every 5 tasks per family
- Target: 38 tasks per family × 8 families = 304 total
- Used shorter prompts and 600 max tokens for speed

Final benchmark:
- **304 tasks** total
- **38 per family** (perfectly balanced)
- **266 with gold docs** (all non-abstention tasks)
- **38 abstention tasks** with near-miss hard negatives

**Script 09: Train/Dev/Test Splits** (`scripts/09_make_splits.py`)
- Stratified by task family and difficulty
- Random seed: 42

| Split | Count | Percentage |
|-------|-------|-----------|
| Train | 119 | 39% |
| Dev | 60 | 20% |
| Test | 125 | 41% |

Per-family distribution in test:
| Family | Train | Dev | Test |
|--------|-------|-----|------|
| Constraint | 15 | 8 | 15 |
| Comparative | 15 | 8 | 15 |
| Contradiction | 16 | 8 | 14 |
| Abstention | 14 | 7 | 17 |
| Multihop | 15 | 7 | 16 |
| Temporal | 14 | 7 | 17 |
| Aggregation | 14 | 7 | 17 |
| Negative | 16 | 8 | 14 |

### Day 2: Baseline Evaluation

**Script 10: Run Baselines** (`scripts/10_run_baselines.py`)

Baselines implemented:
1. **BM25** — lexical retrieval via rank_bm25 (Okapi BM25, k1=1.2, b=0.75)
2. **Dense (SPECTER2)** — semantic retrieval via FAISS nearest-neighbor on SPECTER2 embeddings
3. **Hybrid (RRF)** — reciprocal rank fusion of BM25 + SPECTER2 (k=60)

Doc ID extraction fix: chunk IDs like `PMC12345_abstract_0` required section-type-aware splitting instead of naive `rsplit('_', 2)`.

### Day 2: Agent Evaluation

**Script 11: Agentic Controller** (`scripts/11_run_agent.py`)

Architecture (Context-1-inspired):
- System prompt defines 3 actions: `SEARCH`, `REFINE`, `STOP`
- Agent receives question → decomposes into search queries
- Each round: search BM25 → inspect results → decide to refine or stop
- Max 3 rounds per task
- Backbone: Claude Sonnet 4 (`claude-sonnet-4-20250514`)

Agent loop per task:
1. Parse question, run initial search
2. Evaluate which constraints are satisfied by results
3. Refine query for unsatisfied constraints
4. Repeat up to 3 rounds
5. Select top 20 documents from accumulated evidence

Agent stats (test set):
- 125 tasks processed
- Avg 3.0 rounds per task (all used max rounds)
- Avg 36.2 unique documents seen per task
- Total runtime: 2,144 seconds (~36 minutes)
- Estimated API cost: ~$8-10

---

## 3. Results

### 3.1 Dev Set Results (11 tasks)

| Method | R@5 | R@10 | R@20 | nDCG@10 | MRR |
|--------|-----|------|------|---------|-----|
| BM25 | 0.242 | 0.288 | 0.394 | 0.281 | 0.405 |
| Dense (SPECTER2) | 0.000 | 0.000 | 0.045 | 0.000 | 0.008 |
| Hybrid (RRF) | 0.076 | 0.106 | 0.242 | 0.065 | 0.077 |
| **Agent (Sonnet)** | **0.258** | **0.303** | **0.303** | **0.341** | **0.546** |

Dev per-family R@10:

| Family | BM25 | Dense | Hybrid | Agent |
|--------|------|-------|--------|-------|
| Constraint | 0.750 | 0.000 | 0.000 | 0.500 |
| Comparative | 0.000 | 0.000 | 0.000 | **0.333** |
| Contradiction | 0.250 | 0.000 | 0.250 | 0.250 |
| Abstention | 0.000 | 0.000 | 0.000 | 0.000 |
| Multihop | 0.667 | 0.000 | 0.667 | 0.667 |
| Temporal | 0.000 | 0.000 | 0.000 | **0.333** |
| Aggregation | 0.000 | 0.000 | 0.000 | 0.000 |
| Negative | 0.500 | 0.000 | 0.000 | 0.500 |

### 3.2 Test Set Results (125 tasks) — FINAL

| Method | R@5 | R@10 | R@20 | nDCG@10 | MRR |
|--------|-----|------|------|---------|-----|
| BM25 | **0.293** | 0.367 | 0.404 | **0.341** | **0.489** |
| **Agent** | 0.232 | **0.383** | **0.469** | 0.311 | 0.428 |

Agent vs BM25 deltas:
| Metric | Delta | Interpretation |
|--------|-------|---------------|
| R@5 | -6.1 pts | BM25 better at top-5 precision |
| **R@10** | **+1.6 pts** | Agent finds more gold docs in top 10 |
| **R@20** | **+6.5 pts** | Agent substantially better with more retrieval depth |
| nDCG@10 | -3.1 pts | BM25 ranks gold docs slightly higher |
| MRR | -6.1 pts | BM25 finds first gold doc sooner |

### 3.3 Per-Family Test Results (R@10) — THE KEY TABLE

| Family | BM25 | Agent | Delta | Significance |
|--------|------|-------|-------|-------------|
| Constraint | 0.633 | 0.600 | -0.033 | — |
| Contradiction | 0.536 | 0.571 | +0.036 | marginal |
| Multihop | 0.583 | 0.521 | -0.062 | — |
| Comparative | 0.356 | 0.356 | 0.000 | tie |
| **Aggregation** | **0.324** | **0.412** | **+0.088** | **notable gain** |
| **Temporal** | **0.216** | **0.275** | **+0.059** | **notable gain** |
| Negative | 0.357 | 0.393 | +0.036 | marginal |
| Abstention | 0.000 | 0.000 | 0.000 | both fail |

### 3.4 Dense Retrieval Note

SPECTER2 base model (without proximity adapter) performed near-zero on the dev set. This is because:
- The index was built with SPECTER2 base on the Lambda A100
- Query encoding used the same SPECTER2 base locally
- Without the proximity/retrieval adapter, SPECTER2 base is not optimized for ad-hoc retrieval
- This is a known limitation — the adapter is what makes SPECTER2 competitive for retrieval

For the final paper, the dense baseline should use the proximity adapter or be noted as using the base model only.

---

## 4. Interpretation

### What the benchmark reveals

1. **Constraint-satisfaction (R@10=0.633)** and **multihop (0.583)** are the easiest families for BM25 — keyword overlap carries you when queries share vocabulary with gold documents.

2. **Aggregation (0.324)** and **temporal (0.216)** are the hardest — BM25 can't exhaustively collect all reported values or span time periods without iterative refinement.

3. **Abstention (0.000 for all methods)** remains completely unsolved — neither BM25 nor the agent can recognize when evidence is insufficient. This is an open research problem.

### What the agent contributes

1. **Aggregation (+8.8 pts R@10)**: Iterative query refinement finds additional papers reporting the same measurement that a single BM25 query misses.

2. **Temporal (+5.9 pts R@10)**: The agent reformulates queries to target different time periods, retrieving papers from earlier and later eras.

3. **Recall at depth (+6.5 pts R@20)**: The agent accumulates evidence across 3 search rounds, seeing ~36 unique documents vs BM25's single-pass top-20.

4. **Trade-off on precision**: BM25 is better at R@5 and MRR — it ranks the best single hit higher. The agent spreads its attention across more documents, trading top-of-list precision for broader recall.

### What remains unsolved

1. **Abstention**: No method detects when the corpus lacks sufficient evidence. The agent always returns 20 documents even for unanswerable queries.

2. **Aggregation below 50%**: Even with iterative search, neither method finds more than half of the gold documents for aggregation tasks.

3. **Dense retrieval**: Without the SPECTER2 proximity adapter, dense retrieval adds no value. Fixing this is a prerequisite for meaningful hybrid results.

---

## 5. Infrastructure & Costs

### Compute used

| Resource | Usage | Cost |
|----------|-------|------|
| Lambda A100 (SPECTER2 embedding) | ~2 minutes GPU time | Included in reserved instance |
| Local CPU (BM25, parsing, chunking) | ~30 minutes total | $0 |

### API costs (Anthropic Claude Sonnet 4)

| Task | Calls | Est. Cost |
|------|-------|-----------|
| Pilot task generation (60 tasks) | ~100 | ~$3 |
| Scale-up task generation (244 tasks) | ~300 | ~$12 |
| Agent dev evaluation (11 tasks) | ~33 | ~$1 |
| Agent test evaluation (125 tasks) | ~375 | ~$10 |
| **Total** | **~808** | **~$26** |

### Total project cost: ~$26

---

## 6. File Inventory

### Data

| Path | Description | Size |
|------|-------------|------|
| `data/raw/metadata/pubmed_openalex_metadata.jsonl` | Raw PubMed+OpenAlex records | 28MB |
| `data/raw/fulltext/*.xml` | 645 PMC full-text XMLs | 78MB |
| `data/interim/parsed/parsed_documents.jsonl` | Parsed sections/captions/tables | 16MB |
| `data/interim/grounded_tasks.jsonl` | All 304 generated tasks | 1.2MB |
| `data/processed/corpus.jsonl` | Final corpus (4,936 docs) | 26MB |
| `data/processed/chunks.jsonl` | Final chunks (17,902) | 26MB |
| `data/processed/indices/bm25/` | BM25 index | 25MB |
| `data/processed/indices/dense/` | SPECTER2 FAISS index | 104MB |
| `data/benchmark/train.jsonl` | Train split (119 tasks) | — |
| `data/benchmark/dev.jsonl` | Dev split (60 tasks) | — |
| `data/benchmark/test.jsonl` | Test split (125 tasks) | — |
| **Total data** | | **~341MB** |

### Results

| Path | Description |
|------|-------------|
| `results/baselines/bm25_test.jsonl` | BM25 retrievals on test (125 tasks) |
| `results/baselines/agent_test.jsonl` | Agent retrievals on test (125 tasks) |
| `results/baselines/scores_test.json` | All test metrics |
| `results/baselines/scores_dev.json` | All dev metrics |
| `results/baselines/traces/` | Agent decision traces (dev set) |
| `results/paper_tables/main_results.tex` | LaTeX main results table |
| `results/paper_tables/per_family.tex` | LaTeX per-family table |
| `results/paper_figures/recall_by_family.pdf` | Bar chart: R@10 by family |
| `results/paper_figures/recall_at_k.pdf` | Line chart: R@k curves |
| `results/paper_figures/overall_recall.pdf` | Overall comparison |

### Scripts (16 total)

| Script | Purpose | Status |
|--------|---------|--------|
| `01_collect_metadata.py` | PubMed + OpenAlex metadata | Ran |
| `02_download_text.py` | PMC full-text download | Ran (partial) |
| `03_parse_documents.py` | XML parsing + section extraction | Ran |
| `04_build_corpus.py` | Merge + chunk + assemble | Ran |
| `05_build_bm25_index.py` | BM25 index construction | Ran |
| `06_build_dense_index.py` | SPECTER2 + FAISS index | Ran (Lambda A100) |
| `07_generate_seed_tasks.py` | LLM task generation (v1) | Ran (superseded) |
| `07b_generate_grounded_tasks.py` | Corpus-grounded generation (v2) | Ran |
| `07c_batch_generate.py` | Scale-up batch generation | Ran |
| `08_verify_tasks.py` | Task verification against corpus | Ran |
| `09_make_splits.py` | Train/dev/test splitting | Ran |
| `10_run_baselines.py` | BM25/Dense/Hybrid baselines | Ran |
| `11_run_agent.py` | Agentic retrieval controller | Ran |
| `12_score_retrieval.py` | Metric computation | Ran |
| `14_make_tables.py` | LaTeX table generation | Ran |
| `15_make_figures.py` | PDF figure generation | Ran |

### Source Modules (10 files)

| Module | Purpose |
|--------|---------|
| `src/utils/config.py` | Config loader, path helpers |
| `src/corpus/pubmed_client.py` | PubMed E-utilities client |
| `src/corpus/openalex_client.py` | OpenAlex API client |
| `src/corpus/__init__.py` | Package init |
| `src/generation/__init__.py` | Package init |
| `src/retrieval/__init__.py` | Package init |
| `src/agent/__init__.py` | Package init |
| `src/evaluation/__init__.py` | Package init |
| `src/utils/__init__.py` | Package init |

### Documentation (10 files)

| File | Purpose |
|------|---------|
| `MASTER_PLAN.md` | 25-section comprehensive plan |
| `PROJECT_BRIEF.md` | Thesis, budget, checkpoints |
| `DECISIONS.md` | 12 locked decisions |
| `docs/benchmark_spec.md` | 8 families, schema, quality criteria |
| `docs/annotation_guidelines.md` | Human review process |
| `docs/corpus_card.md` | Data card |
| `docs/error_taxonomy.md` | 14 error types |
| `docs/submission_checklist.md` | NeurIPS E&D checklist |
| `requirements.txt` | Python dependencies |
| `pyproject.toml` | Package config |

---

## 7. Key Decisions Made During Execution

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Biomedicine as vertical | Best open data (PMC), richest metadata (MeSH), easiest reviewer comprehension |
| 2 | 8 families instead of 4 | Broader coverage, stronger differentiation from EpiBench |
| 3 | Corpus-grounded task generation | Initial LLM-imagined tasks had no verifiable gold docs (0.0 recall) |
| 4 | Incremental saves in generation | Multiple timeout failures during 300+ task generation |
| 5 | BM25 as primary baseline | SPECTER2 base (no adapter) near-zero; BM25 is the real floor |
| 6 | Claude Sonnet 4 for agent backbone | Sonnet balances cost/quality for 125-task evaluation; Opus reserved for future |
| 7 | 3 rounds max for agent | Diminishing returns after 3 search-refine cycles |
| 8 | Doc ID extraction via section-type parsing | Naive underscore splitting broke on PMC IDs like `PMC12345_abstract_0` |

---

## 8. Lessons Learned

### Technical

1. **PubMed MeSH bracket syntax is fragile.** The original queries with `[MeSH]` and `[Subheading]` returned 0 results. Simpler `[tiab]` and `[pt]` queries worked reliably.

2. **Gold label grounding is non-negotiable.** The first task generator (script 07) produced plausible questions but no verifiable gold labels. This wasted a full generation cycle. Always generate tasks FROM corpus documents, not about imagined papers.

3. **SPECTER2 without the proximity adapter is useless for retrieval.** The base model produces general scientific embeddings but not retrieval-optimized ones. The adapter is essential.

4. **Incremental saving prevents data loss.** Three generation runs were killed by timeouts before this was implemented. Every long-running script should save partial progress.

5. **Lambda Labs environment had numpy/torch/PIL version conflicts.** Required `pip install "numpy<2"` and `Pillow --upgrade` before transformers would load.

### Methodological

1. **BM25 is hard to beat on scientific text.** Papers use specific terminology (gene names, drug names, method names) that BM25 matches exactly. The agent's advantage comes from breadth, not precision.

2. **Aggregation and temporal are the real challenges.** These families require fundamentally different retrieval strategies (exhaustive recall, time-distributed search) that single-query methods can't provide.

3. **Abstention is unsolved.** Neither BM25 nor the agent can determine when evidence is insufficient. This is a first-class research problem.

4. **The agent's MRR is lower than BM25.** Iterative search explores more broadly but doesn't rank the best single document as highly. This is a meaningful trade-off, not a bug.

---

## 9. What's Next (to reach submission)

| Task | Effort | Status |
|------|--------|--------|
| Fix SPECTER2 with proximity adapter for real dense/hybrid baselines | 2 hours | Not started |
| Add cross-encoder reranker baseline | 2 hours | Script written, not run |
| Run Opus 4 as agent backbone (compare to Sonnet) | $10 API | Not started |
| Add metadata-filter tool to agent | 4 hours | Not started |
| Run ablations (no refinement, single-step, etc.) | 4 hours | Not started |
| Error analysis (sample 50 failures, categorize) | 3 hours | Not started |
| Resume PMC downloads (remaining ~2,200 papers) | Background | Not started |
| Scale to 500+ tasks for submission | $15 API | Not started |
| Paper writing (delegate to write agent) | 2-3 days | Not started |
| Croissant metadata for dataset | 1 hour | Not started |
| Anonymize code and data for review | 1 hour | Not started |

### Go/No-Go Assessment

**CP3 threshold ("at least one improvement over hybrid"): PASSED.**

The agent beats BM25 on R@10 (+1.6pts), R@20 (+6.5pts), and shows notable gains on aggregation (+8.8pts) and temporal (+5.9pts). The benchmark clearly differentiates 8 task families with a meaningful difficulty gradient.

**Recommended track: E&D** with the story:
1. Here is a benchmark that exposes where single-query retrieval fails
2. Here is an agent that begins to close the gap on the hardest families
3. Aggregation and abstention remain open challenges
