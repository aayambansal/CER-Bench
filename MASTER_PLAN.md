# MASTER PLAN: SynthSearch NeurIPS 2026
## Metadata-Aware Self-Editing Search Agent and Benchmark for Biomedical Literature Retrieval

**Last updated**: 2026-04-11
**Status**: Phase 1 (Scope Lock + Scaffolding) — COMPLETE
**Deadline**: May 6, 2026 (AoE)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Strategic Rationale](#2-strategic-rationale)
3. [Submission Target & Venue](#3-submission-target--venue)
4. [Research Thesis](#4-research-thesis)
5. [Prior Work & Positioning](#5-prior-work--positioning)
6. [Scope Lock](#6-scope-lock)
7. [Benchmark Design](#7-benchmark-design)
8. [Corpus Design](#8-corpus-design)
9. [Retrieval Stack & Baselines](#9-retrieval-stack--baselines)
10. [Agent Design (Context-1-Inspired)](#10-agent-design-context-1-inspired)
11. [Synthetic Task Generation Pipeline](#11-synthetic-task-generation-pipeline)
12. [Metrics & Evaluation Protocol](#12-metrics--evaluation-protocol)
13. [Ablation Plan](#13-ablation-plan)
14. [Infrastructure & Compute](#14-infrastructure--compute)
15. [Budget](#15-budget)
16. [Team & Roles](#16-team--roles)
17. [Day-by-Day Execution Schedule](#17-day-by-day-execution-schedule)
18. [Go / No-Go Checkpoints](#18-go--no-go-checkpoints)
19. [Risk Register & Fallbacks](#19-risk-register--fallbacks)
20. [Paper Outline](#20-paper-outline)
21. [Repo Structure & File Map](#21-repo-structure--file-map)
22. [Implementation Task Queue](#22-implementation-task-queue)
23. [Locked Decisions](#23-locked-decisions)
24. [Credential & Security Setup](#24-credential--security-setup)
25. [What Not To Do](#25-what-not-to-do)

---

## 1. Executive Summary

We are building a **benchmark + search agent** paper for NeurIPS 2026.

**What**: A benchmark of realistic biomedical retrieval tasks (constraint-satisfaction,
comparison, contradiction, abstention) plus a metadata-aware, self-editing retrieval
controller inspired by Chroma's Context-1.

**Why**: Existing scientific retrieval benchmarks test single-answer needle-in-a-haystack
questions. Real scientific search requires multi-constraint reasoning, condition tracking,
evidence aggregation, and knowing when to abstain. This gap is under-evaluated.

**How**: Benchmark-first, method-second. Build the evaluation artifact, then show the
agent can improve retrieval over strong non-agentic baselines. If the agent wins, we
have both contributions. If it doesn't, the benchmark alone carries an E&D paper.

**Key numbers**:
- 300-800 benchmark tasks across 4 families
- 50-250K paper corpus from PMC/PubMed/OpenAlex
- 6 baselines (BM25 through agentic controller)
- ~$200-900 total compute/API budget
- 25 days to submission

---

## 2. Strategic Rationale

### Why benchmark-first

The NeurIPS E&D track explicitly welcomes benchmarks, evaluation protocols, and datasets
without requiring a new model. With 25 days, a polished benchmark with strong baselines
is more defensible than a rushed method contribution. If the method happens to work well,
we upgrade the framing.

### Why Context-1 as the blueprint

Chroma's Context-1 contributes a specific and reusable architecture: a retrieval subagent
that decomposes queries, searches iteratively, prunes its own context, and separates
retrieval from answer generation. They trained it on 8,000+ synthetic tasks, released
weights and data-generation code, and showed that a cheap 4x rollout + reciprocal-rank
fusion configuration matches much larger frontier models. Their own "future directions"
names the extensions we want: structured/code search, metadata-aware filtering,
scratchpads, late-interaction retrieval, and tighter orchestrator integration.

We keep the loop. We change the tools and domain.

### Why biomedicine

- Strongest open-access full text: PMC Open Access has ~4M articles
- Richest metadata: MeSH terms, organisms, assays, publication types
- Largest scholarly APIs: PubMed, OpenAlex, Semantic Scholar
- Easiest reviewer comprehension at NeurIPS
- Well-established baselines (SPECTER2 trained across 23 fields)

### Why NOT "all science"

The benchmark space is moving fast. EpiBench (posted this week) already pushes on
multi-turn multimodal research workflows. A generic "research workflow benchmark" risks
overlap. One vertical, done well, differentiates on depth.

---

## 3. Submission Target & Venue

### Primary: NeurIPS 2026 Evaluations & Datasets (E&D) Track

- Abstract deadline: **May 4, 2026 (AoE)**
- Full paper deadline: **May 6, 2026 (AoE)**
- Format: 9-page main body + unlimited references/appendix
- Review: **Double-blind** (new for E&D in 2026)
- Requirements:
  - Code and data must be accessible at submission time
  - New datasets require **Croissant metadata** (ML schema.org)
  - Does NOT require a new model — evaluation contributions are sufficient

### Fallback: NeurIPS 2026 Main Track

Same deadlines. Pivot only if agent shows >5pt Recall@10 gain over best non-agentic
baseline by Checkpoint 3 (Apr 26).

### What E&D explicitly welcomes

Benchmarks, evaluation protocols, datasets, data generators, audits, stress tests,
RL environments. Our benchmark + evaluation protocol + baseline suite fits naturally.

---

## 4. Research Thesis

### Core claim

> Existing scientific retrieval benchmarks under-test real synthetic-science workflows.
> A metadata-aware, self-editing retrieval agent can improve evidence collection for
> biomedical questions, especially when queries require multi-hop reasoning, constraint
> satisfaction, comparison across conditions, and abstention.

### Two-part contribution

1. **SynthSearch-Biomed Benchmark**: A benchmark of 300-800 realistic biomedical
   literature retrieval tasks spanning four families: constraint-satisfaction, comparison,
   contradiction/conditionality, and abstention. Each task includes gold evidence
   documents, gold passages, hard negatives, explicit constraints, and expected answer type.

2. **SynthSearch Agent**: A metadata-aware, self-editing retrieval controller that
   decomposes queries, applies metadata filters, prunes working context, traverses
   citation links, and supports explicit abstention. Implemented as a prompt-based
   controller (no model training required for the initial contribution).

---

## 5. Prior Work & Positioning

### End-to-end scientific literature QA/synthesis

| System | Key feature | Limitation for us |
|--------|------------|-------------------|
| **OpenScholar** (Nature, 2025) | 45M papers, 8B model beats GPT-4o on ScholarQABench | Monolithic answerer, not a retrieval agent. We use it as a downstream comparison. |
| **PaperQA2** | Strong scientific QA pipeline | Closed system, less controllable for ablation. |

### Retrieval backbone

| System | Key feature | Our use |
|--------|------------|---------|
| **SPECTER2** (AllenAI) | Scientific embeddings, retrieval adapter, 23 fields | Our dense retrieval baseline. |
| **BM25** | Lexical floor, strong on exact terminology | Baseline 1. |

### Agentic search

| System | Key feature | Our use |
|--------|------------|---------|
| **Context-1** (Chroma) | 20B search agent, self-editing, 8K+ synthetic tasks | Design blueprint. We copy the loop, change the tools. |

### Benchmarks (competitive landscape)

| Benchmark | What it tests | Gap |
|-----------|--------------|-----|
| **DORIS-MAE** | Complex scientific queries, 17 retrieval methods | Shows the problem exists; doesn't solve it. |
| **LitSearch** | Realistic literature search queries | Dense retrieval +24.8 Recall@5 over BM25. Shows value of dense retrieval but single-turn. |
| **ScholarQABench** | Scientific QA with citations | Answer-focused, not retrieval-focused. |
| **EpiBench** (Apr 2026) | Multi-turn multimodal research workflows | 29.23% hard-split accuracy. Broad but not metadata-aware. Potential overlap risk. |

### Our differentiation

- **Metadata-aware**: We test whether metadata filters (organism, assay, year, etc.)
  improve retrieval beyond pure text matching.
- **Self-editing**: We test whether dynamic context pruning helps.
- **Abstention**: We explicitly evaluate the ability to recognize unanswerable queries.
- **Condition-sensitivity**: Our contradiction/conditionality family tests whether
  systems can track experimental conditions, not just topics.
- **Structured evaluation**: Four task families with separate metrics, not a single score.

---

## 6. Scope Lock

### Domain: Biomedical / life-science literature

Specifically: method- and condition-sensitive literature retrieval. Examples:
- "Find papers that use organism X, assay Y, and intervention Z"
- "Find contradictory findings under different temperature / cell line / substrate conditions"
- "Find all methods satisfying constraints A, B, C"
- "Determine whether the literature actually supports claim Q"

### What is IN scope

- PMC Open Access papers (full text)
- PubMed metadata (abstracts, MeSH, publication types)
- OpenAlex/Semantic Scholar citation graph and topic metadata
- Four task families (constraint, comparison, contradiction, abstention)
- Prompt-based retrieval controller (no model training required)
- Standard retrieval baselines (BM25, SPECTER2, hybrid, reranker)

### What is OUT of scope

- Chemistry, materials science, synthetic biology (future verticals)
- Protocols, patents (future corpus extensions)
- Full end-to-end answer generation (we evaluate retrieval separately)
- Large-model training before CP2
- Tables/figures as first-class retrieval targets (deferred — too hard to parse reliably)
- Non-English papers

---

## 7. Benchmark Design

### Task families (8 total)

| Family | Description | % of tasks | Key challenge |
|--------|------------|------------|---------------|
| **A: Constraint-satisfaction** | Find papers matching multiple experimental conditions | 15% | Intersection over unions |
| **B: Comparative** | Compare methods/organisms/outcomes across papers | 13% | Both-sides retrieval |
| **C: Contradiction/conditionality** | Reconcile conflicting findings under different conditions | 13% | Condition tracking |
| **D: Abstention** | Recognize when corpus doesn't support the query | 15% | Knowing what you don't know |
| **E: Multi-hop evidence chains** | Link evidence across 3+ papers via shared entities | 12% | Bridging entity identification |
| **F: Temporal evolution** | Track how findings/consensus changed over time | 10% | Time-distributed retrieval |
| **G: Aggregation/quantitative** | Collect all reported values for a measurement | 12% | Exhaustive recall over many papers |
| **H: Negative/null results** | Find papers reporting absence of effect | 10% | Overcoming positive-result bias |

### Task schema (JSON)

```json
{
  "task_id": "string",
  "domain": "biomedicine",
  "task_family": "constraint | comparative | contradiction | abstention | multihop | temporal | aggregation | negative",
  "difficulty": "easy | medium | hard",
  "question": "natural-language query",
  "decomposition_hints": ["subquery 1", "subquery 2"],
  "required_constraints": [
    {"type": "organism|assay|intervention|outcome|temporal|design", "value": "string"}
  ],
  "supporting_doc_ids": ["pmcid_1", "pmcid_2"],
  "supporting_passages": [
    {"doc_id": "pmcid_1", "section": "methods", "text": "exact passage", "constraint_satisfied": ["constraint_type"]}
  ],
  "hard_negative_doc_ids": ["pmcid_3", "pmcid_4"],
  "hard_negative_rationale": [
    {"doc_id": "pmcid_3", "reason": "matches topic but wrong organism"}
  ],
  "expected_answer_type": "set | comparison | conditional | abstain | chain | timeline | value_collection | null_result | boolean | freeform",
  "reference_answer": "brief expected answer",
  "metadata_filters": {"year_min": null, "organism": [], "assay": [], "mesh_terms": []},
  "verification_status": "auto_verified | human_verified | needs_review",
  "split": "train | dev | test"
}
```

### Size targets

| Milestone | Tasks | Purpose |
|-----------|-------|---------|
| Pilot | 50-100 | Validate pipeline, calibrate difficulty, human audit |
| Submission | 300-800 | Final benchmark |

### Difficulty distribution

- Easy (20%): BM25 likely finds it. Single constraint. Answer in abstract.
- Medium (40%): 2-3 constraints. Answer in methods/results. Some filtering needed.
- Hard (40%): 4+ constraints. Multi-document. Condition tracking or abstention.

### Splits

- Train: 40% (for few-shot examples or controller training if needed)
- Dev: 20% (prompt iteration, hyperparameter tuning)
- Test: 40% (final evaluation only — locked before any eval)

### Hard negatives

Minimum 2 per gold document. Must:
- Match topic broadly
- Fail on at least one specific constraint
- Be from the same subfield

Strategies: same MeSH / wrong organism, same intervention / wrong outcome, same organism /
wrong method, high BM25 score / fails constraint check, same heading / wrong study type.

### Abstention requirements

At least 15% of tasks. Must be plausible (not nonsensical). Must have near-miss documents.
Verified by confirming no corpus document satisfies all constraints.

---

## 8. Corpus Design

### Three-layer architecture

**Layer 1: Metadata**
- title, abstract, year, venue, DOI, PMID, PMCID
- MeSH terms, publication type, organisms, keywords
- citation count, references, cited-by links
- OpenAlex concepts, S2 fields of study

**Layer 2: Content**
- Abstract text
- Full text sectioned by scientific structure:
  - Introduction, Methods (subsections), Results (subsections), Discussion
- Figure captions
- Table text (where extractable)

**Layer 3: Structured evidence**
- Normalized entities (organisms, assays, interventions)
- Experimental conditions
- Citation graph neighbors

### Data sources

| Source | Provides | License | Access |
|--------|---------|---------|--------|
| PMC Open Access | Full text (XML) | CC BY / CC0 | FTP bulk |
| PubMed | Abstracts, MeSH, metadata | Public domain | E-utilities API |
| OpenAlex | Citation graph, concepts, topics | CC0 | REST API |
| Semantic Scholar | Citations, TLDRs, fields of study | Free API | REST API |

### Chunking strategy

**By scientific structure, not token windows.**

| Chunk type | Max tokens | Source |
|-----------|------------|--------|
| Abstract | 512 | Abstract text |
| Section | 1024 | Split on subsection headings |
| Figure caption | 256 | Figure captions |
| Table text | 512 | Table captions + body |

Each chunk inherits parent document metadata.

### Size targets

| Milestone | Documents |
|-----------|-----------|
| Pilot | 10-50K |
| Full | 50-250K |

### Filtering

- Exclude: editorials, letters, comments, errata
- Year range: 2015-2026
- Require at least abstract
- English only

---

## 9. Retrieval Stack & Baselines

### Baseline progression (evaluated in order)

| # | Baseline | Engine | Purpose |
|---|----------|--------|---------|
| 1 | BM25 | Pyserini (Lucene) | Lexical floor |
| 2 | Dense (SPECTER2) | FAISS + `allenai/specter2` with proximity adapter | Semantic scientific baseline |
| 3 | Hybrid | BM25 + SPECTER2, RRF fusion (k=60) | Combined lexical + semantic |
| 4 | Hybrid + Reranker | Hybrid top-100 reranked by cross-encoder (`ms-marco-MiniLM-L-12-v2`) | Precision-optimized |
| 5 | SynthSearch Agent | Prompt-based controller with tools | Our contribution |
| 6 | OpenScholar 8B | End-to-end answerer (comparison only) | Downstream synthesis baseline |

### Agent tools

```
search(query, filters, top_k)      — hybrid retrieval with optional metadata filters
read(doc_id_or_chunk_id)           — read a specific document or chunk
prune(item_ids, reason)            — remove items from working evidence set
expand_citations(doc_id, direction) — retrieve citing/cited papers
refine_query(query, evidence, constraints) — generate refined subquery
summarize_evidence(item_ids)       — summarize working evidence set
stop(reason)                       — stop with "sufficient" | "abstain" | "max_steps"
```

### Agent control loop

1. Parse the question
2. Extract explicit constraints
3. Run broad hybrid search (top-20)
4. Inspect top documents (read)
5. Drop irrelevant evidence (prune)
6. Refine subqueries for unresolved constraints
7. Run metadata-filtered search
8. Optionally traverse citations
9. Stop when evidence is sufficient or abstention is warranted

### Agent parameters

- Max search steps: 8
- Max total tool calls: 30
- Initial search top-k: 20
- Refinement search top-k: 10
- Max evidence set size: 50 (prune at threshold)
- Prune target: 20
- Abstention confidence threshold: 0.3

### Rollout configuration

- Default: 1x rollout
- RRF configuration: 4x rollout + reciprocal-rank fusion (k=60)
- Context-1 showed 4x rollout + RRF matches much larger models

---

## 10. Agent Design (Context-1-Inspired)

### What we keep from Context-1

- **Explore -> Generate -> Verify -> Distract -> Chain** loop structure
- Iterative query decomposition
- Self-editing (context pruning during search)
- Separation of retrieval from answer generation
- Synthetic task generation with extraction-based verification
- 4x rollout + RRF as a cheap performance boost

### What we add

| Addition | Rationale |
|----------|-----------|
| Metadata filters (organism, assay, year, venue, MeSH) | Scientific search depends on structured conditions, not just text |
| Citation graph hops | Scientific evidence chains through citations |
| Figure/table access | Key results are often in figures and tables |
| Scratchpad / structured fact store | Prune raw text while keeping structured facts |
| Explicit abstention logic | Must know when evidence is insufficient |

### What we change

| Context-1 | SynthSearch |
|-----------|-------------|
| General web search | Biomedical literature search |
| Generic distractors | Domain-specific hard negatives (wrong organism/assay/condition) |
| Text-only | Text + metadata + citation graph |
| No abstention | Explicit abstention evaluation |

### Implementation approach

**Phase 1 (now)**: Prompt-based controller using a frontier LLM as backbone.
The LLM receives the tool definitions, the question, and a system prompt specifying
the control loop. No model training.

**Phase 2 (only if needed)**: LoRA fine-tune of a 7-8B model on agent traces from
Phase 1. Only attempted after CP2 if zero-shot is clearly close but insufficient.

**Phase 3 (future work, NOT for this paper)**: Full RL training in the Context-1 style.
Mentioned in the paper's future work section.

---

## 11. Synthetic Task Generation Pipeline

### Three-stage pipeline

#### Stage A: Seed Extraction

Sample documents from the corpus. For each seed cluster (3 related papers):
- Extract entities: organisms, cell lines, genes/proteins
- Extract methods: assays, techniques, equipment
- Extract conditions: dosage, temperature, duration, controls
- Extract outcomes: measurements, endpoints, findings
- Identify comparison axes
- Identify potential contradictions or conditional effects

Output: Candidate fact graph per seed cluster.

#### Stage B: Task Synthesis

From each fact graph, generate questions requiring:
- Multiple constraints (constraint-satisfaction family)
- Evidence from both sides (comparative family)
- Condition tracking (contradiction family)
- Or deliberately constructed to lack support (abstention family)

Each task must specify:
- What counts as supporting evidence
- What would mislead weaker retrievers (hard negatives)
- Expected answer type

#### Stage C: Verification

For each generated task:
- Confirm supporting documents actually contain the claimed evidence
- Confirm all constraints are satisfied
- Confirm hard negatives match topic but fail on the distinguishing constraint
- Confirm abstention tasks genuinely lack sufficient evidence
- Run BM25 difficulty calibration (easy = BM25 finds it, hard = BM25 misses it)

Auto-repair up to 2 attempts per task. Reject if still failing.

### LLM backbone for generation

- Model: **Claude Opus 4 (claude-opus-4-6)** via user's own Anthropic API key
- Temperature 0.7 for generation, 0.0 for verification
- All calls use the user's own API key (NOT shared keys)
- Estimated cost: $50-150 for 500-1000 tasks with verification

### Hard negative strategies

| Strategy | Weight | Example |
|----------|--------|---------|
| Same topic, wrong organism | 25% | CRISPR paper but in zebrafish, not human |
| Same intervention, wrong outcome | 20% | Metformin paper but measuring weight, not tumor |
| Same organism, wrong method | 20% | Human iPSC paper but using electroporation, not CRISPR |
| High BM25, fails constraint | 20% | Top BM25 hit that doesn't satisfy year/assay constraint |
| Same MeSH, wrong study type | 15% | Same disease topic but review, not RCT |

### Human audit requirements

- **First 50 pilot tasks**: full human review
- **All contradiction tasks**: human review (highest error risk)
- **All abstention tasks**: human review (second highest)
- **Random 10% of remaining**: spot check
- Calibration: 10 tasks reviewed independently by all reviewers before main audit
- Target inter-annotator agreement: >80% on ACCEPT/REJECT

---

## 12. Metrics & Evaluation Protocol

### Layer 1: Retrieval metrics (PRIMARY)

| Metric | Level | k values |
|--------|-------|----------|
| Recall@k | Document + Passage | 5, 10, 20, 50 |
| nDCG@k | Document | 10, 20 |
| MRR | Document | — |
| Evidence set recall | Document | — |
| Passage recall (fuzzy 80%) | Passage | — |

**Primary metric**: Recall@10 (document-level)

### Layer 2: Workflow metrics

- Constraint coverage: fraction of required constraints satisfied by evidence
- Duplicate evidence ratio
- Search iterations used
- Metadata filter usage rate
- Citation expansion utility (fraction of evidence found via citation hops)

### Layer 3: Downstream answer metrics

- Answer correctness (LLM-judged)
- Citation support rate
- Unsupported claim rate

### Layer 4: Abstention metrics

- Abstention precision
- Abstention recall
- Abstention F1

### Layer 5: Efficiency metrics

- Latency (seconds per task)
- Cost (USD per task)
- Total tokens per task
- Model calls per task

### Statistical significance

- Paired bootstrap test, n=1000, alpha=0.05
- Stratified analysis by task family and difficulty

---

## 13. Ablation Plan

| # | Ablation | What it tests |
|---|----------|--------------|
| 1 | No metadata filters | Value of structured filtering |
| 2 | No pruning | Value of self-editing |
| 3 | No citation expansion | Value of citation graph |
| 4 | No scratchpad | Value of structured fact storage |
| 5 | Single-step search | Value of iterative refinement |
| 6 | BM25 only | Lexical floor |
| 7 | Dense only | Semantic floor |
| 8 | Hybrid only | Combined floor |
| 9 | Hybrid + rerank | Non-agentic ceiling |
| 10 | No abstention logic | Value of explicit abstention |

---

## 14. Infrastructure & Compute

### Local machine (your MacBook)

- **Role**: Corpus processing, BM25 indexing, paper writing, script development
- **Requirements**: 32GB+ RAM, 200-500GB free disk
- **No GPU needed locally**

### Lambda Labs A100 (1x A100 SXM4 80GB)

- **Instance**: `gpu_1x_a100_sxm4` (**40GB** VRAM, 216GB RAM, 472GB disk)
- **Access**: `ssh -i "Lambda Cloud MacBook Pro.pem" ubuntu@129.213.26.40`
- **Role**:
  - SPECTER2 embedding generation (50-250K papers)
  - Cross-encoder reranking
  - OpenScholar 8B inference (if used as downstream baseline)
  - Optional: controller fine-tuning (LoRA on 7-8B model)
- **Estimated usage**: 10-30 GPU-hours total across the project
- **Cost**: Lambda reserved instance pricing (already provisioned)
- **Note**: 40GB VRAM (not 80GB). Sufficient for all planned work.

### Anthropic API (Claude Opus 4)

- **Model**: `claude-opus-4-6`
- **Role**: Teacher model / distiller for:
  - Synthetic task generation (Stage B)
  - Task verification (Stage C)
  - Seed extraction (Stage A)
  - Agentic controller backbone (prompt-based agent)
- **Access**: User's own API key (set as `ANTHROPIC_API_KEY` env var)
- **Estimated usage**: $150-800 depending on rollout count

### What runs where

| Task | Where | GPU? | Duration |
|------|-------|------|----------|
| Metadata collection (PubMed/OpenAlex APIs) | Local | No | Hours |
| PMC full-text download | Local | No | Hours |
| XML parsing + chunking | Local | No | Minutes-hours |
| BM25 indexing (Pyserini) | Local | No | Minutes |
| SPECTER2 embedding | Lambda A100 | Yes | 2-6 hours |
| FAISS index building | Lambda A100 | Optional | Minutes |
| Cross-encoder reranking | Lambda A100 | Yes | 1-3 hours |
| Task generation (Claude Opus) | API call | No | Hours (rate-limited) |
| Task verification (Claude Opus) | API call | No | Hours |
| Agent evaluation — 1x rollout | API call | No | Hours |
| Agent evaluation — 4x rollout | API call | No | Hours |
| OpenScholar 8B inference | Lambda A100 | Yes | 2-4 hours |
| Optional LoRA fine-tuning | Lambda A100 | Yes | 4-8 hours |
| Paper writing + figures | Local | No | Days |

---

## 15. Budget

### Compute costs

| Item | Platform | Est. Cost | When |
|------|----------|-----------|------|
| SPECTER2 embedding (50-250K papers) | Lambda A100 | Included in instance | Day 9-10 |
| Cross-encoder reranking | Lambda A100 | Included in instance | Day 10-11 |
| OpenScholar 8B inference | Lambda A100 | Included in instance | Day 19 |
| Optional LoRA fine-tuning | Lambda A100 | Included in instance | Day 17+ |
| **Lambda instance subtotal** | | **~10-30 GPU-hours** | |

### API costs (user's own Anthropic key)

| Item | Est. Cost | When |
|------|-----------|------|
| Seed extraction (Stage A) | $10-30 | Days 6-7 |
| Task synthesis (Stage B) | $30-80 | Days 7-8 |
| Task verification (Stage C) | $10-40 | Days 8-9 |
| Agent evaluation — 1x rollout (300-800 tasks) | $100-300 | Days 12-15 |
| Agent evaluation — 4x rollout (optional) | $300-800 | Days 16-18 |
| **API subtotal** | **$150-800** | |

### Total budget

| Scenario | Total |
|----------|-------|
| Minimal (benchmark + 1x agent rollout only) | $150-350 |
| Standard (full baselines + 1x agent + some 4x rollouts) | $300-600 |
| Full (4x rollout on all tasks + fine-tuning) | $500-1000 |

---

## 16. Team & Roles

### Option A: Lean (current)

| Role | Person | Responsibilities |
|------|--------|-----------------|
| Lead researcher | You | Scope calls, scientific validity, paper ownership, final writing |
| Implementation agent | Claude (Synsc) | Scripts, pipeline, baselines, agent, figures |
| Domain reviewer | TBD (part-time) | Audit first 50-100 tasks, check scientific validity |

### Time commitment

- Lead researcher: ~40-60 hours over 25 days
- Domain reviewer: ~5-10 hours total (task audit phase)
- Claude: continuous

---

## 17. Day-by-Day Execution Schedule

### Phase 1: Days 1-2 (Apr 11-12) — Scope & Scaffold [COMPLETE]

- [x] Lock vertical (biomedicine)
- [x] Lock submission track (E&D default)
- [x] Create repo structure
- [x] Write PROJECT_BRIEF.md
- [x] Write DECISIONS.md
- [x] Write benchmark_spec.md
- [x] Write annotation_guidelines.md
- [x] Write all config files
- [x] Write corpus_card.md, error_taxonomy.md, submission_checklist.md
- [x] Write MASTER_PLAN.md (this file)

### Phase 2: Days 3-5 (Apr 13-16) — Corpus Construction

- [ ] Implement `scripts/01_collect_metadata.py` (PubMed E-utilities + OpenAlex API)
- [ ] Implement `scripts/02_download_text.py` (PMC OA full-text download)
- [ ] Implement `scripts/03_parse_documents.py` (XML parsing, section extraction)
- [ ] Implement `scripts/04_build_corpus.py` (normalize, chunk, assemble JSONL)
- [ ] Run pilot corpus (10-50K papers)
- [ ] Compute corpus statistics
- [ ] Fill in corpus_card.md statistics
- [ ] Human spot-check: inspect 20 random parsed documents

**CP1 (end of Day 5)**: Must have locked vertical, working corpus, benchmark schema, >=20
believable tasks sketched. If not → narrow domain further.

### Phase 3: Days 6-8 (Apr 17-19) — Pilot Benchmark Generation

- [ ] Set up Anthropic API key for bulk generation
- [ ] Implement `scripts/07_generate_seed_tasks.py` (seed extraction + task synthesis)
- [ ] Implement `scripts/08_verify_tasks.py` (automated verification)
- [ ] Generate 100 pilot tasks
- [ ] Run automated verification
- [ ] Create hard negatives
- [ ] Queue tasks for human audit

### Phase 4: Days 9-11 (Apr 20-22) — Indexing & Baseline Retrieval

- [ ] SSH into Lambda A100 instance
- [ ] Implement `scripts/05_build_bm25_index.py`
- [ ] Implement `scripts/06_build_dense_index.py` (SPECTER2 + FAISS)
- [ ] Implement `scripts/10_run_baselines.py` (BM25, dense, hybrid, hybrid+reranker)
- [ ] Implement `scripts/12_score_retrieval.py`
- [ ] Run baselines on pilot benchmark
- [ ] Generate initial results table
- [ ] Identify retrieval gaps and failure modes

**CP2 (end of Day 11)**: Must have 50-100 audited tasks, BM25+dense+hybrid results, at
least one clear retrieval gap. If not → redesign task families.

### Phase 5: Days 12-15 (Apr 23-26) — Agent Implementation & Evaluation

- [ ] Implement `src/agent/` (controller, tools, state management)
- [ ] Implement `scripts/11_run_agent.py`
- [ ] Run agent on pilot benchmark (1x rollout)
- [ ] Run agent on pilot benchmark (4x rollout + RRF)
- [ ] Compare agent vs baselines
- [ ] Inspect agent traces manually
- [ ] Run initial ablations

**CP3 (end of Day 15)**: Must have agent traces, at least one improvement over hybrid.
If not → position paper as benchmark/evaluation first, minimize method claims.

### Phase 6: Days 16-18 (Apr 27-29) — Scale Benchmark & Full Experiments

- [ ] Implement `scripts/09_make_splits.py`
- [ ] Generate full 300-800 task benchmark
- [ ] Second audit pass
- [ ] Rerun all baselines on full benchmark
- [ ] Run full ablation suite
- [ ] Run efficiency measurements
- [ ] Decision: train controller or stay prompt-based?

### Phase 7: Days 19-21 (Apr 30-May 2) — Results & Figures

- [ ] Implement `scripts/13_score_answers.py` (downstream answer eval)
- [ ] Implement `scripts/14_make_tables.py`
- [ ] Implement `scripts/15_make_figures.py`
- [ ] Generate all paper tables
- [ ] Generate all paper figures
- [ ] Write error analysis (sample 50 failures, categorize per error_taxonomy.md)
- [ ] Finalize submission track decision

**CP4 (end of Day 21)**: Must have final experiments, paper outline filled, artifact
packaging started. If not → drop low-value ablations, prioritize clarity.

### Phase 8: Days 22-25 (May 3-6) — Paper Writing & Submission

- [ ] Write Introduction
- [ ] Write Related Work
- [ ] Write Benchmark section
- [ ] Write Method section
- [ ] Write Experiments + Results
- [ ] Write Error Analysis
- [ ] Write Limitations & Ethics
- [ ] Compile paper in NeurIPS LaTeX template
- [ ] Prepare Croissant metadata
- [ ] Anonymize code and data
- [ ] Run submission checklist
- [ ] Submit abstract (May 4 AoE)
- [ ] Submit full paper (May 6 AoE)

---

## 18. Go / No-Go Checkpoints

| Checkpoint | Date | Must Have | If Not |
|------------|------|-----------|--------|
| **CP1** | Apr 16 | Working corpus, schema, >=20 tasks | Narrow domain |
| **CP2** | Apr 22 | 50-100 audited tasks, baseline results, nontrivial gap | Redesign task families |
| **CP3** | Apr 26 | Agent traces, at least one improvement over hybrid | Lead with benchmark, agent is "reference baseline" |
| **CP4** | May 2 | Final experiments, outline filled, artifacts started | Drop ablations, prioritize clarity |

### Track decision rule

| Condition | Track |
|-----------|-------|
| Agent >5pt Recall@10 over best non-agentic baseline | Main track (method + benchmark) |
| Agent improves but <5pt | E&D (benchmark first, method second) |
| Agent does not improve | E&D (benchmark only, agent as reference) |

---

## 19. Risk Register & Fallbacks

| # | Risk | Severity | Probability | Fallback |
|---|------|----------|-------------|----------|
| R1 | Benchmark tasks too easy | High | Medium | Strengthen hard negatives, add more contradiction/abstention |
| R2 | Agent doesn't beat hybrid | Medium | Medium | Lead with benchmark contribution, agent as reference baseline |
| R3 | PMC XML parsing is messy | Medium | High | Fall back to abstract+metadata only |
| R4 | Full-text licensing slows progress | Medium | Low | Abstract-only + metadata benchmark initially |
| R5 | No time for fine-tuning | Low | High | Prompt-based only, training as future work |
| R6 | API costs exceed budget | Medium | Medium | Reduce 4x rollouts, use dev set only for expensive runs |
| R7 | Task generation produces low quality | High | Medium | Strict verification + early human audit |
| R8 | Lambda instance unavailable | Medium | Low | Fall back to Modal (A100-80GB ~$4/hr) |
| R9 | EpiBench overlap concern | Medium | Low | Differentiate on metadata-awareness, abstention, domain depth |
| R10 | Reviewer unfamiliarity with E&D | Low | Low | E&D is established since NeurIPS 2023 |

---

## 20. Paper Outline

### Title

**SynthSearch: A Metadata-Aware Self-Editing Search Agent and Benchmark for Biomedical Literature Retrieval**

### Sections

1. **Introduction** (1.5 pages)
   - Scientific retrieval is hard: DORIS-MAE, LitSearch gaps
   - Existing benchmarks under-test condition-sensitive, multi-constraint workflows
   - Our contributions: benchmark + search agent

2. **Related Work** (1 page)
   - Scientific retrieval systems (OpenScholar, PaperQA2)
   - Retrieval agents (Context-1)
   - Scientific retrieval benchmarks (DORIS-MAE, LitSearch, EpiBench)
   - Benchmark construction methods

3. **SynthSearch-Biomed Benchmark** (2 pages)
   - Domain and corpus description
   - Task taxonomy (4 families)
   - Generation pipeline (extract → synthesize → verify)
   - Hard negative construction
   - Human audit process
   - Dataset statistics

4. **SynthSearch Agent** (1.5 pages)
   - Architecture (tools, state, control loop)
   - Metadata-aware search
   - Self-editing / pruning
   - Citation expansion
   - Abstention logic
   - Rollout and fusion

5. **Experiments** (0.5 pages)
   - Baselines, metrics, setup

6. **Results** (1.5 pages)
   - Main table (all baselines × primary metrics)
   - Per-family breakdown
   - Ablation table
   - Efficiency table
   - Case studies (2-3 examples with agent traces)

7. **Error Analysis** (0.5 pages)
   - Error taxonomy distribution
   - What the agent fixes vs what persists

8. **Limitations & Ethics** (0.5 pages)
   - Synthetic benchmark bias
   - Single-domain scope
   - API dependency
   - Corpus licensing
   - Misuse risk (search manipulation)

**Total**: ~9 pages + references + appendix

---

## 21. Repo Structure & File Map

```
synthetic-science-search/
├── MASTER_PLAN.md              ← This file
├── PROJECT_BRIEF.md            ← Thesis, budget, checkpoints
├── DECISIONS.md                ← Locked decisions log
├── requirements.txt            ← Python dependencies
├── pyproject.toml              ← Package config
├── configs/
│   ├── corpus.yaml             ← Data sources, schema, chunking
│   ├── generation.yaml         ← Task families, pipeline, hard negatives
│   ├── retrieval.yaml          ← BM25/SPECTER2/hybrid/agent configs
│   └── evaluation.yaml         ← Metrics, ablations, tables, figures
├── data/
│   ├── raw/                    ← Downloaded metadata and full text
│   ├── interim/                ← Intermediate processing artifacts
│   ├── processed/              ← Final corpus JSONL, chunks, indices
│   └── benchmark/              ← train.jsonl, dev.jsonl, test.jsonl, schema.json, croissant.json
├── docs/
│   ├── benchmark_spec.md       ← Task families, schema, quality criteria
│   ├── annotation_guidelines.md ← Human review process
│   ├── corpus_card.md          ← Data card (stats TBD)
│   ├── dataset_card.md         ← HuggingFace-style dataset card (TBD)
│   ├── error_taxonomy.md       ← 10 error types for paper analysis
│   └── submission_checklist.md ← NeurIPS E&D requirements
├── scripts/
│   ├── 01_collect_metadata.py  ← PubMed + OpenAlex metadata
│   ├── 02_download_text.py     ← PMC OA full text
│   ├── 03_parse_documents.py   ← XML parsing, section extraction
│   ├── 04_build_corpus.py      ← Normalize, chunk, assemble
│   ├── 05_build_bm25_index.py  ← BM25 index
│   ├── 06_build_dense_index.py ← SPECTER2 + FAISS
│   ├── 07_generate_seed_tasks.py ← Seed extraction + synthesis
│   ├── 08_verify_tasks.py      ← Automated verification
│   ├── 09_make_splits.py       ← Train/dev/test splitting
│   ├── 10_run_baselines.py     ← All non-agentic baselines
│   ├── 11_run_agent.py         ← SynthSearch agent
│   ├── 12_score_retrieval.py   ← Retrieval metrics
│   ├── 13_score_answers.py     ← Downstream answer metrics
│   ├── 14_make_tables.py       ← Paper tables
│   └── 15_make_figures.py      ← Paper figures
├── src/
│   ├── corpus/                 ← Corpus processing modules
│   ├── generation/             ← Task generation modules
│   ├── retrieval/              ← Retrieval engines and hybrid fusion
│   ├── agent/                  ← SynthSearch agent controller
│   ├── evaluation/             ← Metric computation
│   └── utils/                  ← Shared utilities
├── results/
│   ├── pilots/                 ← Pilot experiment results
│   ├── baselines/              ← Baseline results and traces
│   ├── ablations/              ← Ablation results
│   ├── paper_tables/           ← LaTeX-ready tables
│   └── paper_figures/          ← Publication-ready figures
├── paper/
│   ├── neurips_2026.tex        ← Paper source
│   ├── refs.bib                ← Bibliography
│   └── figures/                ← Paper figures
└── tests/                      ← Unit tests
```

---

## 22. Implementation Task Queue

Execute in this exact order. Each task depends on the one before it.

| # | Script / Task | Dependencies | Est. Time |
|---|--------------|--------------|-----------|
| 1 | `scripts/01_collect_metadata.py` | None | 1 day |
| 2 | `scripts/02_download_text.py` | Script 01 (need PMCIDs) | 1 day |
| 3 | `scripts/03_parse_documents.py` | Script 02 (need XML files) | 0.5 days |
| 4 | `scripts/04_build_corpus.py` | Script 03 (need parsed docs) | 0.5 days |
| 5 | `scripts/05_build_bm25_index.py` | Script 04 (need corpus) | 0.5 days |
| 6 | `scripts/06_build_dense_index.py` | Script 04 + Lambda A100 | 0.5 days |
| 7 | `scripts/07_generate_seed_tasks.py` | Script 04 + Anthropic API key | 1 day |
| 8 | `scripts/08_verify_tasks.py` | Script 07 (need raw tasks) | 0.5 days |
| 9 | `scripts/09_make_splits.py` | Script 08 (need verified tasks) | 0.5 days |
| 10 | `scripts/10_run_baselines.py` | Scripts 05, 06, 09 | 1 day |
| 11 | `src/agent/` + `scripts/11_run_agent.py` | Script 10 + Anthropic API key | 2 days |
| 12 | `scripts/12_score_retrieval.py` | Scripts 10, 11 | 0.5 days |
| 13 | `scripts/13_score_answers.py` | Script 12 | 0.5 days |
| 14 | `scripts/14_make_tables.py` | Scripts 12, 13 | 0.5 days |
| 15 | `scripts/15_make_figures.py` | Scripts 12, 13 | 0.5 days |

---

## 23. Locked Decisions

| ID | Decision | Date |
|----|----------|------|
| D001 | Target NeurIPS 2026 E&D track by default | 2026-04-11 |
| D002 | Vertical: biomedical / life-science literature | 2026-04-11 |
| D003 | No large-model training before baseline results | 2026-04-11 |
| D004 | Corpus size: pilot 10-50K, full 50-250K | 2026-04-11 |
| D005 | Benchmark size: pilot 50-100, submission 300-800 | 2026-04-11 |
| D006 | Eight task families: constraint, comparative, contradiction, abstention, multihop, temporal, aggregation, negative | 2026-04-11 |
| D007 | Chunk by scientific structure, not token windows | 2026-04-11 |
| D008 | API provider: Anthropic Claude Opus 4 (user's own key) | 2026-04-11 |
| D009 | Retrieval backbone: SPECTER2 dense + BM25 lexical + cross-encoder reranker | 2026-04-11 |
| D010 | Context-1 as design blueprint, not deployed model. Prompt-based controller. | 2026-04-11 |
| D011 | Compute: Lambda Labs 1x A100-SXM4-40GB at ubuntu@129.213.26.40 | 2026-04-11 |
| D012 | Task families expanded from 4 to 8 for broader coverage | 2026-04-11 |

---

## 24. Credential & Security Setup

### Required credentials

| Credential | Env var | Status | How to set |
|------------|---------|--------|-----------|
| Anthropic API key | `ANTHROPIC_API_KEY` | **SET** (session) | Set for current session. Add to `~/.zshrc` for persistence. |
| Lambda Labs SSH | PEM file | **VERIFIED** | `Lambda Cloud MacBook Pro.pem` → `ubuntu@129.213.26.40` |
| Lambda Labs API | `LAMBDA_API_KEY` | NOT SET | Connect at cli.syntheticsciences.ai → Services (optional, SSH works) |
| NCBI API key (optional) | `NCBI_API_KEY` | — | Register at ncbi.nlm.nih.gov for 10 req/sec |
| OpenAlex (optional) | Polite pool | — | Set email in User-Agent header |

### Security rules

- **NEVER** write API keys into any file that could be committed
- **NEVER** use `echo`, `printenv`, `env` to display keys
- Keys go in environment variables or `.env` files (`.gitignore`'d)
- PEM files go in `.gitignore`

### Before starting implementation

```bash
# 1. Set Anthropic key (if not already in session)
export ANTHROPIC_API_KEY="your-key-here"

# 2. Verify
[ -n "$ANTHROPIC_API_KEY" ] && echo "set" || echo "not set"

# 3. Test Lambda SSH access (VERIFIED - A100-SXM4-40GB online)
ssh -i "Lambda Cloud MacBook Pro.pem" ubuntu@129.213.26.40 nvidia-smi
```

**Status as of 2026-04-11**: Both credentials verified and working.

---

## 25. What Not To Do

1. **Do not decide the scientific claim** without the lead researcher's approval
2. **Do not silently broaden the domain** beyond biomedicine
3. **Do not invent unsupported evidence** in benchmark tasks
4. **Do not treat answer quality as a substitute** for retrieval quality
5. **Do not skip hard-negative construction**
6. **Do not skip abstention evaluation**
7. **Do not train a large policy model** before the benchmark is stable
8. **Do not overclaim "state of the art"** without broad baselines
9. **Do not spend >$100 on API calls** without explicit approval
10. **Do not write API keys into files**
11. **Do not submit to main track** unless agent shows >5pt Recall@10 gain
12. **Do not use shared API keys** for bulk generation/evaluation
13. **Do not optimize for a single metric** — the four-family structure is the point
14. **Do not build the corpus bigger than needed** — 50-250K papers is enough

---

*This plan is a living document. New decisions are appended to DECISIONS.md.
This file is updated only at major milestones.*
