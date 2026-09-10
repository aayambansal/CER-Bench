# SynthSearch: Project Brief

## Working Title

**SynthSearch: A Metadata-Aware Self-Editing Search Agent and Benchmark for Biomedical Literature Retrieval**

## Submission Target

NeurIPS 2026 Evaluations & Datasets (E&D) Track (default).
Main track pivot only if method contribution becomes clearly strong by Day 15.

- Abstract deadline: May 4, 2026 (AoE)
- Full paper deadline: May 6, 2026 (AoE)
- Double-blind review
- E&D requires: accessible code/data at submission + Croissant metadata for new datasets

## Thesis

Existing scientific retrieval benchmarks under-test real experimental-science workflows.
They focus on single-answer needle-in-a-haystack queries and ignore the constraint-heavy,
condition-sensitive, multi-document reasoning that working scientists actually perform.

This paper contributes:

1. **A benchmark** of realistic biomedical literature retrieval tasks spanning four
   families: constraint-satisfaction, comparison, contradiction/conditionality, and
   abstention.

2. **A search agent** inspired by Context-1 that decomposes queries, searches
   iteratively, prunes its working context, applies metadata filters, and traverses
   citation links.

## Vertical

Biomedical / life-science literature.

Rationale: strongest open-access full text (PMC OA), richest metadata (MeSH, organisms,
assays), largest existing scholarly APIs (PubMed, OpenAlex, Semantic Scholar), easiest
reviewer comprehension.

## Corpus

50,000-250,000 biomedical papers from PMC Open Access subset, enriched with:
- Structured metadata (MeSH terms, organisms, publication types, venues, years)
- Section-level chunking (abstract, introduction, methods subsections, results subsections, discussion, figure captions, table text)
- Citation graph edges from OpenAlex/Semantic Scholar

## Benchmark

300-800 tasks across four families:
- **Constraint-satisfaction**: find papers matching multiple experimental conditions
- **Comparative**: compare methods, organisms, outcomes across papers
- **Contradiction/conditionality**: reconcile conflicting findings under different conditions
- **Abstention**: recognize when the corpus does not support a satisfying answer

Each task includes gold evidence documents, gold passages, hard negatives, required
constraints, and expected answer type.

## Baselines

1. BM25 (lexical floor)
2. SPECTER2 dense retrieval (scientific embedding baseline)
3. BM25 + SPECTER2 hybrid
4. Hybrid + cross-encoder reranker
5. SynthSearch agent (prompt-based Context-1-style controller)
6. OpenScholar 8B (downstream answer synthesis comparison)

## Metrics

| Layer | Metrics |
|-------|---------|
| Retrieval | Recall@5/10/20, nDCG@10, MRR, passage recall, evidence set recall |
| Workflow | Constraint coverage, duplicate ratio, search iterations, filter usage |
| Downstream | Answer correctness, citation support rate, unsupported claim rate |
| Abstention | Abstention precision, abstention recall |
| Efficiency | Latency, cost, tokens per task, model calls per task |

## Key Ablations

1. No metadata filters
2. No pruning
3. No citation expansion
4. No scratchpad/summary
5. Single-step vs multi-step search
6. BM25 only / dense only / hybrid only / hybrid+rerank

## Budget

| Category | Estimated Range |
|----------|----------------|
| GPU (indexing, reranking, baselines) | $15-75 |
| LLM API (task gen, agent, verification) | $150-800 |
| Total | $165-875 |

## Team

- Lead researcher: scope decisions, paper writing, scientific validity
- Claude: implementation, corpus processing, baseline runs, figure generation
- Domain reviewer (part-time): audit first 50-100 benchmark tasks

## Go/No-Go Checkpoints

| Checkpoint | Date | Must Have |
|------------|------|-----------|
| CP1 | Apr 16 | Locked vertical, working corpus, schema, 20+ tasks |
| CP2 | Apr 22 | 50-100 audited tasks, BM25+dense+hybrid results, nontrivial retrieval gap |
| CP3 | Apr 26 | Agent traces, at least one improvement over hybrid |
| CP4 | May 2 | Final experiments, paper outline filled, artifacts packaging started |

## Critical Decision Rule

If CP3 fails (agent does not beat hybrid), the paper is positioned as a pure benchmark
contribution for the E&D track. The agent becomes "a reference baseline" rather than
"our proposed method."
