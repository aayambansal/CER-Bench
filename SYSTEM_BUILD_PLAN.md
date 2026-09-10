# SynthSearch System Build Plan
## Scientific Retrieval Subagent — Context-1 for Science

**Created**: 2026-04-13
**Goal**: Build a production scientific search subagent that returns ranked documents,
supporting spans, extracted facts, abstention decisions, and search traces.
**Base model**: gpt-oss-20b (Apache 2.0, tool-capable, 128K context)
**Training recipe**: Context-1-style SFT warmup → on-policy RLVR

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │     Scientific Query         │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Search Controller          │
                    │   (gpt-oss-20b, fine-tuned)  │
                    │                              │
                    │   Actions:                   │
                    │   - search_hybrid            │
                    │   - search_late              │
                    │   - grep_corpus              │
                    │   - read_document            │
                    │   - read_span                │
                    │   - filter_metadata          │
                    │   - expand_citations         │
                    │   - read_trial_record        │
                    │   - structured_query         │
                    │   - prune_chunks             │
                    │   - scratchpad_write         │
                    │   - scratchpad_read          │
                    │   - stop / abstain           │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Retrieval Substrate v3     │
                    │                              │
                    │   Layer A: Scientific Text   │
                    │     PMC OA / BioC full text  │
                    │     PubMed metadata / MeSH   │
                    │                              │
                    │   Layer B: Graph + Metadata   │
                    │     OpenAlex citations/topics │
                    │     Semantic Scholar S2       │
                    │                              │
                    │   Layer C: Structured Data    │
                    │     ClinicalTrials.gov        │
                    │     USPTO office actions      │
                    │                              │
                    │   Layer D: Recency            │
                    │     bioRxiv / medRxiv         │
                    │                              │
                    │   Layer E: Internal (future)  │
                    │     Protocols, ELN, reports   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Search Indices             │
                    │   - BM25 (lexical)           │
                    │   - SPLADE (learned sparse)  │
                    │   - BGE/E5 (dense)           │
                    │   - ColBERTv2 (late-inter.)  │
                    │   - Biomedical reranker      │
                    │   - Metadata filter engine   │
                    │   - Citation graph index     │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Output                     │
                    │   - Ranked documents          │
                    │   - Supporting spans          │
                    │   - Scratchpad (facts)        │
                    │   - Abstain / retrieve        │
                    │   - Search trace              │
                    └─────────────────────────────┘
```

---

## Data Stack (5 Layers)

### Layer A: Reusable Scientific Text
- **PMC Open Access Subset** via BioC API — full text in XML/JSON
- **PubMed** — IDs, MeSH, metadata normalization only (not primary text source)
- Target: 100K-500K articles for initial index

### Layer B: Graph + Metadata
- **OpenAlex** — citation graph, related works, topics, institutions
- **Semantic Scholar** — paper metadata, TLDR, SPECTER embeddings, citation context

### Layer C: Structured Supervision
- **ClinicalTrials.gov** — interventions, outcomes, eligibility, results, timelines
- **USPTO** — office actions, rejections, prior-art citations (via PatentsView API)

### Layer D: Recency
- **bioRxiv / medRxiv** — metadata, abstracts, JATS paths, publication links

### Layer E: Internal (Phase 3+)
- Protocols, experiment logs, assay reports, ELN/LIMS exports

---

## Task Generation: Graph-Motif Sampling

### Core Motifs

Instead of sampling 2-3 random papers, sample **evidence graph motifs**:

1. **Trial → Paper → Follow-up**
   Source: ClinicalTrials.gov trial → linked publications → citation follow-ups
   Families: temporal, contradiction, aggregation

2. **Review claim → Primary studies → Contrary study**
   Source: Review paper cites primary studies; find one that contradicts
   Families: contradiction, comparative

3. **Office action → Cited prior art → Similar uncited patents**
   Source: USPTO rejection → cited references → near-miss prior art
   Families: constraint, abstention

4. **Methods paper → Application paper → Benchmark paper**
   Source: Methods chain via citations
   Families: multihop, comparative

5. **Quantitative result across papers**
   Source: Same measurement (IC50, survival rate) reported in multiple papers
   Families: aggregation

6. **Internal protocol → Experiment report → Decision memo**
   Source: Internal docs (Layer E, future)
   Families: multihop, temporal

### Verification Pipeline

For each generated task:
1. **Extraction-based verification**: LLM reads gold docs and confirms the question
   is answerable from those specific documents
2. **Distractor verification**: Confirm hard negatives match topic but fail on the
   distinguishing constraint
3. **Abstention verification**: Confirm no document in corpus satisfies all constraints
4. **Risk-based human audit**:
   - 100% of contradiction, abstention, negative-result tasks
   - 25% of temporal and aggregation
   - 10% random sample of constraint and multihop
5. **Quality gates** (must pass before scaling):
   - >95% positive precision
   - <1% distractor leakage
   - κ ≥ 0.80 inter-annotator agreement

---

## Training Recipe

### Phase E: Supervised Fine-Tuning

**Base model**: gpt-oss-20b
**Data**: 20K-30K verified tasks × multiple teacher rollouts = 80K-150K trajectories
**Teacher**: gpt-oss-120b or strongest frontier model

Each trajectory stores:
- Initial question + family label
- Tool calls (with tool name, arguments, observations)
- Query refinements
- Pruning actions
- Scratchpad writes/reads
- Final document set
- Abstain/retrieve decision
- Gold labels for reward computation

**Training details**:
- LoRA first (rank 64, alpha 128)
- Learning rate: 2e-4 → 5e-5 cosine decay
- Gradient checkpointing (unsloth-style)
- Mixed precision bf16
- Max sequence length: 16K (tool traces are long)

### Phase F: On-Policy Reinforcement Learning

**Start from**: SFT checkpoint
**Algorithm**: GRPO or CISPO (Context-1 style)
**Batch**: Start with 32 queries × 4 rollouts, scale to 128 × 8

**Reward function**:
```
R = 0.45 * R@10
  + 0.20 * R@20
  + 0.10 * family_metric_bonus
  + 0.15 * abstention_score
  - 0.05 * duplicate_penalty
  - 0.05 * tool_cost_penalty
```

**Family metric bonus** (per-task, based on family):
- Comparative: +0.1 if both sides retrieved
- Multi-hop: +0.1 if chain complete
- Aggregation: +0.1 if value coverage > 0.8
- Temporal: +0.1 if era coverage > 0.7
- Abstention: +0.2 if correctly abstains; -0.2 if false retrieve

**Curriculum** (Context-1 style):
- Stage 1 (weeks 1-2): Recall-heavy. Weight R@20 at 0.40.
- Stage 2 (weeks 3-4): Precision. Weight nDCG@10 at 0.20, add prune reward.
- Stage 3 (week 5+): Abstention + efficiency. Activate abstention reward,
  add latency penalty.

---

## Benchmark Design (Fully Audited)

### Training set (weakly supervised)
- 20K-30K tasks
- Extraction-verified, risk-based human audit
- Used for SFT and RL training only

### Evaluation benchmark (fully audited)
- 800 tasks (100 per family)
- 100% human-audited
- Pooled qrels from all methods (BM25 + dense + sparse + late-interaction + reranker + agent)
- Span-level evidence labels
- Family-specific structured metrics
- Explicit abstention labels with risk-coverage curves

---

## 12-Week Execution Schedule

### Weeks 1-2: Data Stack + Search Indices
- [ ] Build PMC OA ingestion pipeline (BioC API)
- [ ] Build OpenAlex graph ingestion
- [ ] Build ClinicalTrials.gov structured ingestion
- [ ] Build bioRxiv/medRxiv metadata ingestion
- [ ] Index with BM25 + SPLADE + BGE + E5
- [ ] Build metadata filter engine
- [ ] Build citation graph index
- [ ] Target: 100K+ documents indexed

### Weeks 3-4: Graph-Motif Task Generation
- [ ] Implement 5 motif samplers
- [ ] Build extraction-based verification pipeline
- [ ] Generate 2,000 pilot tasks
- [ ] Run full human audit on pilot (100% contradiction/abstention, 25% temporal/aggregation)
- [ ] Quality gate: >95% precision, <1% leakage, κ ≥ 0.80

### Weeks 5-6: Scale Training Data + Build Audited Benchmark
- [ ] Scale to 20K+ training tasks (pass quality gate first)
- [ ] Build 800-task fully audited benchmark
- [ ] Pool qrels from all retrieval methods
- [ ] Human-judge all pooled candidates (document + span level)

### Weeks 7-8: Teacher Traces + SFT
- [ ] Generate teacher trajectories (gpt-oss-120b or frontier model)
- [ ] 80K-150K trajectories from 20K tasks
- [ ] SFT on gpt-oss-20b with LoRA
- [ ] Validate: tool use accuracy, query quality, abstention behavior

### Weeks 9-10: Reinforcement Learning
- [ ] Start GRPO with recall-heavy curriculum
- [ ] Scale rollouts: 32×4 → 128×8
- [ ] Shift curriculum: recall → precision → abstention
- [ ] Train abstention head in parallel

### Weeks 11-12: Evaluation + Productionization
- [ ] Run full evaluation on audited benchmark
- [ ] Compare: 1x rollout, 4x rollout + RRF
- [ ] Run against all baselines (13+ methods)
- [ ] Build production search API
- [ ] Package for deployment

---

## Success Criteria

### Retrieval quality
- Agent R@10 > 0.55 on audited benchmark (current best: 0.443 on synthetic)
- Agent R@20 > 0.70
- Significantly beat SPLADE (current: 0.438 R@10) on family-weighted metrics

### Abstention
- AUROC > 0.85 (current: 0.70 with simple classifier)
- Abstention F1 > 0.60

### Family-specific
- Aggregation: value coverage > 0.60 (current: 0.50 with agent)
- Comparative: both-sides coverage > 0.50 (current: 0.33)
- Multi-hop: chain completion > 0.50 (current: 0.31)
- Temporal: era coverage > 0.50

### Efficiency
- Latency < 10s per query (vs current 17s)
- Cost < $0.02 per query after distillation (vs current $0.08)

---

## Cost Estimate

| Phase | Compute | API | Human | Total |
|-------|---------|-----|-------|-------|
| Data ingestion | $0 (Lambda) | $0 (free APIs) | $0 | $0 |
| Task generation (20K) | $0 | ~$60 (Sonnet) | $0 | $60 |
| Human audit (2K tasks) | $0 | $0 | ~$2,000 | $2,000 |
| Teacher traces (80K) | $0 | ~$200 (frontier) | $0 | $200 |
| SFT training | ~$50 (A100 hours) | $0 | $0 | $50 |
| RL training | ~$200 (A100 hours) | $0 | $0 | $200 |
| Evaluation | ~$20 (compute) | ~$30 (agent runs) | $0 | $50 |
| **Total** | **~$270** | **~$290** | **~$2,000** | **~$2,560** |

The human audit is the largest cost. Everything else is cheap.
