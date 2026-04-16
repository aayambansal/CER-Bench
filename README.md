# CER-Bench: Constrained Evidence Retrieval Benchmark for Biomedical Literature

**NeurIPS 2026 Evaluations & Datasets Track**

CER-Bench is a condition-sensitive retrieval benchmark for biomedical literature comprising 304+ tasks across eight families, designed to evaluate retrieval capabilities that working scientists actually need but existing benchmarks under-test.

## Task Families

| Family | Description | Key Challenge |
|--------|-------------|---------------|
| Constraint-satisfaction | Find papers matching multiple experimental constraints | Intersection over unions |
| Comparative | Retrieve evidence from both sides of a comparison | Balanced both-sides recall |
| Contradiction | Reconcile conflicting findings under different conditions | Condition tracking |
| Abstention | Recognize when corpus lacks sufficient evidence | Resisting false confidence |
| Multi-hop | Chain evidence across 3+ papers via shared entities | Bridging entity discovery |
| Temporal | Track how findings evolved over time | Time-distributed retrieval |
| Aggregation | Collect all reported values for a measurement | Exhaustive recall |
| Negative results | Find papers reporting null or negative findings | Overcoming positive-result bias |

## Key Results

The benchmark reveals three distinct performance regimes:

| Regime | Best Method | R@10 | Strength |
|--------|-------------|------|----------|
| Top-hit precision | BM25 | 0.424 | Exact terminology matching |
| Single-pass recall | SPLADE | 0.438 | Learned sparse expansion |
| Deep evidence collection | Iterative Agent (T=3) | 0.443 | Multi-round query refinement |

Additional findings:
- Classical PRF (RM3) does not help on condition-sensitive queries (R@10: 0.415, below BM25)
- General-purpose rerankers degrade scientific retrieval
- Abstention remains unsolved (best threshold baseline: AUROC 0.714, F1 0.388)

## Benchmark Versions

- **v1**: 304 tasks (38 per family), seeded gold labels (2.4 docs/task), with LLM-adjudicated expansion (12.7 docs/task)
- **v2**: 200 tasks (25 per family), cluster-first generation with complete gold labels by construction (9.8 docs/task)

## Dataset

The benchmark dataset is available on HuggingFace: [[Anonymous — will be revealed upon acceptance]](https://huggingface.co/datasets/[Anonymous — will be revealed upon acceptance])

## Repository Structure

```
CER-Bench/
├── data/benchmark/           # Benchmark tasks (v1 + v2)
├── src/                      # Source code
│   ├── corpus/               # Data ingestion (PubMed, OpenAlex)
│   ├── retrieval/            # Search API (10 tools)
│   ├── agent/                # Iterative search controller + abstention head
│   └── utils/                # Config loader, helpers
├── scripts/                  # Numbered pipeline scripts (01-30)
├── configs/                  # YAML configs (corpus, generation, retrieval, evaluation)
├── docs/                     # Benchmark spec, annotation guidelines, error taxonomy
├── paper/                    # LaTeX source + figures
├── requirements.txt
└── pyproject.toml
```

## Trained Models

| Model | Base | R@10 | HuggingFace |
|-------|------|------|-------------|
| SynthSearch-Qwen3-8B | Qwen/Qwen3-8B | 0.428 | [[Anonymous — will be revealed upon acceptance]](https://huggingface.co/[Anonymous — will be revealed upon acceptance]) |
| SynthSearch-gpt-oss-20b | openai/gpt-oss-20b | 0.384 | [[Anonymous — will be revealed upon acceptance]](https://huggingface.co/[Anonymous — will be revealed upon acceptance]) |

## Installation

```bash
pip install -r requirements.txt
```

## License

Apache 2.0
