# CER-Bench: a relevance-label and source-identity audit of scientific retrieval evaluation

**Paper:** *Fixed Rankings, Moving Leaders: A Relevance-Label and Source-Identity Audit of Scientific Retrieval Evaluation* — Aayam Bansal, Ishaan Gangwani (Synthetic Sciences).
Preprint PDF: [`paper/release/cerbench_arxiv.pdf`](paper/release/cerbench_arxiv.pdf) · anonymized ICLR copy: [`paper/release/cerbench_openreview.pdf`](paper/release/cerbench_openreview.pdf) · claim-to-artifact ledger: [`paper/iclr2027_final/CLAIMS_TO_ARTIFACTS.md`](paper/iclr2027_final/CLAIMS_TO_ARTIFACTS.md).

**Data mirror (tasks, saved rankings, judgments, traces, disclosure arrays, audits):** [huggingface.co/datasets/aayambansall/CER-Bench](https://huggingface.co/datasets/aayambansall/CER-Bench)
**Historical fine-tuned controller adapters:** [synthsearch-qwen3-8b-sft-v1](https://huggingface.co/aayambansall/synthsearch-qwen3-8b-sft-v1) · [synthsearch-gptoss20b-sft-v1](https://huggingface.co/aayambansall/synthsearch-gptoss20b-sft-v1)

## What this repository shows

CER-Bench is a collection of 304 synthetic biomedical retrieval tasks (eight families: constraint, comparative, contradiction, abstention, multi-hop, temporal, aggregation, negative-result) over 4,936 PubMed records, with saved outputs from twelve retrieval configurations. This release audits it rather than extending it:

- **Fixed rankings, moving leaders.** Holding every saved top-20 ranking fixed and swapping seed relevance labels for pooled automated judgments reverses 9 of 45 pairwise system orderings (Kendall τ_b = 0.584) and moves the mean-R@20 leader from a three-round search agent to a dense retriever (BGE). A 1,000-replicate paired, nested disclosure of the existing judgment pool brackets the leader change between 6 and 9 revealed judgments per query; paired empirical intervals overlap zero at both points.
- **Source identity.** Re-fetching all 4,936 PubMed records traced a parser bug (`.//ArticleId` descendant traversal under `PubmedData`) that let cited-paper identifiers replace an article's own identifiers: 247 PMC identifiers change, 226 of them appear in reference lists. A conservative own-article JATS restoration yields a 4,936-document / 10,313-chunk retrieval dataset (525 restored bodies, 21 title-only) with byte-reproducible builds.
- **Verified substrate.** A hash-bound BM25 index (63,109 terms, 2,010,185 postings), tested contracts for structural success, selective answering, and matched candidate budgets, and a 311-test suite.

**What it deliberately does not claim:** no human relevance validation, no new model runs, no corrected-corpus leaderboard, no fresh source-disjoint task partition, no redistribution rights over article text. The paper's Section 3 and Appendix M spell out the boundary between reproducible arithmetic, verified source identity, and still-unvalidated relevance.

## Repository layout

```
├── paper/
│   ├── release/                 # current manuscript: arXiv + OpenReview variants, build.sh, OPENREVIEW_FORM.md
│   ├── iclr2027_final/          # previous audit draft + CLAIMS_TO_ARTIFACTS.md ledger (numbers → files)
│   ├── iclr2027_submission/     # HISTORICAL benchmark manuscript (superseded; kept as construction record)
│   └── neurips_2026.tex         # HISTORICAL first manuscript (superseded)
├── src/
│   ├── corpus/                  # pubmed_xml.py (direct-child parser), verified_jats.py (own-article restoration),
│   │                            #   identity.py, PubMed/OpenAlex/BioC/ClinicalTrials clients
│   ├── retrieval/               # verified_bm25.py (hash-bound index), search_api.py (historical tools)
│   ├── evaluation/              # budget_protocol.py, structural_metrics.py, selective.py, strict_metrics.py
│   ├── agents/                  # provider_clients.py (native clients; keys from environment only)
│   └── agent/, generation/      # historical controller/abstention head and task generation
├── scripts/                     # 01–30 historical pipeline; 31–51 audit, repair, disclosure, verified retrieval
├── tests/                       # 15 modules, 311 tests
├── configs/                     # corpus / generation / retrieval / evaluation / equal-budget protocol
├── docs/                        # protocol and evidence docs (see below)
├── data/
│   ├── benchmark/v1/            # historical 304 tasks, 119/60/125 split (+ schema)
│   ├── benchmark/v2/            # historical 200-task cluster-first collection
│   ├── benchmark/v1_2_repaired_components/   # 238-task post hoc identity-repair candidate, 94/49/95 (exposed; not fresh)
│   ├── processed/authoritative_fulltext_v1/  # manifest, candidate audit, XML audit, summary (text rebuilt locally)
│   ├── processed/authoritative_fulltext_v1_bm25/  # verified BM25 index (index.json + postings.npz)
│   ├── processed/identity_repair_v1/         # mappings, aliases, evidence, quarantined tasks, candidate splits
│   └── raw/pubmed_verified_v1/  # request ledger + per-batch manifests of the authoritative re-fetch
├── results/
│   ├── baselines/               # saved per-task rankings (12 configurations), raw judgments, expanded qrels,
│   │                            #   agent traces, legacy equal-budget shards, equal-budget controls
│   └── readiness/               # authoritative_corpus/, identity/, qrel_sensitivity/ (forensic), qrel_disclosure/
│                                #   (summary + replicates.npz), verified_bm25/, verified_fulltext/, budget/, live_adapter/
├── annotations/                 # human-validation kit manifests/reports/templates (blank; no judgments collected)
└── examples/evaluation_contract # fixtures for the structural/selective evaluator
```

Key documents: [`NONHUMAN_REVISION_STATUS.md`](NONHUMAN_REVISION_STATUS.md) (current status), [`docs/AUTHORITATIVE_CORPUS.md`](docs/AUTHORITATIVE_CORPUS.md), [`docs/VERIFIED_FULLTEXT.md`](docs/VERIFIED_FULLTEXT.md), [`docs/VERIFIED_RETRIEVAL.md`](docs/VERIFIED_RETRIEVAL.md), [`docs/QREL_DISCLOSURE.md`](docs/QREL_DISCLOSURE.md), [`docs/IDENTITY_REPAIR.md`](docs/IDENTITY_REPAIR.md), [`docs/EQUAL_BUDGET_PROTOCOL.md`](docs/EQUAL_BUDGET_PROTOCOL.md), [`docs/EVALUATION_CONTRACT.md`](docs/EVALUATION_CONTRACT.md), [`docs/HUMAN_VALIDATION_PROTOCOL.md`](docs/HUMAN_VALIDATION_PROTOCOL.md), [`docs/benchmark_spec.md`](docs/benchmark_spec.md). Some docs record the absolute paths of the machine they ran on; treat those as provenance, not instructions.

## What is not in git, and why

| Not tracked | Reason | Where to get it |
|---|---|---|
| `data/processed/**/corpus.jsonl`, `chunks.jsonl`, `source_blocks.jsonl`; `data/raw/**/*.xml`; text transport copies under `results/` | Article text is not redistributed (`public_redistribution_authorized=false` in the dataset manifest; see the paper's Ethics statement) | Rebuild from public PubMed endpoints: `scripts/47_build_authoritative_corpus.py`, then `scripts/50_restore_verified_fulltext.py` |
| Blank annotation sheets and per-document evidence under `annotations/` | Contain candidate text; no judgments exist | Regenerate with `scripts/42_prepare_validation_kit.py` |
| `results/baselines/abstention_head.pkl`, legacy dense/BM25 indices | Serialized binaries | Hugging Face mirror (abstention head); legacy indices are rebuilt by `scripts/05`–`06` |

Everything needed to reproduce the paper's numbers without network access — saved rankings, judgments, disclosure replicate arrays, identity comparisons, hashes — is tracked here and mirrored on Hugging Face.

## Reproduce

```sh
pip install -r requirements.txt          # Python 3.11; NumPy, SciPy, pytest suffice for the audits

# Full software suite (expected: 311 passed, 0 failed/skipped)
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider

# Fail-closed submission gate (expected: exit 2, BLOCKED — that is the correct answer)
python scripts/45_submission_readiness.py --require-ready

# Historical label sensitivity (strict mode fails on identity defects by design; use the explicit forensic mode)
python scripts/44_analyze_historical_qrel_sensitivity.py --help

# Controlled nested disclosure, 1,000 replicates (choose a fresh --output; existing outputs are never overwritten)
python scripts/51_qrel_disclosure_experiment.py --output results/readiness/qrel_disclosure/rerun

# Authoritative re-fetch, restoration, verified BM25 (see --help; network needed for the re-fetch)
python scripts/47_build_authoritative_corpus.py --help
python scripts/50_restore_verified_fulltext.py --help
python scripts/49_index_and_run_verified_bm25.py --help
```

Every result directory carries a manifest with SHA-256 hashes of its inputs, code, and outputs. Provider credentials, if you ever run the (unexecuted) live budget conditions, are read from the environment only; none are stored in this repository.

## Historical materials

`paper/neurips_2026.tex` and `paper/iclr2027_submission/` are the original benchmark manuscripts. Their leaderboard claims are superseded by the audit: they were computed on the identity-conflicted corpus with seed or pooled automated labels, and the audit shows the winner depends on the label set. They are kept, unmodified, because the paper cites them as the construction record. The `v2` collection in `data/benchmark/v2/` and the two fine-tuned controller adapters on Hugging Face are likewise historical artifacts that were not re-evaluated.

## Citation

```bibtex
@article{bansal2026fixedrankings,
  title   = {Fixed Rankings, Moving Leaders: A Relevance-Label and Source-Identity Audit of Scientific Retrieval Evaluation},
  author  = {Bansal, Aayam and Gangwani, Ishaan},
  year    = {2026},
  note    = {Preprint. Code and data: https://github.com/aayambansal/CER-Bench}
}
```

## License

Code: Apache License 2.0 (see `LICENSE`). Task files, saved rankings, judgments, and audit outputs: Apache-2.0 as well, matching the Hugging Face dataset card. PubMed metadata and any article text you rebuild locally remain subject to NLM/PMC and publisher terms; this repository does not grant rights over them.
