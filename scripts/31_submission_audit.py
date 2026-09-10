#!/usr/bin/env python3
"""Recompute the paper-facing integrity checks for the ICLR 2027 submission.

The script uses only saved benchmark and retrieval artifacts.  It intentionally
does not call external models or APIs.  Its JSON output is both a provenance
record for the paper and the input to ``32_make_submission_figure.py``.
"""

from __future__ import annotations

import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "baselines"
OUTPUT = ROOT / "results" / "paper_tables" / "submission_audit.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & gold) / len(gold) if gold else 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int = 10) -> float:
    if not gold:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, doc_id in enumerate(retrieved[:k])
        if doc_id in gold
    )
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(gold))))
    return dcg / ideal


def reciprocal_rank(retrieved: list[str], gold: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def score(
    tasks: dict[str, dict[str, Any]],
    retrievals: list[dict[str, Any]],
    allowed_ids: set[str] | None = None,
) -> dict[str, float | int]:
    metrics: defaultdict[str, list[float]] = defaultdict(list)
    for row in retrievals:
        task_id = row["task_id"]
        if allowed_ids is not None and task_id not in allowed_ids:
            continue
        gold = set(tasks[task_id].get("supporting_doc_ids", []))
        if not gold:
            continue
        retrieved = row.get("retrieved_docs", [])
        for k in (5, 10, 20):
            metrics[f"recall@{k}"].append(recall_at_k(retrieved, gold, k))
        metrics["ndcg@10"].append(ndcg_at_k(retrieved, gold))
        metrics["mrr"].append(reciprocal_rank(retrieved, gold))
    return {
        "n_tasks": len(metrics["recall@5"]),
        **{name: round(mean(values), 4) for name, values in metrics.items()},
    }


def per_task_recall(
    tasks: dict[str, dict[str, Any]], retrievals: list[dict[str, Any]], k: int
) -> dict[str, float]:
    values = {}
    for row in retrievals:
        task = tasks.get(row["task_id"])
        if not task:
            continue
        gold = set(task.get("supporting_doc_ids", []))
        if gold:
            values[row["task_id"]] = recall_at_k(row.get("retrieved_docs", []), gold, k)
    return values


def paired_bootstrap(
    left: dict[str, float],
    right: dict[str, float],
    seed: int = 42,
    n_boot: int = 10_000,
) -> dict[str, float | int | list[float]]:
    task_ids = sorted(set(left) & set(right))
    differences = np.asarray([left[task_id] - right[task_id] for task_id in task_ids])
    rng = np.random.default_rng(seed)
    samples = differences[
        rng.integers(0, len(differences), size=(n_boot, len(differences)))
    ].mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    p_two_sided = 2 * min(float(np.mean(samples <= 0)), float(np.mean(samples >= 0)))
    return {
        "n_tasks": len(task_ids),
        "mean_difference": round(float(differences.mean()), 4),
        "ci95": [round(float(lower), 4), round(float(upper), 4)],
        "p_two_sided": round(min(p_two_sided, 1.0), 4),
        "bootstrap_samples": n_boot,
        "seed": seed,
    }


def doc_id_from_chunk(chunk_id: str) -> str:
    section_types = {
        "abstract",
        "other",
        "methods",
        "results",
        "discussion",
        "introduction",
        "conclusion",
        "caption",
        "table",
        "body",
        "results_discussion",
    }
    parts = chunk_id.split("_")
    for index in range(1, len(parts)):
        if "_".join(parts[index:]).split("_")[0] in section_types:
            return "_".join(parts[:index])
    return "_".join(parts[:-2]) if len(parts) >= 3 else chunk_id


def bm25_v2_retrievals(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-run exact saved-index BM25 retrieval for the 70 v2 test tasks."""
    index_path = DATA / "processed" / "indices" / "bm25" / "bm25_index.pkl"
    ids_path = DATA / "processed" / "indices" / "bm25" / "chunk_ids.json"
    with index_path.open("rb") as handle:
        bm25 = pickle.load(handle)
    chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))

    output: list[dict[str, Any]] = []
    for task in tasks:
        tokens = [
            token
            for token in re.sub(r"[^a-z0-9\s\-]", " ", task["question"].lower()).split()
            if len(token) > 1
        ]
        scores = bm25.get_scores(tokens)
        ranked_chunks = np.argsort(scores)[::-1]
        docs: list[str] = []
        seen: set[str] = set()
        for index in ranked_chunks:
            doc_id = doc_id_from_chunk(chunk_ids[int(index)])
            if doc_id not in seen:
                seen.add(doc_id)
                docs.append(doc_id)
            if len(docs) == 20:
                break
        output.append({"task_id": task["task_id"], "retrieved_docs": docs})
    return output


def normalized_verdict(value: str) -> str:
    text = value.strip().upper()
    if text.startswith("NOT_RELEVANT"):
        return "NOT_RELEVANT"
    if text.startswith("RELEVANT"):
        return "RELEVANT"
    return "UNPARSEABLE"


def main() -> None:
    split_rows = {
        split: read_jsonl(DATA / "benchmark" / f"{split}.jsonl")
        for split in ("train", "dev", "test")
    }
    test_tasks = {row["task_id"]: row for row in split_rows["test"]}

    family_counts = {
        split: dict(sorted(Counter(row["task_family"] for row in rows).items()))
        for split, rows in split_rows.items()
    }
    gold_docs = {
        split: {doc_id for row in rows for doc_id in row.get("supporting_doc_ids", [])}
        for split, rows in split_rows.items()
    }
    overlap = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        shared = gold_docs[left] & gold_docs[right]
        overlap[f"{left}_{right}"] = {
            "n_shared_gold_documents": len(shared),
            "jaccard": round(len(shared) / len(gold_docs[left] | gold_docs[right]), 4),
        }

    normalized_queries: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for split, rows in split_rows.items():
        for row in rows:
            key = re.sub(r"\s+", " ", row["question"].strip().lower())
            normalized_queries[key].append({"split": split, "task_id": row["task_id"]})
    duplicate_queries = [
        items for items in normalized_queries.values() if len(items) > 1
    ]

    train_dev_gold = gold_docs["train"] | gold_docs["dev"]
    disjoint_test_ids = {
        row["task_id"]
        for row in split_rows["test"]
        if row.get("supporting_doc_ids")
        and not (set(row["supporting_doc_ids"]) & train_dev_gold)
    }

    method_files = {
        "BM25": "bm25_test.jsonl",
        "SPECTER2": "dense_test.jsonl",
        "BGE": "bge_test.jsonl",
        "E5-large": "e5large_test.jsonl",
        "MedCPT": "medcpt_test.jsonl",
        "SPLADE": "splade_test.jsonl",
        "BM25+RM3": "rm3_test.jsonl",
        "Hybrid": "hybrid_test.jsonl",
        "Hybrid+CE": "hybrid_reranked_test.jsonl",
        "BM25+BGE-reranker": "bge_reranker_test.jsonl",
        "Agent T=1": "agent_single_step_test.jsonl",
        "Agent T=3": "agent_test.jsonl",
    }
    v1_scores = {}
    disjoint_scores = {}
    retrieval_cache = {}
    for method, filename in method_files.items():
        retrievals = read_jsonl(RESULTS / filename)
        retrieval_cache[method] = retrievals
        v1_scores[method] = score(test_tasks, retrievals)
        disjoint_scores[method] = score(test_tasks, retrievals, disjoint_test_ids)

    bootstrap_comparisons = {}
    for k in (10, 20):
        agent_values = per_task_recall(test_tasks, retrieval_cache["Agent T=3"], k)
        for comparator in ("BM25", "SPLADE", "Hybrid", "Agent T=1"):
            comparator_values = per_task_recall(
                test_tasks, retrieval_cache[comparator], k
            )
            bootstrap_comparisons[f"Agent T=3_minus_{comparator}_recall@{k}"] = (
                paired_bootstrap(agent_values, comparator_values)
            )

    adjudications = read_jsonl(RESULTS / "gold_adjudication.jsonl")
    adjudication_counts = Counter(
        normalized_verdict(row.get("judgment", "")) for row in adjudications
    )
    expanded_gold = json.loads(
        (RESULTS / "expanded_gold.json").read_text(encoding="utf-8")
    )
    expanded_sizes = [len(doc_ids) for doc_ids in expanded_gold.values()]
    pooled_scores_raw = json.loads(
        (RESULTS / "scores_test_adjudicated.json").read_text(encoding="utf-8")
    )
    pooled_name_map = {
        "bm25": "BM25",
        "dense": "SPECTER2",
        "bge": "BGE",
        "e5large": "E5-large",
        "medcpt": "MedCPT",
        "splade": "SPLADE",
        "hybrid": "Hybrid",
        "hybrid_reranked": "Hybrid+CE",
        "bge_reranker": "BM25+BGE-reranker",
        "agent": "Agent T=3",
    }
    pooled_scores = {
        pooled_name_map[key]: value["expanded"]
        for key, value in pooled_scores_raw.items()
        if key in pooled_name_map
    }

    v2_rows = read_jsonl(DATA / "benchmark" / "v2" / "test.jsonl")
    v2_tasks = {row["task_id"]: row for row in v2_rows}
    v2_agent = read_jsonl(DATA / "benchmark" / "v2" / "agent_test.jsonl")
    v2_bm25 = bm25_v2_retrievals(v2_rows)
    v2_scores = {
        "BM25": score(v2_tasks, v2_bm25),
        "Agent T=3": score(v2_tasks, v2_agent),
    }

    corpus_stats = json.loads(
        (DATA / "processed" / "corpus_stats.json").read_text(encoding="utf-8")
    )
    abstention = json.loads(
        (RESULTS / "abstention_metrics.json").read_text(encoding="utf-8")
    )

    report = {
        "audit_version": 1,
        "corpus": corpus_stats,
        "v1": {
            "split_counts": {split: len(rows) for split, rows in split_rows.items()},
            "family_counts": family_counts,
            "non_abstention_test_tasks": sum(
                bool(row.get("supporting_doc_ids")) for row in split_rows["test"]
            ),
            "mean_seed_qrels_per_non_abstention_test_task": round(
                mean(
                    [
                        len(row["supporting_doc_ids"])
                        for row in split_rows["test"]
                        if row.get("supporting_doc_ids")
                    ]
                ),
                4,
            ),
            "exact_duplicate_query_groups": duplicate_queries,
            "gold_document_overlap": overlap,
            "document_disjoint_test_tasks": len(disjoint_test_ids),
            "scores": v1_scores,
            "document_disjoint_scores": disjoint_scores,
            "paired_bootstrap": bootstrap_comparisons,
        },
        "pooled_qrels": {
            "n_judgments": len(adjudications),
            "normalization_rule": "strip, uppercase, then prefix-match NOT_RELEVANT before RELEVANT",
            "verdict_counts": dict(adjudication_counts),
            "mean_qrels_per_task": round(mean(expanded_sizes), 4),
            "min_qrels_per_task": min(expanded_sizes),
            "max_qrels_per_task": max(expanded_sizes),
            "scores": pooled_scores,
        },
        "v2_proxy_qrels": {
            "description": "LLM-filtered BM25 topical clusters used as higher-density proxy qrels; not exhaustive document-level judgments against each final query",
            "n_test_tasks": len(v2_rows),
            "mean_qrels_per_task": round(
                mean([len(row.get("supporting_doc_ids", [])) for row in v2_rows]), 4
            ),
            "scores": v2_scores,
        },
        "abstention_diagnostic": {
            key: abstention[key]
            for key in ("auroc", "auprc", "precision", "recall", "f1", "aurc")
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(
        json.dumps(
            {
                "v1_test_non_abstention": report["v1"]["non_abstention_test_tasks"],
                "document_disjoint_test_tasks": len(disjoint_test_ids),
                "pooled_verdict_counts": report["pooled_qrels"]["verdict_counts"],
                "v2_scores": v2_scores,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
