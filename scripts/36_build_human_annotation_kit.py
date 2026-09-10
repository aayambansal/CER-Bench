#!/usr/bin/env python3
"""Build blinded human-qrel, structural-role, and abstention annotation sheets."""

from __future__ import annotations

import csv
import json
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "baselines"
OUT = ROOT / "annotations" / "human_qrel"
SEED = 20260826
SECTION_TYPES = {
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
METHOD_FILES = (
    "bm25_test.jsonl",
    "dense_test.jsonl",
    "bge_test.jsonl",
    "e5large_test.jsonl",
    "medcpt_test.jsonl",
    "splade_test.jsonl",
    "rm3_test.jsonl",
    "hybrid_test.jsonl",
    "hybrid_reranked_test.jsonl",
    "bge_reranker_test.jsonl",
    "agent_single_step_test.jsonl",
    "agent_test.jsonl",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def extract_doc_id(chunk_id: str) -> str:
    parts = chunk_id.split("_")
    for i in range(1, len(parts)):
        suffix = "_".join(parts[i:])
        if any(suffix.startswith(section) for section in SECTION_TYPES):
            return "_".join(parts[:i])
    return "_".join(parts[:-2]) if len(parts) >= 3 else chunk_id


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.sub(r"[^a-z0-9\s\-]", " ", text.lower()).split()
        if len(token) > 1
    ]


def source_url(doc_id: str) -> str:
    if doc_id.startswith("PMC"):
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{doc_id}/"
    if doc_id.isdigit():
        return f"https://pubmed.ncbi.nlm.nih.gov/{doc_id}/"
    return ""


def role_schema(family: str, constraints: list) -> str:
    constraint_ids = [f"C{i + 1}" for i in range(len(constraints))]
    schemas = {
        "constraint": {"covered_constraints": constraint_ids},
        "comparative": {"side": "A|B|BOTH|CONTEXT", "comparison_axis": "text"},
        "contradiction": {
            "role": "FINDING_A|FINDING_B|RECONCILIATION|CONTEXT",
            "condition": "text",
        },
        "multihop": {"hop_indices": [1], "bridge_entity": "text"},
        "temporal": {"temporal_bins": ["EARLY"], "evidence_year": 2020},
        "aggregation": {
            "values": [{"value": "text", "unit": "text", "condition": "text"}]
        },
        "negative": {
            "negative_role": "EXPLICIT_NULL|NEGATIVE_DIRECTION|FAILED_REPLICATION|CONTEXT"
        },
        "abstention": {
            "jointly_supports_query": "YES|NO|UNCERTAIN",
            "violated_constraints": constraint_ids,
        },
    }
    return json.dumps(schemas[family], ensure_ascii=False)


def automatic_label_map() -> dict[tuple[str, str], str]:
    labels = {}
    path = RESULTS / "gold_adjudication.jsonl"
    for row in read_jsonl(path):
        judgment = str(row.get("judgment", "")).strip().upper()
        label = (
            "RELEVANT"
            if judgment.startswith("RELEVANT")
            else "NOT_RELEVANT"
            if judgment.startswith("NOT_RELEVANT")
            else "UNKNOWN"
        )
        labels[(row["task_id"], str(row["doc_id"]))] = label
    return labels


def bm25_tail_candidates(
    tasks: list[dict], pooled: dict[str, set[str]], n: int = 5
) -> dict[str, list[str]]:
    base = DATA / "processed" / "indices" / "bm25"
    with (base / "bm25_index.pkl").open("rb") as handle:
        index = pickle.load(handle)
    chunk_ids = json.load((base / "chunk_ids.json").open())
    output = {}
    for task in tasks:
        scores = index.get_scores(tokenize(task["question"]))
        order = np.argsort(-np.asarray(scores), kind="stable")
        seen, tail = set(), []
        for idx in order:
            doc_id = extract_doc_id(chunk_ids[int(idx)])
            if doc_id in seen:
                continue
            seen.add(doc_id)
            if doc_id not in pooled.get(task["task_id"], set()):
                tail.append(doc_id)
            if len(tail) == n:
                break
        output[task["task_id"]] = tail
    return output


def write_sheet(path: Path, rows: list[dict], include_provenance: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    public = [
        "pair_id",
        "task_id",
        "task_family",
        "question",
        "required_constraints",
        "doc_id",
        "title",
        "year",
        "abstract",
        "source_url",
        "role_schema",
        "relevance",
        "evidence_span",
        "role_json",
        "confidence",
        "notes",
    ]
    master = public[:11] + ["candidate_source", "automatic_label"] + public[11:]
    fields = master if include_provenance else public
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = read_jsonl(DATA / "benchmark" / "test.jsonl")
    docs = {
        row["doc_id"]: row for row in read_jsonl(DATA / "processed" / "corpus.jsonl")
    }
    auto = automatic_label_map()
    pooled: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(RESULTS / "gold_adjudication.jsonl"):
        pooled[row["task_id"]].add(str(row["doc_id"]))
    supported = [task for task in tasks if task.get("supporting_doc_ids")]
    tail = bm25_tail_candidates(supported, pooled, 5)

    qrel_rows = []
    for task in supported:
        tid = task["task_id"]
        candidates = [(doc_id, "pooled_top20") for doc_id in sorted(pooled[tid])]
        candidates += [(doc_id, "bm25_out_of_pool") for doc_id in tail[tid]]
        for doc_id, source in candidates:
            doc = docs.get(doc_id, {})
            qrel_rows.append(
                {
                    "pair_id": f"{tid}::{doc_id}",
                    "task_id": tid,
                    "task_family": task["task_family"],
                    "question": task["question"],
                    "required_constraints": json.dumps(
                        task.get("required_constraints", []), ensure_ascii=False
                    ),
                    "doc_id": doc_id,
                    "title": doc.get("title", ""),
                    "year": doc.get("year", ""),
                    "abstract": doc.get("abstract", ""),
                    "source_url": source_url(doc_id),
                    "candidate_source": source,
                    "automatic_label": auto.get((tid, doc_id), "UNJUDGED"),
                    "role_schema": role_schema(
                        task["task_family"], task.get("required_constraints", [])
                    ),
                    "relevance": "",
                    "evidence_span": "",
                    "role_json": "",
                    "confidence": "",
                    "notes": "",
                }
            )

    # Unsupported-query candidates: diverse top-20 union plus BM25 to 100 unique docs.
    result_by_task: dict[str, set[str]] = defaultdict(set)
    result_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for filename in METHOD_FILES:
        for row in read_jsonl(RESULTS / filename):
            tid = row["task_id"]
            for rank, doc_id in enumerate(map(str, row.get("retrieved_docs", [])[:20])):
                result_by_task[tid].add(doc_id)
                result_scores[tid][doc_id] += 1.0 / (60 + rank + 1)
    abstention = [task for task in tasks if task["task_family"] == "abstention"]
    abstention_tail = bm25_tail_candidates(abstention, result_by_task, 100)
    abstention_rows = []
    for task in abstention:
        tid = task["task_id"]
        candidates = sorted(
            result_by_task[tid],
            key=lambda doc_id: (-result_scores[tid][doc_id], doc_id),
        )
        candidates += [
            doc_id
            for doc_id in task.get("hard_negative_doc_ids", [])
            if doc_id not in candidates
        ]
        candidates += [
            doc_id for doc_id in abstention_tail[tid] if doc_id not in candidates
        ]
        candidates = candidates[:100]
        for doc_id in candidates:
            doc = docs.get(doc_id, {})
            abstention_rows.append(
                {
                    "pair_id": f"{tid}::{doc_id}",
                    "task_id": tid,
                    "task_family": "abstention",
                    "question": task["question"],
                    "required_constraints": json.dumps(
                        task.get("required_constraints", []), ensure_ascii=False
                    ),
                    "doc_id": doc_id,
                    "title": doc.get("title", ""),
                    "year": doc.get("year", ""),
                    "abstract": doc.get("abstract", ""),
                    "source_url": source_url(doc_id),
                    "candidate_source": "abstention_union_top20_or_bm25_tail",
                    "automatic_label": "UNSUPPORTED_PROXY",
                    "role_schema": role_schema(
                        "abstention", task.get("required_constraints", [])
                    ),
                    "relevance": "",
                    "evidence_span": "",
                    "role_json": "",
                    "confidence": "",
                    "notes": "",
                }
            )

    rng = random.Random(SEED)
    annotator_a, annotator_b = qrel_rows.copy(), qrel_rows.copy()
    rng.shuffle(annotator_a)
    rng.shuffle(annotator_b)
    abst_a, abst_b = abstention_rows.copy(), abstention_rows.copy()
    rng.shuffle(abst_a)
    rng.shuffle(abst_b)
    write_sheet(OUT / "qrel_master.csv", qrel_rows, True)
    write_sheet(OUT / "qrel_annotator_A.csv", annotator_a, False)
    write_sheet(OUT / "qrel_annotator_B.csv", annotator_b, False)
    write_sheet(OUT / "abstention_master.csv", abstention_rows, True)
    write_sheet(OUT / "abstention_annotator_A.csv", abst_a, False)
    write_sheet(OUT / "abstention_annotator_B.csv", abst_b, False)

    manifest = {
        "seed": SEED,
        "supported_queries": len(supported),
        "pooled_pairs": sum(len(pooled[task["task_id"]]) for task in supported),
        "out_of_pool_pairs": 5 * len(supported),
        "qrel_pairs_total": len(qrel_rows),
        "abstention_queries": len(abstention),
        "abstention_pairs_total": len(abstention_rows),
        "required_annotators_per_pair": 2,
        "expert_adjudication": "all relevance/role disagreements and any UNCERTAIN label",
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
