#!/usr/bin/env python3
"""Script 08: Verify generated tasks against the corpus.

Checks:
  1. Supporting evidence exists in corpus (keyword + constraint match)
  2. Abstention tasks have no perfect matches
  3. Required constraints are satisfiable
  4. Attaches hard negative doc IDs from BM25 top-k that fail constraints

Usage:
    python scripts/08_verify_tasks.py
"""

import json
import pickle
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import get_data_dir


def tokenize_simple(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def find_supporting_docs(
    query: str,
    constraints: list[dict],
    bm25,
    chunk_ids: list[str],
    chunks_by_id: dict,
    corpus_by_doc: dict,
    top_k: int = 50,
) -> tuple[list[str], list[str], list[dict]]:
    """Find potential supporting and negative docs via BM25.

    Returns:
        (supporting_doc_ids, hard_negative_doc_ids, supporting_passages)
    """
    query_tokens = tokenize_simple(query)
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    supporting = []
    hard_negatives = []
    passages = []
    seen_docs = set()

    for idx in top_indices:
        cid = chunk_ids[idx]
        chunk = chunks_by_id.get(cid, {})
        doc_id = chunk.get("doc_id", "")

        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)

        doc = corpus_by_doc.get(doc_id, {})
        text_lower = chunk.get("text", "").lower()

        # Check constraints
        constraints_met = 0
        for c in constraints:
            val = c.get("value", "").lower()
            if val and val in text_lower:
                constraints_met += 1

        if constraints and constraints_met == len(constraints):
            supporting.append(doc_id)
            passages.append({
                "doc_id": doc_id,
                "section": chunk.get("section_type", ""),
                "text": chunk.get("text", "")[:500],
                "constraint_satisfied": [c["type"] for c in constraints],
            })
        elif constraints and constraints_met > 0:
            hard_negatives.append(doc_id)

    return supporting[:10], hard_negatives[:10], passages[:5]


def main():
    # Load BM25 index
    bm25_dir = get_data_dir("processed/indices/bm25")
    bm25_path = bm25_dir / "bm25_index.pkl"
    ids_path = bm25_dir / "chunk_ids.json"

    if not bm25_path.exists():
        print(f"Error: BM25 index not found. Run scripts/05_build_bm25_index.py first.")
        sys.exit(1)

    print("Loading BM25 index...")
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    with open(ids_path) as f:
        chunk_ids = json.load(f)

    # Load chunks
    print("Loading chunks...")
    chunks_by_id = {}
    with open(get_data_dir("processed") / "chunks.jsonl") as f:
        for line in f:
            chunk = json.loads(line)
            chunks_by_id[chunk["chunk_id"]] = chunk

    # Load corpus for doc-level info
    print("Loading corpus...")
    corpus_by_doc = {}
    with open(get_data_dir("processed") / "corpus.jsonl") as f:
        for line in f:
            doc = json.loads(line)
            corpus_by_doc[doc["doc_id"]] = doc

    # Load raw tasks
    tasks_path = get_data_dir("interim") / "raw_tasks.jsonl"
    if not tasks_path.exists():
        print(f"Error: {tasks_path} not found. Run scripts/07_generate_seed_tasks.py first.")
        sys.exit(1)

    print("Loading tasks...")
    tasks = []
    with open(tasks_path) as f:
        for line in f:
            tasks.append(json.loads(line))
    print(f"  Loaded {len(tasks)} tasks")

    # Verify each task
    print(f"\nVerifying {len(tasks)} tasks...")
    t0 = time.time()

    verified = []
    rejected = []

    for i, task in enumerate(tasks):
        question = task.get("question", "")
        constraints = task.get("required_constraints", [])
        family = task.get("task_family", "")

        # Find supporting docs
        supporting, negatives, passages = find_supporting_docs(
            question, constraints, bm25, chunk_ids, chunks_by_id, corpus_by_doc
        )

        task["supporting_doc_ids"] = supporting
        task["hard_negative_doc_ids"] = negatives
        task["supporting_passages"] = passages

        # Verification logic
        if family == "abstention":
            # Abstention tasks SHOULD have no perfect matches
            if len(supporting) == 0:
                task["verification_status"] = "auto_verified"
                verified.append(task)
            elif len(supporting) <= 2:
                task["verification_status"] = "needs_review"
                task["verification_note"] = f"Found {len(supporting)} possible matches — may not be truly unsupported"
                verified.append(task)
            else:
                task["verification_status"] = "rejected"
                task["verification_note"] = f"Found {len(supporting)} matches — not a valid abstention task"
                rejected.append(task)
        else:
            # Non-abstention tasks SHOULD have supporting evidence
            if len(supporting) >= 1:
                task["verification_status"] = "auto_verified"
                verified.append(task)
            elif len(negatives) >= 1:
                task["verification_status"] = "needs_review"
                task["verification_note"] = "No perfect matches but partial matches found"
                verified.append(task)
            else:
                task["verification_status"] = "rejected"
                task["verification_note"] = "No supporting or partial evidence found in corpus"
                rejected.append(task)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(tasks)}] verified={len(verified)} rejected={len(rejected)}")

    t1 = time.time()

    # Save
    verified_path = get_data_dir("interim") / "verified_tasks.jsonl"
    rejected_path = get_data_dir("interim") / "rejected_tasks.jsonl"

    with open(verified_path, "w") as f:
        for task in verified:
            f.write(json.dumps(task, default=str) + "\n")

    with open(rejected_path, "w") as f:
        for task in rejected:
            f.write(json.dumps(task, default=str) + "\n")

    # Stats
    status_counts = {}
    family_verified = {}
    for t in verified:
        s = t.get("verification_status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1
        fam = t.get("task_family", "unknown")
        family_verified[fam] = family_verified.get(fam, 0) + 1

    print(f"\n{'='*60}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  Verified: {len(verified)}")
    print(f"  Rejected: {len(rejected)}")
    print(f"  Time: {t1-t0:.1f}s")
    print(f"\n  By status:")
    for s, c in sorted(status_counts.items()):
        print(f"    {c:>4}  {s}")
    print(f"\n  Verified by family:")
    for fam, c in sorted(family_verified.items()):
        print(f"    {c:>4}  {fam}")
    print(f"\n  Output:")
    print(f"    Verified: {verified_path}")
    print(f"    Rejected: {rejected_path}")


if __name__ == "__main__":
    main()
