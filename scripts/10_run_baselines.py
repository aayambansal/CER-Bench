#!/usr/bin/env python3
"""Script 10: Run baseline retrieval methods on the benchmark.

Baselines: BM25, Dense (SPECTER2), Hybrid (BM25+Dense RRF), Hybrid+Reranker.

Usage:
    python scripts/10_run_baselines.py [--split dev] [--top-k 20]
"""

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import get_data_dir, load_config


def tokenize_simple(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def load_bm25():
    bm25_dir = get_data_dir("processed/indices/bm25")
    with open(bm25_dir / "bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(bm25_dir / "chunk_ids.json") as f:
        chunk_ids = json.load(f)
    return bm25, chunk_ids


def load_dense():
    dense_dir = get_data_dir("processed/indices/dense")
    index_path = dense_dir / "specter2.index"
    if not index_path.exists():
        return None, None, None
    import faiss
    index = faiss.read_index(str(index_path))
    embeddings = np.load(dense_dir / "embeddings.npy")
    with open(dense_dir / "chunk_ids.json") as f:
        chunk_ids = json.load(f)
    return index, embeddings, chunk_ids


def search_bm25(query: str, bm25, chunk_ids: list, top_k: int = 20) -> list[tuple[str, float]]:
    tokens = tokenize_simple(query)
    scores = bm25.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(chunk_ids[i], float(scores[i])) for i in top_idx]


def search_dense(query: str, index, tokenizer, model, device, chunk_ids: list, top_k: int = 20) -> list[tuple[str, float]]:
    import torch
    inputs = tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
        q_emb = out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    scores, indices = index.search(q_emb, top_k)
    return [(chunk_ids[indices[0][i]], float(scores[0][i])) for i in range(len(indices[0]))]


def reciprocal_rank_fusion(result_lists: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """RRF fusion of multiple ranked lists."""
    scores = {}
    for results in result_lists:
        for rank, (doc_id, _) in enumerate(results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def extract_doc_ids(chunk_results: list[tuple[str, float]]) -> list[str]:
    """Extract unique doc IDs from chunk results, preserving rank order."""
    seen = set()
    doc_ids = []
    for chunk_id, score in chunk_results:
        # chunk_id format: {doc_id}_{section}_{idx}
        parts = chunk_id.rsplit("_", 2)
        doc_id = parts[0] if len(parts) >= 2 else chunk_id
        if doc_id not in seen:
            seen.add(doc_id)
            doc_ids.append(doc_id)
    return doc_ids


def main():
    parser = argparse.ArgumentParser(description="Run baseline retrieval")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    config = load_config("retrieval")

    # Load benchmark split
    split_path = get_data_dir("benchmark") / f"{args.split}.jsonl"
    if not split_path.exists():
        print(f"Error: {split_path} not found. Run scripts/09_make_splits.py first.")
        sys.exit(1)

    tasks = []
    with open(split_path) as f:
        for line in f:
            tasks.append(json.loads(line))
    print(f"Loaded {len(tasks)} {args.split} tasks")

    # Load BM25
    print("Loading BM25 index...")
    bm25, bm25_chunk_ids = load_bm25()

    # Load Dense (optional)
    print("Loading dense index...")
    dense_index, dense_embs, dense_chunk_ids = load_dense()
    has_dense = dense_index is not None

    # Load SPECTER2 model for query embedding (if dense available)
    tokenizer = model = device = None
    if has_dense:
        import torch
        from transformers import AutoTokenizer, AutoModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        model = AutoModel.from_pretrained("allenai/specter2_base").to(device).eval()
        print(f"  SPECTER2 loaded on {device}")
    else:
        print("  Dense index not found — running BM25-only baselines")

    # Run baselines
    baselines = ["bm25"]
    if has_dense:
        baselines.extend(["dense", "hybrid"])

    results = {b: [] for b in baselines}

    print(f"\nRunning baselines: {baselines}")
    t0 = time.time()

    for i, task in enumerate(tasks):
        question = task.get("question", "")

        # BM25
        bm25_results = search_bm25(question, bm25, bm25_chunk_ids, args.top_k * 5)
        bm25_docs = extract_doc_ids(bm25_results)[:args.top_k]
        results["bm25"].append({
            "task_id": task.get("task_id"),
            "retrieved_docs": bm25_docs,
            "retrieved_chunks": [(cid, s) for cid, s in bm25_results[:args.top_k]],
        })

        if has_dense:
            # Dense
            dense_results = search_dense(question, dense_index, tokenizer, model, device, dense_chunk_ids, args.top_k * 5)
            dense_docs = extract_doc_ids(dense_results)[:args.top_k]
            results["dense"].append({
                "task_id": task.get("task_id"),
                "retrieved_docs": dense_docs,
                "retrieved_chunks": [(cid, s) for cid, s in dense_results[:args.top_k]],
            })

            # Hybrid (RRF)
            hybrid_results = reciprocal_rank_fusion([bm25_results, dense_results])
            hybrid_docs = extract_doc_ids(hybrid_results)[:args.top_k]
            results["hybrid"].append({
                "task_id": task.get("task_id"),
                "retrieved_docs": hybrid_docs,
                "retrieved_chunks": hybrid_results[:args.top_k],
            })

        if (i + 1) % 20 == 0 or i + 1 == len(tasks):
            print(f"  [{i+1}/{len(tasks)}] tasks processed")

    t1 = time.time()

    # Save results
    output_dir = get_data_dir("") / ".." / "results" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    for baseline, res in results.items():
        path = output_dir / f"{baseline}_{args.split}.jsonl"
        with open(path, "w") as f:
            for r in res:
                f.write(json.dumps(r, default=str) + "\n")

    print(f"\n{'='*60}")
    print(f"BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"  Split: {args.split}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Baselines: {baselines}")
    print(f"  Time: {t1-t0:.1f}s")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
