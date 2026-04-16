#!/usr/bin/env python3
"""Script 05: Build BM25 index over corpus chunks.

Uses rank_bm25 (pure Python, no Java required) for the pilot.
Switch to Pyserini for production scale.

Usage:
    python scripts/05_build_bm25_index.py
"""

import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import get_data_dir


def tokenize_simple(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    import re
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    tokens = text.split()
    # Remove very short tokens
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


def main():
    chunks_path = get_data_dir("processed") / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"Error: {chunks_path} not found. Run scripts/04_build_corpus.py first.")
        sys.exit(1)

    print("Loading chunks...")
    chunks = []
    chunk_ids = []
    chunk_texts = []

    with open(chunks_path) as f:
        for line in f:
            chunk = json.loads(line)
            chunks.append(chunk)
            chunk_ids.append(chunk["chunk_id"])
            # Combine heading + text for indexing
            text = ""
            if chunk.get("section_heading"):
                text += chunk["section_heading"] + " "
            text += chunk.get("text", "")
            chunk_texts.append(text)

    print(f"  Loaded {len(chunks)} chunks")

    # Tokenize
    print("Tokenizing...")
    t0 = time.time()
    tokenized = [tokenize_simple(text) for text in chunk_texts]
    t1 = time.time()
    print(f"  Tokenized in {t1-t0:.1f}s")

    # Build BM25 index
    print("Building BM25 index...")
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi(tokenized)
    t2 = time.time()
    print(f"  Built in {t2-t1:.1f}s")

    # Save index
    index_dir = get_data_dir("processed/indices/bm25")
    index_path = index_dir / "bm25_index.pkl"
    ids_path = index_dir / "chunk_ids.json"

    with open(index_path, "wb") as f:
        pickle.dump(bm25, f)
    with open(ids_path, "w") as f:
        json.dump(chunk_ids, f)

    # Test query
    print("\nTest query: 'CRISPR gene editing efficiency human cells'")
    query_tokens = tokenize_simple("CRISPR gene editing efficiency human cells")
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]

    for rank, idx in enumerate(top_indices):
        c = chunks[idx]
        print(f"  #{rank+1} [score={scores[idx]:.2f}] {c['chunk_id']}")
        print(f"       {c['text'][:120]}...")

    t3 = time.time()
    print(f"\n{'='*60}")
    print(f"BM25 INDEX BUILT")
    print(f"{'='*60}")
    print(f"  Chunks indexed: {len(chunks)}")
    print(f"  Index size: {index_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Output: {index_path}")
    print(f"  Total time: {t3-t0:.1f}s")


if __name__ == "__main__":
    main()
