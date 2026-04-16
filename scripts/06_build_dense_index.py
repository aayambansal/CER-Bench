#!/usr/bin/env python3
"""Script 06: Build SPECTER2 dense index over corpus chunks.

Embeds all chunks with SPECTER2 (allenai/specter2 + proximity adapter)
and builds a FAISS index for similarity search.

Can run on CPU (slower) or GPU (fast).

Usage:
    python scripts/06_build_dense_index.py [--batch-size 64] [--device cuda]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import get_data_dir


def main():
    parser = argparse.ArgumentParser(description="Build SPECTER2 dense index")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (auto-detect)")
    args = parser.parse_args()

    chunks_path = get_data_dir("processed") / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"Error: {chunks_path} not found. Run scripts/04_build_corpus.py first.")
        sys.exit(1)

    # Load chunks
    print("Loading chunks...")
    chunk_ids = []
    chunk_texts = []
    with open(chunks_path) as f:
        for line in f:
            chunk = json.loads(line)
            chunk_ids.append(chunk["chunk_id"])
            text = ""
            if chunk.get("section_heading"):
                text += chunk["section_heading"] + ": "
            text += chunk.get("text", "")
            # SPECTER2 max length is 512 tokens, truncate long text
            chunk_texts.append(text[:2000])
    print(f"  Loaded {len(chunk_ids)} chunks")

    # Detect device
    import torch
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Load SPECTER2 with retrieval adapter
    print("Loading SPECTER2 model...")
    t0 = time.time()
    from transformers import AutoTokenizer, AutoModel, AutoAdapterModel

    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")

    # Try loading with adapter support first, fall back to base model
    try:
        model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        model.load_adapter("allenai/specter2", source="hf", load_as="specter2_proximity", set_active=True)
        print("  Loaded with proximity adapter")
    except Exception:
        print("  Adapter loading failed, using base model")
        model = AutoModel.from_pretrained("allenai/specter2_base")

    model = model.to(device)
    model.eval()
    t1 = time.time()
    print(f"  Model loaded in {t1-t0:.1f}s")

    # Embed in batches
    print(f"Embedding {len(chunk_texts)} chunks (batch_size={args.batch_size})...")
    all_embeddings = []

    for i in range(0, len(chunk_texts), args.batch_size):
        batch = chunk_texts[i : i + args.batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = embeddings.cpu().numpy()

        all_embeddings.append(embeddings)

        done = min(i + args.batch_size, len(chunk_texts))
        if done % 1000 == 0 or done == len(chunk_texts):
            elapsed = time.time() - t1
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  [{done}/{len(chunk_texts)}] {rate:.0f} chunks/s")

    t2 = time.time()
    embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
    print(f"  Embedding shape: {embeddings_matrix.shape}")
    print(f"  Embedding time: {t2-t1:.1f}s")

    # Normalize for cosine similarity (inner product on normalized vectors = cosine)
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    embeddings_matrix = embeddings_matrix / norms

    # Build FAISS index
    print("Building FAISS index...")
    import faiss

    dim = embeddings_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)
    index.add(embeddings_matrix)
    t3 = time.time()
    print(f"  FAISS index built in {t3-t2:.1f}s")

    # Save
    index_dir = get_data_dir("processed/indices/dense")
    faiss.write_index(index, str(index_dir / "specter2.index"))
    np.save(index_dir / "embeddings.npy", embeddings_matrix)
    with open(index_dir / "chunk_ids.json", "w") as f:
        json.dump(chunk_ids, f)

    # Test query
    print("\nTest query: 'CRISPR gene editing efficiency in human cells'")
    query = "CRISPR gene editing efficiency in human cells"
    q_inputs = tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        q_out = model(**q_inputs)
        q_emb = q_out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

    scores, indices = index.search(q_emb, 5)
    for rank in range(5):
        idx = indices[0][rank]
        score = scores[0][rank]
        print(f"  #{rank+1} [score={score:.4f}] {chunk_ids[idx]}")

    t4 = time.time()
    print(f"\n{'='*60}")
    print(f"DENSE INDEX BUILT")
    print(f"{'='*60}")
    print(f"  Chunks indexed: {len(chunk_ids)}")
    print(f"  Embedding dim: {dim}")
    print(f"  Index file: {index_dir / 'specter2.index'}")
    print(f"  Total time: {t4-t0:.1f}s")


if __name__ == "__main__":
    main()
