#!/usr/bin/env python3
"""Script 04: Build the final corpus — merge metadata + parsed text, chunk by structure.

Merges:
  - PubMed/OpenAlex metadata (from script 01)
  - Parsed full text (from script 03, where available)

Produces:
  - data/processed/corpus.jsonl  (one record per document)
  - data/processed/chunks.jsonl  (one record per chunk)
  - data/processed/corpus_stats.json

Usage:
    python scripts/04_build_corpus.py
"""

import json
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, get_data_dir


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English)."""
    return len(text) // 4


def chunk_document(doc: dict, config: dict) -> list[dict]:
    """Chunk a document by scientific structure.

    Args:
        doc: Merged document record.
        config: Chunking config from corpus.yaml.

    Returns:
        List of chunk dicts, each inheriting doc metadata.
    """
    chunks = []
    doc_id = doc.get("pmcid") or doc["pmid"]
    chunk_idx = 0

    # Shared metadata for all chunks
    meta = {
        "doc_id": doc_id,
        "pmid": doc.get("pmid", ""),
        "pmcid": doc.get("pmcid", ""),
        "year": doc.get("year"),
        "venue": doc.get("venue", ""),
        "mesh_terms": doc.get("mesh_terms", []),
        "seed_query": doc.get("seed_query", ""),
    }

    # Chunk 1: Abstract (always present if doc passed filtering)
    abstract = doc.get("abstract", "")
    if abstract:
        chunks.append({
            "chunk_id": f"{doc_id}_abstract_{chunk_idx}",
            **meta,
            "section_type": "abstract",
            "section_heading": "Abstract",
            "text": abstract[:2048],  # Cap at ~512 tokens
            "token_estimate": _estimate_tokens(abstract[:2048]),
            "position": chunk_idx,
        })
        chunk_idx += 1

    # Chunk from parsed sections (only for papers with full text)
    sections = doc.get("sections", [])
    for sec in sections:
        section_type = sec.get("section_type", "other")
        heading = sec.get("heading", "")
        text = sec.get("text", "")

        if not text or len(text) < 50:
            continue

        # Get max tokens for this chunk type (from config)
        max_tokens = 1024  # default
        for ct in config.get("chunk_types", []):
            if ct["type"] == "section":
                max_tokens = ct.get("max_tokens", 1024)
                break

        max_chars = max_tokens * 4  # rough conversion

        # If section is too long, split on paragraph boundaries
        if len(text) > max_chars:
            # Split into paragraphs (approximation)
            paragraphs = text.split(". ")
            current = ""
            for para in paragraphs:
                if len(current) + len(para) + 2 > max_chars and current:
                    chunks.append({
                        "chunk_id": f"{doc_id}_{section_type}_{chunk_idx}",
                        **meta,
                        "section_type": section_type,
                        "section_heading": heading,
                        "text": current.strip(),
                        "token_estimate": _estimate_tokens(current),
                        "position": chunk_idx,
                    })
                    chunk_idx += 1
                    current = para + ". "
                else:
                    current += para + ". "
            if current.strip():
                chunks.append({
                    "chunk_id": f"{doc_id}_{section_type}_{chunk_idx}",
                    **meta,
                    "section_type": section_type,
                    "section_heading": heading,
                    "text": current.strip(),
                    "token_estimate": _estimate_tokens(current),
                    "position": chunk_idx,
                })
                chunk_idx += 1
        else:
            chunks.append({
                "chunk_id": f"{doc_id}_{section_type}_{chunk_idx}",
                **meta,
                "section_type": section_type,
                "section_heading": heading,
                "text": text,
                "token_estimate": _estimate_tokens(text),
                "position": chunk_idx,
            })
            chunk_idx += 1

    # Chunk from figure captions
    for fig in doc.get("figure_captions", []):
        caption = fig.get("caption", "")
        if caption and len(caption) > 30:
            chunks.append({
                "chunk_id": f"{doc_id}_caption_{chunk_idx}",
                **meta,
                "section_type": "caption",
                "section_heading": fig.get("label", "Figure"),
                "text": caption[:1024],
                "token_estimate": _estimate_tokens(caption[:1024]),
                "position": chunk_idx,
            })
            chunk_idx += 1

    # Chunk from table text
    for tbl in doc.get("table_texts", []):
        text_parts = []
        if tbl.get("caption"):
            text_parts.append(tbl["caption"])
        if tbl.get("body_text"):
            text_parts.append(tbl["body_text"])
        text = " ".join(text_parts)
        if text and len(text) > 30:
            chunks.append({
                "chunk_id": f"{doc_id}_table_{chunk_idx}",
                **meta,
                "section_type": "table",
                "section_heading": tbl.get("label", "Table"),
                "text": text[:2048],
                "token_estimate": _estimate_tokens(text[:2048]),
                "position": chunk_idx,
            })
            chunk_idx += 1

    return chunks


def main():
    config = load_config("corpus")
    chunking_config = config.get("chunking", {})

    # Load metadata
    metadata_path = get_data_dir("raw/metadata") / "pubmed_openalex_metadata.jsonl"
    if not metadata_path.exists():
        print(f"Error: metadata not found at {metadata_path}")
        sys.exit(1)

    print("Loading metadata...")
    metadata = {}
    with open(metadata_path) as f:
        for line in f:
            rec = json.loads(line)
            metadata[rec["pmid"]] = rec

    print(f"  Loaded {len(metadata)} metadata records")

    # Load parsed full text (if available)
    parsed_path = get_data_dir("interim/parsed") / "parsed_documents.jsonl"
    parsed = {}
    if parsed_path.exists():
        print("Loading parsed full text...")
        with open(parsed_path) as f:
            for line in f:
                doc = json.loads(line)
                parsed[doc["pmcid"]] = doc
        print(f"  Loaded {len(parsed)} parsed documents")
    else:
        print("  No parsed documents found — using abstracts only")

    # Merge metadata + parsed text
    print("\nMerging and building corpus...")
    t0 = time.time()

    corpus_path = get_data_dir("processed") / "corpus.jsonl"
    chunks_path = get_data_dir("processed") / "chunks.jsonl"

    total_docs = 0
    total_chunks = 0
    docs_with_fulltext = 0
    section_type_counts = {}
    chunk_token_counts = []

    with open(corpus_path, "w") as corpus_f, open(chunks_path, "w") as chunks_f:
        for pmid, meta in metadata.items():
            pmcid = meta.get("pmcid", "")

            # Build merged document
            doc = {
                "doc_id": pmcid or pmid,
                "pmid": pmid,
                "pmcid": pmcid,
                "doi": meta.get("doi", ""),
                "title": meta.get("title", ""),
                "abstract": meta.get("abstract", ""),
                "year": meta.get("year"),
                "venue": meta.get("venue", ""),
                "venue_abbrev": meta.get("venue_abbrev", ""),
                "authors": meta.get("authors", []),
                "mesh_terms": meta.get("mesh_terms", []),
                "publication_types": meta.get("publication_types", []),
                "keywords": meta.get("keywords", []),
                "seed_query": meta.get("seed_query", ""),
                # OpenAlex enrichment
                "openalex_id": meta.get("openalex_id", ""),
                "concepts": meta.get("concepts", []),
                "topics": meta.get("topics", []),
                "cited_by_count": meta.get("cited_by_count", 0),
                "referenced_work_ids": meta.get("referenced_work_ids", []),
                "is_oa": meta.get("is_oa", False),
                # Full text fields (populated if available)
                "sections": [],
                "figure_captions": [],
                "table_texts": [],
                "has_fulltext": False,
            }

            # Merge parsed full text if available
            if pmcid and pmcid in parsed:
                p = parsed[pmcid]
                doc["sections"] = p.get("sections", [])
                doc["figure_captions"] = p.get("figure_captions", [])
                doc["table_texts"] = p.get("table_texts", [])
                doc["has_fulltext"] = True
                docs_with_fulltext += 1

                # Use parsed title/abstract if metadata versions are empty
                if not doc["title"] and p.get("title"):
                    doc["title"] = p["title"]
                if not doc["abstract"] and p.get("abstract"):
                    doc["abstract"] = p["abstract"]

            # Write corpus record
            corpus_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            total_docs += 1

            # Generate chunks
            doc_chunks = chunk_document(doc, chunking_config)
            for chunk in doc_chunks:
                chunks_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1
                st = chunk["section_type"]
                section_type_counts[st] = section_type_counts.get(st, 0) + 1
                chunk_token_counts.append(chunk["token_estimate"])

            if total_docs % 1000 == 0:
                print(f"  Processed {total_docs} docs, {total_chunks} chunks")

    t1 = time.time()

    # Compute stats
    stats = {
        "total_documents": total_docs,
        "documents_with_fulltext": docs_with_fulltext,
        "documents_abstract_only": total_docs - docs_with_fulltext,
        "total_chunks": total_chunks,
        "avg_chunks_per_doc": round(total_chunks / max(total_docs, 1), 1),
        "section_type_distribution": dict(sorted(
            section_type_counts.items(), key=lambda x: -x[1]
        )),
        "chunk_token_stats": {},
        "build_time_seconds": round(t1 - t0, 1),
    }

    if chunk_token_counts:
        sorted_tokens = sorted(chunk_token_counts)
        stats["chunk_token_stats"] = {
            "min": sorted_tokens[0],
            "median": sorted_tokens[len(sorted_tokens) // 2],
            "mean": sum(sorted_tokens) // len(sorted_tokens),
            "max": sorted_tokens[-1],
            "total": sum(sorted_tokens),
        }

    stats_path = get_data_dir("processed") / "corpus_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"CORPUS BUILD SUMMARY")
    print(f"{'='*60}")
    print(f"  Total documents: {total_docs}")
    print(f"  With full text: {docs_with_fulltext} ({100*docs_with_fulltext/max(total_docs,1):.1f}%)")
    print(f"  Abstract only: {total_docs - docs_with_fulltext}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Avg chunks/doc: {stats['avg_chunks_per_doc']}")
    print(f"\n  Chunk types:")
    for st, count in stats["section_type_distribution"].items():
        print(f"    {count:>6}  {st}")
    if stats["chunk_token_stats"]:
        ts = stats["chunk_token_stats"]
        print(f"\n  Token stats:")
        print(f"    Median: {ts['median']}")
        print(f"    Mean: {ts['mean']}")
        print(f"    Total: {ts['total']:,}")
    print(f"\n  Output:")
    print(f"    Corpus: {corpus_path}")
    print(f"    Chunks: {chunks_path}")
    print(f"    Stats: {stats_path}")
    print(f"  Time: {t1-t0:.1f}s")


if __name__ == "__main__":
    main()
