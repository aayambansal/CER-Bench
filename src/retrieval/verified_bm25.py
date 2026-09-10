"""Offline, hash-bound chunk BM25. No provider, pickle, or metadata retrieval."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

CONFIG = {"version": "verified-bm25-v1", "k1": 1.5, "b": 0.75,
          "epsilon": 0.25, "tokenizer": "lower-ascii-alphanumeric-v1",
          "tie_break": "input_chunk_order", "document_score": "max_chunk"}
STATUSES = {"authoritative_validated", "synthetic_structural", "candidate_diagnostic"}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path):
    with Path(path).open() as f:
        rows = [parse_json(line) for line in f if line.strip()]
    if any(not isinstance(r, dict) for r in rows):
        raise ValueError("JSONL rows must be objects")
    return rows


def parse_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite JSON value")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def tokenize(text):
    """Lowercase then [a-z0-9]+; keep one-character tokens and repetitions."""
    if not isinstance(text, str):
        raise ValueError("Text/query must be a string")
    return re.findall(r"[a-z0-9]+", text.lower())


def _ids(rows, field):
    ids = [r.get(field) for r in rows]
    if any(not isinstance(i, str) or not i or i != i.strip() for i in ids):
        raise ValueError(f"Invalid {field}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate {field}")
    return ids


def validate_inputs(corpus, chunks, manifest, *, mode="authoritative_validated"):
    """Manifest is an explicit identity attestation, not proof of scientific truth."""
    if mode not in STATUSES:
        raise ValueError("Unsupported dataset mode")
    m = parse_json(Path(manifest).read_text())
    if not isinstance(m, dict):
        raise ValueError("Dataset manifest must be an object")
    if m.get("schema") != "verified-retrieval-dataset-v1" or m.get("status") != mode:
        raise ValueError("Dataset manifest schema/status mismatch")
    if m.get("identity_validated") is not True or m.get("canonical_ids_unique") is not True:
        raise ValueError("Explicit validated identity and canonical uniqueness required")
    if not isinstance(m.get("corpus_id"), str) or not m["corpus_id"].strip():
        raise ValueError("Missing corpus identity")
    hashes = {"corpus_sha256": file_hash(corpus), "chunks_sha256": file_hash(chunks)}
    if any(m.get(k) != v for k, v in hashes.items()):
        raise ValueError("Dataset input hash mismatch")
    docs, rows = read_jsonl(corpus), read_jsonl(chunks)
    if not docs or not rows:
        raise ValueError("Empty corpus/chunks")
    doc_ids, chunk_ids = _ids(docs, "doc_id"), _ids(rows, "chunk_id")
    article_ids = _ids(docs, "article_id")
    by_doc = dict(zip(doc_ids, article_ids))
    for row in docs + rows:
        if row.get("corpus_id") != m["corpus_id"]:
            raise ValueError("Cross-corpus identity")
    for row in rows:
        if row.get("doc_id") not in by_doc:
            raise ValueError("Orphan chunk")
        if row.get("article_id") != by_doc[row["doc_id"]]:
            raise ValueError("Cross-article chunk")
        if not isinstance(row.get("text"), str):
            raise ValueError("Chunk text must be explicit string")
    if set(by_doc) != {r["doc_id"] for r in rows}:
        raise ValueError("Every document must have a chunk")
    if m.get("document_count") != len(docs) or m.get("chunk_count") != len(rows):
        raise ValueError("Dataset count mismatch")
    return docs, rows, m, {**hashes, "manifest_sha256": file_hash(manifest)}


def _arrays(rows):
    # Term-major postings; query work touches only postings of query vocabulary.
    postings = {}
    lengths = np.empty(len(rows), dtype=np.int64)
    for i, row in enumerate(rows):
        counts = Counter(tokenize(row["text"]))
        lengths[i] = sum(counts.values())
        for term, count in counts.items():
            postings.setdefault(term, []).append((i, count))
    vocab = sorted(postings)
    if not vocab:
        raise ValueError("Corpus has no indexable tokens")
    ptr, indices, tf = [0], [], []
    for term in vocab:
        for i, count in postings[term]:
            indices.append(i)
            tf.append(count)
        ptr.append(len(indices))
    df = np.diff(ptr)
    raw_idf = np.log((len(rows) - df + 0.5) / (df + 0.5))
    idf = np.where(raw_idf < 0, CONFIG["epsilon"] * raw_idf.mean(), raw_idf)
    return vocab, {"indptr": np.array(ptr, dtype=np.int64),
                   "indices": np.array(indices, dtype=np.int64),
                   "tf": np.array(tf, dtype=np.float64), "lengths": lengths,
                   "idf": idf}


def _array_hashes(arrays):
    return {k: digest({"dtype": str(v.dtype), "shape": list(v.shape),
                       "bytes": hashlib.sha256(v.tobytes()).hexdigest()})
            for k, v in arrays.items()}


def build_index(corpus, chunks, manifest, output, *, mode="authoritative_validated"):
    start = time.perf_counter()
    docs, rows, m, hashes = validate_inputs(corpus, chunks, manifest, mode=mode)
    vocab, arrays = _arrays(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "index.json").exists() or (output / "postings.npz").exists():
        raise ValueError("Refusing to overwrite existing index")
    np.savez_compressed(output / "postings.npz", **arrays)
    meta = {"schema": CONFIG["version"], "config": dict(CONFIG), "config_sha256": digest(CONFIG),
            "code_sha256": file_hash(__file__), "input_hashes": hashes,
            "status": mode, "corpus_id": m["corpus_id"], "vocabulary": vocab,
            "doc_ids": [d["doc_id"] for d in docs], "chunk_ids": [r["chunk_id"] for r in rows],
            "chunk_doc_ids": [r["doc_id"] for r in rows],
            "array_sha256": _array_hashes(arrays),
            "npz_sha256": file_hash(output / "postings.npz"),
            "build_seconds": time.perf_counter() - start}
    meta["integrity_sha256"] = digest(meta)
    write_json(output / "index.json", meta)
    return meta


class VerifiedBM25:
    """Callable full chunk ranking, plus bounded search and document ranking.

    Never accepts tasks, annotations, qrels or oracle labels. Selector scores are
    original-query max chunk BM25 within discovered documents, not relevance.
    """

    @classmethod
    def load(cls, index, *, corpus, chunks, manifest, mode="authoritative_validated"):
        docs, rows, m, hashes = validate_inputs(corpus, chunks, manifest, mode=mode)
        index = Path(index)
        meta = parse_json((index / "index.json").read_text())
        if not isinstance(meta, dict):
            raise ValueError("Index manifest must be an object")
        integrity = meta.get("integrity_sha256")
        if integrity != digest({k: v for k, v in meta.items() if k != "integrity_sha256"}):
            raise ValueError("Index JSON integrity mismatch")
        expected = {"schema": CONFIG["version"], "config": CONFIG,
                    "config_sha256": digest(CONFIG), "code_sha256": file_hash(__file__),
                    "input_hashes": hashes, "status": mode, "corpus_id": m["corpus_id"],
                    "doc_ids": [d["doc_id"] for d in docs],
                    "chunk_ids": [r["chunk_id"] for r in rows],
                    "chunk_doc_ids": [r["doc_id"] for r in rows],
                    "npz_sha256": file_hash(index / "postings.npz")}
        if any(meta.get(k) != v for k, v in expected.items()):
            raise ValueError("Index provenance/hash/order mismatch")
        with np.load(index / "postings.npz", allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        if _array_hashes(arrays) != meta.get("array_sha256"):
            raise ValueError("Index array integrity mismatch")
        # Recompute at load to verify semantic binding, not just mutable checksums.
        vocab, expected_arrays = _arrays(rows)
        if vocab != meta.get("vocabulary") or arrays.keys() != expected_arrays.keys() or any(
                not np.array_equal(arrays[k], expected_arrays[k]) for k in expected_arrays):
            raise ValueError("Index postings/vocabulary do not match input")
        self = cls()
        self.meta, self.rows, self.doc_ids = meta, rows, expected["doc_ids"]
        self.arrays, self.terms = arrays, {t: i for i, t in enumerate(vocab)}
        self.norm = CONFIG["k1"] * (1 - CONFIG["b"] + CONFIG["b"] * arrays["lengths"] / arrays["lengths"].mean())
        self.last_trace = None
        return self

    def scores(self, query):
        scores = np.zeros(len(self.rows), dtype=np.float64)
        a = self.arrays
        # Repeated query terms contribute repeatedly, as in BM25Okapi.get_scores.
        for term in tokenize(query):
            t = self.terms.get(term)
            if t is None:
                continue
            lo, hi = a["indptr"][t:t + 2]
            ix, tf = a["indices"][lo:hi], a["tf"][lo:hi]
            scores[ix] += a["idf"][t] * tf * (CONFIG["k1"] + 1) / (tf + self.norm[ix])
        return scores

    def search(self, query, top_k=None, *, exclude_doc_ids=(), max_unique_docs=None):
        """Sorted chunk prefix, including zero/negative scores; cap <=24 unique docs.

        With a unique cap, stop immediately upon admitting its final new document.
        Exclusions are explicit IDs. top_k instead caps chunk rows, not documents.
        """
        if top_k is not None and (type(top_k) is not int or top_k < 0):
            raise ValueError("Invalid top_k")
        if max_unique_docs is not None and (type(max_unique_docs) is not int or not 1 <= max_unique_docs <= 24):
            raise ValueError("Unique candidate cap must be 1..24")
        excluded = set(exclude_doc_ids)
        if not excluded <= set(self.doc_ids):
            raise ValueError("Unknown excluded document")
        start = time.perf_counter()
        scores = self.scores(query)
        order = np.argsort(-scores, kind="stable")
        hits, seen = [], set()
        for i in order:
            row = self.rows[int(i)]
            if row["doc_id"] in excluded:
                continue
            if top_k is not None and len(hits) >= top_k:
                break
            hits.append({"doc_id": row["doc_id"], "chunk_id": row["chunk_id"],
                         "text": row["text"], "score": float(scores[i])})
            seen.add(row["doc_id"])
            if max_unique_docs is not None and len(seen) >= max_unique_docs:
                break
        self.last_trace = {"query": query, "chunk_hits": hits,
                           "seconds": time.perf_counter() - start,
                           "oracle_labels_seen": False}
        return hits

    def __call__(self, query):
        return self.search(query)

    def rank(self, query, candidates, top_k=20):
        """Max score of ALL chunks in each candidate; stable winning-chunk ties."""
        candidates = list(candidates)
        if len(candidates) > 24 or len(set(candidates)) != len(candidates) or not set(candidates) <= set(self.doc_ids):
            raise ValueError("Candidates must be <=24 unique known document IDs")
        if type(top_k) is not int or top_k < 0:
            raise ValueError("Invalid top_k")
        allowed, seen, result = set(candidates), set(), []
        for hit in self.search(query):
            if hit["doc_id"] in allowed and hit["doc_id"] not in seen:
                seen.add(hit["doc_id"])
                result.append({k: hit[k] for k in ("doc_id", "chunk_id", "score")})
        return result[:top_k]
