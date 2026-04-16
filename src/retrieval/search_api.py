"""Unified scientific search API — the retrieval substrate v2.

Exposes the full tool set for the Context-1-style controller:
  - search_lexical(query) → BM25
  - search_dense(query, model) → BGE / E5 / MedCPT / SPECTER2
  - search_late(query) → ColBERTv2
  - search_sparse(query) → SPLADE
  - search_hybrid(query, methods, fusion) → RRF / learned fusion
  - filter_metadata(docs, filters) → year, organism, assay, MeSH
  - expand_citations(doc_id, direction) → cited / citing
  - grep_corpus(pattern) → regex over chunks
  - read_document(doc_id) → full sectioned view
  - prune_evidence(doc_ids, reason) → remove from working set
  - estimate_confidence(evidence_set) → abstention signal
"""

import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class SearchResult:
    doc_id: str
    chunk_id: str
    score: float
    title: str = ""
    year: Optional[int] = None
    section_type: str = ""
    text: str = ""
    source_method: str = ""


@dataclass
class RetrievalSubstrate:
    """Unified retrieval backend with all search tools."""

    corpus_dir: Path
    index_dir: Path

    # Loaded lazily
    _bm25: object = field(default=None, repr=False)
    _bm25_ids: list = field(default_factory=list, repr=False)
    _docs: dict = field(default_factory=dict, repr=False)
    _chunks: dict = field(default_factory=dict, repr=False)
    _dense_indices: dict = field(default_factory=dict, repr=False)
    _colbert_index: object = field(default=None, repr=False)
    _splade_index: object = field(default=None, repr=False)
    _loaded: bool = False

    def load(self):
        """Load all indices and corpus data."""
        if self._loaded:
            return

        # BM25
        bm25_dir = self.index_dir / "bm25"
        if (bm25_dir / "bm25_index.pkl").exists():
            with open(bm25_dir / "bm25_index.pkl", "rb") as f:
                self._bm25 = pickle.load(f)
            with open(bm25_dir / "chunk_ids.json") as f:
                self._bm25_ids = json.load(f)

        # Corpus documents
        corpus_path = self.corpus_dir / "corpus.jsonl"
        if corpus_path.exists():
            with open(corpus_path) as f:
                for line in f:
                    doc = json.loads(line)
                    self._docs[doc["doc_id"]] = doc

        # Chunks
        chunks_path = self.corpus_dir / "chunks.jsonl"
        if chunks_path.exists():
            with open(chunks_path) as f:
                for line in f:
                    chunk = json.loads(line)
                    self._chunks[chunk["chunk_id"]] = chunk

        # Dense indices (load all available)
        dense_dir = self.index_dir / "dense"
        if dense_dir.exists():
            import faiss
            for idx_file in dense_dir.glob("*.index"):
                name = idx_file.stem
                ids_file = dense_dir / f"{name}_chunk_ids.json"
                if not ids_file.exists():
                    ids_file = dense_dir / "chunk_ids.json"
                if ids_file.exists():
                    index = faiss.read_index(str(idx_file))
                    with open(ids_file) as f:
                        ids = json.load(f)
                    self._dense_indices[name] = (index, ids)

        # ColBERTv2 (if available)
        colbert_dir = self.index_dir / "colbert"
        if colbert_dir.exists() and (colbert_dir / "colbert_index.pt").exists():
            self._colbert_index = colbert_dir

        # SPLADE (if available)
        splade_dir = self.index_dir / "splade"
        if splade_dir.exists() and (splade_dir / "splade_index.npz").exists():
            import scipy.sparse as sp
            self._splade_index = {
                "matrix": sp.load_npz(str(splade_dir / "splade_index.npz")),
                "ids": json.load(open(splade_dir / "chunk_ids.json")),
            }

        self._loaded = True
        print(f"Substrate loaded: {len(self._docs)} docs, {len(self._chunks)} chunks, "
              f"{len(self._dense_indices)} dense indices, "
              f"colbert={'yes' if self._colbert_index else 'no'}, "
              f"splade={'yes' if self._splade_index else 'no'}")

    # ─── Tool: Lexical search ─────────────────────────────────────────

    def search_lexical(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """BM25 lexical search."""
        self.load()
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k * 3]
        return self._dedup_to_docs(
            [(self._bm25_ids[i], float(scores[i])) for i in top_idx],
            top_k, source="bm25"
        )

    # ─── Tool: Dense search ───────────────────────────────────────────

    def search_dense(self, query_embedding: np.ndarray, model_name: str = "specter2",
                     top_k: int = 20) -> list[SearchResult]:
        """Dense nearest-neighbor search using pre-computed query embedding."""
        self.load()
        if model_name not in self._dense_indices:
            return []
        index, ids = self._dense_indices[model_name]
        q = query_embedding.reshape(1, -1).astype(np.float32)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
        scores, indices = index.search(q, top_k * 3)
        return self._dedup_to_docs(
            [(ids[indices[0][i]], float(scores[0][i])) for i in range(len(indices[0]))],
            top_k, source=f"dense_{model_name}"
        )

    # ─── Tool: Hybrid search ─────────────────────────────────────────

    def search_hybrid(self, query: str, query_embedding: np.ndarray = None,
                      dense_model: str = "specter2", top_k: int = 20,
                      rrf_k: int = 60) -> list[SearchResult]:
        """Reciprocal rank fusion of lexical + dense."""
        lexical = self.search_lexical(query, top_k * 3)
        if query_embedding is not None:
            dense = self.search_dense(query_embedding, dense_model, top_k * 3)
        else:
            dense = []

        # RRF
        scores = {}
        for rank, r in enumerate(lexical):
            scores[r.doc_id] = scores.get(r.doc_id, 0) + 1.0 / (rrf_k + rank + 1)
        for rank, r in enumerate(dense):
            scores[r.doc_id] = scores.get(r.doc_id, 0) + 1.0 / (rrf_k + rank + 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for doc_id, score in ranked:
            doc = self._docs.get(doc_id, {})
            results.append(SearchResult(
                doc_id=doc_id, chunk_id="", score=score,
                title=doc.get("title", "")[:200], year=doc.get("year"),
                text=doc.get("abstract", "")[:300], source_method="hybrid_rrf"
            ))
        return results

    # ─── Tool: Metadata filter ────────────────────────────────────────

    def filter_metadata(self, doc_ids: list[str],
                        year_min: int = None, year_max: int = None,
                        organisms: list[str] = None,
                        mesh_terms: list[str] = None,
                        publication_types: list[str] = None) -> list[str]:
        """Filter documents by metadata constraints."""
        self.load()
        filtered = []
        for did in doc_ids:
            doc = self._docs.get(did, {})
            if year_min and (doc.get("year") or 0) < year_min:
                continue
            if year_max and (doc.get("year") or 9999) > year_max:
                continue
            if organisms:
                doc_text = (doc.get("abstract", "") + " " + doc.get("title", "")).lower()
                if not any(o.lower() in doc_text for o in organisms):
                    continue
            if mesh_terms:
                doc_mesh = set(m.lower() for m in doc.get("mesh_terms", []))
                if not any(m.lower() in doc_mesh for m in mesh_terms):
                    continue
            if publication_types:
                doc_pt = set(p.lower() for p in doc.get("publication_types", []))
                if not any(p.lower() in doc_pt for p in publication_types):
                    continue
            filtered.append(did)
        return filtered

    # ─── Tool: Citation expansion ─────────────────────────────────────

    def expand_citations(self, doc_id: str, direction: str = "both",
                         max_hops: int = 1) -> list[str]:
        """Expand via citation graph. direction: 'references', 'cited_by', 'both'."""
        self.load()
        doc = self._docs.get(doc_id, {})
        expanded = set()

        if direction in ("references", "both"):
            # OpenAlex reference IDs are openalex IDs, need to map to our doc_ids
            for ref_oa_id in doc.get("referenced_work_ids", []):
                # Find doc with this openalex_id
                for d in self._docs.values():
                    if d.get("openalex_id", "").endswith(ref_oa_id):
                        expanded.add(d["doc_id"])
                        break

        if direction in ("cited_by", "both"):
            # Find docs that reference this doc's openalex_id
            my_oa_id = doc.get("openalex_id", "")
            if my_oa_id:
                short_id = my_oa_id.replace("https://openalex.org/", "")
                for d in self._docs.values():
                    if short_id in d.get("referenced_work_ids", []):
                        expanded.add(d["doc_id"])

        return list(expanded)[:50]  # cap

    # ─── Tool: Grep corpus ────────────────────────────────────────────

    def grep_corpus(self, pattern: str, max_results: int = 20) -> list[SearchResult]:
        """Regex search over all chunk text."""
        self.load()
        compiled = re.compile(pattern, re.IGNORECASE)
        results = []
        for cid, chunk in self._chunks.items():
            text = chunk.get("text", "")
            if compiled.search(text):
                doc_id = chunk.get("doc_id", "")
                doc = self._docs.get(doc_id, {})
                results.append(SearchResult(
                    doc_id=doc_id, chunk_id=cid, score=1.0,
                    title=doc.get("title", "")[:200], year=doc.get("year"),
                    section_type=chunk.get("section_type", ""),
                    text=text[:300], source_method="grep"
                ))
                if len(results) >= max_results:
                    break
        return results

    # ─── Tool: Read document ──────────────────────────────────────────

    def read_document(self, doc_id: str) -> dict:
        """Return full document with all sections."""
        self.load()
        doc = self._docs.get(doc_id, {})
        # Collect all chunks for this doc
        doc_chunks = [c for c in self._chunks.values() if c.get("doc_id") == doc_id]
        doc_chunks.sort(key=lambda c: c.get("position", 0))
        return {
            "doc_id": doc_id,
            "title": doc.get("title", ""),
            "year": doc.get("year"),
            "venue": doc.get("venue", ""),
            "abstract": doc.get("abstract", ""),
            "mesh_terms": doc.get("mesh_terms", []),
            "sections": [
                {"heading": c.get("section_heading", ""), "type": c.get("section_type", ""),
                 "text": c.get("text", "")}
                for c in doc_chunks if c.get("section_type") != "abstract"
            ],
            "cited_by_count": doc.get("cited_by_count", 0),
            "has_fulltext": doc.get("has_fulltext", False),
        }

    # ─── Tool: Estimate confidence ────────────────────────────────────

    def estimate_confidence(self, evidence: list[SearchResult],
                            query: str) -> dict:
        """Estimate retrieval confidence for abstention decisions."""
        if not evidence:
            return {"confidence": 0.0, "should_abstain": True, "reason": "no evidence found"}

        scores = [r.score for r in evidence]
        top1 = scores[0] if scores else 0
        top5_mean = np.mean(scores[:5]) if len(scores) >= 5 else np.mean(scores)
        score_gap = scores[0] - scores[4] if len(scores) >= 5 else 0

        # Simple heuristic: low top-1 + low gap = low confidence
        confidence = min(1.0, top1 / 20.0)  # normalize BM25 scores
        should_abstain = confidence < 0.3 and score_gap < 2.0

        return {
            "confidence": round(confidence, 3),
            "should_abstain": should_abstain,
            "top1_score": round(top1, 3),
            "top5_mean": round(top5_mean, 3),
            "score_gap": round(score_gap, 3),
            "n_evidence": len(evidence),
        }

    # ─── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s\-]", " ", text)
        return [t for t in text.split() if len(t) > 1]

    @staticmethod
    def _extract_doc_id(chunk_id: str) -> str:
        section_types = {'abstract', 'other', 'methods', 'results', 'discussion',
                         'introduction', 'conclusion', 'caption', 'table', 'body',
                         'results_discussion'}
        parts = chunk_id.split('_')
        for i in range(1, len(parts)):
            if '_'.join(parts[i:]).split('_')[0] in section_types:
                return '_'.join(parts[:i])
        return '_'.join(parts[:-2]) if len(parts) >= 3 else chunk_id

    def _dedup_to_docs(self, chunk_results: list[tuple[str, float]],
                       top_k: int, source: str) -> list[SearchResult]:
        """Deduplicate chunk results to document-level, preserving rank order."""
        seen = set()
        results = []
        for cid, score in chunk_results:
            doc_id = self._extract_doc_id(cid)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = self._docs.get(doc_id, {})
            chunk = self._chunks.get(cid, {})
            results.append(SearchResult(
                doc_id=doc_id, chunk_id=cid, score=score,
                title=doc.get("title", "")[:200], year=doc.get("year"),
                section_type=chunk.get("section_type", ""),
                text=chunk.get("text", "")[:300], source_method=source
            ))
            if len(results) >= top_k:
                break
        return results


def create_substrate(project_root: str = ".") -> RetrievalSubstrate:
    """Create a substrate from the standard project layout."""
    root = Path(project_root)
    return RetrievalSubstrate(
        corpus_dir=root / "data" / "processed",
        index_dir=root / "data" / "processed" / "indices",
    )
