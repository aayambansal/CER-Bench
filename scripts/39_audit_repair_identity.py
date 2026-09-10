#!/usr/bin/env python3
"""Local-only identity audit and deliberately lossy metadata/abstract candidate.

No full text is promoted. Tasks touching fulltext-bearing or conflicted records
are quarantined even when a copied chunk PMID seems to disambiguate an old ID.
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.corpus.identity import (SPLITS, alias_key, annotated_ids, identity_index,
                                normalize_id, publish_directory, read_jsonl,
                                resolve, sha256, transform_refs, unique,
                                write_json, write_jsonl)


def normalized_text(value):
    return " ".join(str(value).split())


def xml_evidence(path):
    """Read article-front IDs only, never IDs in references or filenames."""
    row = {"source": str(path), "sha256": sha256(path)}
    try:
        root = ET.parse(path).getroot()  # stdlib does not fetch external DTDs
        for node in root.iter():
            node.tag = node.tag.rsplit("}", 1)[-1]
        articles = [root] if root.tag == "article" else root.findall("./article")
        if len(articles) != 1:
            raise ValueError("Expected exactly one top-level article")
        meta = articles[0].find("./front/article-meta")
        if meta is None:
            raise ValueError("Missing article-front metadata")
        ids = defaultdict(set)
        for item in meta.findall("./article-id"):
            kind = item.get("pub-id-type", "").lower()
            if kind in {"pmid", "pmc", "pmcid"}:
                ns = "PMID" if kind == "pmid" else "PMCID"
                ids[ns].add(normalize_id(item.text, ns))
        row["pmids"] = sorted(ids["PMID"])
        row["pmcids"] = sorted(ids["PMCID"])
        title = meta.find("./title-group/article-title")
        row["title"] = normalized_text("".join(title.itertext())) if title is not None else ""
        row["status"] = "local_front_identity" if len(ids["PMID"]) == len(ids["PMCID"]) == 1 else "unresolved_front_identity"
    except (ET.ParseError, ValueError) as exc:
        row.update(status="unreadable_identity", error=str(exc), pmids=[], pmcids=[])
    return row


def audit(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    corpus_path = root / "data/processed/corpus.jsonl"
    chunks_path = root / "data/processed/chunks.jsonl"
    metadata_path = root / "data/raw/metadata/pubmed_openalex_metadata.jsonl"
    parsed_path = root / "data/interim/parsed/parsed_documents.jsonl"
    task_paths = {s: root / f"data/benchmark/{s}.jsonl" for s in SPLITS}
    paths = [corpus_path, chunks_path, metadata_path, *task_paths.values()]
    if parsed_path.exists():
        paths.append(parsed_path)
    paths += sorted((root / "data/raw/fulltext").glob("*.xml"))
    paths += sorted((root / "data/benchmark/v1_1_component_disjoint").glob("*json*"))
    paths += [Path(__file__), ROOT / "src/corpus/identity.py", ROOT / "scripts/04_build_corpus.py"]
    source_hashes = {str(p): sha256(p) for p in paths}
    corpus, chunks, metadata = map(read_jsonl, (corpus_path, chunks_path, metadata_path))
    tasks = {s: read_jsonl(p) for s, p in task_paths.items()}
    unique([t for s in SPLITS for t in tasks[s]], "task_id")
    parsed = read_jsonl(parsed_path) if parsed_path.exists() else []
    evidence = [xml_evidence(p) for p in paths if p.suffix == ".xml"]
    by_pmc = defaultdict(list)
    for e in evidence:
        for pmc in e["pmcids"]:
            by_pmc[pmc].append(e)
    valid_rows, invalid_rows = [], []
    for line, row in enumerate(corpus, 1):
        try:
            identity_index([row])
            valid_rows.append(row)
        except (ValueError, KeyError) as exc:
            invalid_rows.append({"source_line": line, "reason": str(exc), "record": row})
    aliases = identity_index(valid_rows)
    meta_by_pmid = defaultdict(list)
    invalid_metadata = []
    for line, row in enumerate(metadata, 1):
        try:
            pmid = normalize_id(row["pmid"], "PMID")
            meta_by_pmid[pmid].append(row)
            for key, values in identity_index([row]).items():
                aliases.setdefault(key, set()).update(values)
        except (ValueError, KeyError) as exc:
            invalid_metadata.append({"source_line": line, "reason": str(exc)})
    # A raw front identity conflicting with metadata invalidates that PMCID alias.
    for pmc, entries in by_pmc.items():
        for e in entries:
            for key in (pmc, pmc[3:]):
                aliases.setdefault(key, set()).update(e["pmids"])
    canonical_counts = Counter(normalize_id(r["pmid"], "PMID") for r in valid_rows)
    parsed_by_pmc = defaultdict(list)
    for row in parsed:
        try:
            parsed_by_pmc[normalize_id(row["pmcid"], "PMCID")].append(row)
        except (ValueError, KeyError):
            pass  # counted below; never used to promote content
    candidate, quarantine_docs, mapping, unsafe = [], list(invalid_rows), [], set()
    fulltext_evidence = []
    for line, row in enumerate(corpus, 1):
        try:
            canonical = normalize_id(row["pmid"], "PMID")
            pmc = normalize_id(row["pmcid"], "PMCID") if row.get("pmcid") else ""
        except (ValueError, KeyError):
            continue
        reasons = []
        raw = by_pmc.get(pmc, [])
        raw_pmids = sorted({v for e in raw for v in e["pmids"]})
        has_text = bool(row.get("has_fulltext") or any(row.get(k) for k in ("sections", "figure_captions", "table_texts")))
        if has_text:
            reasons.append("fulltext_removed_generation_context_untrusted")
        if pmc and aliases.get(pmc, set()) != {canonical}:
            reasons.append("pmcid_alias_conflict")
        if raw_pmids and raw_pmids != [canonical]:
            reasons.append("raw_article_front_pmid_conflicts_with_metadata")
        if has_text or raw:
            fulltext_evidence.append({
                "source_line": line, "old_id": row.get("doc_id"), "canonical_id": canonical,
                "claimed_pmcid": pmc, "metadata_title": row.get("title", ""),
                "raw_pmids": raw_pmids, "raw_titles": sorted({e.get("title", "") for e in raw}),
                "raw_sources": [e["source"] for e in raw],
                "parsed_titles": sorted({p.get("title", "") for p in parsed_by_pmc.get(pmc, [])}),
                "has_fulltext": has_text, "issues": reasons.copy(),
                "decision": "Fulltext excluded regardless of apparent identity agreement",
            })
        fatal = []
        metas = meta_by_pmid.get(canonical, [])
        if canonical_counts[canonical] != 1:
            fatal.append("duplicate_canonical_pmid")
        if len(metas) != 1:
            fatal.append("missing_or_duplicate_source_metadata_pmid")
        elif any(row.get(k, "") != metas[0].get(k, "") for k in ("title", "abstract")):
            fatal.append("metadata_title_or_abstract_disagreement")
        if not row.get("abstract"):
            fatal.append("missing_abstract")
        if reasons or fatal:
            unsafe.add(canonical)
        mapping.append({"source_line": line, "old_id": row.get("doc_id"), "canonical_id": canonical,
                        "claimed_pmcid": pmc, "old_id_candidates": sorted(aliases.get(alias_key(row.get("doc_id")), set())),
                        "candidate_included": not fatal, "issues": sorted(set(reasons + fatal))})
        if fatal:
            quarantine_docs.append({"source_line": line, "reasons": fatal, "record": row})
            continue
        doc = dict(row)
        doc.update(doc_id=canonical, pmid=canonical.split(":")[1], pmcid="", has_fulltext=False,
                   sections=[], figure_captions=[], table_texts=[],
                   identity_provenance={"legacy_id": row.get("doc_id"), "claimed_pmcid": pmc,
                                        "status": "metadata_only_candidate_not_adjudicated",
                                        "fulltext_removed": has_text, "issues": reasons})
        candidate.append(doc)
    unique(candidate, "doc_id")
    candidates = {d["doc_id"]: d for d in candidate}
    old_chunk_counts = Counter(c.get("chunk_id") for c in chunks)
    eligible, quarantine_chunks = [], []
    for line, chunk in enumerate(chunks, 1):
        reasons = []
        canonical = None
        try:
            canonical = resolve(chunk.get("doc_id"), aliases)
            if normalize_id(chunk.get("pmid"), "PMID") != canonical:
                reasons.append("chunk_pmid_disagrees_with_resolved_identity")
        except ValueError as exc:
            reasons.append(str(exc))
        if old_chunk_counts[chunk.get("chunk_id")] != 1:
            reasons.append("duplicate_old_chunk_id")
        if chunk.get("section_type") != "abstract":
            reasons.append("fulltext_chunk_not_promoted")
        if canonical not in candidates:
            reasons.append("document_not_in_candidate")
        elif chunk.get("text") != candidates[canonical]["abstract"][:2048]:
            reasons.append("not_exact_metadata_abstract_prefix")
        if reasons:
            quarantine_chunks.append({"source_line": line, "reasons": reasons, "record": chunk})
        else:
            eligible.append((line, canonical, chunk))
    new_counts = Counter(canonical for _, canonical, _ in eligible)
    kept_chunks, chunk_map = [], {}
    for line, canonical, chunk in eligible:
        if new_counts[canonical] != 1:
            quarantine_chunks.append({"source_line": line, "reasons": ["candidate_chunk_collision"], "record": chunk})
            continue
        new_id = f"{canonical}_abstract_0"
        chunk_map[chunk["chunk_id"]] = new_id
        kept_chunks.append({**chunk, "doc_id": canonical, "chunk_id": new_id,
                            "pmid": canonical.split(":")[1], "pmcid": "",
                            "identity_provenance": {"legacy_id": chunk["chunk_id"], "status": "metadata_abstract_only"}})
    unique(kept_chunks, "chunk_id")
    chunk_docs = {c["doc_id"] for c in kept_chunks}
    kept_tasks, quarantine_tasks = {s: [] for s in SPLITS}, []
    for split in SPLITS:
        for line, task in enumerate(tasks[split], 1):
            legacy_map = {}
            def doc_fn(old):
                new = resolve(old, aliases)
                if new not in candidates or new in unsafe or new not in chunk_docs:
                    raise ValueError(f"Unsafe or uncovered task document: {old} -> {new}")
                legacy_map[str(old)] = new
                return new
            def chunk_fn(old):
                if old not in chunk_map:
                    raise ValueError(f"Quarantined or unknown task chunk: {old}")
                new = chunk_map[old]
                doc_fn(new.rsplit("_abstract_0", 1)[0])
                return new
            try:
                repaired = transform_refs(task, doc_fn, chunk_fn)
                if not annotated_ids(repaired)[0]:
                    raise ValueError("Task has no annotated documents")
                for passage in repaired.get("supporting_passages", []):
                    doc = candidates.get(passage.get("doc_id"))
                    if doc is None or passage.get("section", "abstract") != "abstract" or not passage.get("text"):
                        raise ValueError("Nonabstract or unidentified supporting passage")
                    if normalized_text(passage["text"]) not in normalized_text(doc["abstract"]):
                        raise ValueError("Supporting passage does not match metadata abstract")
                repaired.update(split=split, source_split=split,
                                identity_provenance={"status": "unadjudicated_proxy_candidate", "legacy_citation_map": legacy_map,
                                                     "prose_rewritten": False, "previously_exposed": True})
                kept_tasks[split].append(repaired)
            except (ValueError, KeyError, TypeError) as exc:
                quarantine_tasks.append({"source_split": split, "source_line": line,
                                         "reason": str(exc), "record": task})
    conflicts = [{"alias": a, "canonical_candidates": sorted(v)} for a, v in sorted(aliases.items()) if len(v) > 1]
    report = {
        "version": "identity-repair-v1-candidate", "release_blocked": True,
        "source_hashes": source_hashes,
        "original": {"corpus_records": len(corpus), "unique_doc_ids": len({r.get('doc_id') for r in corpus}),
                     "chunk_records": len(chunks), "unique_chunk_ids": len(old_chunk_counts),
                     "tasks": sum(map(len, tasks.values()))},
        "candidate": {"corpus_records": len(candidate), "chunk_records": len(kept_chunks),
                      "tasks": sum(map(len, kept_tasks.values())), "task_source_splits": {s: len(v) for s, v in kept_tasks.items()},
                      "fulltext_records": 0, "documents_without_retained_chunks": len(candidates.keys() - chunk_docs)},
        "quarantine": {"corpus_records": len(quarantine_docs), "chunk_records": len(quarantine_chunks), "tasks": len(quarantine_tasks)},
        "conflicting_aliases": len(conflicts), "invalid_metadata": invalid_metadata,
        "raw_xml_records": len(evidence), "raw_xml_status_counts": dict(Counter(e["status"] for e in evidence)),
        "parsed_records": len(parsed), "parsed_records_with_valid_pmcid": sum(map(len, parsed_by_pmc.values())),
        "raw_front_pmid_disagreement_records": sum("raw_article_front_pmid_conflicts_with_metadata" in e["issues"] for e in fulltext_evidence),
        "fulltext_records_removed": sum(e["has_fulltext"] for e in fulltext_evidence),
        "task_quarantine_reasons": dict(Counter(q["reason"] for q in quarantine_tasks)),
        "unresolved": [
            "Local metadata identity is not externally adjudicated; PMCID claims are removed from active candidate fields.",
            "All parsed/fulltext content is excluded. Renaming IDs does not repair misattached fulltext or scientific claims.",
            "Tasks touching original fulltext-bearing/conflicted records are excluded, not regenerated or reassigned.",
            "Prose, constraints and answers remain synthetic/unverified and may contain legacy citations; explicit per-task citation maps are provided.",
            "Old indices, qrels, results, and historical splits remain invalidated for candidate use; nothing is migrated implicitly.",
            "Previously exposed test tasks are not fresh test data. This candidate is not gold or release-ready.",
        ],
    }
    def build(stage):
        outputs = {"corpus.jsonl": sorted(candidate, key=lambda r: r["doc_id"]),
                   "chunks.jsonl": sorted(kept_chunks, key=lambda r: r["chunk_id"]),
                   "mapping.jsonl": mapping, "alias_conflicts.jsonl": conflicts,
                   "aliases.jsonl": [{"alias": a, "canonical_candidates": sorted(v)} for a, v in sorted(aliases.items())],
                   "raw_identity_evidence.jsonl": evidence, "fulltext_identity_evidence.jsonl": fulltext_evidence,
                   "quarantine_corpus.jsonl": quarantine_docs, "quarantine_chunks.jsonl": quarantine_chunks,
                   "quarantine_tasks.jsonl": quarantine_tasks,
                   "chunk_mapping.jsonl": [{"old_id": a, "canonical_id": b} for a, b in sorted(chunk_map.items())]}
        outputs.update({f"{s}.jsonl": sorted(v, key=lambda t: t["task_id"]) for s, v in kept_tasks.items()})
        for name, rows in outputs.items():
            write_jsonl(stage / name, rows)
        unique(read_jsonl(stage / "corpus.jsonl"), "doc_id")
        unique(read_jsonl(stage / "chunks.jsonl"), "chunk_id")
        if len(candidate) + len(quarantine_docs) != len(corpus) or len(kept_chunks) + len(quarantine_chunks) != len(chunks):
            raise ValueError("Record accounting failed")
        if sum(map(len, kept_tasks.values())) + len(quarantine_tasks) != sum(map(len, tasks.values())):
            raise ValueError("Task accounting failed")
        if any(sha256(p) != digest for p, digest in source_hashes.items()):
            raise ValueError("Source changed during audit")
        report["output_hashes"] = {name: sha256(stage / name) for name in sorted(outputs)}
        write_json(stage / "report.json", report)
        return report
    return publish_directory(output, build)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/identity_repair_v1")
    args = parser.parse_args()
    report = audit(args.root, args.output)
    print(json.dumps({k: v for k, v in report.items() if k not in {"source_hashes", "output_hashes", "task_quarantine_reasons"}}, indent=2))


if __name__ == "__main__":
    main()
