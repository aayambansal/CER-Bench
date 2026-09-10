#!/usr/bin/env python3
"""Local-only, fresh-output verified JATS restoration and retrieval integration."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.corpus.pubmed_xml import inspect_fulltext, parse_pubmed_xml
from src.corpus.verified_jats import CORPUS_ID, extract_jats, chunk_blocks


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(path, values):
    with path.open("x") as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def write_json(path, value):
    with path.open("x") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def validate_documents(docs, chunks, blocks, max_chars=4096):
    """Validate IDs, exact source spans, and complete character coverage."""
    if not __debug__:
        raise RuntimeError("optimized Python is forbidden: release assertions must execute")
    by_id = {d["doc_id"]: d for d in docs}
    assert len(by_id) == len(docs) == len({d["article_id"] for d in docs})
    indexed = {b["block_id"]: b for b in blocks if b["indexed"]}
    doc_blocks = defaultdict(list)
    for b in indexed.values():
        doc_blocks[b["doc_id"]].append(b)
    assert len(indexed) == sum(b["indexed"] for b in blocks)
    groups = defaultdict(list)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    for c in chunks:
        d = by_id[c["doc_id"]]
        assert c["article_id"] == d["article_id"] == d["doc_id"] == "PMID:" + d["pmid"]
        assert c["corpus_id"] == d["corpus_id"] == CORPUS_ID
        lo, hi = c["document_start"], c["document_end"]
        assert c["text"] == d["text"][lo:hi] and 0 < len(c["text"]) <= max_chars
        for span in c["source_spans"]:
            b = indexed[span["block_id"]]
            assert b["doc_id"] == c["doc_id"]
            assert c["text"][span["chunk_start"]:span["chunk_end"]] == b["text"][span["block_start"]:span["block_end"]]
            assert span["source_sha256"] == b["source_sha256"]
            assert span["document_start"] == lo + span["chunk_start"] == b["document_start"] + span["block_start"]
            assert span["document_end"] == lo + span["chunk_end"] == b["document_start"] + span["block_end"]
        # No source text may be silently left unbound; separators are explicit.
        expected = [(b["block_id"], max(lo, b["document_start"]), min(hi, b["document_end"]))
                    for b in doc_blocks[d["doc_id"]] if max(lo, b["document_start"]) < min(hi, b["document_end"])]
        assert [(s["block_id"], s["document_start"], s["document_end"]) for s in c["source_spans"]] == expected
        groups[d["doc_id"]].append(c)
    assert set(groups) == set(by_id)
    for did, group in groups.items():
        end = 0
        for c in group:
            assert c["document_start"] <= end
            end = max(end, c["document_end"])
        assert end == len(by_id[did]["text"])
    return dict(identity_consistent=True, every_document_has_chunks=True, exact_source_spans=True,
                full_document_character_coverage=True, bounded_chunks=True)


def build(base, output):
    if output.exists():
        raise FileExistsError("output must be fresh; refusing overwrite: " + str(output))
    upstream = base / "data/processed/authoritative_v1/corpus.jsonl"
    original = read_rows(upstream)
    input_hashes = {str(upstream.relative_to(base)): sha(upstream)}
    if len(original) != len({d["pmid"] for d in original}):
        raise ValueError("duplicate authoritative PMID")
    # Verify the inherited source corpus against its immutable-in-place manifest.
    upstream_manifest = base / "results/readiness/authoritative_corpus/output_manifest.json"
    pinned = json.loads(upstream_manifest.read_text())
    pin = next(r for r in pinned if r["path"] == str(upstream.relative_to(base)))
    if sha(upstream) != pin["sha256"]:
        raise ValueError("authoritative_v1 corpus checksum changed")
    input_hashes[str(upstream_manifest.relative_to(base))] = sha(upstream_manifest)
    # Recheck each inherited PMID/PMC/title/abstract against own PubMed XML.
    authoritative = {}
    for rel, expected in sorted({(r["provenance"]["raw_path"], r["provenance"]["raw_sha256"]) for r in original}):
        p = base / rel
        if sha(p) != expected:
            raise ValueError("authoritative XML checksum changed: " + rel)
        input_hashes[rel] = expected
        parsed = parse_pubmed_xml(p.read_bytes())
        if parsed["problems"] or parsed["duplicate_pmids"]:
            raise ValueError("invalid authoritative source records")
        for r in parsed["records"]:
            if r["pmid"] in authoritative:
                raise ValueError("duplicate authoritative source record")
            authoritative[r["pmid"]] = r
    for d in original:
        if any(d[k] != authoritative[d["pmid"]][k] for k in ("pmid", "pmcid", "title", "abstract", "doi", "year")):
            raise ValueError("authoritative row disagrees with own PubMed metadata: " + d["pmid"])

    raw_audit, pmid_files, pmc_files = [], defaultdict(list), defaultdict(list)
    for p in sorted((base / "data/raw/fulltext").glob("*.xml")):
        rel = str(p.relative_to(base))
        digest = sha(p)
        input_hashes[rel] = digest
        try:
            identity = inspect_fulltext(p.read_bytes())
        except Exception as exc:
            identity = dict(status="invalid_xml", pmids=[], pmcids=[], error=str(exc))
        raw_audit.append(dict(path=rel, sha256=digest, identity=identity))
        for pmid in identity["pmids"]:
            pmid_files[pmid].append(rel)
        for pmcid in identity["pmcids"]:
            pmc_files[pmcid].append(rel)
    pmc_owners = Counter(d["pmcid"] for d in original if d["pmcid"])
    docs, chunks, all_blocks, candidate_audit, excluded = [], [], [], [], []
    summary = Counter()
    omission_counts, license_counts, section_types, block_kinds = Counter(), Counter(), Counter(), Counter()
    paragraph_counts = Counter()
    for old in sorted(original, key=lambda r: int(r["pmid"])):
        pmid, pmcid = old["pmid"], old["pmcid"]
        did = "PMID:" + pmid
        parsed, path = None, None
        fulltext_status = "no_previously_verified_local_candidate"
        if old["fulltext_status"] == "identity_matched_raw_candidate_text_unvalidated":
            summary["input_matching_candidates"] += 1
            candidate = old["fulltext_candidates"]
            try:
                if len(candidate) != 1 or not pmcid or pmc_owners[pmcid] != 1:
                    raise ValueError("candidate or authoritative PMC not unique")
                rel = candidate[0]["path"]
                path = base / rel
                if path.resolve().parent != (base / "data/raw/fulltext").resolve():
                    raise ValueError("candidate outside local raw fulltext directory")
                if pmid_files[pmid] != [rel] or pmc_files[pmcid] != [rel]:
                    raise ValueError("local front PMID/PMCID not unique or conflicting")
                if input_hashes[rel] != candidate[0]["sha256"]:
                    raise ValueError("candidate XML checksum changed")
                parsed = extract_jats(path.read_bytes(), pmid, pmcid, old["title"])
                fulltext_status = parsed["status"]
                candidate_audit.append(dict(doc_id=did, raw_path=rel, **{k:v for k,v in parsed.items() if k not in {"blocks", "sections"}}))
            except Exception as exc:
                fulltext_status = "candidate_quarantined_error"
                candidate_audit.append(dict(doc_id=did, raw_path=str(path.relative_to(base)) if path else None,
                                            status=fulltext_status, error=str(exc)))
        metadata_section = did + ":section:metadata"
        blocks = []
        for key in ("title", "abstract"):
            if old[key]:
                blocks.append(dict(block_id=did + ":pubmed:" + key, text=old[key], kind="authoritative_" + key,
                                   source_type="authoritative_pubmed", source_sha256=old["provenance"]["raw_sha256"],
                                   xml_path="/PubmedArticleSet/PubmedArticle[MedlineCitation/PMID='" + pmid + "']/" + old["field_paths"][key],
                                   source_path=old["provenance"]["raw_path"], section_id=metadata_section,
                                   section_type="authoritative_title_abstract", section_heading="Title and abstract",
                                   rendering_addition="title_is_explicit_indexed_text_not_hidden_metadata_boost" if key == "title" else "structured_abstract_labels_preserved"))
        sections = [dict(section_id=metadata_section, section_type="authoritative_title_abstract", heading="Title and abstract",
                         parent_section_id=None, xml_path=None, source_sha256=old["provenance"]["raw_sha256"])]
        raw_blocks = []
        license_info = dict(category="license_unreviewed", permissions=[], machine_readable_licenses=[],
                            use_scope="local_research_only", public_redistribution_authorized=False)
        body_blocks = []
        if parsed is not None:
            license_info = parsed["license"]
        if parsed and parsed["status"] == "extracted_source_identity_validated":
            summary["title_accepted_candidates"] += 1
            summary["normalized_exact_title_matches"] += parsed["title_comparison"]["normalized_equal"]
            omission_counts.update(o["reason"] for o in parsed["omissions"])
            for k in ("source_paragraphs", "handled_paragraphs", "handled_with_text", "explicitly_omitted_paragraphs"):
                paragraph_counts[k] += parsed["coverage"][k]
            for sec in parsed["sections"]:
                sec = dict(sec, section_id=did + ":" + sec["section_id"],
                           parent_section_id=did + ":" + sec["parent_section_id"] if sec["parent_section_id"] else None,
                           source_sha256=parsed["source_sha256"])
                sections.append(sec)
            for b in parsed["blocks"]:
                b = dict(b, block_id=did + ":" + b["block_id"], section_id=did + ":" + b["section_id"],
                         source_path=str(path.relative_to(base)))
                is_body = "/body[1]/" in b["xml_path"] or b["xml_path"].endswith("/body[1]")
                b["indexed"] = is_body or not old["abstract"]
                b["indexing_reason"] = "own_body" if is_body else "front_abstract_fallback" if not old["abstract"] else "front_abstract_retained_not_duplicated_with_authoritative_abstract"
                if b["indexed"]:
                    blocks.append(b)
                    if is_body:
                        body_blocks.append(b)
                else:
                    raw_blocks.append(b)
            substantial = [b for b in body_blocks if b["kind"] not in {"section_heading", "title", "label", "attrib"}]
            if substantial:
                fulltext_status = "restored_conservative_own_body"
                summary["restored_documents"] += 1
            else:
                fulltext_status = "identity_title_validated_no_extractable_body"
                summary["accepted_candidates_without_body"] += 1
        elif parsed and parsed["status"] == "major_title_mismatch_quarantined":
            summary["title_mismatch_quarantined"] += 1
        if not blocks:
            excluded.append(dict(pmid=pmid, doc_id=did, reason="no_indexable_title_abstract_or_validated_body"))
            continue
        for b in blocks:
            b.update(doc_id=did, article_id=did, corpus_id=CORPUS_ID, indexed=True)
        text, doc_chunks = chunk_blocks(did, blocks)
        for b in raw_blocks:
            b.update(doc_id=did, article_id=did, corpus_id=CORPUS_ID, document_start=None, document_end=None)
        for b in blocks + raw_blocks:
            b.update(block_start=0, block_end=len(b["text"]), offset_unit="unicode_codepoints_in_rendered_text")
        for sec in sections:
            covered = [b for b in blocks if b["section_id"] == sec["section_id"] or
                       (sec["xml_path"] and (b["xml_path"] == sec["xml_path"] or b["xml_path"].startswith(sec["xml_path"] + "/")))]
            sec.update(document_start=min((b["document_start"] for b in covered), default=None),
                       document_end=max((b["document_end"] for b in covered), default=None),
                       indexed=bool(covered), offset_unit="unicode_codepoints_in_rendered_document")
            if covered:
                section_types[sec["section_type"]] += 1
        title_only = all(b["kind"] == "authoritative_title" for b in blocks)
        summary["title_only_documents"] += title_only
        summary["no_authoritative_abstract_documents"] += not bool(old["abstract"])
        summary["body_source_chunks"] += sum(any(s["block_id"] in {b["block_id"] for b in body_blocks} for s in c["source_spans"]) for c in doc_chunks)
        block_kinds.update(b["kind"] for b in blocks)
        license_counts[license_info["category"]] += 1
        docs.append(dict(doc_id=did, article_id=did, corpus_id=CORPUS_ID, pmid=pmid, pmcid=pmcid,
                         title=old["title"], abstract=old["abstract"], doi=old["doi"], year=old["year"],
                         text=text, sections=sections, fulltext_status=fulltext_status,
                         has_fulltext=fulltext_status == "restored_conservative_own_body", title_only=title_only,
                         fulltext_source_sha256=parsed["source_sha256"] if parsed else None,
                         fulltext_source_path=str(path.relative_to(base)) if path else None,
                         provenance=old["provenance"], fulltext_license=license_info,
                         identity_validated=True, human_validation=False, human_validation_status="waived_not_done",
                         biomedical_relevance_validated=False, public_redistribution_authorized=False,
                         text_validation="structural_source_ownership_and_offsets_only; partial_JATS_not_semantic_validation"))
        chunks.extend(doc_chunks)
        all_blocks.extend(blocks + raw_blocks)
    checks = validate_documents(docs, chunks, all_blocks)
    for rel, digest in input_hashes.items():
        if sha(base / rel) != digest:
            raise ValueError("input changed during build: " + rel)
    summary.update(input_documents=len(original), document_count=len(docs), chunk_count=len(chunks),
                   excluded_documents=len(excluded), local_xml_files=len(raw_audit), indexed_blocks=sum(b["indexed"] for b in all_blocks),
                   retained_nonindexed_front_blocks=sum(not b["indexed"] for b in all_blocks),
                   indexed_document_characters=sum(len(d["text"]) for d in docs),
                   chunk_characters_with_overlap=sum(len(c["text"]) for c in chunks),
                   chunk_token_proxy_with_overlap=sum(c["token_proxy"] for c in chunks),
                   document_token_proxy_without_overlap=sum(len(re.findall(r"[a-z0-9]+", d["text"].lower())) for d in docs),
                   max_chunk_characters=max(len(c["text"]) for c in chunks))
    report = dict(counts=dict(summary), checks=checks, omission_counts=dict(omission_counts),
                  license_categories=dict(license_counts), indexed_section_types=dict(section_types), indexed_block_kinds=dict(block_kinds),
                  paragraph_coverage=dict(paragraph_counts), fulltext_status_counts=dict(Counter(d["fulltext_status"] for d in docs)),
                  token_proxy_definition="lowercase Unicode then regex [a-z0-9]+; lexical token count only, not model tokens",
                  human_validation=False, human_validation_status="waived_not_done", biomedical_relevance_validated=False,
                  historical_tasks_or_qrels_used=False, public_redistribution_authorized=False)
    output.mkdir(parents=True, exist_ok=False)
    for filename, data in (("corpus.jsonl", docs), ("chunks.jsonl", chunks), ("source_blocks.jsonl", all_blocks),
                           ("candidate_audit.jsonl", candidate_audit), ("local_xml_audit.jsonl", raw_audit), ("excluded_metadata.jsonl", excluded)):
        write_rows(output / filename, data)
    write_json(output / "summary.json", report)
    sourcehash = hashlib.sha256(json.dumps(input_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = dict(schema="verified-retrieval-dataset-v1", status="authoritative_validated", corpus_id=CORPUS_ID,
                    identity_validated=True, canonical_ids_unique=True, document_count=len(docs), chunk_count=len(chunks),
                    corpus_sha256=sha(output / "corpus.jsonl"), chunks_sha256=sha(output / "chunks.jsonl"),
                    sourcehash=sourcehash, source_hash_algorithm="sha256_of_sorted_compact_json_source_hashes",
                    source_hashes=input_hashes, source_blocks_sha256=sha(output / "source_blocks.jsonl"),
                    code_sha256={r:sha(base / r) for r in ("src/corpus/verified_jats.py", "scripts/50_restore_verified_fulltext.py", "src/corpus/pubmed_xml.py")},
                    source_identity_scope="own_PubMed_PMID_PMC_and_own_JATS_front_IDs_title_gate_only_not_relevance_gold",
                    textvalidationlimits=["partial_JATS_whitelist", "formula_linearization_or_TeX_not_semantically_validated",
                                          "images_and_supplements_omitted", "back_references_other_articles_excluded",
                                          "title_explicitly_indexed", "codepoint_offsets_in_rendered_text_not_XML_bytes"],
                    human_validation=False, human_validation_status="waived_not_done", biomedical_relevance_validated=False,
                    historical_tasks_or_qrels_used=False, public_redistribution_authorized=False, use_scope="local_research_only",
                    chunk_policy=dict(target_characters=4096, max_characters=4096, overlap_max_characters=256,
                                      prefer="rendered_block_boundaries_then_whitespace", separator="\n\n",
                                      offset_unit="Unicode_codepoints", cross_document_overlap=False),
                    output_hashes={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    write_json(output / "dataset_manifest.json", manifest)
    print(json.dumps({k:v for k,v in report.items() if k not in {"indexed_section_types", "indexed_block_kinds"}}, indent=2))
    return report


def compare_builds(first, second, verification):
    """Compare independently produced files and exercise worker49's real gate."""
    if first.resolve() == second.resolve() or verification.exists():
        raise ValueError("comparison requires distinct builds and a fresh verification file")
    if verification.resolve().is_relative_to(first.resolve()) or verification.resolve().is_relative_to(second.resolve()):
        raise ValueError("verification report must be outside both immutable dataset builds")
    hashes1 = {p.name:sha(p) for p in sorted(first.iterdir()) if p.is_file()}
    hashes2 = {p.name:sha(p) for p in sorted(second.iterdir()) if p.is_file()}
    if hashes1 != hashes2:
        raise ValueError("independent builds are not byte-identical")
    from src.retrieval.verified_bm25 import validate_inputs
    docs, chunks, manifest, bound = validate_inputs(first / "corpus.jsonl", first / "chunks.jsonl", first / "dataset_manifest.json")
    verification.parent.mkdir(parents=True, exist_ok=True)
    value = dict(independent_builds_byte_identical=True, files_compared=len(hashes1), output_hashes=hashes1,
                 first_build=str(first), second_build=str(second), worker49_validate_inputs_passed=True,
                 document_count=len(docs), chunk_count=len(chunks), verified_binding=bound,
                 sourcehash=manifest["sourcehash"], identity_scope="source_identity_only_not_relevance_gold",
                 human_validation=False)
    write_json(verification, value)
    print(json.dumps({k:v for k,v in value.items() if k != "output_hashes"}, indent=2))
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--output", type=Path, default=BASE / "data/processed/authoritative_fulltext_v1")
    p.add_argument("--compare-with", type=Path, help="existing independent build to compare after producing fresh output")
    p.add_argument("--verification", type=Path, help="fresh comparison report outside both dataset directories")
    args = p.parse_args()
    if bool(args.compare_with) != bool(args.verification):
        p.error("--compare-with and --verification must be supplied together")
    build(args.base, args.output)
    if args.compare_with:
        compare_builds(args.compare_with, args.output, args.verification)


if __name__ == "__main__":
    main()
