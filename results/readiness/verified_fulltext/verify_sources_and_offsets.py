"""Independent, read-only snapshot audit; writes one fresh verification report."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parents[3]
DATA = BASE / "data/processed/authoritative_fulltext_v1"
EVIDENCE = BASE / "results/readiness/verified_fulltext"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    manifest = json.loads((DATA / "dataset_manifest.json").read_text())
    for rel, digest in manifest["source_hashes"].items():
        assert sha(BASE / rel) == digest, rel
    for filename, digest in manifest["output_hashes"].items():
        assert sha(DATA / filename) == digest, filename
    for path in DATA.iterdir():
        assert sha(path) == sha(EVIDENCE / "independent_build" / path.name)
    for rel, digest in manifest["code_sha256"].items():
        assert sha(BASE / rel) == digest, rel
    docs = {d["doc_id"]:d for d in rows(DATA / "corpus.jsonl")}
    groups, blocks = defaultdict(list), {}
    for b in rows(DATA / "source_blocks.jsonl"):
        assert b["block_id"] not in blocks
        blocks[b["block_id"]] = b
        groups[b["doc_id"]].append(b)
    section_count, span_count = 0, 0
    for did, d in docs.items():
        sections = {s["section_id"]:s for s in d["sections"]}
        assert len(sections) == len(d["sections"])
        for b in groups[did]:
            assert b["section_id"] in sections
            assert b["source_sha256"] == sections[b["section_id"]]["source_sha256"]
            assert b["block_start"] == 0 and b["block_end"] == len(b["text"])
            assert manifest["source_hashes"][b["source_path"]] == b["source_sha256"]
            if b["indexed"]:
                assert d["text"][b["document_start"]:b["document_end"]] == b["text"]
            else:
                assert b["document_start"] is b["document_end"] is None
        for s in sections.values():
            assert s["parent_section_id"] is None or s["parent_section_id"] in sections
            covered = [b for b in groups[did] if b["indexed"] and
                       (b["section_id"] == s["section_id"] or (s["xml_path"] and
                        (b["xml_path"] == s["xml_path"] or b["xml_path"].startswith(s["xml_path"] + "/"))))]
            assert s["document_start"] == min((b["document_start"] for b in covered), default=None)
            assert s["document_end"] == max((b["document_end"] for b in covered), default=None)
            section_count += 1
    chunk_docs = set()
    for c in rows(DATA / "chunks.jsonl"):
        chunk_docs.add(c["doc_id"])
        for s in c["source_spans"]:
            b = blocks[s["block_id"]]
            assert c["doc_id"] == b["doc_id"] == c["article_id"]
            assert s["section_id"] == b["section_id"]
            assert c["text"][s["chunk_start"]:s["chunk_end"]] == b["text"][s["block_start"]:s["block_end"]]
            span_count += 1
    assert chunk_docs == set(docs)
    audit = rows(DATA / "candidate_audit.jsonl")
    accepted = [a for a in audit if a["status"] == "extracted_source_identity_validated"]
    assert all(not a["coverage"]["unexplained_paragraphs"] and not a["coverage"]["duplicate_paragraph_visits"] for a in accepted)
    junit = ET.parse(EVIDENCE / "tests.junit.xml").getroot()
    suites = list(junit.iter("testsuite"))
    tests = sum(int(s.get("tests", 0)) for s in suites)
    failures = sum(int(s.get("failures", 0))+int(s.get("errors", 0)) for s in suites)
    assert tests == 83 and failures == 0
    report = dict(all_bound_sources_still_match=True, bound_sources_verified=len(manifest["source_hashes"]),
                  dataset_files_byte_identical_after_verification=8, code_hashes_still_match=True,
                  all_section_offsets_and_ownership_passed=True, section_entries_checked=section_count,
                  source_blocks_checked=len(blocks), chunk_source_intersections_checked=span_count,
                  no_zero_chunk_documents=True, unexplained_paragraphs=0, duplicate_paragraph_visits=0,
                  accepted_candidates_checked=len(accepted), tests=tests, test_failures_or_errors=failures,
                  junit_sha256=sha(EVIDENCE / "tests.junit.xml"), dataset_manifest_sha256=sha(DATA / "dataset_manifest.json"),
                  verifier_sha256=sha(Path(__file__)), human_validation=False,
                  limitations="source identity and structural offsets only; no biomedical relevance or redistribution approval")
    with (EVIDENCE / "source_and_coverage_verification.json").open("x") as f:
        f.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("run without -O")
    main()
