"""Verify real-smoke receipts after pytest without printing corpus text."""
from pathlib import Path
import json
import statistics
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.retrieval.verified_bm25 import digest, file_hash, write_json
from src.evaluation.budget_protocol import validate_manifest

p = Path(__file__).resolve().parent
report = json.loads((p / "summary.json").read_text())
assert report["provenance_id"] == digest({k: v for k, v in report.items() if k != "provenance_id"})
for relative, sha in report["output_sha256"].items():
    assert file_hash(p / relative) == sha, relative
for relative, sha in report["source_sha256"].items():
    assert file_hash(ROOT / relative) == sha, relative
for filename, sha in report["index_sha256"].items():
    assert file_hash(ROOT / "data/processed/authoritative_fulltext_v1_bm25" / filename) == sha
plan = json.loads((p / "predeclared_plan.json").read_text())
for relative, sha in plan["synthetic_preserved_sha256"].items():
    assert file_hash(ROOT / relative) == sha, relative
assert file_hash(sys.executable) == plan["executable_sha256"]
adapter = json.loads((p / "adapter_dry/manifest.json").read_text())
assert validate_manifest(adapter)
for source, sha in adapter["source_sha256"].items():
    assert file_hash(source) == sha, source
runtime = adapter["config"]["runtime_dependencies"]
for source, sha in runtime.items():
    assert file_hash(source) == sha, source
tests = ET.parse(p / "tests.xml")
cases = list(tests.iter("testcase"))
assert len(cases) == 68 and not any(list(tests.iter(tag)) for tag in ("failure", "error", "skipped"))
queries = json.loads((p / "query_diagnostics.json").read_text())
times = [q["wall_seconds"] for q in queries]
receipt = {"status": "verified", "label_status": "unscored_retrieval_smoke_no_qrels",
           "summary_provenance_id": report["provenance_id"],
           "summary_sha256": file_hash(p / "summary.json"), "tests_sha256": file_hash(p / "tests.xml"),
           "test_count": len(cases), "failures": 0, "skips": 0,
           "adapter_source_hashes_verified": len(adapter["source_sha256"]),
           "runtime_dependency_hashes_verified": len(runtime),
           "synthetic_files_verified_unchanged": len(plan["synthetic_preserved_sha256"]),
           "query_prefix_seconds": {"minimum": min(times), "median": statistics.median(times), "maximum": max(times)},
           "query_diagnostics": [{"intention": q["task_id"], "seconds": q["wall_seconds"],
                                  "prefix_chunk_count": q["prefix_chunk_count"],
                                  "positive_scoring_chunks": q["positive_scoring_chunks"],
                                  "saved_hit_rows": len(q["bounded_chunk_hits"])} for q in queries],
           "rss_mib": {"after_build": report["resources"]["build"]["process_peak_rss_after_bytes"] / 2**20,
                       "after_load": report["resources"]["verified_load"]["process_peak_rss_after_bytes"] / 2**20,
                       "final": report["process_peak_rss_final_bytes"] / 2**20},
           "verification_code_sha256": file_hash(__file__)}
write_json(p / "verification.json", receipt)
print(json.dumps(receipt, indent=2))
