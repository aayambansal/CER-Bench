"""Post-suite receipt: never changes the frozen execution summary or native manifest."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.retrieval.verified_bm25 import digest, file_hash, write_json
from src.evaluation.budget_protocol import validate_manifest


def verify():
    p = Path(__file__).resolve().parent
    report = json.loads((p / "summary.json").read_text())
    assert report["provenance_id"] == digest({k: v for k, v in report.items() if k != "provenance_id"})
    for name, sha in report["source_sha256"].items():
        assert file_hash(ROOT / name) == sha, name
    for name, sha in report["output_sha256"].items():
        assert file_hash(p / name) == sha, name
    for name, sha in report["index_sha256"].items():
        assert file_hash(ROOT / "data/processed/authoritative_fulltext_v1_bm25" / name) == sha
    plan = json.loads((p / "plan.json").read_text())
    for name, sha in plan["historical_files_sha256"].items():
        assert file_hash(ROOT / name) == sha, name
    manifest = json.loads((p / "native48/manifest.json").read_text())
    assert validate_manifest(manifest)
    for mapping in (manifest["source_sha256"], manifest["index_sha256"], manifest["config"]["runtime_dependencies"]):
        for name, sha in mapping.items():
            assert file_hash(name) == sha, name
    cases = ET.parse(p / "all_tests.xml")
    counts = {"tests": len(list(cases.iter("testcase"))),
              **{name: len(list(cases.iter(tag))) for name, tag in
                 (("failures", "failure"), ("errors", "error"), ("skips", "skipped"))}}
    assert counts["tests"] > 0 and counts["failures"] == counts["errors"] == 0
    receipt = {"status": "verified", "full_configured_repository_suite": True,
               "suite_command": "python -m pytest -q --junitxml=results/readiness/verified_bm25/real_corpus_v2/all_tests.xml",
               "counts": counts, "human_validation": False, "live": False, "synthetic": False,
               "label_status": "unscored_retrieval_smoke_no_qrels", "executed_model_calls": 0,
               "summary_provenance_id": report["provenance_id"],
               "summary_sha256": file_hash(p / "summary.json"),
               "native_manifest_sha256": file_hash(p / "native48/manifest.json"),
               "tests_xml_sha256": file_hash(p / "all_tests.xml"),
               "verification_code_sha256": file_hash(__file__),
               "current_source_bindings_verified": len(report["source_sha256"]),
               "runtime_bindings_verified": len(manifest["config"]["runtime_dependencies"]),
               "historical_files_unchanged": len(plan["historical_files_sha256"])}
    write_json(p / "verification.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    verify()
