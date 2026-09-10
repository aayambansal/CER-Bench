import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("readiness45", ROOT / "scripts/45_submission_readiness.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_missing_evidence_cannot_clear(tmp_path):
    report = gate.evaluate(tmp_path)
    assert report["status"] == "BLOCKED"
    assert not report["submission_ready"]
    assert any(b["code"] == "missing_or_invalid_identity" for b in report["blockers"])


def test_software_success_is_not_scientific_readiness(tmp_path):
    report = tmp_path / "tests.xml"
    report.write_text('<testsuites><testsuite><testcase name="pass"/></testsuite></testsuites>')
    result = gate.evaluate(tmp_path, report)
    assert result["software_tests"]["tests"] == 1
    assert result["software_tests"]["failures"] == 0
    assert result["release_blocked"]


def test_failed_and_skipped_cases_are_reported(tmp_path):
    p = tmp_path / "tests.xml"
    p.write_text('<testsuite><testcase><failure/></testcase><testcase><skipped/></testcase><testcase><error/></testcase></testsuite>')
    r = gate.read_test_summary(p)
    assert (r["tests"], r["failures"], r["skipped"], r["errors"]) == (3, 1, 1, 1)


def test_evidence_path_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        gate.checked_path(tmp_path, "../outside")


def test_stale_source_binding_blocks(tmp_path):
    identity = tmp_path / gate.SOURCES["identity"]
    identity.parent.mkdir(parents=True)
    source = tmp_path / "source.json"
    source.write_text("changed")
    identity.write_text(json.dumps({"source_hashes": {"source.json": "0" * 64}}))
    result = gate.evaluate(tmp_path)
    assert any(b["code"] == "stale_identity_evidence" for b in result["blockers"])


def test_manually_flipping_status_does_not_clear_missing_experiments(tmp_path):
    p = tmp_path / gate.SOURCES["clearance"]
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"status": "ready", "required_author_review": {"all": True}, "reviewer_attestation": "placeholder"}))
    r = gate.evaluate(tmp_path)
    assert not r["submission_ready"]
    assert any(b["code"] == "budget_runs_not_complete" for b in r["blockers"])


def test_candidate_flat_overlap_manifest(tmp_path):
    p = tmp_path / gate.SOURCES["splits"]
    p.parent.mkdir(parents=True)
    counts = {k: 0 for k in ["documents", "chunks", "clusters", "exact_questions", "lexical_near_duplicates"]}
    p.write_text(json.dumps({"cross_split_overlap": counts, "release_blocked": True}))
    r = gate.evaluate(tmp_path)
    assert not any(b["code"] == "split_invariants" for b in r["blockers"])
    assert r["local_checks"] and r["release_blocked"]
    counts["documents"] = 1
    p.write_text(json.dumps({"cross_split_overlap": counts, "release_blocked": True}))
    assert any(b["code"] == "split_invariants" for b in gate.evaluate(tmp_path)["blockers"])
