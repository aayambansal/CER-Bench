#!/usr/bin/env python3
"""Fail-closed local submission gate, separate from software test success.

No network, models, installs, pickle loading, or paper mutation. This snapshot
cannot certify a future release merely by changing an attestation boolean.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "identity": "data/processed/identity_repair_v1/report.json",
    "splits": "data/benchmark/v1_2_repaired_components/manifest.json",
    "qrels": "results/readiness/qrel_sensitivity/v1_final/INPUT_AUDIT.json",
    "human_kit": "annotations/human_qrel_v2/manifest.json",
    "human_queries": "annotations/human_queries_v1/manifest.json",
    "budget": "results/readiness/budget/legacy_audit_final.json",
    "clearance": "configs/submission_clearance.json",
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_path(root, path):
    p = Path(path)
    resolved = (p if p.is_absolute() else root / p).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("evidence path is outside the benchmark root")
    return resolved


def read_test_summary(path):
    tree = ET.parse(path).getroot()
    cases = list(tree.iter("testcase"))
    return {
        "tests": len(cases),
        "failures": sum(c.find("failure") is not None for c in cases),
        "errors": sum(c.find("error") is not None for c in cases),
        "skipped": sum(c.find("skipped") is not None for c in cases),
        "note": "Software-only evidence; not human validation or empirical readiness.",
    }


def evaluate(root=ROOT, test_report=None):
    root = root.resolve()
    blockers, hashes, reports, checks = [], {}, {}, []

    def block(code, reason):
        blockers.append({"code": code, "reason": reason})

    for name, relative in SOURCES.items():
        try:
            p = checked_path(root, relative)
            reports[name] = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(reports[name], dict):
                raise ValueError("expected JSON object")
            hashes[relative] = digest(p)
        except (OSError, ValueError) as e:
            block("missing_or_invalid_" + name, str(e))

    # Validate stored source bindings, not just the existence of an audit file.
    bindings = 0
    for name, key in (("identity", "source_hashes"), ("splits", "source_hashes"),
                      ("human_kit", "input_sha256")):
        for source, expected in reports.get(name, {}).get(key, {}).items():
            try:
                p = checked_path(root, source)
                if digest(p) != expected:
                    raise ValueError("source hash changed: " + str(p.relative_to(root)))
                bindings += 1
            except (OSError, ValueError) as e:
                block("stale_" + name + "_evidence", str(e))

    identity = reports.get("identity", {})
    if identity.get("release_blocked") is not False:
        block("identity_unadjudicated", "Identity repair is a lossy candidate, not adjudicated source content.")
    if identity.get("raw_front_pmid_disagreement_records", 0):
        block("historical_text_misattachment", str(identity["raw_front_pmid_disagreement_records"]) + " corpus records disagree with local article-front PMID evidence.")
    splits = reports.get("splits", {})
    overlap = splits.get("cross_split_overlap", {})
    required_overlap_keys = {"documents", "chunks", "clusters", "exact_questions", "lexical_near_duplicates"}
    if (not isinstance(overlap, dict) or not required_overlap_keys <= set(overlap)
            or any(type(v) is not int or v != 0 for v in overlap.values())):
        block("split_invariants", "Missing or nonzero component-split overlap evidence.")
    else:
        checks.append("Candidate split has zero checked annotated/lexical cross-split overlap; this is not a fresh holdout.")
    if splits.get("release_blocked") is not False:
        block("exposed_proxy_test", "Reassigned existing questions remain exposed synthetic proxy tasks.")
    qrels = reports.get("qrels", {})
    if qrels.get("status") != "pass" or qrels.get("errors"):
        block("historical_qrel_contract", "Strict historical input audit fails; stored expanded labels omit a normalized relevant judgment and corpus IDs conflict.")
    if reports.get("human_kit", {}).get("release_blocked") is not False:
        block("human_validation_absent", "The provisional kit contains no completed independent human validation and must be rebuilt after source repair.")
    if reports.get("human_queries", {}).get("status") != "validated_human_queries":
        block("independent_queries_absent", "Human-written query slots are blank, not experimental data.")

    # The checked legacy records are all unbound. A new, versioned live runner and
    # locked corpus are prerequisites; dry-run plans cannot satisfy this gate.
    block("budget_runs_not_complete", "Legacy controller shards are incomplete and lack required provenance; seven-condition plans and fake-provider tests are not model runs.")
    block("structural_and_selective_results_absent", "Structural and selective evaluators are tested, but adjudicated real-data inputs and held-out results are absent.")
    block("new_pdf_not_verified", "No revised empirical manuscript PDF has been compiled and independently checked in this revision.")

    clearance = reports.get("clearance", {})
    review = clearance.get("required_author_review", {})
    unreviewed = sorted(k for k, v in review.items() if v is not True)
    if not review or unreviewed or not clearance.get("reviewer_attestation"):
        block("author_and_scientific_clearance", "Unresolved author/scientific review: " + ", ".join(unreviewed or ["missing attestation"]))

    software = None
    if test_report is not None:
        try:
            p = checked_path(root, test_report)
            software = read_test_summary(p)
            hashes[str(p.relative_to(root))] = digest(p)
            if not software["tests"] or software["failures"] or software["errors"]:
                block("software_tests", "Software suite is empty or failing.")
        except (OSError, ValueError, ET.ParseError) as e:
            block("software_test_evidence", str(e))
    script = root / "scripts/45_submission_readiness.py"
    if script.exists():
        hashes["scripts/45_submission_readiness.py"] = digest(script)
    return {"version": "cerbench.submission-readiness.v1", "submission_ready": not blockers,
            "release_blocked": bool(blockers), "status": "BLOCKED" if blockers else "READY",
            "scope": "local_revision_snapshot_not_acceptance_prediction", "blockers": blockers,
            "local_checks": checks, "verified_source_bindings": bindings,
            "software_tests": software, "evidence_sha256": hashes,
            "authority": clearance.get("authorization", {}),
            "deadlines": {"abstract_and_author_list": "2026-09-18 23:59 AoE",
                          "full_paper_and_supplement": "2026-09-25 23:59 AoE",
                          "official_source": "https://iclr.cc/Conferences/2027/AuthorGuidelines",
                          "main_text_page_limit": 9}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--test-report", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    result = evaluate(test_report=args.test_report)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as f:
            f.write(text)
    print(json.dumps({"status": result["status"], "submission_ready": result["submission_ready"],
                      "blockers": result["blockers"], "software_tests": result["software_tests"]}, indent=2))
    return 2 if args.require_ready and result["release_blocked"] else 0


if __name__ == "__main__":
    sys.exit(main())
