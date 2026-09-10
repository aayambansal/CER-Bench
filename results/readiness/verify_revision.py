"""Reproducible local integration verification; never invokes a model or network."""
from __future__ import annotations

import ast
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/readiness/integration_final")
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    if not OUT.is_relative_to(ROOT):
        raise ValueError("Verification output must be within the benchmark root")
    OUT.mkdir(parents=True, exist_ok=False)
    def relative(name):
        return str((OUT / name).relative_to(ROOT))
    final = OUT / "revision_verification.json"
    if final.exists():
        raise FileExistsError("Choose a separately versioned verification run; do not overwrite prior evidence")
    receipts = []
    tracked = [ROOT / "paper/iclr2027_submission/cerbench_iclr2027.tex"]
    tracked += [p for p in (ROOT / "paper/iclr2027_submission/dist").iterdir()
                if p.suffix in {".pdf", ".zip", ".json"}]
    tracked += [p for p in (ROOT / "results/baselines").glob("*.json*")]
    tracked += [ROOT / "data/processed/corpus.jsonl", ROOT / "data/processed/chunks.jsonl"]
    before = {str(p.relative_to(ROOT)): sha(p) for p in tracked}
    source_files = sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    for p in source_files:
        ast.parse(p.read_text(), filename=str(p))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")

    def run(name, command, expected=0):
        proc = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=120)
        log = OUT / (name + ".log")
        with log.open("x") as handle:
            handle.write("STDOUT\n" + proc.stdout + "\nSTDERR\n" + proc.stderr)
        receipts.append({"name": name, "command": command, "returncode": proc.returncode,
                         "expected_returncode": expected, "log": str(log.relative_to(ROOT)), "sha256": sha(log)})
        if proc.returncode != expected:
            raise RuntimeError(f"{name}: expected {expected}, got {proc.returncode}; see {log}")
        return proc

    run("revision_tests", [PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           "--junitxml=" + relative("final_tests.xml")])
    run("revision_gate", [PYTHON, "scripts/45_submission_readiness.py", "--require-ready",
                          "--test-report", relative("final_tests.xml"),
                          "--output", relative("submission_status.json")], expected=2)
    # Verify that historical packaging/builds stop before touching old artifacts.
    run("revision_build_blocked", ["bash", "scripts/run_iclr2027_submission.sh"], expected=2)
    run("revision_package_blocked", [PYTHON, "scripts/33_build_review_packages.py"], expected=1)
    prefix = [PYTHON, "scripts/40_evaluate_validated_runs.py", "--tasks", "examples/evaluation_contract/tasks.json",
              "--runs", "examples/evaluation_contract/runs.json", "--corpus", "examples/evaluation_contract/corpus.json"]
    run("revision_evaluator_verified_fixture", prefix + ["--annotations", "examples/evaluation_contract/annotations.json",
        "--selective-judgments", "examples/evaluation_contract/selective_judgments.json",
        "--calibration", "examples/evaluation_contract/calibration.json", "--calibration-split", "dev",
        "--output", relative("final_synthetic_verified.json")])
    run("revision_evaluator_proxy_fixture", prefix + ["--annotations", "examples/evaluation_contract/annotations_proxy.json",
        "--allow-proxy", "--output", relative("final_synthetic_proxy.json")])
    kit_report = ROOT / "annotations/human_qrel_v2" / (OUT.name + "_blank_validation.json")
    run("revision_blank_kit", [PYTHON, "scripts/43_validate_human_labels.py", "--blank-kit",
        "--report", str(kit_report.relative_to(ROOT))])

    tex_path = ROOT / "paper/iclr2027_revision/cerbench_revision.tex"
    tex = tex_path.read_text()
    bib = (tex_path.parent / "refs.bib").read_text()
    keys = set(re.findall(r"@\w+\s*\{\s*([^,]+),", bib))
    citations = {key.strip() for group in re.findall(r"\\cite\w*\{([^}]+)\}", tex) for key in group.split(",")}
    if not citations <= keys:
        raise AssertionError("Missing bibliography entries: " + str(citations - keys))
    stack = []
    for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", tex):
        kind, name = match.groups()
        if kind == "begin":
            stack.append(name)
        elif not stack or stack.pop() != name:
            raise AssertionError("Unbalanced LaTeX environment: " + name)
    if stack:
        raise AssertionError("Open LaTeX environments: " + str(stack))
    after = {str(p.relative_to(ROOT)): sha(p) for p in tracked}
    if before != after:
        raise AssertionError("Historical files changed during integration verification")
    status = json.loads((OUT / "submission_status.json").read_text())
    if status["submission_ready"] or not status["release_blocked"]:
        raise AssertionError("Blocked evidence unexpectedly cleared")
    result = {"status": "local_software_verified_submission_blocked", "submission_ready": False,
              "software_tests": status["software_tests"], "syntax_checked_python_files": len(source_files),
              "historical_files_unchanged": len(before), "historical_sha256": after,
              "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in source_files},
              "receipts": receipts, "python_version": sys.version,
              "blank_kit_validation": {"path": str(kit_report.relative_to(ROOT)), "sha256": sha(kit_report)},
              "draft_check": {"bibliography_keys_resolve": True, "environments_balanced": True,
                              "latex_compiled": False, "pdf_generated": False, "sha256": sha(tex_path)},
              "external_calls": 0, "human_judgments": 0, "empirical_model_runs": 0,
              "notes": ["Fixture adjudication flags simulate humans; no biomedical validation occurred.",
                        "Expected exit codes 1/2 establish rejection behavior, not successful empirical experiments."]}
    with final.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: result[k] for k in ["status", "software_tests", "syntax_checked_python_files", "historical_files_unchanged", "draft_check"]}, indent=2))


if __name__ == "__main__":
    main()
