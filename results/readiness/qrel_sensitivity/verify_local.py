"""Bounded offline verification of the fail-closed historical diagnostic."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/readiness/qrel_sensitivity"
PYTHON = "/Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    # All subprocesses are local, bounded and contain no retrieval or installation.
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    commands = []
    tests = [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "tests/test_qrel_sensitivity.py",
             "tests/test_strict_metrics.py", "--junitxml=results/readiness/qrel_sensitivity/tests_final.xml"]
    r = subprocess.run(tests, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    (OUT / "tests_final.stdout.txt").write_text(r.stdout)
    (OUT / "tests_final.stderr.txt").write_text(r.stderr)
    commands.append({"command": tests, "returncode": r.returncode, "expected": 0})
    if r.returncode:
        raise RuntimeError("tests failed; see tests_final.stdout.txt")
    for name in ("v1_final", "v1_repeat"):
        cmd = [PYTHON, "scripts/44_analyze_historical_qrel_sensitivity.py", "--output",
               "results/readiness/qrel_sensitivity/" + name]
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        (OUT / f"{name}.stdout.txt").write_text(r.stdout)
        (OUT / f"{name}.stderr.txt").write_text(r.stderr)
        commands.append({"command": cmd, "returncode": r.returncode, "expected": 1})
        if r.returncode != 1:
            raise RuntimeError(f"unexpected exit status: {name}/{r.returncode}")
    files_a = {p.name: sha(p) for p in (OUT / "v1_final").iterdir()}
    files_b = {p.name: sha(p) for p in (OUT / "v1_repeat").iterdir()}
    assert files_a == files_b, "nondeterministic audit outputs"
    manifest = json.loads((OUT / "v1_final/manifest.json").read_text())
    assert manifest["source_changes_during_audit"] == []
    for name, h in manifest["outputs_sha256"].items():
        assert sha(OUT / "v1_final" / name) == h
    for path, h in manifest["inputs_and_code_sha256"].items():
        assert sha(path) == h
    previous = json.loads((OUT / "v1_audit/manifest.json").read_text())
    historical = {p: h for p, h in previous["inputs_and_code_sha256"].items()
                  if "/data/" in p or "/results/baselines/" in p}
    assert all(sha(p) == h for p, h in historical.items()), "historical file changed"
    audit = json.loads((OUT / "v1_final/INPUT_AUDIT.json").read_text())
    assert len(audit["errors"]) == 2
    assert audit["corpus"]["duplicate_id_count"] == 10
    assert len(audit["runs"]) == 12
    assert all(r["row_count"] == 125 and not r["missing_ids"] and not r["unknown_ids"] for r in audit["runs"].values())
    assert audit["judgments"]["expanded_qrel_pairs"] == 1370
    mismatch = audit["judgments"]["stored_expanded_set_mismatches"]
    assert len(mismatch) == 1 and mismatch[0]["task_id"] == "multihop_0189"
    assert mismatch[0]["reconstructed_only"] == ["37307965"]
    assert mismatch[0]["raw_evidence"][0]["physical_row"] == 1337
    assert set(files_a) == {"INPUT_AUDIT.json", "FAILURE.json", "REPORT.md", "manifest.json"}
    result = {"verification_status": "passed", "diagnostic_status": "fail", "release_blocked": True,
              "commands": commands, "byte_identical_independent_runs": True,
              "historical_inputs_unchanged_count": len(historical), "manifest_hashes_verified": True,
              "numerical_outputs_absent_as_required": True, "output_sha256": files_a,
              "verification_script_sha256": sha(__file__),
              "test_outputs_sha256": {p.name: sha(p) for p in OUT.glob("tests_final.*")}}
    (OUT / "VERIFICATION.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
