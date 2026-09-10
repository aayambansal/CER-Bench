"""Exact local execution/provenance driver; no downloads or optional packages."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PYTHON = "/Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python"
sys.path.insert(0, str(ROOT))
from src.corpus.identity import sha256, write_json


def main():
    originals = [ROOT / f"data/processed/{n}" for n in ("corpus.jsonl", "chunks.jsonl", "corpus_stats.json")]
    originals += sorted(p for p in (ROOT / "data/benchmark").rglob("*") if p.is_file() and "v1_2_repaired_components" not in p.parts)
    originals += sorted(p for p in (ROOT / "results").rglob("*") if p.is_file() and OUT not in p.parents)
    hashes = {str(p): sha256(p) for p in originals}
    write_json(OUT / "historical_hashes_before.json", hashes)
    commands = [
        [PYTHON, "-m", "pytest", "-q", "tests/test_identity_repair.py", "tests/test_component_splits_identity.py", "-p", "no:cacheprovider", f"--junitxml={OUT / 'pytest.xml'}"],
        [PYTHON, "scripts/39_audit_repair_identity.py", "--root", str(ROOT), "--output", str(ROOT / "data/processed/identity_repair_v1")],
        [PYTHON, "scripts/38_make_component_disjoint_splits.py", "--source", str(ROOT / "data/processed/identity_repair_v1"), "--output", str(ROOT / "data/benchmark/v1_2_repaired_components"), "--seed", "20260826"],
    ]
    executions = []
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    for name, command in zip(("pytest", "audit", "splits"), commands):
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
        (OUT / f"{name}.stdout.txt").write_text(result.stdout, encoding="utf-8")
        (OUT / f"{name}.stderr.txt").write_text(result.stderr, encoding="utf-8")
        executions.append({"name": name, "argv": command, "cwd": str(ROOT), "returncode": result.returncode,
                           "environment_overrides": {k: env[k] for k in ("PYTHONDONTWRITEBYTECODE", "PYTEST_DISABLE_PLUGIN_AUTOLOAD")}})
        write_json(OUT / "commands.json", executions)
        print(name, "returncode", result.returncode, flush=True)
        print(result.stdout, flush=True)
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            raise SystemExit(result.returncode)
    after = {str(p): sha256(p) for p in originals}
    write_json(OUT / "historical_hashes_after.json", after)
    changed = [p for p, digest in hashes.items() if after[p] != digest]
    write_json(OUT / "preservation.json", {"historical_files_checked": len(hashes), "changed": changed,
                                          "preserved": not changed, "release_blocked": True})
    if changed:
        raise SystemExit("Historical files changed (possibly concurrent work); inspect preservation.json")


if __name__ == "__main__":
    main()
