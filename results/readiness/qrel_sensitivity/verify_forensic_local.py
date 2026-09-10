"""Offline verification, not a biomedical provenance override."""
import hashlib
import json
import math
import os
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/readiness/qrel_sensitivity"
PY = "/Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python"
LABEL = "historical_string_label_diagnostic_not_validated_biomedical_relevance"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def main():
    frozen_before = {p.name: sha(p) for p in (OUT / "v1_final").iterdir()}
    manifest = read(OUT / "v1_forensic/manifest.json")
    commands = []
    for mode, target, expected in [(True, "v1_forensic_repeat", 0), (False, "v1_strict_after_forensic", 1)]:
        cmd = [PY, "scripts/44_analyze_historical_qrel_sensitivity.py", "--output", str(OUT / target)]
        if mode:
            cmd.append("--forensic-token-analysis")
        r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                           capture_output=True, text=True, timeout=60)
        commands.append({"command": cmd, "actual_exit": r.returncode, "expected_exit": expected,
                         "stdout": r.stdout, "stderr": r.stderr})
        assert r.returncode == expected
    a = {p.name: sha(p) for p in (OUT / "v1_forensic").iterdir()}
    b = {p.name: sha(p) for p in (OUT / "v1_forensic_repeat").iterdir()}
    assert a == b
    assert frozen_before == {p.name: sha(p) for p in (OUT / "v1_final").iterdir()}
    for p, h in manifest["inputs_and_code_sha256"].items():
        assert sha(p) == h
    for p, h in manifest["outputs_sha256"].items():
        assert sha(OUT / "v1_forensic" / p) == h
    report = read(OUT / "v1_forensic/analysis.json")
    strict = read(OUT / "v1_strict_after_forensic/FAILURE.json")
    assert strict["errors"] == report["original_failure"]["errors"]
    assert strict["release_blocked"] is True
    for name in ("analysis.json", "manifest.json", "qrel_versions.json"):
        obj = read(OUT / "v1_forensic" / name)
        assert obj["interpretation"] == LABEL and obj["release_blocked"] and obj["release_ready"] is False
        assert obj["input_validation_status"] == "fail"
    for line in (OUT / "v1_forensic/per_task_metrics.jsonl").open():
        row = json.loads(line)
        assert row["interpretation"] == LABEL and row["release_blocked"] and not row["release_ready"]
    # Independent Fraction-based R20 recomputation, not calls to the production evaluator.
    qrels = read(OUT / "v1_forensic/qrel_versions.json")["versions"]
    methods = list(report["means"]["seed"])
    exact = {}
    for regime, gold in qrels.items():
        exact[regime] = {}
        for m in methods:
            rr = [json.loads(line) for line in (ROOT / f"results/baselines/{m}_test.jsonl").open()]
            values = [Fraction(sum(d in gold[r['task_id']] for d in r['retrieved_docs'][:20]), len(gold[r['task_id']]))
                      for r in rr if gold[r['task_id']]]
            assert len(values) == 108
            value = sum(values, Fraction())/108
            exact[regime][m] = value
            assert str(value) == report["recall20_exact_rational_means"][regime][m]
            assert abs(float(value)-report["means"][regime][m]["Recall@20"]) < 1e-15
    counts = {}
    for comparison, metrics in report["primary_10_system_rank_comparisons"].items():
        left, right = comparison.split("_vs_")
        concordant = discordant = tie_left = tie_right = 0
        reversed_pairs = set()
        for a, b in combinations(report["pool_systems"], 2):
            x = exact[left][a]-exact[left][b]
            y = exact[right][a]-exact[right][b]
            tie_left += x == 0
            tie_right += y == 0
            concordant += x*y > 0
            discordant += x*y < 0
            if x*y < 0:
                reversed_pairs.add((a, b))
        tau = (concordant-discordant)/math.sqrt((45-tie_left)*(45-tie_right))
        reported = metrics["Recall@20"]
        assert abs(tau-reported["kendall_tau_b"]) < 1e-15
        assert reversed_pairs == {(p["left"], p["right"]) for p in reported["strict_reversals"]}
        counts[comparison] = {"concordant": concordant, "discordant": discordant,
                              "left_tied_pairs": tie_left, "right_tied_pairs": tie_right, "exact_arithmetic_tau_b": tau}
    for effect in report["missing_pair_effects"]:
        m = effect["method"]
        assert exact["reconstructed_expanded"][m]-exact["stored_expanded"][m] == Fraction(effect["mean_delta_exact"])
    result = {"interpretation": LABEL, "release_blocked": True, "release_ready": False,
              "input_validation_status": "fail", "status": "forensic_token_diagnostic_only",
              "checks": {"byte_identical_independent_forensic_runs": True, "strict_default_still_exits_1": True,
                         "strict_failures_identical": True, "v1_final_immutable": True, "source_and_output_hashes_match": True,
                         "all_36_R20_means_independently_verified_with_fractions": True,
                         "R20_reversals_and_tau_independently_verified_with_fractions": True,
                         "exact_missing_pair_mean_effects_verified": True},
              "pair_counts": counts, "commands": commands, "strict_v1_final_sha256": frozen_before,
              "forensic_output_sha256": {p.name: sha(p) for p in (OUT / "v1_forensic").iterdir()},
              "verification_script_sha256": sha(__file__), "tests_xml_sha256": sha(OUT / "tests_forensic.xml")}
    (OUT / "FORENSIC_VERIFICATION.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"interpretation": LABEL, "release_blocked": True, "checks": result["checks"], "pair_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
