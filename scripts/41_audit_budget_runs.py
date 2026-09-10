#!/usr/bin/env python3
"""Read-only shard audit. Filename groups are hypotheses, never merged runs."""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.budget_protocol import CONDITIONS, file_hash, validate_manifest


def audit(paths, tasks, split, expected_shards=None, task_sha256=None):
    tids = [t["task_id"] for t in tasks]
    expected = set(tids)
    groups = defaultdict(list)
    files = []
    for path in sorted(map(Path, paths)):
        group = re.sub(r"_shard\d+$", "", path.stem)
        errors, rows = [], []
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("task_id"), str):
                    raise ValueError("Invalid row/task identity")
                rows.append((line_no, row))
            except Exception as exc:
                errors.append({"line": line_no, "error": str(exc)})
        sidecar = path.with_suffix(path.suffix + ".manifest.json")
        bound = None
        if sidecar.exists():
            try:
                bound = json.loads(sidecar.read_text())
                if not validate_manifest(bound):
                    errors.append({"error": "invalid manifest hash/schema"})
                    bound = None
            except Exception as exc:
                errors.append({"error": f"invalid manifest: {exc}"})
        file_report = {"path": str(path), "sha256": file_hash(path), "rows": len(rows),
                       "parse_or_manifest_errors": errors}
        files.append(file_report)
        for line_no, row in rows:
            groups[group].append((path, line_no, row, bound))
        groups[group]  # retain empty shards
    reports = {}
    for group, entries in groups.items():
        counts = Counter()
        models, provenance = set(), set()
        unbound, issues, conditions = [], [], defaultdict(set)
        for path, line_no, row, bound in entries:
            loc = {"file": path.name, "line": line_no, "task_id": row["task_id"]}
            condition = row.get("condition", "legacy_combined")
            identity = (row["task_id"], condition)
            counts[identity] += 1
            conditions[condition].add(row["task_id"])
            models.add((row.get("model"), row.get("provider")))
            pid = row.get("provenance_id")
            provenance.add(pid)
            if not bound or pid != bound["provenance_id"]:
                unbound.append(loc)
                if pid:
                    issues.append({**loc, "error": "row provenance not bound to valid sidecar"})
            else:
                for declared_condition in bound["config"].get("conditions", []):
                    conditions[declared_condition]
                if task_sha256 is not None and bound["task_sha256"] != task_sha256:
                    issues.append({**loc, "error": "manifest task hash mismatch"})
                if bound["split"] != split or row["task_id"] not in bound["task_ids"]:
                    issues.append({**loc, "error": "manifest task/split mismatch"})
                if (row.get("model"), row.get("provider")) != (bound["model"], bound["provider"]):
                    issues.append({**loc, "error": "manifest model/provider mismatch"})
            if row["task_id"] not in expected:
                issues.append({**loc, "error": "unexpected task"})
            if condition != "legacy_combined" and condition not in CONDITIONS:
                issues.append({**loc, "error": "unknown condition"})
            if row.get("status") == "dry_run":
                issues.append({**loc, "error": "dry_run is not experimental completion"})
                continue
            rounds = row.get("iterative_round_candidates") if condition == "legacy_combined" else [
                [c.get("doc_id") for c in rd.get("candidate_chunks", [])] for rd in row.get("rounds", [])]
            if not isinstance(rounds, list) or any(not isinstance(r, list) for r in rounds):
                issues.append({**loc, "error": "invalid/missing round candidates"})
                continue
            flat = [d for r in rounds for d in r]
            if any(not isinstance(d, str) or not d for d in flat):
                issues.append({**loc, "error": "invalid candidate identity"})
                continue
            expected_sizes = [24] if condition in ("original_top24", "one_shot_rewrite") else [8, 8, 8]
            if [len(r) for r in rounds] != expected_sizes or len(flat) != 24 or len(set(flat)) != 24:
                issues.append({**loc, "error": "invalid candidate counts/dedup", "round_sizes": [len(r) for r in rounds], "unique_docs": len(set(flat))})
            queries = row.get("iterative_queries", []) if condition == "legacy_combined" else row.get("queries", [])
            if len(queries) != len(expected_sizes) or any(not isinstance(q, str) or not q.strip() for q in queries):
                issues.append({**loc, "error": "invalid query counts/strings"})
            if condition == "legacy_combined":
                if not isinstance(row.get("one_shot_query"), str) or not row.get("one_shot_query", "").strip():
                    issues.append({**loc, "error": "invalid one-shot query"})
                decision = row.get("final_decision", {})
                selected = decision.get("selected_doc_ids", [])
                if decision.get("decision") not in ("ANSWER", "ABSTAIN", "UNCERTAIN"):
                    issues.append({**loc, "error": "invalid decision"})
            else:
                selected = row.get("common_selector_docs", [])
                if row.get("candidate_docs") != flat:
                    issues.append({**loc, "error": "candidate list differs from rounds"})
                if row.get("status") != "complete":
                    issues.append({**loc, "error": "noncomplete execution", "status": row.get("status")})
                for field in ("common_selector_docs", "rrf_docs"):
                    values = row.get(field, [])
                    if not isinstance(values, list) or len(values) != min(20, len(set(flat))) or any(not isinstance(d, str) for d in values) or len(set(values)) != len(values) or not set(values) <= set(flat):
                        issues.append({**loc, "error": f"invalid {field}"})
            if not isinstance(selected, list) or any(not isinstance(d, str) for d in selected) or len(selected) != len(set(selected)) or not set(selected) <= set(flat):
                issues.append({**loc, "error": "invalid final selection", "selected_doc_ids": selected,
                               "candidate_doc_ids": flat})
        group_files = [f for f in files if re.sub(r"_shard\d+$", "", Path(f["path"]).stem) == group]
        shard_ids = sorted(int(m.group(1)) for f in group_files if (m := re.search(r"_shard(\d+)\.jsonl$", f["path"])))
        n_shards = (expected_shards or {}).get(group)
        missing_shards = sorted(set(range(n_shards)) - set(shard_ids)) if n_shards else None
        shard_coverage = {}
        if n_shards:
            for shard in range(n_shards):
                assigned = {t for i, t in enumerate(tids) if i % n_shards == shard}
                present = {r["task_id"] for p, _, r, _ in entries if p.stem.endswith(f"_shard{shard}")}
                shard_coverage[str(shard)] = {"expected": len(assigned), "covered": len(present & assigned),
                                              "missing_task_ids": sorted(assigned - present),
                                              "misassigned_task_ids": sorted(present - assigned)}
        coverage = {c: {"covered": len(ids & expected), "expected": len(expected),
                        "missing_task_ids": sorted(expected - ids), "unexpected_task_ids": sorted(ids - expected)} for c, ids in conditions.items()}
        duplicates = [{"task_id": t, "condition": c, "count": n} for (t, c), n in sorted(counts.items()) if n > 1]
        reports[group] = {"rows": len(entries), "coverage_by_condition": coverage,
                          "shards_present": shard_ids, "missing_shards": missing_shards,
                          "shard_coverage_modulo_task_order": shard_coverage,
                          "unexpected_shards": sorted(set(shard_ids) - set(range(n_shards))) if n_shards else None,
                          "shard_count_declared": n_shards, "duplicate_identities": duplicates,
                          "models_and_providers": sorted([list(m) for m in models], key=str),
                          "provenance_conflict": len(provenance) > 1 or len(models) > 1,
                          "unbound_rows": len(unbound), "issues": issues,
                          "publication_ready": False,
                          "selection_ranking_verified": False,
                          "note": "Structural audit only. Legacy query provenance, selector order, and actual provider observations cannot be reconstructed."}
    return {"split": split, "expected_tasks": len(expected),
            "duplicate_task_ids": sorted(t for t, n in Counter(tids).items() if n > 1),
            "files": files, "groups": reports,
            "cross_group_overlap_policy": "Separate repeats: never combine coverage or model usage."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--expected-shards", action="append", default=[], metavar="GROUP=N")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = [json.loads(s) for s in args.tasks.read_text().splitlines() if s.strip()]
    declarations = {g: int(n) for g, n in (s.rsplit("=", 1) for s in args.expected_shards)}
    report = audit(args.shards, tasks, args.split, declarations, file_hash(args.tasks))
    report["task_path"] = str(args.tasks.resolve())
    report["task_sha256"] = file_hash(args.tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
    for name, group in report["groups"].items():
        print(name, json.dumps({k: group[k] for k in ("rows", "unbound_rows", "shards_present", "missing_shards", "provenance_conflict")}))
        print(" coverage", {c: (v["covered"], v["expected"]) for c, v in group["coverage_by_condition"].items()}, "issues", len(group["issues"]), "duplicates", len(group["duplicate_identities"]))


if __name__ == "__main__":
    main()
