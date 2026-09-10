"""Independent disk checks and byte-for-byte replay of published candidates."""
import importlib.util
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.corpus.identity import SPLITS, annotated_ids, read_jsonl, sha256, unique, write_json


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    repair = ROOT / "data/processed/identity_repair_v1"
    split_dir = ROOT / "data/benchmark/v1_2_repaired_components"
    report = json.loads((repair / "report.json").read_text())
    manifest = json.loads((split_dir / "manifest.json").read_text())
    checked_sources, checked_outputs = 0, 0
    for directory, record in ((repair, report), (split_dir, manifest)):
        for path, digest in record["source_hashes"].items():
            assert sha256(path) == digest, path
            checked_sources += 1
        for path, digest in record["output_hashes"].items():
            assert sha256(directory / path) == digest, path
            checked_outputs += 1
    docs = read_jsonl(repair / "corpus.jsonl")
    chunks = read_jsonl(repair / "chunks.jsonl")
    unique(docs, "doc_id")
    unique(chunks, "chunk_id")
    doc_ids = {d["doc_id"] for d in docs}
    chunk_ids = {c["chunk_id"] for c in chunks}
    covered = {c["doc_id"] for c in chunks}
    assert covered <= doc_ids
    assert all(not d["has_fulltext"] and not d["pmcid"] and not any(d.get(k) for k in ("sections", "figure_captions", "table_texts")) for d in docs)
    rows = {s: read_jsonl(split_dir / f"{s}.jsonl") for s in SPLITS}
    split_module = load("38_make_component_disjoint_splits.py")
    overlap = split_module.validate(rows)
    tasks = [t for s in SPLITS for t in rows[s]]
    source_tasks = [t for s in SPLITS for t in read_jsonl(repair / f"{s}.jsonl")]
    assert {t["task_id"] for t in tasks} == {t["task_id"] for t in source_tasks}
    for t in tasks:
        d, c = annotated_ids(t)
        assert d <= covered and c <= chunk_ids, t["task_id"]
    # The only permitted task changes in 38 are split/component provenance.
    lookup = {t["task_id"]: t for t in source_tasks}
    for t in tasks:
        strip = lambda row: {k: v for k, v in row.items() if k not in {"split", "component_id", "source_split"}}
        assert strip(t) == strip(lookup[t["task_id"]])
    with tempfile.TemporaryDirectory(prefix="replay-", dir=OUT) as temp:
        temp = Path(temp)
        load("39_audit_repair_identity.py").audit(ROOT, temp / "repair")
        split_module.make_splits(repair, temp / "splits", manifest["seed"])
        for original, replay in ((repair, temp / "repair"), (split_dir, temp / "splits")):
            assert {p.name for p in original.iterdir()} == {p.name for p in replay.iterdir()}
            for p in original.iterdir():
                assert sha256(p) == sha256(replay / p.name), p.name
    historical = json.loads((OUT / "historical_hashes_before.json").read_text())
    changed = [p for p, h in historical.items() if sha256(p) != h]
    # Other workers own other readiness files. Report concurrent changes without
    # rewriting them; data source changes still fail the checks above/below.
    assert not [p for p in changed if Path(p).is_relative_to(ROOT / "data")], changed
    old_tasks = [t for s in SPLITS for t in read_jsonl(ROOT / f"data/benchmark/{s}.jsonl")]
    original_family = Counter(t["task_family"] for t in old_tasks)
    retained_family = Counter(t["task_family"] for t in tasks)
    result = {
        "status": "passed", "release_blocked": True,
        "source_hash_entries_checked": checked_sources, "output_hash_entries_checked": checked_outputs,
        "historical_files_unchanged": len(historical) - len(changed),
        "other_results_changed_since_initial_snapshot": changed,
        "preservation_note": "Data source hashes must match. Changes to other workers' results are reported, not repaired or attributed by this verifier.",
        "byte_identical_audit_replay": True,
        "byte_identical_split_replay": True, "task_payloads_preserved_except_split_provenance": True,
        "candidate_task_document_and_chunk_references_resolve": True,
        "cross_split_overlap": overlap,
        "coverage_by_family": {f: {"original": n, "retained": retained_family[f], "quarantined": n - retained_family[f]} for f, n in sorted(original_family.items())},
        "raw_xml_unreadable": [e for e in read_jsonl(repair / "raw_identity_evidence.jsonl") if e["status"] != "local_front_identity"],
        "known_6286148_evidence": [e for e in read_jsonl(repair / "fulltext_identity_evidence.jsonl") if e["old_id"] == "6286148"],
    }
    write_json(OUT / "verification.json", result)
    commands = json.loads((OUT / "commands.json").read_text())
    commands.append({"name": "independent_verification", "argv": [sys.executable, "-B", "results/readiness/identity/verify_local.py"], "cwd": str(ROOT), "returncode": 0})
    write_json(OUT / "commands.json", commands)
    print(json.dumps({k: v for k, v in result.items() if k not in {"known_6286148_evidence", "raw_xml_unreadable"}}, indent=2))


if __name__ == "__main__":
    main()
