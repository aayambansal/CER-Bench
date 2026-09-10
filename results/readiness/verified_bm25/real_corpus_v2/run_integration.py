"""Current native48 offline integration, reusing immutable index; no build/API calls."""
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.retrieval.verified_bm25 import VerifiedBM25, digest, file_hash, write_json
from src.evaluation import budget_protocol as bp

OUT = Path(__file__).resolve().parent
OLD = OUT.parent / "real_corpus"
DATA = ROOT / "data/processed/authoritative_fulltext_v1"
INDEX = ROOT / "data/processed/authoritative_fulltext_v1_bm25"


def snapshot(directory):
    return {str(p.relative_to(ROOT)): file_hash(p) for p in directory.rglob("*") if p.is_file()}


def verify_sources(bindings):
    for name, sha in bindings.items():
        if file_hash(ROOT / name) != sha:
            raise ValueError(f"Stale source binding: {name}")


def measured(fn):
    start = time.perf_counter()
    result = fn()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return result, {"measurement_origin": "current_v2_execution", "wall_seconds": time.perf_counter() - start,
                    "process_peak_rss_after_bytes": int(rss if sys.platform == "darwin" else rss * 1024)}


def independent_selector(idx, question, candidates):
    scores, best, allowed = idx.scores(question), {}, set(candidates)
    for i, chunk in enumerate(idx.rows):
        d = chunk["doc_id"]
        if d in allowed and (d not in best or scores[i] > best[d][0]):
            best[d] = (float(scores[i]), i)
    return [{"doc_id": d, "chunk_id": idx.rows[best[d][1]]["chunk_id"], "score": best[d][0]}
            for d in sorted(best, key=lambda d: (-best[d][0], best[d][1]))]


def run():
    if (OUT / "plan.json").exists():
        raise ValueError("Fresh v2 output required; never overwrite historical receipts")
    old = json.loads((OLD / "summary.json").read_text())
    assert old["provenance_id"] == digest({k: v for k, v in old.items() if k != "provenance_id"})
    index_hashes = {n: file_hash(INDEX / n) for n in ("index.json", "postings.npz")}
    assert index_hashes == old["index_sha256"]
    inputs = {"corpus": DATA / "corpus.jsonl", "chunks": DATA / "chunks.jsonl", "manifest": DATA / "dataset_manifest.json"}
    assert {k + "_sha256": file_hash(p) for k, p in inputs.items()} == old["input_hashes"]
    # Final test source must exist before this snapshot and native48 manifest freeze.
    source_paths = sorted(set(ROOT.glob("src/**/*.py")) | set(ROOT.glob("scripts/*.py")) |
                          set(ROOT.glob("tests/**/*.py")) | set(ROOT.glob("*.py")) |
                          {Path(__file__), OUT / "verify_integration.py"})
    sources = {str(p.relative_to(ROOT)): file_hash(p) for p in source_paths}
    historical = {**snapshot(OLD), **snapshot(OUT.parent / "synthetic")}
    plan = {"schema": "real-corpus-native48-v2-plan", "source_sha256": sources,
            "index_sha256": index_hashes, "input_hashes": old["input_hashes"],
            "historical_files_sha256": historical, "index_rebuilt": False,
            "probes_sha256": file_hash(OLD / "probes.json"), "live": False,
            "human_validation": False, "publication_claim": False}
    write_json(OUT / "plan.json", plan)
    idx, load_resources = measured(lambda: VerifiedBM25.load(INDEX, **inputs))
    path = ROOT / "scripts/48_run_live_budget.py"
    spec = importlib.util.spec_from_file_location("current_native48", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    config = OUT / "config.json"
    conditions = ["original_top24", "repeated_original_3x8", "heuristic_keywords_3x8"]
    write_json(config, {"conditions": conditions, "model": "NO_MODEL_EXECUTED", "provider": "none"})
    args = SimpleNamespace(config=config, tasks=OLD / "probes.json", corpus=inputs["corpus"],
                           chunks=inputs["chunks"], dataset_manifest=inputs["manifest"], index=INDEX,
                           backend="verified_bm25", stage="dev-smoke", split="dev", repeat=1,
                           live=False, dry_retrieval_check=True, campaign_budget=None, output=OUT / "native48")
    # Native defaults only: no replacement client/search factory, no transport copies.
    dry_rows, adapter_resources = measured(lambda: adapter.execute(args))
    manifest = adapter.load(OUT / "native48/manifest.json")
    assert bp.validate_manifest(manifest)
    assert manifest["config"]["synthetic"] is False and manifest["config"]["live"] is False
    assert manifest["config"]["retrieval"]["backend"] == "verified_bm25"
    for p in (INDEX / "index.json", INDEX / "postings.npz", inputs["manifest"]):
        assert manifest["index_sha256"][str(p.resolve())] == file_hash(p)
    assert len(dry_rows) == 24 and all(r["status"] == "dry_run" and not r["synthetic"] and
                                      r["executed_model_calls"] == 0 for r in dry_rows)
    native = adapter.load(OUT / "native48/retrieval_checks.json")
    probes = adapter.load(OLD / "probes.json")["probes"]
    questions = {p["task_id"]: p["question"] for p in probes}
    assert len(native) == 24
    bounded = []
    start = time.perf_counter()
    for row in native:
        assert row["status"] == "complete" and row["synthetic"] is False
        assert row["execution_kind"] == "offline_unscored_retrieval_check"
        assert row["model_runs"] == 0 and row["accounting"]["model_calls"] == 0 and row["calls"] == []
        assert len(row["candidate_docs"]) == len(set(row["candidate_docs"])) == 24
        q = questions[row["task_id"]]
        ranking = idx.rank(q, row["candidate_docs"], top_k=24)
        assert ranking == independent_selector(idx, q, row["candidate_docs"])
        assert row["common_selector_docs"] == [h["doc_id"] for h in ranking[:20]]
        bounded.append({"task_id": row["task_id"], "condition": row["condition"],
                        "status": "complete", "candidate_docs": row["candidate_docs"],
                        "common_selector_docs": row["common_selector_docs"], "rrf_docs": row["rrf_docs"],
                        "queries": row["queries"], "document_ranking": ranking,
                        "accounting": row["accounting"], "original_query_max_selector_verified": True,
                        "feedback_generation_executed": False, "synthetic": False,
                        "human_validation": False, "label_status": "unscored_retrieval_smoke_no_qrels",
                        "rounds": [{"query": rd["query"], "shortfall": rd["shortfall"],
                                    "admitted_chunks": [{k: h[k] for k in ("doc_id", "chunk_id", "score")} for h in rd["candidate_chunks"]]}
                                   for rd in row["rounds"]]})
    for task_id in questions:
        group = {r["condition"]: r for r in bounded if r["task_id"] == task_id}
        assert set(group) == set(conditions)
        for field in ("candidate_docs", "common_selector_docs", "rrf_docs"):
            assert group[conditions[0]][field] == group[conditions[1]][field]
    selector_seconds = time.perf_counter() - start
    write_json(OUT / "protocol_diagnostics.json", bounded)
    integration = {"backend": "verified_bm25", "dependency_injection": False, "synthetic": False,
                   "original_jsonl_inputs": True, "unicode_jsonl_reader_verified": True,
                   "status": "offline_native_dry_only", "dry_plan_rows": len(dry_rows),
                   "actual_native_retrieval_checks": len(native), "probe_count": len(questions),
                   "executed_provider_calls": 0, "executed_generation_calls": 0,
                   "native_manifest_provenance_id": manifest["provenance_id"]}
    write_json(OUT / "adapter_integration.json", integration)
    verify_sources(sources)
    assert all(file_hash(ROOT / p) == sha for p, sha in historical.items())
    assert {n: file_hash(INDEX / n) for n in index_hashes} == index_hashes
    assert {k + "_sha256": file_hash(p) for k, p in inputs.items()} == old["input_hashes"]
    summary = {"schema": "verified-real-corpus-native48-v2", "status": "complete_unscored_smoke",
               "label_status": "unscored_retrieval_smoke_no_qrels", "human_validation": False,
               "publication_claim": False, "live": False, "synthetic": False,
               "executed_model_calls": 0, "feedback_generation_executed": False, "index_rebuilt": False,
               "documents": len(idx.doc_ids), "chunks": len(idx.rows), "probe_count": len(questions),
               "protocol_runs": len(native), "adapter_dry_plan_rows": len(dry_rows),
               "source_sha256": sources, "index_sha256": index_hashes, "input_hashes": old["input_hashes"],
               "resources": {"verified_load": load_resources, "native_adapter": adapter_resources,
                             "selector_checks_seconds": selector_seconds,
                             "memory_note": "Cumulative process-lifetime RSS, not isolated phase allocations."},
               "historical_evidence": {"measurement_origin": "copied_historical_v1_not_rerun",
                                       "summary_path": str((OLD / "summary.json").relative_to(ROOT)),
                                       "summary_sha256": file_hash(OLD / "summary.json"),
                                       "build_resources": old["resources"]["build"],
                                       "formula_comparisons": old["formula_comparisons"],
                                       "formula_max_absolute_error": old["formula_max_absolute_error"]},
               "checks": {"native_verified_backend": True, "original_jsonl_reader": True,
                          "original_max_selector": True, "repeated_original_equivalence": True,
                          "all_three_nonllm_conditions": True, "input_index_code_hashes": True,
                          "historical_outputs_unchanged": True, "adapter_offline_dry": True},
               "output_sha256": {str(p.relative_to(OUT)): file_hash(p) for p in OUT.rglob("*")
                                 if p.is_file() and p.suffix in (".json", ".jsonl")}}
    summary["provenance_id"] = digest(summary)
    write_json(OUT / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "documents", "chunks", "protocol_runs", "resources", "provenance_id")}, indent=2))


if __name__ == "__main__":
    run()
