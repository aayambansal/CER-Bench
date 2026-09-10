"""Offline assertion summary for the original-file provider48 dry integration."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = HERE / "verified_originals_dry"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


manifest = load(OUT / "manifest.json")
plans = load(OUT / "runs.json")
checks = load(OUT / "retrieval_checks.json")
tasks = load(OUT / "selected_tasks.json")
assert manifest["config"]["synthetic"] is False
assert manifest["config"]["live"] is False
assert manifest["config"]["retrieval"]["backend"] == "verified_bm25"
assert len(tasks) == 8 and len(plans) == 56 and len(checks) == 24
assert all(not set(t).intersection({"gold_doc_ids", "qrels", "decision", "confidence"}) for t in tasks)
assert all(p["status"] == "dry_run" and p["synthetic"] is False and
           p["executed_model_calls"] == p["executed_search_calls"] == 0 and
           p["provenance_id"] == manifest["provenance_id"] for p in plans)
assert all(r["status"] == "complete" and r["synthetic"] is False and not r["calls"] and
           len(set(r["candidate_docs"])) == 24 and len(r["common_selector_docs"]) == len(r["rrf_docs"]) == 20
           and r["provenance_id"] == manifest["provenance_id"] for r in checks)
assert not (OUT / "ledger.jsonl").exists()
assert not (OUT / "pending.json").exists()
assert {Path(p).name for p in manifest["index_sha256"]} == {"index.json", "postings.npz", "dataset_manifest.json"}
assert all(sha(p) == h for key in ("index_sha256", "source_sha256") for p, h in manifest[key].items())
assert sha(ROOT / "data/processed/authoritative_fulltext_v1/corpus.jsonl") == manifest["corpus_sha256"]
assert sha(ROOT / "data/processed/authoritative_fulltext_v1/chunks.jsonl") == manifest["config"]["chunks_sha256"]
assert sha(ROOT / "results/readiness/verified_bm25/real_corpus/probes.json") == manifest["task_sha256"]
record = dict(status="verified_offline_only", synthetic=False, new_model_runs=0, provider_calls=0,
              human_validation=False, label_status="unscored_retrieval_smoke_no_qrels",
              input_transport="original_corpus_and_chunks_jsonl_no_array_copy",
              tasks=len(tasks), dry_condition_plans=len(plans), deterministic_retrieval_checks=len(checks),
              planned_model_calls=sum(p["planned_model_calls"] for p in plans),
              configured_max_attempts=manifest["config"]["max_attempts"],
              full_plan_fits_request_cap=sum(p["planned_model_calls"] for p in plans) <= manifest["config"]["max_attempts"],
              documents=manifest["config"]["retrieval"]["document_count"],
              chunks=manifest["config"]["retrieval"]["chunk_count"],
              local_search_invocations=sum(len(r["search_calls"]) for r in checks),
              local_uncached_searches=sum(not c["cache_hit"] for r in checks for c in r["search_calls"]),
              max_protocol_trace_bytes=max(len(json.dumps(r, ensure_ascii=False).encode()) for r in checks),
              trace_max_bytes=manifest["config"]["trace_max_bytes"],
              original_selector="original_query_max_chunk_rank_restricted_to_candidates_top20",
              candidate_unique_doc_budget=24, source_hashes_checked=len(manifest["source_sha256"]),
              index_hashes=manifest["index_sha256"], provenance_id=manifest["provenance_id"],
              output_sha256={p.name: sha(p) for p in OUT.glob("*.json")}, verifier_sha256=sha(__file__))
assert record["max_protocol_trace_bytes"] < record["trace_max_bytes"]
destination = HERE / "verified_originals_summary.json"
with destination.open("x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, ensure_ascii=False, allow_nan=False)
    handle.write("\n")
print(json.dumps({k: v for k, v in record.items() if k not in ("output_sha256", "index_hashes")}))
