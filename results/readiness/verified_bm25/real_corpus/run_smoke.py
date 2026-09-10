"""Reproducible real-corpus, unscored offline smoke. Never calls a provider.

Fresh index/output required. resource.ru_maxrss is process-lifetime high water,
not allocation delta or isolated per-phase memory. No tracemalloc timing overhead.
"""
from collections import Counter
import argparse
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.retrieval.verified_bm25 import (VerifiedBM25, build_index, file_hash,
                                       digest, parse_json, tokenize, write_json, read_jsonl)
from src.evaluation import budget_protocol as bp

PINNED_MANIFEST = "59d7751daeb67925903629ed6c542c1e98874d17b0499e43f3a387cc1168b5ae"
CONDITIONS = ("original_top24", "repeated_original_3x8", "heuristic_keywords_3x8")


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def measured(fn):
    before = peak_rss_bytes()
    start = time.perf_counter()
    value = fn()
    return value, {"wall_seconds": time.perf_counter() - start,
                   "process_peak_rss_before_bytes": before,
                   "process_peak_rss_after_bytes": peak_rss_bytes()}


def independent_selector(idx, query, candidates):
    """Independent document reduction over the unsorted original-query scores."""
    allowed = set(candidates)
    scores, best = idx.scores(query), {}
    for i, row in enumerate(idx.rows):
        did = row["doc_id"]
        if did in allowed and (did not in best or scores[i] > best[did][0]):
            best[did] = (float(scores[i]), i)
    order = sorted(best, key=lambda d: (-best[d][0], best[d][1]))
    return [{"doc_id": d, "chunk_id": idx.rows[best[d][1]]["chunk_id"],
             "score": best[d][0]} for d in order]


def formula_check(idx, queries):
    # Re-tokenize source text; no index df, idf, tf or lengths used as reference.
    df, total_tokens = Counter(), 0
    for row in idx.rows:
        counts = Counter(tokenize(row["text"]))
        df.update(counts.keys())
        total_tokens += sum(counts.values())
    n = len(idx.rows)
    raw = {t: math.log(n - f + .5) - math.log(f + .5) for t, f in df.items()}
    average = math.fsum(raw.values()) / len(raw)
    idf = {t: .25 * average if value < 0 else value for t, value in raw.items()}
    # Fixed positional samples plus high-scoring chunks; selection has no labels.
    sample = {0, n // 4, n // 2, 3 * n // 4, n - 1}
    for q in queries:
        sample.update(int(i) for i in np.argsort(-idx.scores(q), kind="stable")[:3])
    checks = []
    for q in [*queries, "the the cancer", "zzzznotavocabularyterm999"]:
        actual = idx.scores(q)
        for i in sorted(sample):
            counts = Counter(tokenize(idx.rows[i]["text"]))
            expected = sum(idf.get(t, 0) * counts[t] * 2.5 /
                           (counts[t] + 1.5 * (.25 + .75 * sum(counts.values()) / (total_tokens / n)))
                           for t in tokenize(q))
            error = abs(float(actual[i]) - expected)
            if not math.isclose(float(actual[i]), expected, rel_tol=1e-12, abs_tol=1e-12):
                raise AssertionError("Independent BM25 formula disagreement")
            checks.append({"query": q, "chunk_id": idx.rows[i]["chunk_id"],
                           "actual": float(actual[i]), "expected": expected, "absolute_error": error})
    return {"passed": True, "comparison_count": len(checks), "sampled_chunks": len(sample),
            "max_absolute_error": max(c["absolute_error"] for c in checks),
            "reference_total_tokens": total_tokens, "reference_vocabulary": len(df),
            "negative_raw_idf_terms": sum(v < 0 for v in raw.values()),
            "reference_average_idf": average, "checks": checks}


def verified_factory(index_dir, inputs):
    """Adapter seam: validates original inputs AND both index files at invocation."""
    def factory(chunks, metadata):
        idx = VerifiedBM25.load(index_dir, **inputs)
        if chunks != idx.rows or metadata != idx.meta:
            raise ValueError("Injected adapter inputs differ from verified index binding")
        return idx
    return factory


def adapter_dry_run(idx, index_dir, inputs, probes, output):
    path = ROOT / "scripts/48_run_live_budget.py"
    spec = importlib.util.spec_from_file_location("verified_real_adapter", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    tasks_path, config_path = output / "adapter_tasks.jsonl", output / "adapter_config.json"
    tasks_path.write_text("".join(json.dumps(p) + "\n" for p in probes))
    write_json(config_path, {"conditions": list(CONDITIONS), "model": "none", "provider": "offline"})
    # The adapter's JSONL reader uses str.splitlines(), which incorrectly splits
    # embedded U+2028/U+2029. Its whole-JSON-array path preserves these records.
    # Keep explicit, hash-bound transport copies; never rewrite original inputs.
    transport = output / "adapter_transport"
    transport.mkdir()
    for key in ("corpus", "chunks"):
        write_json(transport / (key + ".json"), read_jsonl(inputs[key]))
    invoked = []
    factory = verified_factory(index_dir, inputs)

    def instrumented(chunks, metadata):
        verified = factory(chunks, metadata)
        for probe in probes:
            hits = verified.search(probe["question"], max_unique_docs=24)
            docs = list(dict.fromkeys(r["doc_id"] for r in hits))
            assert len(docs) == 24
            assert docs == list(dict.fromkeys(r["doc_id"] for r in idx.search(probe["question"], max_unique_docs=24)))
            invoked.append({"task_id": probe["task_id"], "candidate_docs": docs,
                            "seconds": verified.last_trace["seconds"]})
        return verified

    def forbidden_provider(*args, **kwargs):
        raise AssertionError("Provider construction prohibited in offline smoke")

    args = SimpleNamespace(config=config_path, tasks=tasks_path, corpus=transport / "corpus.json",
                           chunks=transport / "chunks.json", index=index_dir / "index.json",
                           stage="dev-smoke", split="dev", repeat=1, live=False,
                           output=output / "adapter_dry")
    rows = adapter.execute(args, client_factory=forbidden_provider, search_factory=instrumented)
    assert len(rows) == 24 and len(invoked) == 8
    assert all(r["status"] == "dry_run" and r["executed_model_calls"] == 0 and
               r["executed_search_calls"] == 0 for r in rows)
    manifest = parse_json((output / "adapter_dry/manifest.json").read_text())
    assert bp.validate_manifest(manifest)
    return {"status": "offline_dry_only", "dry_plan_rows": len(rows),
            "actual_searches_in_injected_factory": invoked,
            "executed_provider_calls": 0, "executed_generation_calls": 0,
            "note": "Adapter marks injected dependencies synthetic=true; retained unchanged. Real corpus local searches occurred in factory only. Dry plans did not execute protocol searches or feedback generation.",
            "adapter_provenance_id": manifest["provenance_id"]}


def run(args):
    output, index_dir = Path(args.output), Path(args.index)
    inputs = {k: Path(getattr(args, k)) for k in ("corpus", "chunks", "manifest")}
    if file_hash(inputs["manifest"]) != PINNED_MANIFEST:
        raise ValueError("Pinned authoritative manifest mismatch")
    if (index_dir / "index.json").exists() or (index_dir / "postings.npz").exists():
        raise ValueError("Fresh real index destination required")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "predeclared_plan.json").exists():
        raise ValueError("Fresh smoke output required")
    probes_file = Path(args.probes)
    declared = parse_json(probes_file.read_text())
    probes = declared["probes"]
    if len(probes) != 8 or len({p["task_family"] for p in probes}) != 8:
        raise ValueError("Exactly eight capability intentions required")
    sources = [Path(__file__), ROOT / "src/retrieval/verified_bm25.py",
               ROOT / "scripts/49_index_and_run_verified_bm25.py",
               ROOT / "src/evaluation/budget_protocol.py", ROOT / "scripts/48_run_live_budget.py",
               ROOT / "src/agents/provider_clients.py"]
    source_hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in sources}
    preserved = ROOT / "results/readiness/verified_bm25/synthetic"
    preserved_hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in preserved.rglob("*") if p.is_file()}
    plan = {"schema": "verified-real-corpus-smoke-plan-v1", "probes": declared,
            "probes_sha256": file_hash(probes_file), "source_sha256": source_hashes,
            "input_sha256": {k: file_hash(p) for k, p in inputs.items()},
            "conditions": list(CONDITIONS), "max_unique_candidates": 24,
            "saved_search_hit_limit": 24, "saved_text_characters": 0,
            "label_status": "unscored_retrieval_smoke_no_qrels", "publication_claim": False,
            "historical_tasks_or_qrels_used": False, "human_validation": False,
            "synthetic_preserved_sha256": preserved_hashes,
            "python": sys.version, "executable": sys.executable,
            "executable_sha256": file_hash(sys.executable), "numpy_version": np.__version__}
    # Durable declaration precedes build and any retrieval, not a post-hoc probe selection.
    write_json(output / "predeclared_plan.json", plan)
    meta, build_resource = measured(lambda: build_index(output=index_dir, **inputs))
    idx, load_resource = measured(lambda: VerifiedBM25.load(index_dir, **inputs))
    if len(idx.doc_ids) != 4936 or len(idx.rows) != 10313:
        raise AssertionError("Unexpected authoritative corpus counts")
    resources = {"measurement": "Actual single-process wall time without tracemalloc. RSS is cumulative process-lifetime high water including interpreter/native allocations, not a phase-specific delta.",
                 "build": build_resource, "verified_load": load_resource}
    write_json(output / "build_load_resources.json", resources)
    records, query_diagnostics = [], []
    for probe in probes:
        query = probe["question"]
        prefix, timing = measured(lambda: idx.search(query, max_unique_docs=24))
        candidates = list(dict.fromkeys(r["doc_id"] for r in prefix))
        assert len(candidates) == 24
        scored = idx.scores(query)
        query_diagnostics.append({"task_id": probe["task_id"], "query": query, **timing,
                                  "query_tokens": tokenize(query),
                                  "oov_tokens": [t for t in tokenize(query) if t not in idx.terms],
                                  "positive_scoring_chunks": int(np.sum(scored > 0)),
                                  "prefix_chunk_count": len(prefix), "candidate_docs": candidates,
                                  "bounded_chunk_hits": [{k: h[k] for k in ("doc_id", "chunk_id", "score")} for h in prefix[:24]],
                                  "truncated_saved_hit_rows": max(0, len(prefix) - 24)})
        condition_rows = {}
        for condition in CONDITIONS:
            row, timing = measured(lambda: bp.run_protocol(probe, condition, idx, model="none", provider="offline"))
            assert row["status"] == "complete" and len(set(row["candidate_docs"])) == 24
            assert row["accounting"]["model_calls"] == 0 and row["calls"] == []
            ranking = idx.rank(query, row["candidate_docs"], top_k=24)
            assert ranking == independent_selector(idx, query, row["candidate_docs"])
            assert row["common_selector_docs"] == [r["doc_id"] for r in ranking[:20]]
            condition_rows[condition] = row
            # Keep admitted hits only (24/condition), no full ranking or article text.
            records.append({"task_id": probe["task_id"], "capability_intention": probe["task_family"],
                            "condition": condition, "status": row["status"], "queries": row["queries"],
                            "candidate_docs": row["candidate_docs"], "document_ranking": ranking,
                            "common_selector_docs": row["common_selector_docs"], "rrf_docs": row["rrf_docs"],
                            "rounds": [{"query": rd["query"], "requested_new_docs": rd["requested_new_docs"],
                                        "shortfall": rd["shortfall"], "inspected_chunk_count": len(rd["inspected_chunks"]),
                                        "admitted_chunks": [{k: h[k] for k in ("doc_id", "chunk_id", "score")} for h in rd["candidate_chunks"]]}
                                       for rd in row["rounds"]],
                            "accounting": row["accounting"], "resources": timing,
                            "original_query_max_selector_verified": True,
                            "oracle_labels_seen": False, "label_status": plan["label_status"],
                            "feedback_generation_executed": False, "publication_claim": False})
        original, repeated = (condition_rows[c] for c in CONDITIONS[:2])
        for key in ("candidate_docs", "common_selector_docs", "rrf_docs"):
            assert original[key] == repeated[key], f"Repeated-original mismatch: {key}"
        assert original["candidate_docs"] == candidates
        query_diagnostics[-1]["same_query_budget_equivalence_verified"] = True
    write_json(output / "query_diagnostics.json", query_diagnostics)
    write_json(output / "protocol_diagnostics.json", records)
    reference, reference_resources = measured(lambda: formula_check(idx, [p["question"] for p in probes]))
    reference["resources"] = reference_resources
    write_json(output / "independent_formula_checks.json", reference)
    adapter, adapter_resources = measured(lambda: adapter_dry_run(idx, index_dir, inputs, probes, output))
    adapter["resources"] = adapter_resources
    write_json(output / "adapter_integration.json", adapter)
    assert all(file_hash(ROOT / p) == h for p, h in source_hashes.items()), "Source changed during execution"
    assert all(file_hash(p) == plan["input_sha256"][k] for k, p in inputs.items()), "Input changed during execution"
    assert all(file_hash(ROOT / p) == h for p, h in preserved_hashes.items()), "Synthetic outputs changed"
    report = {"schema": "verified-real-corpus-smoke-v1", "status": "complete_unscored_smoke",
              "label_status": plan["label_status"], "human_validation": False, "publication_claim": False,
              "documents": len(idx.doc_ids), "chunks": len(idx.rows), "vocabulary": len(idx.terms),
              "tokens": int(idx.arrays["lengths"].sum()), "postings": len(idx.arrays["tf"]),
              "probe_count": len(probes), "protocol_runs": len(records),
              "index_paths": [str((index_dir / f).resolve()) for f in ("index.json", "postings.npz")],
              "index_sha256": {f: file_hash(index_dir / f) for f in ("index.json", "postings.npz")},
              "index_bytes": {f: (index_dir / f).stat().st_size for f in ("index.json", "postings.npz")},
              "input_hashes": meta["input_hashes"], "source_sha256": source_hashes,
              "resources": resources, "process_peak_rss_final_bytes": peak_rss_bytes(),
              "formula_comparisons": reference["comparison_count"],
              "formula_max_absolute_error": reference["max_absolute_error"],
              "adapter_dry_plan_rows": adapter["dry_plan_rows"],
              "executed_model_calls": 0, "feedback_generation_executed": False,
              "synthetic_outputs_preserved": True,
              "checks": {"original_max_selector": True, "repeated_original_equivalence": True,
                         "all_three_nonllm_conditions": True, "input_index_code_hashes": True,
                         "independent_naive_formula": True, "adapter_offline_dry": True},
              "output_sha256": {str(p.relative_to(output)): file_hash(p) for p in output.rglob("*")
                                if p.is_file() and "attempt1" not in p.parts and p.name not in ("summary.json", "run_smoke.py", "probes.json")}}
    report["provenance_id"] = digest(report)
    write_json(output / "summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "chunks", "manifest", "index", "probes", "output"):
        parser.add_argument("--" + name, required=True)
    report = run(parser.parse_args())
    print(json.dumps({k: report[k] for k in ("status", "documents", "chunks", "resources",
                                            "formula_comparisons", "formula_max_absolute_error",
                                            "protocol_runs", "adapter_dry_plan_rows", "provenance_id")}, indent=2))
