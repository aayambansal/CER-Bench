#!/usr/bin/env python3
"""Explicit offline index/build/run/synthetic entry point. See VERIFIED_RETRIEVAL.md."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import tempfile
import statistics
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.verified_bm25 import (VerifiedBM25, build_index, file_hash,
                                        read_jsonl, write_json, parse_json, STATUSES)
from src.evaluation.budget_protocol import run_protocol


def write_jsonl(path, rows):
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def validate_evaluation(tasks_path, qrels_path, expected_path, index):
    tasks = read_jsonl(tasks_path)
    expected = parse_json(Path(expected_path).read_text())
    ids = [t.get("task_id") for t in tasks]
    if (not isinstance(expected, list) or not expected or
            any(not isinstance(i, str) or not i for i in expected) or
            len(set(expected)) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected)):
        raise ValueError("Tasks must match complete explicit expected universe")
    if any(not isinstance(t.get("question"), str) or not t["question"].strip() for t in tasks):
        raise ValueError("Every task needs a nonempty question")
    qrels = parse_json(Path(qrels_path).read_text())
    label = {"synthetic_structural": "synthetic_structural_not_gold",
             "candidate_diagnostic": "candidate_not_gold",
             "authoritative_validated": "explicit_qrels_not_human_validated"}[index.meta["status"]]
    if (not isinstance(qrels, dict) or qrels.get("schema") != "verified-retrieval-qrels-v1" or
            qrels.get("label_status") != label or
            qrels.get("corpus_sha256") != index.meta["input_hashes"]["corpus_sha256"] or
            qrels.get("tasks_sha256") != file_hash(tasks_path) or
            not isinstance(qrels.get("qrels"), dict) or set(qrels["qrels"]) != set(expected)):
        raise ValueError("Qrels status/hash/complete universe mismatch")
    for judgments in qrels["qrels"].values():
        if not isinstance(judgments, dict) or not set(judgments) <= set(index.doc_ids) or any(
                type(g) is not int or g < 0 for g in judgments.values()):
            raise ValueError("Invalid qrel document/grade")
    return tasks, qrels


def run_baseline(args):
    started = time.perf_counter()
    index = VerifiedBM25.load(args.index, corpus=args.corpus, chunks=args.chunks,
                             manifest=args.manifest, mode=args.mode)
    load_seconds = time.perf_counter() - started
    tasks, qrels = validate_evaluation(args.tasks, args.qrels, args.expected_task_ids, index)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "runs.jsonl").exists() or (output / "run_manifest.json").exists():
        raise ValueError("Refusing to overwrite baseline output")
    statuses = []
    (output / "runs.jsonl").touch(exist_ok=False)
    for task in tasks:
        traces = []

        def search(query):
            hits = index.search(query)
            traces.append(dict(index.last_trace))
            return hits

        start = time.perf_counter()
        # Only these two fields cross the retrieval boundary; qrels stay here.
        record = run_protocol({"task_id": task["task_id"], "question": task["question"]},
                              args.condition, search, model="none", provider="offline")
        candidate_ranking = index.rank(task["question"], record["candidate_docs"], top_k=24)
        ranking = candidate_ranking[:20]
        if [r["doc_id"] for r in ranking] != record["common_selector_docs"]:
            raise ValueError("Budget protocol/common-selector ranking disagreement")
        relevant = {d for d, g in qrels["qrels"][task["task_id"]].items() if g > 0}
        selected = set(record["common_selector_docs"])
        record.update(document_ranking=ranking, candidate_document_ranking=candidate_ranking,
                      exact_search_traces=traces,
                      selector_trace=dict(index.last_trace),
                      elapsed_seconds=time.perf_counter() - start,
                      oracle_labels_seen_by_retriever=False,
                      label_status=qrels["label_status"], publication_claim=False,
                      diagnostic_recall_at20=len(selected & relevant) / len(relevant) if relevant else None)
        with (output / "runs.jsonl").open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        statuses.append(record["status"])
    manifest = {"schema": "verified-retrieval-run-v1", "status": index.meta["status"],
                "label_status": qrels["label_status"], "publication_claim": False,
                "oracle_labels_seen_by_retriever": False, "condition": args.condition,
                "expected_task_ids": json.loads(Path(args.expected_task_ids).read_text()),
                "inputs": {k: file_hash(getattr(args, k)) for k in
                           ("corpus", "chunks", "manifest", "tasks", "qrels", "expected_task_ids")},
                "index_json_sha256": file_hash(Path(args.index) / "index.json"),
                "index_npz_sha256": file_hash(Path(args.index) / "postings.npz"),
                "source_sha256": {p.name: file_hash(p) for p in
                                  (Path(__file__), ROOT / "src/retrieval/verified_bm25.py",
                                   ROOT / "src/evaluation/budget_protocol.py")},
                "load_seconds": load_seconds, "total_seconds": time.perf_counter() - started,
                "task_count": len(statuses), "run_sha256": file_hash(output / "runs.jsonl"),
                "all_complete": all(s == "complete" for s in statuses)}
    write_json(output / "run_manifest.json", manifest)
    return manifest


def generate_synthetic(output, n_docs=48, chunks_per_doc=3, seed=1729):
    """Programmatic fiction; structural annotations live outside retrieval text."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Synthetic destination must be empty")
    if n_docs < 24 or chunks_per_doc < 1:
        raise ValueError("Synthetic integration requires >=24 documents and >=1 chunk")
    rng = random.Random(seed)
    corpus_id = f"fiction-structural-{seed}-{n_docs}-{chunks_per_doc}"
    systems = ["algae", "yeast", "bacteria", "moss"]
    processes = ["transport", "repair", "growth"]
    contexts = ["cold", "saline"]
    words = ["measured", "culture", "signal", "response", "assay", "increase", "decrease", "control", "time", "sample"]
    docs, chunks, structure = [], [], {}
    for i in range(n_docs):
        did, article = f"D{i:06d}", f"fiction-article-{i}"
        factors = {"system": systems[i % 4], "process": processes[(i // 4) % 3],
                   "context": contexts[(i // 12) % 2]}
        docs.append({"doc_id": did, "article_id": article, "corpus_id": corpus_id})
        structure[did] = factors
        for j in range(chunks_per_doc):
            text = (f"We measured {factors['process']} in {factors['system']} under {factors['context']} conditions. "
                    + " ".join(rng.choices(words, k=70)))
            chunks.append({"chunk_id": f"opaque-{i * chunks_per_doc + j:07d}",
                           "doc_id": did, "article_id": article, "corpus_id": corpus_id,
                           "text": text})
    write_jsonl(output / "corpus.jsonl", docs)
    write_jsonl(output / "chunks.jsonl", chunks)
    tasks, judgments = [], {}
    for t in range(4):
        task_id = f"T{t:02d}"
        constraints = structure[docs[t]["doc_id"]]
        relevant = [d for d, factors in structure.items() if factors == constraints]
        tasks.append({"task_id": task_id,
                      "question": f"Which studies measured {constraints['process']} in {constraints['system']} under {constraints['context']} conditions?",
                      "synthetic_contract": {"label_status": "synthetic_structural_not_gold",
                                             "constraints": constraints,
                                             "relevance_rule": "exact equality of all three generated factors",
                                             "required_evidence_roles": ["system", "process", "context"],
                                             "supporting_doc_ids": relevant}})
        judgments[task_id] = {d: int(d in relevant) for d in structure}
    write_jsonl(output / "tasks.jsonl", tasks)
    write_json(output / "expected_task_ids.json", [t["task_id"] for t in tasks])
    write_json(output / "structural_labels.json", {"schema": "synthetic-factor-contract-v1",
               "label_status": "synthetic_structural_not_gold", "seed": seed,
               "generated_fiction": True, "human_validated": False,
               "documents": structure, "retrieval_text_contains_label_metadata": False})
    write_json(output / "dataset_manifest.json", {"schema": "verified-retrieval-dataset-v1",
               "status": "synthetic_structural", "corpus_id": corpus_id,
               "identity_validated": True, "canonical_ids_unique": True,
               "document_count": len(docs), "chunk_count": len(chunks),
               "corpus_sha256": file_hash(output / "corpus.jsonl"),
               "chunks_sha256": file_hash(output / "chunks.jsonl")})
    write_json(output / "qrels.json", {"schema": "verified-retrieval-qrels-v1",
               "label_status": "synthetic_structural_not_gold",
               "corpus_sha256": file_hash(output / "corpus.jsonl"),
               "tasks_sha256": file_hash(output / "tasks.jsonl"), "qrels": judgments})


def benchmark(output):
    """Bounded fiction only. Projection is explicitly NOT a real-corpus timing."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "benchmark.json").exists():
        raise ValueError("Refusing to overwrite benchmark")
    measurements = []
    for n_docs in (200, 2000):
        with tempfile.TemporaryDirectory(prefix="verified-bm25-", dir=output) as tmp:
            p = Path(tmp) / "data"
            generate_synthetic(p, n_docs=n_docs)
            inputs = dict(corpus=p / "corpus.jsonl", chunks=p / "chunks.jsonl",
                          manifest=p / "dataset_manifest.json", mode="synthetic_structural")
            started = time.perf_counter()
            meta = build_index(output=p / "index", **inputs)
            build_seconds = time.perf_counter() - started
            started = time.perf_counter()
            idx = VerifiedBM25.load(p / "index", **inputs)
            load_seconds = time.perf_counter() - started
            queries = ["algae transport cold", "yeast repair saline", "moss growth cold",
                       "bacteria transport saline", "culture measured signal"]
            scoring, full = [], []
            for query in queries * 3:
                started = time.perf_counter()
                idx.scores(query)
                scoring.append(time.perf_counter() - started)
                started = time.perf_counter()
                idx.search(query)
                full.append(time.perf_counter() - started)
            measurements.append({"documents": n_docs, "chunks": len(idx.rows), "seed": 1729,
                                 "tokens": int(idx.arrays["lengths"].sum()),
                                 "vocabulary": len(meta["vocabulary"]),
                                 "postings": len(idx.arrays["tf"]), "queries": queries,
                                 "query_repetitions": 3, "build_seconds": build_seconds,
                                 "verified_load_seconds": load_seconds,
                                 "score_seconds": scoring, "full_search_seconds": full,
                                 "score_median_seconds": statistics.median(scoring),
                                 "full_search_median_seconds": statistics.median(full),
                                 "input_hashes": meta["input_hashes"],
                                 "npz_bytes": (p / "index/postings.npz").stat().st_size})
    largest, target = measurements[-1], 1_000_000
    factor = target / largest["chunks"]
    report = {"schema": "verified-bm25-synthetic-benchmark-v1",
              "label_status": "synthetic_structural_not_gold", "publication_claim": False,
              "executable": sys.executable, "python_version": sys.version,
              "source_sha256": {"cli": file_hash(__file__), "retriever": file_hash(ROOT / "src/retrieval/verified_bm25.py")},
              "measurements": measurements,
              "unmeasured_projection": {"target_chunks": target,
                  "assumption": "Same 78-token lengths, vocabulary, query posting density; linear build/load and N log N full-search scaling from largest synthetic point. Not a confidence interval or measured real corpus.",
                  "build_seconds": largest["build_seconds"] * factor,
                  "verified_load_seconds": largest["verified_load_seconds"] * factor,
                  "full_search_seconds": largest["full_search_median_seconds"] * factor * math.log(target) / math.log(largest["chunks"])},
              "limitations": "Small vocabulary and warm local caches; no peak-RSS measurement; full hit JSON serialization excluded; extrapolation can be very inaccurate. Temporary inputs regenerated by seed and generator code."}
    write_json(output / "benchmark.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "run"):
        p = sub.add_parser(command)
        for flag in ("corpus", "chunks", "manifest", "index"):
            p.add_argument("--" + flag, required=True)
        p.add_argument("--mode", choices=sorted(STATUSES), default="authoritative_validated")
        if command == "run":
            for flag in ("tasks", "qrels", "expected-task-ids", "output"):
                p.add_argument("--" + flag, required=True)
            p.add_argument("--condition", choices=("original_top24", "repeated_original_3x8", "heuristic_keywords_3x8"), default="original_top24")
    p = sub.add_parser("synthetic")
    p.add_argument("--output", required=True)
    p.add_argument("--documents", type=int, default=48)
    p.add_argument("--chunks-per-document", type=int, default=3)
    p.add_argument("--seed", type=int, default=1729)
    p = sub.add_parser("benchmark")
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_index(args.corpus, args.chunks, args.manifest, args.index, mode=args.mode)
        print(json.dumps({k: result[k] for k in ("status", "build_seconds", "input_hashes")}))
    elif args.command == "run":
        print(json.dumps(run_baseline(args)))
    elif args.command == "benchmark":
        print(json.dumps(benchmark(args.output)))
    else:
        generate_synthetic(args.output, args.documents, args.chunks_per_document, args.seed)
        print(json.dumps({"output": args.output, "label_status": "synthetic_structural_not_gold"}))


if __name__ == "__main__":
    main()
