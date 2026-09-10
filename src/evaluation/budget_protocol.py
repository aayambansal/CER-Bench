"""Offline matched candidate-budget protocol. No provider or network dependencies.

search(query) returns ranked chunk dictionaries with explicit doc_id/chunk_id.
client(request) returns a JSON-compatible provider envelope containing `parsed`
and optionally usage/finish_reason. Neither dependency is constructed here.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

VERSION = "equal-budget-v2"
CONDITIONS = (
    "original_top24", "repeated_original_3x8", "heuristic_keywords_3x8",
    "one_shot_rewrite", "upfront_three_queries", "evidence_feedback",
    "no_accumulated_evidence",
)
PROMPTS = {
    "rewrite": 'Design a lexical query for the question. Return {"query":"..."}.',
    "upfront": 'Before any retrieval, design exactly three lexical queries. Return {"queries":["...","...","..."]}.',
    "refine": 'Refine the lexical query using only the supplied question, query history and evidence. Do not assume evidence is relevant. Return {"query":"..."}.',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest(*, tasks, split, corpus, index_files, source_files, model, provider,
             repeat, config, task_ids):
    if not split or not model or not provider or not repeat or not index_files or not source_files:
        raise ValueError("Explicit split/model/provider/repeat/index/source required")
    value = {
        "protocol_version": VERSION, "split": split, "task_path": str(Path(tasks).resolve()),
        "task_sha256": file_hash(tasks), "corpus_sha256": file_hash(corpus),
        "index_sha256": {str(Path(p).resolve()): file_hash(p) for p in index_files},
        "source_sha256": {str(Path(p).resolve()): file_hash(p) for p in source_files},
        "prompt_sha256": digest(PROMPTS), "model": model, "provider": provider,
        "repeat": repeat, "config": config, "task_ids": list(task_ids),
    }
    value["provenance_id"] = digest(value)
    return value


def validate_manifest(value):
    required = {"protocol_version", "split", "task_path", "task_sha256", "corpus_sha256",
                "index_sha256", "source_sha256", "prompt_sha256", "model", "provider",
                "repeat", "config", "task_ids", "provenance_id"}
    if not isinstance(value, dict) or not required <= value.keys():
        return False
    if any(not isinstance(value[k], dict) for k in ("config", "index_sha256", "source_sha256")) or not isinstance(value["task_ids"], list):
        return False
    return value["protocol_version"] == VERSION and value["provenance_id"] == digest({k: v for k, v in value.items() if k != "provenance_id"})


def token_count(usage, key):
    value = usage.get(key) if isinstance(usage, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return value


def validate_resume(old_manifest, new_manifest, rows):
    if not validate_manifest(old_manifest) or old_manifest != new_manifest:
        raise ValueError("Resume provenance mismatch (legacy/unbound runs cannot resume)")
    seen = set()
    for row in rows:
        identity = (row.get("task_id"), row.get("condition"))
        if identity in seen:
            raise ValueError("Duplicate resume identity")
        if row.get("provenance_id") != new_manifest["provenance_id"]:
            raise ValueError("Resume row provenance mismatch")
        if row.get("task_id") not in new_manifest["task_ids"] or row.get("condition") not in CONDITIONS:
            raise ValueError("Unexpected resume identity")
        if (row.get("model"), row.get("provider")) != (new_manifest["model"], new_manifest["provider"]):
            raise ValueError("Resume model/provider mismatch")
        if row.get("condition") not in new_manifest["config"].get("conditions", CONDITIONS):
            raise ValueError("Resume condition not in manifest")
        if row.get("status") != "complete":
            raise ValueError("Incomplete/error/dry-run rows cannot silently resume")
        candidates = row.get("candidate_docs", [])
        if len(candidates) != 24 or len(set(candidates)) != 24:
            raise ValueError("Resume invalid candidate budget")
        for field in ("common_selector_docs", "rrf_docs"):
            selected = row.get(field, [])
            if len(selected) != 20 or len(set(selected)) != 20 or not set(selected) <= set(candidates):
                raise ValueError("Resume invalid selection")
        seen.add(identity)
    return seen


def request(kind, question, model, provider, history=(), evidence=()):
    # Task annotations never enter requests; history contains query strings only.
    content = {"question": question}
    if kind == "refine":
        content.update(query_history=list(history), evidence=list(evidence))
    return {"model": model, "provider": provider, "temperature": 0,
            "max_tokens": 400, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": PROMPTS[kind]},
                         {"role": "user", "content": json.dumps(content, ensure_ascii=False)}]}


def dry_plan(task, condition, model, provider):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    kind = {"one_shot_rewrite": "rewrite", "upfront_three_queries": "upfront",
            "evidence_feedback": "refine", "no_accumulated_evidence": "refine"}.get(condition)
    return {"task_id": task["task_id"], "condition": condition, "status": "dry_run",
            "model": model, "provider": provider, "executed_model_calls": 0,
            "executed_search_calls": 0,
            "planned_model_calls": 2 if kind == "refine" else int(kind is not None),
            "request_template": request(kind, task["question"], model, provider) if kind else None,
            "deferred_evidence": kind == "refine",
            "note": "Plan only; iterative evidence and later query history are populated at execution."}


def unique_docs(rows):
    return list(dict.fromkeys(row["doc_id"] for row in rows))


def heuristic_queries(question):
    """Deterministic keyword windows; NOT RM3 or relevance feedback."""
    tokens = list(dict.fromkeys(re.findall(r"[a-z0-9-]{2,}", question.lower())))
    return [question, " ".join(tokens[:24]) or question, " ".join(tokens[-24:]) or question]


def select_original(search, question, candidates):
    ranking = unique_docs(search(question))
    ranks = {doc: i for i, doc in enumerate(ranking)}
    return sorted(candidates, key=lambda d: (ranks.get(d, float("inf")), d))[:20]


def rrf(search, queries, candidates):
    scores = Counter()
    for query in queries:
        for rank, doc in enumerate(unique_docs(search(query)), 1):
            if doc in candidates:
                scores[doc] += 1 / (60 + rank)
    return sorted(candidates, key=lambda d: (-scores[d], d))[:20]


def run_protocol(task, condition, search, client=None, *, model="deterministic-test",
                 provider="injected", provenance_id=None, snippet_chars=420):
    """Execute with injected local dependencies; failures return retained traces.

    Cap is 24 UNIQUE documents, not compute equality. Search returns full rankings;
    inspected rows, admitted chunks, displayed evidence and selector work differ.
    No-accumulated ablation sees ONLY the latest round, retaining query history.
    """
    if condition not in CONDITIONS or snippet_chars < 0:
        raise ValueError("Invalid condition/snippet limit")
    question = task["question"]
    record = {"task_id": task["task_id"], "condition": condition, "model": model,
              "provider": provider, "provenance_id": provenance_id, "status": "running",
              "queries": [], "rounds": [], "calls": [], "search_calls": [],
              "candidate_docs": [], "common_selector_docs": [], "rrf_docs": []}
    cache = {}

    def ranked(query, purpose):
        hit = query in cache
        event = {"query": query, "purpose": purpose, "cache_hit": hit}
        record["search_calls"].append(event)
        try:
            if not hit:
                cache[query] = list(search(query))
            rows = cache[query]
            if any(not isinstance(r.get("doc_id"), str) or not r.get("doc_id") or
                   not isinstance(r.get("chunk_id"), str) or not r.get("chunk_id") for r in rows):
                raise ValueError("Search requires explicit nonempty doc_id and chunk_id")
            event.update(ranked_chunk_rows=len(rows), ranked_unique_chunks=len({r["chunk_id"] for r in rows}),
                         ranked_unique_docs=len(unique_docs(rows)))
            return rows
        except Exception as exc:
            event["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise

    def generate(kind, evidence=()):
        displayed = [{"doc_id": r["doc_id"], "chunk_id": r["chunk_id"],
                      "text": str(r.get("text", ""))[:snippet_chars]} for r in evidence]
        req = request(kind, question, model, provider, record["queries"], displayed)
        call = {"raw_request": req, "raw_response": None, "observation": displayed,
                "observed_unique_docs": len(unique_docs(displayed)),
                "observed_unique_chunks": len({r["chunk_id"] for r in displayed}),
                "truncated_characters": sum(max(0, len(str(r.get("text", ""))) - snippet_chars) for r in evidence),
                "usage": None}
        record["calls"].append(call)
        start = time.perf_counter()
        try:
            if client is None:
                raise RuntimeError("No injected client; live provider execution is disabled")
            raw = client(req)
            call["raw_response"] = raw
            call["usage"] = raw.get("usage")
            if call["usage"] is not None:
                usage = call["usage"]
                if not isinstance(usage, dict) or any(
                    v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0)
                    for k, v in usage.items() if k in ("prompt_tokens", "completion_tokens")
                ):
                    raise ValueError("Invalid token usage accounting")
            call["finish_reason"] = raw.get("finish_reason")
            if raw.get("finish_reason") in ("length", "content_filter"):
                raise ValueError("Truncated or filtered model response")
            parsed = raw["parsed"]
            values = parsed.get("queries") if kind == "upfront" else [parsed.get("query")]
            if not isinstance(values, list) or len(values) != (3 if kind == "upfront" else 1) or any(not isinstance(q, str) or not q.strip() for q in values):
                raise ValueError("Invalid query response")
            return [q.strip() for q in values]
        except Exception as exc:
            call["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            call["latency_seconds"] = time.perf_counter() - start

    try:
        if condition == "one_shot_rewrite":
            planned = generate("rewrite")
        elif condition == "upfront_three_queries":
            planned = generate("upfront")  # All three fixed BEFORE any search.
        elif condition == "heuristic_keywords_3x8":
            planned = heuristic_queries(question)
        else:
            planned = [question] * (1 if condition == "original_top24" else 3)
        admitted = []
        for i in range(len(planned)):
            if i and condition in ("evidence_feedback", "no_accumulated_evidence"):
                evidence = admitted if condition == "evidence_feedback" else record["rounds"][-1]["candidate_chunks"]
                query = generate("refine", evidence)[0]
            else:
                query = planned[i]
            record["queries"].append(query)
            cap = 24 if len(planned) == 1 else 8
            seen = set(record["candidate_docs"])
            chosen, inspected = [], []
            for row in ranked(query, "candidate_generation"):
                inspected.append({"doc_id": row["doc_id"], "chunk_id": row["chunk_id"]})
                if row["doc_id"] in seen:
                    continue
                seen.add(row["doc_id"])
                chosen.append(dict(row))
                if len(chosen) == cap:
                    break
            admitted.extend(chosen)
            record["candidate_docs"].extend(unique_docs(chosen))
            record["rounds"].append({"query": query, "requested_new_docs": cap,
                                      "candidate_chunks": chosen, "inspected_chunks": inspected,
                                      "shortfall": cap - len(chosen)})
        record["common_selector_docs"] = select_original(lambda q: ranked(q, "original_selector"), question, record["candidate_docs"])
        record["rrf_docs"] = rrf(lambda q: ranked(q, "rrf_diagnostic"), record["queries"], record["candidate_docs"])
        record["status"] = "complete" if len(record["candidate_docs"]) == 24 else "underfilled"
    except Exception as exc:
        record["status"] = "error"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    observations = [r for call in record["calls"] for r in call["observation"]]
    record["accounting"] = {
        "candidate_unique_docs": len(set(record["candidate_docs"])),
        "candidate_unique_chunks": len({r["chunk_id"] for rd in record["rounds"] for r in rd["candidate_chunks"]}),
        "observed_chunk_exposures": len(observations),
        "observed_unique_chunks": len({r["chunk_id"] for r in observations}),
        "observed_unique_docs": len(unique_docs(observations)),
        "model_calls": len(record["calls"]),
        "usage_complete": all(token_count(c["usage"], key) is not None for c in record["calls"] for key in ("prompt_tokens", "completion_tokens")),
        "reported_usage": [c["usage"] for c in record["calls"]],
        "reported_token_totals": {key: sum(token_count(c["usage"], key) or 0 for c in record["calls"]) for key in ("prompt_tokens", "completion_tokens")},
        "calls_missing_or_invalid_token_counts": sum(any(token_count(c["usage"], key) is None for key in ("prompt_tokens", "completion_tokens")) for c in record["calls"]),
        "truncated_evidence_characters": sum(c["truncated_characters"] for c in record["calls"]),
        "compute_equal": False,
    }
    return record
