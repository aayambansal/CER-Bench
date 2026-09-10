#!/usr/bin/env python3
"""Locked, provenance-bound executor. Default is a network-free plan."""
import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.agents.provider_clients import Ledger, CampaignLedger, NativeClient, SafetyError, dumps, sanitize
from src.evaluation import budget_protocol as bp


def load(path):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise SafetyError("Duplicate JSON key")
            result[k] = v
        return result
    def invalid(value):
        raise SafetyError("Nonfinite JSON")
    parse = lambda s: json.loads(s, object_pairs_hook=pairs, parse_constant=invalid)
    # JSON permits literal U+2028/U+2029 inside strings. str.splitlines() does not.
    with Path(path).open(encoding="utf-8", newline="") as handle:
        if Path(path).suffix == ".jsonl":
            return [parse(line) for line in handle if line.strip()]
        return parse(handle.read())


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        handle.write(dumps(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def portable_search(chunks, index):
    """Explicit portable BM25 index, NOT a silent replacement for legacy pickle."""
    tokenize = lambda text: re.findall(r"[a-z0-9]+", text.lower())
    expected = [{"chunk_id": r["chunk_id"], "tokens": tokenize(r["text"])} for r in chunks]
    if index != {"schema": "cerbench.portable-bm25.v1", "rows": expected}:
        raise SafetyError("Index/chunks mismatch or unsupported index schema")
    counts = [Counter(row["tokens"]) for row in expected]
    n = len(counts)
    avg = sum(sum(c.values()) for c in counts) / n if n else 0
    if not avg:
        raise SafetyError("Empty lexical index")
    df = Counter(t for c in counts for t in c)
    def search(query):
        terms = tokenize(query)
        scores = [sum(math.log(1 + (n-df[t]+0.5)/(df[t]+0.5)) * c[t]*2.5 /
                      (c[t]+1.5*(0.25+0.75*sum(c.values())/avg)) for t in terms if c[t]) for c in counts]
        return [chunks[i] for i in sorted(range(n), key=lambda i: (-scores[i], chunks[i]["chunk_id"]))]
    return search


TRACE_MAX_BYTES = 8 * 1024 * 1024


def bounded_protocol(*args, **kwargs):
    row = bp.run_protocol(*args, **kwargs)
    if len(dumps(row).encode("utf-8")) > TRACE_MAX_BYTES:
        raise SafetyError("Protocol trace exceeds bound; no silent trace truncation")
    return row


def execute(args, *, client_factory=NativeClient, search_factory=None):
    config = load(args.config)
    # Do not write arbitrary config until secret redaction has been checked.
    if sanitize(config) != config:
        raise SafetyError("Configuration must contain no runtime secrets")
    allowed = {"approved", "provider", "model", "input_usd_per_million", "output_usd_per_million", "input_token_reserve", "output_token_reserve", "timeout_seconds", "max_retries", "max_attempts", "max_usd", "conditions", "model_evidence", "pricing_evidence", "token_reserve_evidence", "input_verification"}
    if set(config) - allowed:
        raise SafetyError("Unknown configuration fields")
    conditions = config.get("conditions", [])
    if not conditions or len(set(conditions)) != len(conditions) or not set(conditions) <= set(bp.CONDITIONS):
        raise SafetyError("Explicit unique protocol conditions required")
    backend = getattr(args, "backend", "portable")  # Legacy programmatic fixtures only.
    if backend not in ("verified_bm25", "portable"):
        raise SafetyError("Explicit supported backend required")
    tasks, corpus, chunks = [load(getattr(args, k)) for k in ("tasks", "corpus", "chunks")]
    task_schema = "task_records"
    if isinstance(tasks, dict):
        if tasks.get("schema") != "verified-bm25-unscored-probes-v1" or tasks.get("label_status") != "unscored_retrieval_smoke_no_qrels":
            raise SafetyError("Unsupported task envelope")
        task_schema = tasks["schema"]
        tasks = tasks["probes"]
    index_json = Path(args.index) / "index.json" if Path(args.index).is_dir() else Path(args.index)
    index_files = [index_json]
    if backend == "verified_bm25":
        dataset_manifest = getattr(args, "dataset_manifest", None)
        if not dataset_manifest or index_json.name != "index.json":
            raise SafetyError("Verified backend requires original index.json and dataset manifest")
        index_files += [index_json.parent / "postings.npz", Path(dataset_manifest)]
        from src.retrieval.verified_bm25 import VerifiedBM25
        search = VerifiedBM25.load(index_json.parent, corpus=args.corpus, chunks=args.chunks,
                                   manifest=dataset_manifest)
        retrieval_metadata = {"backend": backend, "index_config": search.meta["config"],
                              "dataset_status": search.meta["status"], "document_count": len(search.doc_ids),
                              "chunk_count": len(search.rows)}
        if search_factory is not None:
            search = search_factory(chunks, load(index_json))
    else:
        if getattr(args, "dataset_manifest", None):
            raise SafetyError("Dataset manifest is only accepted with verified_bm25")
        search = (search_factory or portable_search)(chunks, load(index_json))
        retrieval_metadata = {"backend": backend, "legacy_equivalence_claimed": False}
    ids = [r["doc_id"] if isinstance(r, dict) else r for r in corpus]
    if len(set(ids)) != len(ids) or any(not isinstance(d, str) or not d for d in ids):
        raise SafetyError("Invalid corpus identifiers")
    task_ids = [t["task_id"] for t in tasks]
    if not tasks or len(set(task_ids)) != len(task_ids) or any(not t.get("question") or not t.get("task_family") or t.get("split") != args.split for t in tasks):
        raise SafetyError("Tasks require unique IDs, questions, families and explicit split")
    chunk_ids = [r["chunk_id"] for r in chunks]
    if len(set(chunk_ids)) != len(chunk_ids) or any(r["doc_id"] not in ids or not isinstance(r.get("text"), str) for r in chunks):
        raise SafetyError("Invalid chunk identities")
    if args.stage == "dev-smoke":
        if args.split != "dev" or args.repeat != 1:
            raise SafetyError("Dev smoke requires dev split and repeat 1")
        selected = {}
        for task in sorted(tasks, key=lambda t: t["task_id"]):
            selected.setdefault(task["task_family"], task)
        tasks = list(selected.values())
    elif args.repeat not in (1, 2, 3):
        raise SafetyError("Frozen phase allows repeat 1, 2, 3 only")
    source_files = sorted(set(ROOT.glob("src/**/*.py")) | set(ROOT.glob("scripts/*.py")) | set(ROOT.glob("tests/**/*.py")) | set(ROOT.glob("*.py")))
    evidence = {}
    if args.live:
        verification = config.get("input_verification") or {}
        verified_paths = {k: getattr(args, k) for k in ("tasks", "corpus", "chunks")}
        verified_paths["index"] = index_json
        if backend == "verified_bm25":
            verified_paths.update(postings=index_files[1], dataset_manifest=index_files[2])
        if verification.get("approved") is not True or any(verification.get(k + "_sha256") != bp.file_hash(path) for k, path in verified_paths.items()):
            raise SafetyError("Explicit verified input hashes required")
        for key in ("model_evidence", "pricing_evidence", "token_reserve_evidence"):
            spec = config.get(key)
            if not isinstance(spec, dict) or not spec.get("path") or not spec.get("sha256") or bp.file_hash(spec["path"]) != spec["sha256"]:
                raise SafetyError("Approved nonsecret evidence files and matching hashes required")
            evidence[key] = spec["sha256"]
    synthetic = client_factory is not NativeClient or search_factory is not None
    campaign_path = getattr(args, "campaign_budget", None)
    campaign = load(campaign_path) if campaign_path else None
    if args.live and not synthetic and not campaign_path:
        raise SafetyError("Live native execution requires a shared campaign budget file")
    if campaign is not None and (set(campaign) != {"schema", "approved", "campaign_id", "max_usd", "max_attempts"}
                                 or campaign.get("schema") != "cerbench.campaign-budget.v1"
                                 or campaign.get("approved") is not True or not campaign.get("campaign_id")
                                 or sanitize(campaign) != campaign):
        raise SafetyError("Invalid nonsecret campaign approval")
    bound_config = dict(config, conditions=conditions, stage=args.stage, live=args.live, synthetic=synthetic,
                        retrieval=retrieval_metadata, task_input_schema=task_schema, trace_max_bytes=TRACE_MAX_BYTES,
                        dry_retrieval_check=getattr(args, "dry_retrieval_check", False),
                        campaign_budget_path=str(Path(campaign_path).resolve()) if campaign_path else None,
                        campaign_budget_sha256=bp.file_hash(campaign_path) if campaign_path else None,
                        chunks_sha256=bp.file_hash(args.chunks), configuration_sha256=bp.file_hash(args.config), evidence_sha256=evidence,
                        python_version=sys.version, python_executable_sha256=bp.file_hash(sys.executable),
                        runtime_dependencies={str(Path(m.__file__).resolve()): bp.file_hash(m.__file__)
                                              for m in list(sys.modules.values()) if getattr(m, "__file__", None)
                                              and Path(m.__file__).is_file() and str(Path(m.__file__).resolve()).startswith(str(Path(os.__file__).resolve().parent))})
    manifest = bp.manifest(tasks=args.tasks, split=args.split, corpus=args.corpus, index_files=index_files,
                           source_files=source_files, model=config.get("model") or "UNVERIFIED", provider=config.get("provider") or "UNVERIFIED",
                           repeat=args.repeat, config=bound_config, task_ids=[t["task_id"] for t in tasks])
    if sanitize(manifest) != manifest:
        raise SafetyError("Manifest contains sensitive runtime values")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "output.lock").open("a") as lock, ExitStack() as stack:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SafetyError("Output locked by another executor") from None
        mpath, rpath = out / "manifest.json", out / "runs.json"
        rows = load(rpath) if rpath.exists() else []
        if mpath.exists() and load(mpath) != manifest:
            raise SafetyError("Resume manifest mismatch")
        if rows and not args.live:
            raise SafetyError("Dry plan already exists; choose a fresh output")
        if not mpath.exists() and (rows or (out / "ledger.jsonl").exists()):
            raise SafetyError("Unbound outputs cannot resume")
        done = bp.validate_resume(manifest, manifest, rows) if args.live else set()
        # Persist row pending state separately: a crash between calls must not rerun a paid row.
        pending = out / "pending.json"
        if pending.exists():
            raise SafetyError("Partial pending task: manual reconciliation required")
        client = None
        if args.live:
            ledger = Ledger(out / "ledger.jsonl", config.get("max_usd"), config.get("max_attempts"))
            if campaign_path:
                campaign_file = Path(campaign_path).resolve()
                campaign_lock = stack.enter_context(Path(str(campaign_file) + ".lock").open("a"))
                try:
                    fcntl.flock(campaign_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise SafetyError("Campaign locked by another executor") from None
                campaign_binding = Path(str(campaign_file) + ".binding.json")
                binding = {"sha256": bp.file_hash(campaign_file), "campaign_id": campaign["campaign_id"]}
                if campaign_binding.exists() and load(campaign_binding) != binding:
                    raise SafetyError("Campaign approval changed; refusing budget reset")
                write(campaign_binding, binding)
                aggregate = Ledger(Path(str(campaign_file) + ".ledger.jsonl"), campaign["max_usd"], campaign["max_attempts"])
                ledger = CampaignLedger(ledger, aggregate, manifest["provenance_id"])
            client = client_factory(config, ledger)
        write(mpath, manifest)
        write(out / "selected_tasks.json", tasks)
        if getattr(args, "dry_retrieval_check", False):
            if args.live:
                raise SafetyError("Dry retrieval checks cannot accompany live execution")
            checks = []
            for task in tasks:
                for condition in ("original_top24", "repeated_original_3x8", "heuristic_keywords_3x8"):
                    row = bounded_protocol(task, condition, search, model="NO_MODEL_EXECUTED", provider="none",
                                           provenance_id=manifest["provenance_id"])
                    row.update(synthetic=synthetic, execution_kind="offline_unscored_retrieval_check", model_runs=0)
                    checks.append(row)
                    if row["status"] != "complete":
                        raise SafetyError("Offline retrieval candidate shortfall")
            write(out / "retrieval_checks.json", checks)
        for task in tasks:
            for condition in conditions:
                if (task["task_id"], condition) in done:
                    continue
                if not args.live:
                    row = bp.dry_plan(task, condition, manifest["model"], manifest["provider"])
                else:
                    write(pending, {"task_id": task["task_id"], "condition": condition})
                    attempt_start = len(ledger.events)
                    row = bounded_protocol(task, condition, search, client, model=config["model"], provider=config["provider"], provenance_id=manifest["provenance_id"])
                    row["provider_attempt_events"] = ledger.events[attempt_start:]
                    row["spend_accounting"] = {"policy": "all_attempt_reservations_retained", "actual_usd": None
                                              if any(e.get("actual_usd") is None for e in ledger.events[attempt_start:] if e["state"] != "pending")
                                              else sum(e["actual_usd"] for e in ledger.events[attempt_start:] if e["state"] != "pending"),
                                              "reserved_usd": sum(e["reserved_usd"] for e in ledger.events[attempt_start:] if e["state"] == "pending")}
                rows.append(sanitize(row))
                rows[-1]["synthetic"] = synthetic
                rows[-1]["provenance_id"] = manifest["provenance_id"]
                write(rpath, rows)
                if args.live:
                    if row["status"] != "complete":
                        raise SafetyError("Incomplete protocol row; retained pending state; no automatic retry")
                    pending.unlink()
        if args.live:
            budget_path = out / "budget_runs.jsonl"
            tmp = out / "budget_runs.jsonl.tmp"
            with tmp.open("w") as handle:
                for row in rows:
                    handle.write(dumps(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, budget_path)
            write(out / "budget_runs.jsonl.manifest.json", manifest)
            for condition in conditions:
                for selector in ("common_selector_docs", "rrf_docs"):
                    write(out / (condition + "." + selector + ".eval.json"),
                          [{"task_id": r["task_id"], "retrieved_doc_ids": r[selector]} for r in rows if r["condition"] == condition])
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    for name in ("config", "tasks", "corpus", "chunks", "index", "output", "split"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--stage", choices=("dev-smoke", "frozen"), default="dev-smoke")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--backend", choices=("verified_bm25", "portable"), default="verified_bm25")
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--campaign-budget", help="Shared approved nonsecret campaign JSON; required for native live runs")
    parser.add_argument("--dry-retrieval-check", action="store_true", help="Run only deterministic, unscored local checks in addition to zero-call plans")
    args = parser.parse_args()
    try:
        execute(args)
    except Exception:
        # Unknown exceptions may include provider/data secrets; CLI never prints them.
        print("Live executor blocked or incomplete; inspect sanitized output state and prerequisites.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
