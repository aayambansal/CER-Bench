"""All responses and credentials below are synthetic. No external calls."""
import importlib.util
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from src.agents.provider_clients import KEYS, Ledger, CampaignLedger, NativeClient, SafetyError, dumps, https_transport
from src.evaluation import budget_protocol as bp

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("live_runner", ROOT / "scripts/48_run_live_budget.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for key in KEYS.values():
        monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("SHARED_" + key, raising=False)
    def forbidden(*a, **k):
        pytest.fail("Network forbidden in offline adapter tests")
    monkeypatch.setattr(socket, "create_connection", forbidden)


def config(provider="openai"):
    return dict(provider=provider, model="SYNTHETIC-model", approved=True,
                input_usd_per_million=1, output_usd_per_million=2,
                input_token_reserve=32768, output_token_reserve=400,
                timeout_seconds=1, max_retries=0, model_evidence="synthetic",
                pricing_evidence="synthetic", token_reserve_evidence="synthetic")


def response(provider, text='{"query":"SYNTHETIC"}', usage=True):
    if provider == "openai":
        raw = {"id": "synthetic-request", "choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
        if usage:
            raw["usage"] = {"prompt_tokens": 12, "completion_tokens": 4}
    elif provider == "anthropic":
        raw = {"id": "synthetic-request", "content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}
        if usage:
            raw["usage"] = {"input_tokens": 12, "output_tokens": 4}
    else:
        raw = {"responseId": "synthetic-request", "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}
        if usage:
            raw["usageMetadata"] = {"promptTokenCount": 12, "candidatesTokenCount": 4}
    return 200, {}, json.dumps(raw).encode()


@pytest.mark.parametrize("provider", list(KEYS))
def test_native_shapes_and_redaction(tmp_path, monkeypatch, provider):
    key = 'SYNTHETIC-secret-"-value'
    monkeypatch.setenv(KEYS[provider], key)
    observed = []
    def transport(host, path, headers, payload, timeout):
        observed.append((host, path, payload))
        assert key not in path
        assert timeout == 1
        status, _, body = response(provider, json.dumps({"query": key}))
        return status, {"x-request-id": key, "Authorization": key}, body
    ledger = Ledger(tmp_path / "ledger", 1, 2)
    result = NativeClient(config(provider), ledger, transport=transport)(bp.request("rewrite", "question", "SYNTHETIC-model", provider))
    assert result["parsed"]["query"] == "[REDACTED]"
    assert "SYNTHETIC-secret" not in (tmp_path / "ledger").read_text()
    assert result["usage"] == {"prompt_tokens": 12, "completion_tokens": 4}
    assert len(observed) == 1


def test_absent_env_and_unverified_fail_before_dispatch(tmp_path):
    c = config()
    with pytest.raises(SafetyError, match="credential"):
        NativeClient(c, Ledger(tmp_path / "ledger", 1, 1))
    c["approved"] = False
    with pytest.raises(SafetyError, match="Unverified"):
        NativeClient(c, Ledger(tmp_path / "ledger", 1, 1))
    assert not (tmp_path / "ledger").exists()


@pytest.mark.parametrize("status", [301, 302, 307, 308, 400, 401, 429, 500, 502, 503, 504])
def test_error_bodies_redirects_retry_caps(tmp_path, monkeypatch, status):
    monkeypatch.setenv("OPENAI_API_KEY", "SYNTHETIC-secret")
    calls = []
    def transport(*args):
        calls.append(1)
        return status, {"Location": "http://127.0.0.1/secret"}, b"SYNTHETIC-secret"
    c = config(); c["max_retries"] = 2
    ledger = Ledger(tmp_path / "ledger", 1, 2)
    with pytest.raises(SafetyError) as exc:
        NativeClient(c, ledger, transport=transport, sleep=lambda _: None)(bp.request("rewrite", "q", c["model"], c["provider"]))
    assert len(calls) == (2 if status in (429, 500, 502, 503, 504) else 1)
    assert "SYNTHETIC-secret" not in str(exc.value) + (tmp_path / "ledger").read_text()
    assert len([e for e in ledger.events if e["state"] == "pending"]) == len(calls)


@pytest.mark.parametrize("host,path", [("localhost", "/"), ("api.openai.com.evil", "/"), ("api.openai.com", "//evil?key=x"), ("api.openai.com", "/../")])
def test_ssrf_rejected(host, path):
    with pytest.raises(SafetyError):
        https_transport(host, path, {}, {}, 1)


def test_timeout_unknown_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "SYNTHETIC-secret")
    def transport(*args):
        raise TimeoutError("SYNTHETIC-secret")
    ledger = Ledger(tmp_path / "ledger", 1, 5)
    with pytest.raises(SafetyError, match="unknown billed"):
        NativeClient(config(), ledger, transport=transport)(bp.request("rewrite", "q", "SYNTHETIC-model", "openai"))
    assert ledger.events[0]["timeout_seconds"] == 1
    assert ledger.events[1]["actual_usd"] is None
    assert "SYNTHETIC-secret" not in (tmp_path / "ledger").read_text()
    with pytest.raises(SafetyError, match="reconciliation"):
        Ledger(tmp_path / "ledger", 1, 5)


def test_missing_usage_and_preflight_reserve(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "SYNTHETIC-secret")
    request = bp.request("rewrite", "q", "SYNTHETIC-model", "openai")
    ledger = Ledger(tmp_path / "ledger", 1, 1)
    with pytest.raises(SafetyError, match="Usage absent"):
        NativeClient(config(), ledger, transport=lambda *a: response("openai", usage=False))(request)
    assert ledger.events[-1]["actual_usd"] is None
    cheap = Ledger(tmp_path / "cheap", 0.00001, 1)
    with pytest.raises(SafetyError, match="before dispatch"):
        NativeClient(config(), cheap)(request)
    assert not cheap.events
    with pytest.raises(SafetyError, match="Input exceeds"):
        NativeClient(config(), cheap)(dict(request, messages=[{"content": "x" * 40000}]))


@pytest.mark.parametrize("body", [b"SYNTHETIC-secret", b"{}", b'{"error":"SYNTHETIC-secret"}'])
def test_malformed_success_suppressed(tmp_path, monkeypatch, body):
    monkeypatch.setenv("OPENAI_API_KEY", "SYNTHETIC-secret")
    ledger = Ledger(tmp_path / "ledger", 1, 2)
    with pytest.raises(SafetyError) as exc:
        NativeClient(config(), ledger, transport=lambda *a: (200, {}, body))(bp.request("rewrite", "q", "SYNTHETIC-model", "openai"))
    assert "SYNTHETIC-secret" not in str(exc.value) + (tmp_path / "ledger").read_text()
    assert ledger.events[-1]["state"] == "unknown"


@pytest.mark.parametrize("model", ["https://evil/model", "a?key=x", "../model", "model/other"])
def test_model_path_injection_blocked(tmp_path, model):
    c = config(); c["model"] = model
    with pytest.raises(SafetyError, match="model ID"):
        NativeClient(c, Ledger(tmp_path / "ledger", 1, 2))


def test_crash_pending_ledger_cannot_resume(tmp_path):
    ledger = Ledger(tmp_path / "ledger", 1, 2)
    ledger.reserve(0.1, 1)
    with pytest.raises(SafetyError, match="reconciliation"):
        Ledger(tmp_path / "ledger", 1, 2)


def fixture_inputs(tmp_path):
    corpus = ["SYNTHETIC-doc-%02d" % i for i in range(30)]
    chunks = [{"doc_id": d, "chunk_id": d + "-chunk", "text": "synthetic evidence"} for d in corpus]
    tasks = [{"task_id": "SYNTHETIC-task", "task_family": "constraint", "question": "synthetic question", "split": "dev"}]
    index = {"schema": "cerbench.portable-bm25.v1", "rows": [{"chunk_id": r["chunk_id"], "tokens": ["synthetic", "evidence"]} for r in chunks]}
    paths = {}
    for name, obj in [("corpus", corpus), ("chunks", chunks), ("tasks", tasks), ("index", index)]:
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(obj))
        paths[name] = str(path)
    evidence = tmp_path / "SYNTHETIC-evidence.txt"
    evidence.write_text("SYNTHETIC test only; not model or price verification")
    c = config()
    for field in ("model_evidence", "pricing_evidence", "token_reserve_evidence"):
        c[field] = {"path": str(evidence), "sha256": bp.file_hash(evidence)}
    c.update(max_usd=1, max_attempts=20, conditions=list(bp.CONDITIONS), input_verification=dict(approved=True, **{k + "_sha256": bp.file_hash(v) for k, v in paths.items()}))
    path = tmp_path / "config.json"; path.write_text(json.dumps(c))
    return SimpleNamespace(**paths, config=str(path), output=str(tmp_path / "SYNTHETIC-output"), split="dev", stage="dev-smoke", repeat=1, live=True)


def test_offline_full_integration_resume_and_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "SYNTHETIC-secret")
    args = fixture_inputs(tmp_path)
    def factory(c, ledger):
        def transport(host, path, headers, payload, timeout):
            text = '{"queries":["synthetic a","synthetic b","synthetic c"]}' if "exactly three" in payload["messages"][0]["content"] else '{"query":"synthetic refinement"}'
            return response("openai", text)
        return NativeClient(c, ledger, transport=transport)
    rows = runner.execute(args, client_factory=factory)
    assert len(rows) == 7 and all(r["status"] == "complete" for r in rows)
    assert [len(r["calls"]) for r in rows] == [0, 0, 0, 1, 1, 2, 2]
    assert all(len(r["candidate_docs"]) == 24 and len(r["common_selector_docs"]) == len(r["rrf_docs"]) == 20 for r in rows)
    assert len(runner.execute(args, client_factory=factory)) == 7
    exported = runner.load(Path(args.output) / "evidence_feedback.common_selector_docs.eval.json")
    assert set(exported[0]) == {"task_id", "retrieved_doc_ids"}
    manifest = runner.load(Path(args.output) / "manifest.json")
    assert bp.validate_manifest(manifest)
    assert bp.validate_resume(manifest, manifest, rows)
    audit_spec = importlib.util.spec_from_file_location("budget_auditor", ROOT / "scripts/41_audit_budget_runs.py")
    auditor = importlib.util.module_from_spec(audit_spec); audit_spec.loader.exec_module(auditor)
    audit = auditor.audit([Path(args.output) / "budget_runs.jsonl"], runner.load(args.tasks), "dev")
    (Path(args.output) / "SYNTHETIC-budget-audit.json").write_text(json.dumps(audit, indent=2))
    assert all(not f["parse_or_manifest_errors"] for f in audit["files"])
    args.repeat = 2; args.stage = "frozen"
    with pytest.raises(SafetyError, match="manifest mismatch"):
        runner.execute(args, client_factory=factory)


def test_pending_and_lock_and_dry_run(tmp_path):
    import fcntl
    args = fixture_inputs(tmp_path); args.live = False
    rows = runner.execute(args)
    assert all(r["executed_model_calls"] == r["executed_search_calls"] == 0 for r in rows)
    out = Path(args.output)
    assert not (out / "ledger.jsonl").exists()
    with (out / "output.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SafetyError, match="locked"):
            runner.execute(args)
    (out / "runs.json").unlink()
    (out / "pending.json").write_text("{}")
    with pytest.raises(SafetyError, match="Partial pending"):
        runner.execute(args)


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
def test_jsonl_literal_unicode_separators(tmp_path, separator):
    records = [{"text": "before" + separator + "after", "task_id": "one"}, {"text": separator, "task_id": "two"}]
    path = tmp_path / "physical.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    assert runner.load(path) == records
    ledger = Ledger(tmp_path / "ledger.jsonl", 1, 2)
    attempt = ledger.reserve(.1, 1)
    ledger.append({"attempt": attempt, "state": "complete", "raw_content": separator})
    assert Ledger(ledger.path, 1, 2).events[-1]["raw_content"] == separator


@pytest.mark.parametrize("line", ['{"x":1,"x":2}', '{"x":NaN}'])
def test_jsonl_keeps_strict_validation(tmp_path, line):
    path = tmp_path / "invalid.jsonl"; path.write_text(line + "\n")
    with pytest.raises(SafetyError):
        runner.load(path)


def verified_fixture(tmp_path):
    """Invented unit-test documents and simulated identity attestation, never gold."""
    from src.retrieval.verified_bm25 import build_index
    args = fixture_inputs(tmp_path)
    corpus = [{"doc_id": "SYNTHETIC-%02d" % i, "article_id": "SYNTHETIC-article-%02d" % i, "corpus_id": "SYNTHETIC"} for i in range(30)]
    chunks = [dict(doc, chunk_id=doc["doc_id"] + "-" + str(j), text=("generic text" if j == 0 else "synthetic evidence \u2028 target \u2029")) for doc in corpus for j in range(2)]
    for name, rows in (("corpus", corpus), ("chunks", chunks)):
        path = tmp_path / (name + ".jsonl")
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        setattr(args, name, str(path))
    manifest = dict(schema="verified-retrieval-dataset-v1", status="authoritative_validated",
                    identity_validated=True, canonical_ids_unique=True, corpus_id="SYNTHETIC",
                    document_count=len(corpus), chunk_count=len(chunks),
                    corpus_sha256=bp.file_hash(args.corpus), chunks_sha256=bp.file_hash(args.chunks))
    path = tmp_path / "dataset_manifest.json"; path.write_text(json.dumps(manifest))
    args.dataset_manifest = str(path)
    args.backend = "verified_bm25"
    args.index = str(tmp_path / "verified_index")
    build_index(args.corpus, args.chunks, args.dataset_manifest, args.index)
    args.live = False
    return args


def test_native_verified_backend_and_hashes(tmp_path):
    from src.retrieval.verified_bm25 import VerifiedBM25
    args = verified_fixture(tmp_path)
    args.dry_retrieval_check = True
    rows = runner.execute(args)
    out = Path(args.output)
    assert all(r["synthetic"] is False and r["executed_model_calls"] == 0 for r in rows)
    manifest = runner.load(out / "manifest.json")
    assert bp.validate_manifest(manifest)
    assert manifest["config"]["retrieval"]["backend"] == "verified_bm25"
    for path in (Path(args.index) / "index.json", Path(args.index) / "postings.npz", Path(args.dataset_manifest)):
        assert manifest["index_sha256"][str(path.resolve())] == bp.file_hash(path)
    idx = VerifiedBM25.load(args.index, corpus=args.corpus, chunks=args.chunks, manifest=args.dataset_manifest)
    checks = runner.load(out / "retrieval_checks.json")
    assert len(checks) == 3
    for row in checks:
        assert row["status"] == "complete" and len(row["candidate_docs"]) == 24 and not row["calls"]
        assert row["common_selector_docs"] == [hit["doc_id"] for hit in idx.rank("synthetic question", row["candidate_docs"])]
        assert len(dumps(row).encode()) < runner.TRACE_MAX_BYTES


def test_injected_search_is_still_synthetic(tmp_path):
    args = verified_fixture(tmp_path)
    rows = runner.execute(args, search_factory=lambda chunks, index: lambda query: chunks)
    assert all(r["synthetic"] for r in rows)


def test_verified_index_requires_original_manifest(tmp_path):
    args = verified_fixture(tmp_path); args.dataset_manifest = None
    with pytest.raises(SafetyError, match="dataset manifest"):
        runner.execute(args)


def test_campaign_retains_attempts_across_run_directories(tmp_path):
    campaign = Ledger(tmp_path / "campaign.jsonl", .15, 2)
    a = CampaignLedger(Ledger(tmp_path / "a.jsonl", 1, 2), campaign, "run-a")
    n = a.reserve(.1, 1)
    a.append({"attempt": n, "state": "http_error", "http_status": 429, "actual_usd": None})
    b = CampaignLedger(Ledger(tmp_path / "b.jsonl", 1, 2), Ledger(campaign.path, .15, 2), "run-b")
    with pytest.raises(SafetyError, match="cap reached"):
        b.reserve(.1, 1)
    assert not b.events
    assert [e["run_provenance_id"] for e in campaign.events if e["state"] == "binding"] == ["run-a"]


def test_campaign_binding_does_not_settle_unknown(tmp_path):
    campaign = Ledger(tmp_path / "campaign.jsonl", 1, 2)
    ledger = CampaignLedger(Ledger(tmp_path / "a.jsonl", 1, 2), campaign, "run-a")
    ledger.reserve(.1, 1)
    with pytest.raises(SafetyError, match="Unknown billed outcome"):
        Ledger(campaign.path, 1, 2)


def test_campaign_lock_blocks_second_executor(tmp_path):
    import fcntl
    args = fixture_inputs(tmp_path)
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps(dict(schema="cerbench.campaign-budget.v1", approved=True,
                                        campaign_id="SYNTHETIC-campaign", max_usd=1, max_attempts=10)))
    args.campaign_budget = str(campaign)
    with Path(str(campaign) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SafetyError, match="Campaign locked"):
            runner.execute(args, client_factory=lambda c, ledger: pytest.fail("must not construct client"))


def test_native_live_requires_campaign(tmp_path):
    args = fixture_inputs(tmp_path)
    with pytest.raises(SafetyError, match="shared campaign budget"):
        runner.execute(args)


def test_trace_bound_fails_closed(tmp_path, monkeypatch):
    args = fixture_inputs(tmp_path)
    monkeypatch.setattr(runner, "TRACE_MAX_BYTES", 10)
    with pytest.raises(SafetyError, match="trace exceeds bound"):
        runner.bounded_protocol(runner.load(args.tasks)[0], "original_top24",
                                runner.portable_search(runner.load(args.chunks), runner.load(args.index)))


def test_protocol_preserves_observation_schedule(tmp_path):
    args = verified_fixture(tmp_path)
    from src.retrieval.verified_bm25 import VerifiedBM25
    idx = VerifiedBM25.load(args.index, corpus=args.corpus, chunks=args.chunks, manifest=args.dataset_manifest)
    task = runner.load(args.tasks)[0]
    for condition, expected in (("evidence_feedback", [8, 16]), ("no_accumulated_evidence", [8, 8])):
        seen = []
        def fake(request):
            observed = json.loads(request["messages"][1]["content"])["evidence"]
            seen.append(len(observed))
            assert all(len(r["text"]) <= 420 for r in observed)
            return {"parsed": {"query": "target"}, "usage": {"prompt_tokens": 12, "completion_tokens": 4}}
        row = runner.bounded_protocol(task, condition, idx, fake)
        assert seen == expected and row["status"] == "complete"
        assert len(row["candidate_docs"]) == 24
