import importlib.util
import json
import runpy
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation import budget_protocol as bp


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network forbidden in budget tests")
    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


TASK = {"task_id": "t", "question": "original question", "supporting_doc_ids": ["SECRET_GOLD"]}


def search(query):
    docs = list(range(40))
    if query != TASK["question"]:
        docs.reverse()
    return [{"doc_id": f"d{i:02}", "chunk_id": f"d{i:02}_c{j}", "text": "x" * 500}
            for i in docs for j in range(2)]


def client(req):
    upfront = "exactly three" in req["messages"][0]["content"]
    return {"parsed": {"queries": ["a", "b", "c"]} if upfront else {"query": "revised"},
            "usage": {"prompt_tokens": 12, "completion_tokens": 3}, "finish_reason": "stop"}


@pytest.mark.parametrize("condition", bp.CONDITIONS)
def test_all_branches_budget_dedup_shared_selector(condition):
    row = bp.run_protocol(TASK, condition, search, client)
    assert row["status"] == "complete"
    assert len(row["candidate_docs"]) == len(set(row["candidate_docs"])) == 24
    assert row["common_selector_docs"] == bp.select_original(search, TASK["question"], row["candidate_docs"])
    assert row["rrf_docs"] == bp.rrf(search, row["queries"], row["candidate_docs"])
    assert len(row["common_selector_docs"]) == 20
    assert set(row["rrf_docs"]) <= set(row["candidate_docs"])
    assert "SECRET_GOLD" not in json.dumps(row)
    assert row["accounting"]["candidate_unique_chunks"] == 24
    assert not row["accounting"]["compute_equal"]
    for rd in row["rounds"]:
        assert len(rd["candidate_chunks"]) == rd["requested_new_docs"]
        assert len(rd["inspected_chunks"]) >= len(rd["candidate_chunks"])


def test_repeat_is_original_and_no_feedback_upfront_order():
    a = bp.run_protocol(TASK, "original_top24", search)
    b = bp.run_protocol(TASK, "repeated_original_3x8", search)
    assert a["candidate_docs"] == b["candidate_docs"]
    events = []
    def fake_client(req):
        events.append("client")
        content = json.loads(req["messages"][1]["content"])
        assert set(content) == {"question"}
        return client(req)
    def fake_search(query):
        events.append("search")
        return search(query)
    row = bp.run_protocol(TASK, "upfront_three_queries", fake_search, fake_client)
    assert events[0] == "client" and events.count("client") == 1
    assert row["queries"] == ["a", "b", "c"]
    assert row["accounting"]["observed_unique_docs"] == 0


def test_accumulation_ablation_and_truncation():
    full = bp.run_protocol(TASK, "evidence_feedback", search, client)
    last = bp.run_protocol(TASK, "no_accumulated_evidence", search, client)
    assert [c["observed_unique_docs"] for c in full["calls"]] == [8, 16]
    assert [c["observed_unique_docs"] for c in last["calls"]] == [8, 8]
    assert full["calls"][1]["truncated_characters"] == 16 * 80
    latest = {r["doc_id"] for r in last["rounds"][1]["candidate_chunks"]}
    assert {r["doc_id"] for r in last["calls"][1]["observation"]} == latest
    assert full["accounting"]["observed_chunk_exposures"] == 24
    assert full["accounting"]["observed_unique_chunks"] == 16


@pytest.mark.parametrize("condition", bp.CONDITIONS)
def test_short_corpus_never_duplicates_to_fill(condition):
    row = bp.run_protocol(TASK, condition, lambda q: search(q)[:10], client)
    assert row["status"] == "underfilled"
    assert len(row["candidate_docs"]) == len(set(row["candidate_docs"])) < 24


@pytest.mark.parametrize("response", [{"parsed": {"query": ""}}, {"parsed": {"query": "ok"}, "finish_reason": "length"}, {"raw_text": "not json"}, {"parsed": {"query": "ok"}, "usage": {"prompt_tokens": "bad"}}])
def test_invalid_raw_response_retained(response):
    row = bp.run_protocol(TASK, "one_shot_rewrite", search, lambda r: response)
    assert row["status"] == "error"
    assert row["calls"][0]["raw_response"] == response
    assert row["calls"][0]["error"]
    assert not row["search_calls"]


def test_timeout_retained_without_retry_or_fallback():
    def timeout(req):
        raise TimeoutError("fake local timeout")
    row = bp.run_protocol(TASK, "evidence_feedback", search, timeout)
    assert row["status"] == "error"
    assert len(row["candidate_docs"]) == 8
    assert len(row["calls"]) == 1
    assert row["calls"][0]["error"] == {"type": "TimeoutError", "message": "fake local timeout"}
    assert row["calls"][0]["raw_request"] and row["calls"][0]["observation"]
    assert not row["accounting"]["usage_complete"]


def make_manifest(tmp_path):
    f = tmp_path / "input.jsonl"
    f.write_text(json.dumps(TASK) + "\n")
    return bp.manifest(tasks=f, split="test", corpus=f, index_files=[f], source_files=[f],
                       model="fake", provider="local", repeat="r1", config={}, task_ids=["t"])


def test_manifest_hash_and_resume(tmp_path):
    m = make_manifest(tmp_path)
    assert bp.validate_manifest(m)
    row = bp.run_protocol(TASK, "original_top24", search, model="fake", provider="local", provenance_id=m["provenance_id"])
    assert bp.validate_resume(m, m, [row]) == {("t", "original_top24")}
    for key in ("model", "provider", "repeat", "task_sha256", "corpus_sha256", "prompt_sha256", "split", "index_sha256", "source_sha256", "config"):
        changed = {**m, key: "changed"}
        with pytest.raises(ValueError, match="provenance mismatch"):
            bp.validate_resume(m, changed, [row])
    with pytest.raises(ValueError, match="Duplicate"):
        bp.validate_resume(m, m, [row, row])
    with pytest.raises(ValueError, match="row provenance"):
        bp.validate_resume(m, m, [{**row, "provenance_id": "other"}])
    with pytest.raises(ValueError, match="Incomplete"):
        bp.validate_resume(m, m, [{**row, "status": "dry_run"}])
    with pytest.raises(ValueError):
        bp.validate_resume({}, m, [])


def test_cli_default_offline_and_live_gate(tmp_path, monkeypatch):
    f = tmp_path / "tasks.jsonl"
    f.write_text(json.dumps(TASK) + "\n")
    output = tmp_path / "plans.jsonl"
    script = ROOT / "scripts/35_generate_equal_budget_queries.py"
    argv = [str(script), "--tasks", str(f), "--split", "test", "--corpus", str(f),
            "--index-file", str(f), "--repeat", "r1", "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(script), run_name="__main__")
    rows = [json.loads(s) for s in output.read_text().splitlines()]
    assert len(rows) == 7
    assert all(r["status"] == "dry_run" and r["executed_model_calls"] == r["executed_search_calls"] == 0 for r in rows)
    assert all("SECRET_GOLD" not in json.dumps(r) for r in rows)
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        runpy.run_path(str(script), run_name="__main__")
    monkeypatch.setattr(sys, "argv", argv + ["--live"])
    with pytest.raises(SystemExit):
        runpy.run_path(str(script), run_name="__main__")
    assert output.read_bytes() == original


def test_scorer_matched_subset_and_uncertain():
    scorer = load_script("34_equal_budget_controls.py")
    rows, ids = scorer.matched_conditions({"base": [{"task_id": "a"}, {"task_id": "b"}], "incomplete": [{"task_id": "b"}]})
    assert ids == ["b"]
    assert all(len(r) == 1 for r in rows.values())
    assert scorer.matched_conditions({"a": [], "b": [{"task_id": "a"}]})[1] == []
    m = scorer.decision_metrics([(True, "UNCERTAIN"), (False, "UNCERTAIN"), (False, "ANSWER"), (True, "ABSTAIN")])
    assert m["answer_coverage"] == 0.25 and m["nonanswer_rate"] == 0.75
    assert m["tp"] == 1 and m["fn"] == 1 and m["tn"] == 2
    assert m["selective_accuracy"] is None
    assert m["answered_answerability_proxy_accuracy"] == 1
    assert scorer.bootstrap_delta([], [], {}, "common_selector_docs", 20)["n_tasks"] == 0


def test_scorer_rejects_duplicate_mixed_unbound(tmp_path):
    scorer = load_script("34_equal_budget_controls.py")
    f = tmp_path / "rows.jsonl"
    row = {"task_id": "t", "model": "m"}
    f.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="unbound"):
        scorer.load_llm_rows(f)
    assert scorer.load_llm_rows(f, True)
    f.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        scorer.load_llm_rows(f, True)
    f.write_text(json.dumps(row) + "\n" + json.dumps({**row, "model": "other"}) + "\n")
    with pytest.raises(ValueError, match="Mixed"):
        scorer.load_llm_rows(f, True)


def test_audit_missing_duplicates_invalid_and_provenance(tmp_path):
    audit = load_script("41_audit_budget_runs.py").audit
    f = tmp_path / "run_shard0.jsonl"
    row = {"task_id": "t", "model": "m", "one_shot_query": "q", "iterative_queries": ["q"] * 3,
           "iterative_round_candidates": [[f"d{i}" for i in range(8)]] * 3,
           "final_decision": {"decision": "ANSWER", "selected_doc_ids": ["outside"]}}
    f.write_text(json.dumps(row) + "\n" + json.dumps({**row, "provenance_id": "other"}) + "\n{broken\n")
    report = audit([f], [TASK, {"task_id": "missing"}], "test", {"run": 2})
    g = report["groups"]["run"]
    assert g["missing_shards"] == [1]
    assert g["coverage_by_condition"]["legacy_combined"]["missing_task_ids"] == ["missing"]
    assert g["duplicate_identities"][0]["count"] == 2
    assert g["provenance_conflict"] and g["unbound_rows"] == 2
    assert not g["publication_ready"]
    assert {e["error"] for e in g["issues"]} >= {"invalid candidate counts/dedup", "invalid final selection"}
    assert report["files"][0]["parse_or_manifest_errors"]


def test_audit_keeps_repeats_separate(tmp_path):
    audit = load_script("41_audit_budget_runs.py").audit
    paths = []
    for i in range(2):
        f = tmp_path / f"repeat{i}_shard0.jsonl"
        f.write_text(json.dumps({"task_id": "t", "model": "m"}) + "\n")
        paths.append(f)
    result = audit(paths, [TASK], "test")
    assert len(result["groups"]) == 2
    assert all(not g["duplicate_identities"] for g in result["groups"].values())


def test_scorer_main_matched_metrics_and_immutable_outputs(tmp_path, monkeypatch):
    scorer = load_script("34_equal_budget_controls.py")
    class FakeBM25:
        def ranked_docs(self, query):
            ids = bp.unique_docs(search(query))
            return ids, {d: float(len(ids) - i) for i, d in enumerate(ids)}
        def take_new(self, query, n, excluded):
            return [d for d in self.ranked_docs(query)[0] if d not in excluded][:n]
    tasks = [{**TASK, "task_family": "constraint", "supporting_doc_ids": ["d00"]},
             {**TASK, "task_id": "extra", "task_family": "constraint", "supporting_doc_ids": ["d00"]}]
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    (tmp_path / "expanded_gold.json").write_text(json.dumps({"t": ["d00"], "extra": ["d00"]}))
    legacy = {"task_id": "t", "model": "fake", "one_shot_query": "rewrite", "iterative_queries": [TASK["question"]] * 3,
              "iterative_round_candidates": [[f"d{i:02}" for i in range(j, j + 8)] for j in (0, 8, 16)],
              "final_decision": {"decision": "UNCERTAIN"}}
    query_file = tmp_path / "legacy.jsonl"
    query_file.write_text(json.dumps(legacy) + "\n")
    output = tmp_path / "scored"
    monkeypatch.setattr(scorer, "BM25Search", FakeBM25)
    monkeypatch.setattr(scorer, "RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["34", "--tasks", str(task_file), "--split", "test", "--llm-queries", str(query_file),
                                      "--allow-unbound-exploratory", "--output-dir", str(output)])
    scorer.main()
    metrics = json.loads((output / "equal_budget_metrics.json").read_text())
    assert metrics["protocol"]["matched_task_ids"] == ["t"]
    for family in ("seed_qrels", "pooled_qrels"):
        assert {r["common_selector"]["n_tasks"] for r in metrics[family].values()} == {1}
    assert metrics["proxy_abstention"]["answer_coverage"] == 0
    assert metrics["proxy_abstention"]["selective_accuracy"] is None
    saved = {f.name: f.read_bytes() for f in output.iterdir()}
    with pytest.raises(SystemExit):
        scorer.main()
    assert saved == {f.name: f.read_bytes() for f in output.iterdir()}
