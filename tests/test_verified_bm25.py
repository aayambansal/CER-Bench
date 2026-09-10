import importlib.util
import json
import math
from pathlib import Path
from collections import Counter

import numpy as np
import pytest

from src.retrieval.verified_bm25 import (VerifiedBM25, build_index, tokenize,
                                        file_hash, write_json, digest, validate_inputs)
from src.evaluation.budget_protocol import run_protocol

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verified_cli", ROOT / "scripts/49_index_and_run_verified_bm25.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.fixture
def data(tmp_path):
    p = tmp_path / "dataset"
    cli.generate_synthetic(p)
    return p


def paths(p):
    return dict(corpus=p / "corpus.jsonl", chunks=p / "chunks.jsonl",
                manifest=p / "dataset_manifest.json", mode="synthetic_structural")


def index(p):
    build_index(output=p / "index", **paths(p))
    return VerifiedBM25.load(p / "index", **paths(p))


def test_tokenizer():
    assert tokenize("TNF-α A_1 naïve 3.2 X X") == ["tnf", "a", "1", "na", "ve", "3", "2", "x", "x"]


@pytest.mark.parametrize("query", ["algae transport cold", "culture culture", "missingword", "", "measured saline algae"])
def test_okapi_reference(data, query):
    idx = index(data)
    counts = [Counter(tokenize(r["text"])) for r in idx.rows]
    vocabulary = set().union(*counts)
    n = len(counts)
    idf = {t: math.log(n - sum(t in c for c in counts) + .5) - math.log(sum(t in c for c in counts) + .5) for t in vocabulary}
    average = sum(idf.values()) / len(idf)
    idf = {t: .25 * average if v < 0 else v for t, v in idf.items()}
    avgdl = sum(sum(c.values()) for c in counts) / n
    expected = [sum(idf.get(t, 0) * c[t] * 2.5 / (c[t] + 1.5 * (.25 + .75 * sum(c.values()) / avgdl)) for t in tokenize(query)) for c in counts]
    np.testing.assert_allclose(idx.scores(query), expected, rtol=1e-13, atol=1e-13)


def test_stable_prefix_and_selector(data):
    idx = index(data)
    hits = idx.search("unknown", max_unique_docs=24)
    assert len(hits) == 70  # 23 complete three-chunk docs plus final first chunk.
    assert [r["chunk_id"] for r in hits] == idx.meta["chunk_ids"][:70]
    assert len({r["doc_id"] for r in hits}) == 24
    candidates = list(reversed(idx.doc_ids[:24]))
    assert [r["doc_id"] for r in idx.rank("unknown", candidates)] == idx.doc_ids[:20]
    assert idx.search("unknown", top_k=0) == []
    assert idx.search("unknown", exclude_doc_ids=idx.doc_ids[:24], max_unique_docs=1)[0]["doc_id"] == idx.doc_ids[24]
    with pytest.raises(ValueError):
        idx.search("x", max_unique_docs=25)
    with pytest.raises(ValueError):
        idx.rank("x", [idx.doc_ids[0]] * 2)


@pytest.mark.parametrize("condition", ["original_top24", "repeated_original_3x8", "heuristic_keywords_3x8"])
def test_budget_protocol_integration(data, condition):
    idx = index(data)
    task = {"task_id": "t", "question": "algae transport cold", "oracle": "NEVER_VISIBLE"}
    row = run_protocol(task, condition, idx)
    assert row["status"] == "complete"
    assert len(set(row["candidate_docs"])) == 24
    assert row["common_selector_docs"] == [r["doc_id"] for r in idx.rank(task["question"], row["candidate_docs"])]
    assert "NEVER_VISIBLE" not in json.dumps(row)


@pytest.mark.parametrize("mutation,match", [("orphan", "Orphan"), ("article", "Cross-article"),
                                            ("corpus", "Cross-corpus"), ("duplicate", "Duplicate"),
                                            ("canonical", "Duplicate"), ("empty", "text")])
def test_invalid_identity(data, mutation, match):
    docs = cli.read_jsonl(data / "corpus.jsonl")
    chunks = cli.read_jsonl(data / "chunks.jsonl")
    if mutation == "orphan":
        chunks[0]["doc_id"] = "absent"
    elif mutation == "article":
        chunks[0]["article_id"] = "other"
    elif mutation == "corpus":
        chunks[0]["corpus_id"] = "other"
    elif mutation == "duplicate":
        chunks[1]["chunk_id"] = chunks[0]["chunk_id"]
    elif mutation == "canonical":
        docs[1]["article_id"] = docs[0]["article_id"]
    else:
        chunks[0]["text"] = None
    cli.write_jsonl(data / "corpus.jsonl", docs)
    cli.write_jsonl(data / "chunks.jsonl", chunks)
    m = json.loads((data / "dataset_manifest.json").read_text())
    m.update(corpus_sha256=file_hash(data / "corpus.jsonl"), chunks_sha256=file_hash(data / "chunks.jsonl"))
    write_json(data / "dataset_manifest.json", m)
    with pytest.raises(ValueError, match=match):
        validate_inputs(**paths(data))


@pytest.mark.parametrize("mutation", ["npz", "config", "code", "vocab", "order", "input"])
def test_load_hashes(data, mutation):
    index(data)
    p = data / "index/index.json"
    m = json.loads(p.read_text())
    if mutation == "npz":
        with (data / "index/postings.npz").open("ab") as f:
            f.write(b"tamper")
    elif mutation == "input":
        with (data / "chunks.jsonl").open("a") as f:
            f.write("\n")
    else:
        if mutation == "config":
            m["config"]["b"] = 0
        elif mutation == "code":
            m["code_sha256"] = "0" * 64
        elif mutation == "vocab":
            m["vocabulary"].reverse()
        else:
            m["chunk_ids"].reverse()
        m["integrity_sha256"] = digest({k: v for k, v in m.items() if k != "integrity_sha256"})
        write_json(p, m)
    with pytest.raises(ValueError):
        VerifiedBM25.load(data / "index", **paths(data))


def test_modes_and_universe(data):
    with pytest.raises(ValueError, match="status"):
        validate_inputs(**{**paths(data), "mode": "authoritative_validated"})
    idx = index(data)
    inputs = (data / "tasks.jsonl", data / "qrels.json", data / "expected_task_ids.json", idx)
    tasks, qrels = cli.validate_evaluation(*inputs)
    assert len(tasks) == 4 and qrels["label_status"] == "synthetic_structural_not_gold"
    write_json(data / "expected_task_ids.json", ["T00"])
    with pytest.raises(ValueError, match="universe"):
        cli.validate_evaluation(*inputs)


def test_cli_end_to_end(data):
    common = ["--corpus", str(data / "corpus.jsonl"), "--chunks", str(data / "chunks.jsonl"),
              "--manifest", str(data / "dataset_manifest.json"), "--index", str(data / "index"),
              "--mode", "synthetic_structural"]
    cli.main(["build", *common])
    cli.main(["run", *common, "--tasks", str(data / "tasks.jsonl"), "--qrels", str(data / "qrels.json"),
              "--expected-task-ids", str(data / "expected_task_ids.json"), "--output", str(data / "run")])
    runs = cli.read_jsonl(data / "run/runs.jsonl")
    assert len(runs) == 4
    assert all(r["status"] == "complete" and not r["publication_claim"] for r in runs)
    assert all(r["exact_search_traces"][0]["chunk_hits"] and r["document_ranking"] for r in runs)


def test_single_token_negative_idf(tmp_path):
    data = tmp_path / "data"
    cli.generate_synthetic(data, n_docs=24, chunks_per_doc=1)
    rows = cli.read_jsonl(data / "chunks.jsonl")
    for r in rows:
        r["text"] = "ubiquitous"
    cli.write_jsonl(data / "chunks.jsonl", rows)
    m = json.loads((data / "dataset_manifest.json").read_text())
    m["chunks_sha256"] = file_hash(data / "chunks.jsonl")
    write_json(data / "dataset_manifest.json", m)
    idx = index(data)
    assert np.all(idx.scores("ubiquitous") < 0)  # Do not incorrectly clip epsilon floor to zero.
    np.testing.assert_allclose(idx.scores("ubiquitous"), .25 * math.log(.5 / 24.5))


@pytest.mark.parametrize("value", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_strict_json(value):
    from src.retrieval.verified_bm25 import parse_json
    with pytest.raises(ValueError):
        parse_json(value)


def test_live_executor_injected_search_no_provider(data):
    """Exercise the live-worker executor's dependency seam in dry mode only."""
    from types import SimpleNamespace
    path = ROOT / "scripts/48_run_live_budget.py"
    if not path.exists():
        pytest.skip("Independent live worker adapter not present")
    spec = importlib.util.spec_from_file_location("live_executor_integration", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    idx = index(data)
    tasks = cli.read_jsonl(data / "tasks.jsonl")
    for task in tasks:
        task.update(task_family="synthetic", split="dev")
    cli.write_jsonl(data / "adapter_tasks.jsonl", tasks)
    write_json(data / "adapter_config.json", {"conditions": ["original_top24"], "model": "none", "provider": "offline"})
    invoked = []

    def factory(chunks, metadata):
        assert chunks == idx.rows
        assert metadata == idx.meta
        invoked.append(idx.search(tasks[0]["question"], max_unique_docs=24))
        return idx

    def no_provider(*args):
        raise AssertionError("Provider construction forbidden")

    args = SimpleNamespace(config=data / "adapter_config.json", tasks=data / "adapter_tasks.jsonl",
                           corpus=data / "corpus.jsonl", chunks=data / "chunks.jsonl",
                           index=data / "index/index.json", stage="dev-smoke", split="dev",
                           repeat=1, live=False, output=data / "adapter_run")
    result = adapter.execute(args, client_factory=no_provider, search_factory=factory)
    assert len(invoked) == 1 and len({h["doc_id"] for h in invoked[0]}) == 24
    assert all(r["status"] == "dry_run" for r in result)


def test_selector_max_all_chunks_original_query(data):
    idx = index(data)
    candidates = list(dict.fromkeys(r["doc_id"] for r in idx.search("saline yeast", max_unique_docs=24)))
    scores = idx.scores("cold algae")
    ranking = idx.rank("cold algae", candidates, top_k=24)
    for hit in ranking:
        assert hit["score"] == max(scores[i] for i, row in enumerate(idx.rows) if row["doc_id"] == hit["doc_id"])
    assert set(h["doc_id"] for h in ranking) == set(candidates)


@pytest.mark.parametrize("mutation", ["label", "task_hash", "orphan", "missing_task"])
def test_invalid_qrels(data, mutation):
    idx = index(data)
    p = data / "qrels.json"
    q = json.loads(p.read_text())
    if mutation == "label":
        q["label_status"] = "gold"
    elif mutation == "task_hash":
        q["tasks_sha256"] = "bad"
    elif mutation == "orphan":
        q["qrels"]["T00"]["absent"] = 1
    else:
        del q["qrels"]["T00"]
    write_json(p, q)
    with pytest.raises(ValueError):
        cli.validate_evaluation(data / "tasks.jsonl", p, data / "expected_task_ids.json", idx)


def test_structural_contract(data):
    labels = json.loads((data / "structural_labels.json").read_text())["documents"]
    qrels = json.loads((data / "qrels.json").read_text())["qrels"]
    for task in cli.read_jsonl(data / "tasks.jsonl"):
        contract = task["synthetic_contract"]
        expected = {d for d, factors in labels.items() if factors == contract["constraints"]}
        assert set(contract["supporting_doc_ids"]) == expected
        assert {d for d, g in qrels[task["task_id"]].items() if g > 0} == expected
    for chunk in cli.read_jsonl(data / "chunks.jsonl"):
        assert "synthetic_contract" not in chunk["text"]
        assert chunk["doc_id"] not in chunk["text"]
        assert all(value in chunk["text"] for value in labels[chunk["doc_id"]].values())


def test_real_smoke_probe_contract():
    p = ROOT / "results/readiness/verified_bm25/real_corpus/probes.json"
    probes = json.loads(p.read_text())
    assert probes["label_status"] == "unscored_retrieval_smoke_no_qrels"
    assert len(probes["probes"]) == 8
    assert {p["task_family"] for p in probes["probes"]} == {
        "constraint", "comparative", "contradiction", "abstention", "multihop",
        "temporal", "aggregation", "negative"}
    assert all(set(p) == {"task_id", "task_family", "split", "question"} for p in probes["probes"])


def test_adapter_array_transport_unicode(tmp_path):
    path = ROOT / "scripts/48_run_live_budget.py"
    spec = importlib.util.spec_from_file_location("unicode_transport_adapter", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    rows = [{"text": "paragraph\u2029second\u2028line", "doc_id": "D1"}]
    p = tmp_path / "corpus.json"
    write_json(p, rows)
    assert adapter.load(p) == rows
    p = tmp_path / "corpus.jsonl"
    p.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n")
    assert adapter.load(p) == rows  # Native48 now preserves embedded U+2028/U+2029.


@pytest.fixture(scope="module")
def actual_smoke():
    p = ROOT / "results/readiness/verified_bm25/real_corpus_v2/summary.json"
    if not p.exists():
        pytest.skip("Explicit real-corpus smoke has not been run in this checkout")
    return p.parent, json.loads(p.read_text())


def test_actual_smoke_provenance(actual_smoke):
    p, report = actual_smoke
    assert report["provenance_id"] == digest({k: v for k, v in report.items() if k != "provenance_id"})
    assert report["input_hashes"]["manifest_sha256"] == "59d7751daeb67925903629ed6c542c1e98874d17b0499e43f3a387cc1168b5ae"
    assert (report["documents"], report["chunks"]) == (4936, 10313)
    for relative, sha in report["output_sha256"].items():
        assert file_hash(p / relative) == sha
    for relative, sha in report["source_sha256"].items():
        assert file_hash(ROOT / relative) == sha
    for filename, sha in report["index_sha256"].items():
        assert file_hash(ROOT / "data/processed/authoritative_fulltext_v1_bm25" / filename) == sha
    from src.evaluation.budget_protocol import validate_manifest
    native = json.loads((p / "native48/manifest.json").read_text())
    assert validate_manifest(native)
    for bindings in (native["source_sha256"], native["index_sha256"], native["config"]["runtime_dependencies"]):
        for name, sha in bindings.items():
            assert file_hash(name) == sha


def test_actual_smoke_protocols(actual_smoke):
    p, report = actual_smoke
    rows = json.loads((p / "protocol_diagnostics.json").read_text())
    assert len(rows) == 24
    for task_id in {r["task_id"] for r in rows}:
        by_condition = {r["condition"]: r for r in rows if r["task_id"] == task_id}
        a, b = by_condition["original_top24"], by_condition["repeated_original_3x8"]
        assert a["candidate_docs"] == b["candidate_docs"]
        assert a["common_selector_docs"] == b["common_selector_docs"]
    for row in rows:
        assert row["original_query_max_selector_verified"]
        assert len(set(row["candidate_docs"])) == 24
        assert row["accounting"]["model_calls"] == 0
        assert not row["feedback_generation_executed"]
        assert sum(len(r["admitted_chunks"]) for r in row["rounds"]) == 24
        assert all("text" not in h for r in row["rounds"] for h in r["admitted_chunks"])


def test_actual_smoke_formula_resources_adapter(actual_smoke):
    p, report = actual_smoke
    historical = report["historical_evidence"]
    assert historical["measurement_origin"] == "copied_historical_v1_not_rerun"
    assert file_hash(ROOT / historical["summary_path"]) == historical["summary_sha256"]
    assert historical["formula_comparisons"] >= 50
    assert historical["formula_max_absolute_error"] < 1e-10
    assert all(report["checks"].values())
    assert report["adapter_dry_plan_rows"] == 24
    assert report["human_validation"] is False and report["publication_claim"] is False
    assert report["index_rebuilt"] is False
    for phase in ("native_adapter", "verified_load"):
        assert report["resources"][phase]["measurement_origin"] == "current_v2_execution"
        assert report["resources"][phase]["wall_seconds"] > 0
        assert report["resources"][phase]["process_peak_rss_after_bytes"] > 0
    adapter = json.loads((p / "adapter_integration.json").read_text())
    assert adapter["actual_native_retrieval_checks"] == 24
    assert adapter["dependency_injection"] is False
    assert adapter["synthetic"] is False and report["live"] is False
    assert adapter["executed_provider_calls"] == 0
    assert adapter["executed_generation_calls"] == 0


def test_historical_v1_stale_binding_detected_explicitly():
    """An old receipt stays valid history, but cannot attest to revised current code."""
    path = ROOT / "results/readiness/verified_bm25/real_corpus_v2/run_integration.py"
    spec = importlib.util.spec_from_file_location("v2_binding_check", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    old = json.loads((ROOT / "results/readiness/verified_bm25/real_corpus/summary.json").read_text())
    assert old["provenance_id"] == digest({k: v for k, v in old.items() if k != "provenance_id"})
    with pytest.raises(ValueError, match="Stale source binding"):
        runner.verify_sources(old["source_sha256"])
