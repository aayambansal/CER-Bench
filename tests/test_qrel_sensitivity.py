"""Offline formula and fail-closed tests; no scientific validation implied."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

P = Path(__file__).resolve().parents[1] / "scripts/44_analyze_historical_qrel_sensitivity.py"
spec = importlib.util.spec_from_file_location("qrel_sensitivity", P)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def test_tau_ties_and_strict_reversals():
    result = q.rank_comparison(dict(a=1., b=1., c=2.), dict(a=1., b=2., c=3.), ("a", "b", "c"))
    assert result["kendall_tau_b"] == pytest.approx(2 / np.sqrt(6))
    assert result["spearman"] == pytest.approx(np.sqrt(3)/2)
    assert result["seed_tied_pairs"] == 1
    assert result["strict_reversals"] == []
    json.dumps(result, allow_nan=False)
    r = q.rank_comparison(dict(a=1., b=2., c=3.), dict(a=3., b=2., c=1.), ("a", "b", "c"))
    assert r["kendall_tau_b"] == pytest.approx(-1)
    assert len(r["strict_reversals"]) == 3


def test_unrounded_reversal():
    r = q.rank_comparison(dict(a=.100001, b=.100002), dict(a=.100002, b=.100001), ("a", "b"))
    assert len(r["strict_reversals"]) == 1


def test_cluster_bootstrap_deterministic_task_weighted():
    a = q.paired_cluster([1, 1, -1], ["a", "a", "b"], 1000, 42)
    assert a == q.paired_cluster([1, 1, -1], ["a", "a", "b"], 1000, 42)
    assert a["difference"] == pytest.approx(1/3)
    assert a["ci95_percentile"] == [-1, 1]
    assert a["p_two_sided"] == (a["exceedances"]+1)/1001
    assert q.paired_cluster([0, 0], ["a", "b"], 100, 42)["p_two_sided"] == 1


def test_holm():
    assert q.holm([.01, .04, .03]) == pytest.approx([.03, .06, .06])


def test_reconstruction_and_negative_prefix():
    seed = {"t": ["s"], "empty": []}
    labels = [{"task_id": "t", "doc_id": "p", "judgment": "RELEVANT\nexplanation"},
              {"task_id": "t", "doc_id": "n", "judgment": "NOT_RELEVANT\nRELEVANT mentioned"}]
    expanded, normalized = q.reconstruct(seed, labels, ["s", "p", "n"], 2)
    assert expanded == {"t": ["p", "s"], "empty": []}
    assert normalized[1]["relevant"] is False
    assert q.subsample(seed, normalized, 0, np.random.default_rng(1))[0] == seed
    assert q.subsample(seed, normalized, 1, np.random.default_rng(1))[0] == expanded


@pytest.mark.parametrize("label", ["MAYBE", "", None, "RELEVANTNESS"])
def test_unknown_labels_fail(label):
    with pytest.raises(ValueError, match="unknown judgment"):
        q.verdict(label)


def test_contradictions_duplicates_missing_rows():
    row = {"task_id": "t", "doc_id": "s", "judgment": "NOT_RELEVANT"}
    with pytest.raises(ValueError, match="seed/judgment contradiction t/s"):
        q.reconstruct({"t": ["s"]}, [row], ["s"])
    row["judgment"] = "RELEVANT"
    with pytest.raises(ValueError, match="duplicate/contradictory"):
        q.reconstruct({"t": []}, [row, row], ["s"])
    with pytest.raises(ValueError, match="missing/excess judgment rows"):
        q.reconstruct({"t": ["s"]}, [], ["s"], 30)
    with pytest.raises(ValueError, match="missing rows=.*t"):
        q.exact_keys({}, {"t": []}, "run")
    with pytest.raises(ValueError, match="unknown IDs=.*extra"):
        q.exact_keys({"extra": []}, {}, "run")


def test_unknown_and_duplicate_documents_rows_keys():
    with pytest.raises(ValueError, match="unknown doc IDs.*bad"):
        q.validate_docs(["bad"], ["s"], "t")
    with pytest.raises(ValueError, match="duplicate"):
        q.validate_docs(["s", "s"], ["s"], "t")
    with pytest.raises(ValueError, match="duplicate task_id"):
        q.index_rows([{"task_id": "t"}, {"task_id": "t"}], "run")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        q.no_duplicate_keys([("t", 1), ("t", 2)])


def test_strict_doc_metrics_and_empty():
    r = q.doc_metrics(["n", "p"], ["p", "s"])
    assert r["Recall@20"] == .5
    assert r["MRR"] == .5
    assert r["nDCG@10"] == pytest.approx((1/np.log2(3))/(1+1/np.log2(3)))
    assert all(v is None for v in q.doc_metrics([], []).values())


def test_jsonl_unicode_separator_not_record_boundary(tmp_path):
    p = tmp_path / "unicode.jsonl"
    p.write_text('{"text":"a\u2028b\u2029c"}\n')
    assert q.rows(p) == [{"text": "a\u2028b\u2029c"}]


def test_component_transitive_bridge_across_all_splits():
    tasks = [dict(task_id="testA", question="one question", supporting_doc_ids=["a"], hard_negative_doc_ids=[]),
             dict(task_id="train", question="second query", supporting_doc_ids=["a"], hard_negative_doc_ids=["b"]),
             dict(task_id="testB", question="third text", supporting_doc_ids=["b"], hard_negative_doc_ids=[])]
    comp, _ = q.component_map(tasks)
    assert len(set(comp.values())) == 1


def test_duplicate_corpus_audit_preserves_conflicting_records():
    records = [{"doc_id": "x", "pmid": "1", "title": "first"},
               {"doc_id": "y", "pmid": "2"}, {"doc_id": "x", "pmid": "3", "title": "other"}]
    result = q.duplicate_corpus_records(records)
    assert len(result) == 1
    assert result[0]["doc_id"] == "x"
    assert result[0]["physical_rows"] == [1, 3]
    assert result[0]["differing_fields"] == ["pmid", "title"]
    assert [r["pmid"] for r in result[0]["records"]] == ["1", "3"]


def test_forensic_safety_fields_cannot_be_overridden():
    e = q.forensic_envelope(release_ready=True, release_blocked=False, status="passed", input_validation_status="passed")
    assert e["release_ready"] is False
    assert e["release_blocked"] is True
    assert e["input_validation_status"] == "fail"
    assert e["status"] == "forensic_token_diagnostic_only"
    assert e["interpretation"] == q.FORENSIC_LABEL


def test_forensic_never_reads_or_indexes_corpus_preserves_failures(monkeypatch):
    # Offline integration against immutable failed inputs, not validated relevance.
    frozen = q.ROOT / "results/readiness/qrel_sensitivity/v1_final"
    before = {p.name: q.digest(p) for p in frozen.iterdir()}
    original_rows, original_read = q.rows, q.read
    def guarded_rows(path):
        assert "data/processed" not in str(path), "forensic mode must not parse corpus or candidate data"
        return original_rows(path)
    def guarded_read(path):
        assert "data/processed" not in str(path), "forensic mode must not dereference metadata"
        return original_read(path)
    monkeypatch.setattr(q, "rows", guarded_rows)
    monkeypatch.setattr(q, "read", guarded_read)
    monkeypatch.setattr(q, "audit_inputs", lambda *a: pytest.fail("must reuse frozen failure, not parse corpus"))
    report, per_task, gold = q.forensic_compute(q.ROOT, resamples=100)
    assert report["release_blocked"] and not report["release_ready"]
    assert report["unresolved_input_audit"] == original_read(frozen / "INPUT_AUDIT.json")
    assert report["original_failure"] == original_read(frozen / "FAILURE.json")
    assert len(report["unresolved_input_audit"]["errors"]) == 2
    assert {p.name: q.digest(p) for p in frozen.iterdir()} == before
    assert report["denominator"]["qrel_pairs"] == {"seed": 264, "stored_expanded": 1369, "reconstructed_expanded": 1370}
    assert len(per_task) == 12*125*3
    assert all(r["interpretation"] == q.FORENSIC_LABEL and r["release_blocked"] for r in per_task)
    assert "37307965" not in gold["stored_expanded"]["multihop_0189"]
    assert "37307965" in gold["reconstructed_expanded"]["multihop_0189"]
    for e in report["missing_pair_effects"]:
        assert q.Fraction(e["mean_delta_exact"]) == q.Fraction(e["task_delta_exact"])/108


def test_forensic_cli_is_explicit_and_dispatches_separately(monkeypatch, tmp_path):
    monkeypatch.setattr(q, "ROOT", tmp_path)
    called = []
    monkeypatch.setattr(q, "run_forensic", lambda root, output, resamples: called.append((output, resamples)) or 0)
    monkeypatch.setattr(q.sys, "argv", [str(P), "--forensic-token-analysis"])
    assert q.main() == 0
    assert called == [(tmp_path / "results/readiness/qrel_sensitivity/v1_forensic", 10000)]
    assert not (tmp_path / "results/readiness/qrel_sensitivity/v1_final").exists()
