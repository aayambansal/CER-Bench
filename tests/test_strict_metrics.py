import importlib.util
import json
from pathlib import Path
import pytest
from src.evaluation.strict_metrics import doc_metrics, evaluate_documents


def test_metrics_and_duplicate_ndcg_attack():
    assert doc_metrics(["a", "b"], ["a", "b"])["nDCG@10"] == 1
    assert doc_metrics(["z", "a"], ["a"])["MRR"] == 0.5
    with pytest.raises(ValueError, match="duplicate"):
        doc_metrics(["a"] * 10, ["a"])
    with pytest.raises(ValueError, match="duplicate"):
        doc_metrics(["a"], ["a", "a"])
    assert doc_metrics([], ["a"])["MRR"] == 0


def test_empty_and_missing_judgments_distinct():
    out = evaluate_documents([{"task_id": "empty", "gold_doc_ids": []}, {"task_id": "unknown"}],
        [{"task_id": "empty", "retrieved_doc_ids": []}, {"task_id": "unknown", "retrieved_doc_ids": []}])
    assert out["judgments"] == {"nonempty": 0, "empty_gold": 1, "missing": 1}
    assert all(x == {"value": None, "denominator": 0} for x in out["metrics"].values())
    assert out["status"] == "not_evaluated"


def test_missing_fail_or_explicit_zero():
    tasks = [{"task_id": "a", "gold_doc_ids": ["d"]}]
    with pytest.raises(ValueError, match="missing run"):
        evaluate_documents(tasks, [])
    out = evaluate_documents(tasks, [], "zero")
    assert out["metrics"]["Recall@5"] == {"value": 0, "denominator": 1}
    assert out["coverage"]["fraction"] == 0
    assert out["per_task"][0]["missing_output_penalty"]


@pytest.mark.parametrize("tasks,runs,match", [
    ([{"task_id": "a"}] * 2, [], "duplicate"),
    ([{"task_id": "a"}], [{"task_id": "a"}] * 2, "duplicate"),
    ([{"task_id": "a"}], [{"task_id": "z"}], "unknown"),
    ([{"task_id": "a"}], [{"task_id": "a"}], "retrieved_doc_ids"),
])
def test_invalid_universe(tasks, runs, match):
    with pytest.raises(ValueError, match=match):
        evaluate_documents(tasks, runs)


def test_corpus_and_empty_universe():
    with pytest.raises(ValueError, match="unknown corpus"):
        evaluate_documents([{"task_id": "a", "gold_doc_ids": ["bad"]}],
                           [{"task_id": "a", "retrieved_doc_ids": []}], corpus_ids=["good"])
    assert evaluate_documents([], [])["coverage"]["fraction"] is None


def cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts/40_evaluate_validated_runs.py"
    spec = importlib.util.spec_from_file_location("strict_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_hashes_no_overwrite_and_errors(tmp_path):
    cli = cli_module()
    tasks, runs, output = (tmp_path / x for x in ("tasks.json", "runs.json", "report.json"))
    tasks.write_text('[{"task_id":"t","gold_doc_ids":["a"]}]')
    runs.write_text('[{"task_id":"t","retrieved_doc_ids":["a"]}]')
    args = ["--tasks", str(tasks), "--runs", str(runs), "--output", str(output)]
    assert cli.main(args) == 0
    original = output.read_bytes()
    report = json.loads(original)
    assert len(report["hashes"]["inputs"]) == 2
    assert len(report["hashes"]["scripts"]) == 5
    import hashlib
    assert report["hashes"]["inputs"][str(tasks.resolve())] == hashlib.sha256(tasks.read_bytes()).hexdigest()
    with pytest.raises(SystemExit):
        cli.main(args)
    assert output.read_bytes() == original
    runs.write_text('[]')
    args[-1] = str(tmp_path / "error.json")
    assert cli.main(args) == 2
    assert not Path(args[-1]).exists()


def test_cli_rejects_duplicate_keys_and_nonfinite():
    cli = cli_module()
    for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
        with pytest.raises(ValueError):
            cli.decode(text)


def test_all_synthetic_interfaces_and_proxy_gate(tmp_path):
    cli = cli_module()
    examples = Path(__file__).resolve().parents[1] / "examples/evaluation_contract"
    output = tmp_path / "smoke.json"
    args = ["--tasks", str(examples / "tasks.json"), "--runs", str(examples / "runs.json"),
            "--annotations", str(examples / "annotations.json"),
            "--selective-judgments", str(examples / "selective_judgments.json"),
            "--corpus", str(examples / "corpus.json"), "--calibration", str(examples / "calibration.json"),
            "--calibration-split", "dev", "--output", str(output)]
    assert cli.main(args) == 0
    report = json.loads(output.read_text())
    assert len(report["hashes"]["inputs"]) == 6
    assert report["documents"]["metrics"]["Recall@5"]["value"] == 0.75
    assert report["structural"]["summary"]["evaluated"] == {
        "denominator": 2, "metrics": {"unit_coverage": 1, "set_completion": 1, "family_success": 0}}
    assert report["selective"]["coverage"] == 0.5 and report["selective"]["selective_risk"] == 1
    assert len(report["selective"]["risk_coverage"]) == 2
    assert report["calibration"]["threshold"] == 0.9
    args[args.index("--annotations") + 1] = str(examples / "annotations_proxy.json")
    args[-1] = str(tmp_path / "unverified.json")
    assert cli.main(args) == 0
    assert json.loads(Path(args[-1]).read_text())["structural"]["status"] == "not_evaluated"
    args[-1] = str(tmp_path / "proxy.json")
    assert cli.main(args + ["--allow-proxy"]) == 0
    report = json.loads(Path(args[-1]).read_text())
    assert report["warnings"] and report["structural"]["status"] == "descriptive_proxy"
    assert report["structural"]["summary"]["evaluated"]["denominator"] == 0


def test_metric_bounds_exhaustive_small_rankings():
    from itertools import permutations, combinations
    docs = ["a", "b", "c", "d"]
    for n in range(5):
        for rank in permutations(docs, n):
            for m in range(1, 5):
                for gold in combinations(docs, m):
                    assert all(0 <= v <= 1 for v in doc_metrics(list(rank), list(gold)).values())
