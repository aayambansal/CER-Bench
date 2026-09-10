import pytest
from src.evaluation.selective import evaluate_selective, risk_coverage, select_dev_threshold


def label(key, failure=False, support=True):
    return {"task_id": key, "evidence_set_failure": failure, "verified_support": support,
            "annotation_status": "human_adjudicated", "provenance": "SYNTHETIC external judgment"}


def test_decisions_support_and_actual_failure_separate():
    tasks = [{"task_id": str(i), "gold_doc_ids": []} for i in range(3)]
    runs = [{"task_id": str(i), "decision": d, "confidence": 0.8}
            for i, d in enumerate(["ANSWER", "ABSTAIN", "UNCERTAIN"])]
    out = evaluate_selective(tasks, runs, [label("0", True, True), label("1", False, False)])
    assert out["coverage"] == 1 / 3
    assert out["selective_risk"] == 1
    assert out["support_confusion"]["ABSTAIN"]["unsupported"] == 1
    assert out["support_confusion"]["UNCERTAIN"]["unjudged"] == 1


def test_missing_losses_and_all_empty():
    tasks = [{"task_id": "a"}]
    runs = [{"task_id": "a", "decision": "ANSWER", "confidence": 1}]
    out = evaluate_selective(tasks, runs, [])
    assert out["selective_risk"] is None and out["missing_answer_judgments"] == 1
    assert out["risk_coverage"][-1]["risk"] is None
    assert evaluate_selective(tasks, [], [])["coverage"] == 0
    assert evaluate_selective([], [], [])["coverage"] is None
    assert risk_coverage([])[0]["risk"] is None


def test_tied_scores_are_blocks_and_order_invariant():
    rows = [{"task_id": "a", "confidence": 0.8, "evidence_set_failure": False},
            {"task_id": "b", "confidence": 0.8, "evidence_set_failure": True}]
    points = risk_coverage(rows)
    assert points == risk_coverage(rows[::-1])
    assert len(points) == 2 and points[-1]["risk"] == 0.5
    with pytest.raises(ValueError, match="duplicate"):
        risk_coverage(rows + rows)


def test_dev_calibration_no_leakage_or_guarantees():
    rows = [{**label("dev1"), "confidence": 0.8}, {**label("dev2", True), "confidence": 0.8}]
    result = select_dev_threshold(rows, split="dev", evaluation_task_ids=["test"], max_risk=0.1)
    assert result["policy"] == "answer_none"
    assert "no conformal" in result["warning"]
    assert select_dev_threshold(rows, split="dev", evaluation_task_ids=["test"], max_risk=0.5)["threshold"] == 0.8
    with pytest.raises(ValueError, match="dev split"):
        select_dev_threshold(rows, split="test", evaluation_task_ids=["test"])
    with pytest.raises(ValueError, match="leakage"):
        select_dev_threshold(rows, split="dev", evaluation_task_ids=["dev1"])
    rows[0]["evidence_set_failure"] = None
    with pytest.raises(ValueError, match="every external"):
        select_dev_threshold(rows, split="dev", evaluation_task_ids=["test"])


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), True, -1, 1.01, "0.5"])
def test_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        evaluate_selective([{"task_id": "a"}], [{"task_id": "a", "decision": "ANSWER", "confidence": confidence}], [])


def test_invalid_decision_and_unverified_labels():
    with pytest.raises(ValueError, match="decision"):
        evaluate_selective([{"task_id": "a"}], [{"task_id": "a", "decision": "", "confidence": 0}], [])
    bad = label("a")
    bad["annotation_status"] = "unvalidated"
    with pytest.raises(ValueError, match="externally verified"):
        evaluate_selective([{"task_id": "a"}], [], [bad])
