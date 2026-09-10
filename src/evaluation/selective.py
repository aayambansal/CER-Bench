"""Selective evidence-set evaluation and explicitly empirical dev calibration."""
import math
from .strict_metrics import index_rows, check_subset

DECISIONS = {"ANSWER", "ABSTAIN", "UNCERTAIN"}


def validate_decision(row):
    if row.get("decision") not in DECISIONS:
        raise ValueError("decision must be ANSWER, ABSTAIN, or UNCERTAIN")
    c = row.get("confidence")
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or not 0 <= c <= 1:
        raise ValueError("confidence must be finite in [0,1]")


def validate_judgment(row):
    if row.get("annotation_status") not in {"human_adjudicated", "expert_adjudicated"}:
        raise ValueError("selective judgments must be externally verified")
    p = row.get("provenance")
    if not isinstance(p, str) or not p.strip():
        raise ValueError("selective judgments require nonempty provenance")
    for field in ("evidence_set_failure", "verified_support"):
        if field in row and row[field] is not None and type(row[field]) is not bool:
            raise ValueError(f"{field} must be boolean or null")


def risk_coverage(rows, expected_count=None):
    """Confidence sweep over supplied candidate answers; equal scores enter together.

    Rows need confidence and external evidence_set_failure. No guessed loss for
    missing labels: risk is null unless every selected candidate has a label.
    """
    index_rows(rows, "risk candidates")
    n = len(rows) if expected_count is None else expected_count
    if type(n) is not int or n < len(rows):
        raise ValueError("expected_count cannot be smaller than candidates")
    for row in rows:
        validate_decision({**row, "decision": "ANSWER"})
        if row.get("evidence_set_failure") is not None and type(row["evidence_set_failure"]) is not bool:
            raise ValueError("evidence_set_failure must be boolean or null")
    points = [{"threshold": None, "answered": 0, "coverage": 0.0 if n else None,
               "judged": 0, "risk": None}]
    ordered = sorted(rows, key=lambda r: -r["confidence"])
    selected, judged, failures, i = 0, 0, 0, 0
    while i < len(ordered):
        threshold = ordered[i]["confidence"]
        while i < len(ordered) and ordered[i]["confidence"] == threshold:
            loss = ordered[i].get("evidence_set_failure")
            selected += 1
            judged += loss is not None
            failures += int(loss) if loss is not None else 0
            i += 1
        points.append({"threshold": threshold, "answered": selected, "coverage": selected / n,
                       "judged": judged, "risk": failures / selected if judged == selected else None})
    return points


def evaluate_selective(tasks, runs, judgments):
    expected, actual, labels = index_rows(tasks, "tasks"), index_rows(runs, "runs"), index_rows(judgments, "selective judgments")
    check_subset(actual, expected, "runs")
    check_subset(labels, expected, "selective judgments")
    for row in labels.values():
        validate_judgment(row)
    confusion = {d: {"supported": 0, "unsupported": 0, "unjudged": 0} for d in sorted(DECISIONS)}
    candidates = []
    counts = dict.fromkeys(sorted(DECISIONS), 0)
    for key, row in actual.items():
        validate_decision(row)
        decision = row["decision"]
        counts[decision] += 1
        label = labels.get(key, {})
        support = label.get("verified_support")
        confusion[decision]["unjudged" if support is None else "supported" if support else "unsupported"] += 1
        if decision == "ANSWER":
            candidates.append({"task_id": key, "confidence": row["confidence"],
                               "evidence_set_failure": label.get("evidence_set_failure")})
    judged = [r for r in candidates if r["evidence_set_failure"] is not None]
    failures = sum(r["evidence_set_failure"] for r in judged)
    risk = failures / len(candidates) if candidates and len(judged) == len(candidates) else None
    return {"status": "evaluated" if risk is not None else "partially_evaluated" if actual else "not_evaluated",
            "expected": len(expected), "actual": len(actual), "missing_outputs": len(expected) - len(actual),
            "decisions": counts, "coverage": len(candidates) / len(expected) if expected else None,
            "selective_risk": risk, "answered": len(candidates), "judged_answers": len(judged),
            "missing_answer_judgments": len(candidates) - len(judged),
            "observed_judged_answer_risk": failures / len(judged) if judged else None,
            "support_confusion": confusion, "risk_coverage_scope": "submitted ANSWER candidates only",
            "risk_coverage": risk_coverage(candidates, len(expected))}


def select_dev_threshold(rows, *, split, evaluation_task_ids, max_risk=0.1):
    """Maximal empirical dev coverage; NOT a conformal or statistical guarantee.

    Caller supplies *all candidate answers* with verified external losses. A frozen
    threshold selects confidence >= threshold on disjoint evaluation tasks.
    """
    if split != "dev":
        raise ValueError("threshold selection requires dev split")
    from .strict_metrics import strings
    evaluation = strings(evaluation_task_ids, "evaluation_task_ids", True)
    dev = index_rows(rows, "calibration")
    if set(dev) & set(evaluation):
        raise ValueError("calibration/evaluation task leakage")
    if isinstance(max_risk, bool) or not isinstance(max_risk, (float, int)) or not math.isfinite(max_risk) or not 0 <= max_risk <= 1:
        raise ValueError("max_risk must be finite in [0,1]")
    for row in rows:
        validate_judgment(row)
        if type(row.get("evidence_set_failure")) is not bool:
            raise ValueError("calibration requires every external failure label")
    points = risk_coverage(rows)
    feasible = [p for p in points if p["risk"] is not None and p["risk"] <= max_risk]
    best = max(feasible, key=lambda p: p["answered"]) if feasible else None
    return {"status": "selected" if best else "no_feasible_threshold", "method": "empirical_dev_only",
            "warning": "descriptive calibration; no conformal or population risk guarantee; task-ID disjointness is not component disjointness",
            "threshold": best["threshold"] if best else None, "policy": "confidence >= threshold" if best else "answer_none",
            "max_risk": max_risk, "dev_count": len(rows), "selected_point": best}
