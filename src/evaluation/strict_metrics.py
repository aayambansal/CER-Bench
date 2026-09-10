"""Strict binary-qrel retrieval evaluation; no implicit task/judgment inference."""
import math

VERSION = "cerbench.evaluation.v1"
METRICS = ("Recall@5", "Recall@10", "Recall@20", "nDCG@10", "MRR")


def strings(value, label, nonempty=False):
    if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError(f"{label}: expected list of nonempty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{label}: duplicate IDs/units")
    if nonempty and not value:
        raise ValueError(f"{label}: must not be empty")
    return value


def index_rows(rows, label):
    out = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("task_id"), str) or not row["task_id"].strip():
            raise ValueError(f"{label}: missing/invalid task_id")
        key = row["task_id"]
        if key in out:
            raise ValueError(f"{label}: duplicate task_id {key}")
        out[key] = row
    return out


def check_subset(rows, expected, label):
    unknown = set(rows) - set(expected)
    if unknown:
        raise ValueError(f"{label}: unknown task IDs {sorted(unknown)}")


def doc_metrics(retrieved, gold):
    strings(retrieved, "retrieved_doc_ids")
    strings(gold, "gold_doc_ids")
    if not gold:
        return dict.fromkeys(METRICS)
    relevant = set(gold)
    result = {f"Recall@{k}": len(set(retrieved[:k]) & relevant) / len(relevant) for k in (5, 10, 20)}
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(retrieved[:10]) if d in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
    result["nDCG@10"] = dcg / ideal
    result["MRR"] = next((1 / (i + 1) for i, d in enumerate(retrieved) if d in relevant), 0.0)
    return result


def evaluate_documents(tasks, runs, missing_policy="error", corpus_ids=None):
    if missing_policy not in ("error", "zero"):
        raise ValueError("missing_policy must be error or zero")
    expected, actual = index_rows(tasks, "tasks"), index_rows(runs, "runs")
    check_subset(actual, expected, "runs")
    missing = sorted(set(expected) - set(actual))
    if missing and missing_policy == "error":
        raise ValueError(f"missing run rows: {missing}")
    corpus = None if corpus_ids is None else set(strings(corpus_ids, "corpus IDs"))
    records = []
    for key, task in expected.items():
        retrieved = strings(actual[key].get("retrieved_doc_ids"), "retrieved_doc_ids") if key in actual else []
        gold = task.get("gold_doc_ids")
        if gold is not None:
            strings(gold, "gold_doc_ids")
        if corpus is not None:
            unknown = (set(retrieved) | set(gold or [])) - corpus
            if unknown:
                raise ValueError(f"{key}: unknown corpus doc IDs {sorted(unknown)}")
        status = "missing_judgments" if gold is None else "empty_gold" if not gold else "evaluated"
        records.append({"task_id": key, "status": status, "output_present": key in actual,
                        "missing_output_penalty": key not in actual and status == "evaluated",
                        "metrics": dict.fromkeys(METRICS) if gold is None else doc_metrics(retrieved, gold)})
    eligible = [r for r in records if r["status"] == "evaluated"]
    return {"schema_version": VERSION, "status": "evaluated" if eligible else "not_evaluated",
            "missing_policy": missing_policy,
            "coverage": {"expected": len(expected), "actual": len(actual), "missing": len(missing),
                         "fraction": len(actual) / len(expected) if expected else None, "missing_task_ids": missing},
            "judgments": {"nonempty": len(eligible), "empty_gold": sum(r["status"] == "empty_gold" for r in records),
                          "missing": sum(r["status"] == "missing_judgments" for r in records)},
            "metrics": {m: {"value": sum(r["metrics"][m] for r in eligible) / len(eligible) if eligible else None,
                            "denominator": len(eligible)} for m in METRICS}, "per_task": records}
