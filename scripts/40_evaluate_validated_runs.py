#!/usr/bin/env python3
"""Local-only strict evaluator. JSON/JSONL in; exclusive-create versioned JSON out."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.strict_metrics import VERSION, evaluate_documents, index_rows, check_subset, strings
from src.evaluation.structural_metrics import score_structure, summarize_structure
from src.evaluation.selective import evaluate_selective, select_dev_threshold


def reject_constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def unique_object(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate JSON key: {k}")
        out[k] = v
    return out


def decode(text):
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", required=True, help="JSON array or JSONL; explicit expected task universe")
    p.add_argument("--runs", required=True)
    p.add_argument("--annotations")
    p.add_argument("--selective-judgments")
    p.add_argument("--corpus", help="JSON array of document IDs; activates strict referential integrity")
    p.add_argument("--missing-policy", choices=("error", "zero"), default="error")
    p.add_argument("--allow-proxy", action="store_true")
    p.add_argument("--structural-k", type=int, default=10)
    p.add_argument("--calibration", help="Dev-only externally judged candidate answers")
    p.add_argument("--calibration-split", choices=("dev", "test", "train"))
    p.add_argument("--max-risk", type=float, default=0.1)
    p.add_argument("--output", required=True, help="Must not exist; no overwrite option")
    args = p.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        p.error(f"refusing to overwrite existing output: {output}")
    hashes = {}
    def load(path):
        if not path:
            return None
        raw = Path(path).read_bytes()
        hashes[str(Path(path).resolve())] = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8")
        value = decode(text) if text.lstrip().startswith("[") else [decode(line) for line in text.splitlines() if line.strip()]
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected JSON array or JSONL rows")
        return value
    try:
        if args.structural_k <= 0:
            raise ValueError("structural-k must be positive")
        if args.calibration_split and not args.calibration:
            raise ValueError("calibration-split requires calibration")
        tasks, runs = load(args.tasks), load(args.runs)
        annotations, judgments, corpus, calibration = (load(path) for path in
            (args.annotations, args.selective_judgments, args.corpus, args.calibration))
        documents = evaluate_documents(tasks, runs, args.missing_policy, corpus)
        expected, actual = index_rows(tasks, "tasks"), index_rows(runs, "runs")
        anns = index_rows(annotations or [], "annotations")
        check_subset(anns, expected, "annotations")
        structural = []
        for key, task in expected.items():
            a = anns.get(key)
            if a is not None and a["task_family"] != task.get("task_family"):
                raise ValueError(f"{key}: annotation/task family mismatch")
            score = score_structure(actual.get(key, {}).get("retrieved_doc_ids", [])[:args.structural_k], a, args.allow_proxy, corpus)
            structural.append({"task_id": key, "task_family": task.get("task_family"),
                               "output_present": key in actual,
                               "missing_output_penalty": key not in actual and score["denominator"] == 1, **score})
        selective_requested = judgments is not None or any("decision" in r or "confidence" in r for r in runs)
        selective = evaluate_selective(tasks, runs, judgments or []) if selective_requested else {"status": "not_evaluated", "reason": "no_selective_inputs"}
        threshold = select_dev_threshold(calibration, split=args.calibration_split,
                                        evaluation_task_ids=list(expected), max_risk=args.max_risk) if calibration is not None else None
        if corpus is not None and calibration is not None:
            # Calibration rows need not contain retrieved docs, but if supplied they must resolve.
            for row in calibration:
                if "retrieved_doc_ids" in row and not set(strings(row["retrieved_doc_ids"], "calibration doc IDs")) <= set(corpus):
                    raise ValueError("calibration references unknown corpus document")
        modules = [Path(__file__).resolve()] + [ROOT / "src/evaluation" / name for name in
                    ("strict_metrics.py", "structural_metrics.py", "selective.py", "__init__.py")]
        scripts = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in modules}
        warnings = sorted({w for r in structural for w in r.get("warnings", [])})
        structural_status = ("evaluated" if any(r["status"] == "evaluated" for r in structural) else
                             "descriptive_proxy" if warnings else "not_evaluated")
        report = {"schema_version": VERSION, "status": "completed", "warnings": warnings,
                  "configuration": vars(args), "hashes": {"algorithm": "sha256", "inputs": hashes, "scripts": scripts},
                  "documents": documents, "structural": {"status": structural_status, "cutoff": args.structural_k,
                    "summary": summarize_structure(structural), "by_family": {
                        family: summarize_structure([r for r in structural if r["task_family"] == family])
                        for family in sorted({r["task_family"] for r in structural if isinstance(r["task_family"], str)})},
                    "per_task": structural}, "selective": selective, "calibration": threshold}
        serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as f:
            f.write(serialized)
    except (ValueError, OSError, KeyError, TypeError) as e:
        print(json.dumps({"schema_version": VERSION, "status": "error", "error": str(e)}, allow_nan=False), file=sys.stderr)
        return 2
    print(json.dumps({"status": "completed", "output": str(output), "warnings": warnings}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
