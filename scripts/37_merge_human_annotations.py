#!/usr/bin/env python3
"""Validate, merge, and summarize two blinded CER-Bench annotation sheets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


VALID = {"0", "1", "2", "U"}


def load(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for row in rows:
        pair = row["pair_id"]
        label = row["relevance"].strip().upper()
        if label not in VALID:
            raise ValueError(f"{path}: {pair} has invalid/missing relevance {label!r}")
        if label == "2" and not row["evidence_span"].strip():
            raise ValueError(
                f"{path}: {pair} is directly relevant but has no evidence span"
            )
        if row["role_json"].strip():
            json.loads(row["role_json"])
        output[pair] = row
    return output


def cohen_kappa(a: list[str], b: list[str], binary: bool = False) -> float:
    if binary:
        a = ["2" if x == "2" else "N" for x in a]
        b = ["2" if x == "2" else "N" for x in b]
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[label] / n) * (cb[label] / n) for label in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def main() -> None:
    raise SystemExit(
        'Legacy merge disabled: unsafe identity, uncertainty, span and provenance checks. '
        'Use scripts/43_validate_human_labels.py with the provisional human_qrel_v2 kit; '
        'no final human-qrel export is authorized.'
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("annotator_a", type=Path)
    parser.add_argument("annotator_b", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    a, b = load(args.annotator_a), load(args.annotator_b)
    if set(a) != set(b):
        raise ValueError("Annotator sheets contain different pair IDs")
    pairs = sorted(a)
    la = [a[p]["relevance"].strip().upper() for p in pairs]
    lb = [b[p]["relevance"].strip().upper() for p in pairs]
    disagreements = []
    agreements = []
    for pair in pairs:
        ra, rb = a[pair], b[pair]
        row = {
            "pair_id": pair,
            "task_id": ra["task_id"],
            "task_family": ra["task_family"],
            "question": ra["question"],
            "doc_id": ra["doc_id"],
            "title": ra["title"],
            "label_A": ra["relevance"],
            "evidence_A": ra["evidence_span"],
            "role_A": ra["role_json"],
            "label_B": rb["relevance"],
            "evidence_B": rb["evidence_span"],
            "role_B": rb["role_json"],
            "adjudicated_relevance": ra["relevance"]
            if ra["relevance"] == rb["relevance"]
            else "",
            "adjudicated_evidence_span": ra["evidence_span"]
            if ra["relevance"] == rb["relevance"]
            else "",
            "adjudicated_role_json": ra["role_json"]
            if ra["role_json"] == rb["role_json"]
            else "",
            "adjudicator_notes": "",
        }
        (
            agreements
            if ra["relevance"] == rb["relevance"] and ra["role_json"] == rb["role_json"]
            else disagreements
        ).append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list((disagreements or agreements)[0])
    for name, rows in [
        ("agreements.csv", agreements),
        ("adjudication_required.csv", disagreements),
    ]:
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "n_pairs": len(pairs),
        "exact_label_agreement": round(
            sum(x == y for x, y in zip(la, lb)) / len(pairs), 4
        ),
        "cohen_kappa_four_way": round(cohen_kappa(la, lb), 4),
        "cohen_kappa_direct_relevance": round(cohen_kappa(la, lb, True), 4),
        "role_and_label_agreements": len(agreements),
        "adjudication_required": len(disagreements),
        "label_counts_A": Counter(la),
        "label_counts_B": Counter(lb),
    }
    (args.output_dir / "agreement_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
