#!/usr/bin/env python3
"""Script 14: Generate paper-ready LaTeX tables from results.

Usage:
    python scripts/14_make_tables.py [--split test]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import get_data_dir


def make_main_table(scores: dict, split: str) -> str:
    """Generate the main results table in LaTeX."""
    k_values = [5, 10, 20]
    baselines = sorted(scores.keys())

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Retrieval results on SynthSearch-Biomed (" + split + r" split). Best results in \textbf{bold}.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{l" + "c" * (len(k_values) + 2) + "}",
        r"\toprule",
        r"Method & " + " & ".join([f"R@{k}" for k in k_values]) + r" & nDCG@10 & MRR \\",
        r"\midrule",
    ]

    # Find best per metric
    bests = {}
    for metric in [f"recall@{k}" for k in k_values] + ["ndcg@10", "mrr"]:
        best_val = max(scores[b]["aggregate"].get(metric, 0) for b in baselines)
        bests[metric] = best_val

    for baseline in baselines:
        agg = scores[baseline]["aggregate"]
        vals = []
        for metric in [f"recall@{k}" for k in k_values] + ["ndcg@10", "mrr"]:
            v = agg.get(metric, 0)
            s = f"{v:.3f}"
            if abs(v - bests[metric]) < 0.0001:
                s = r"\textbf{" + s + "}"
            vals.append(s)
        name = baseline.replace("_", " ").title()
        lines.append(f"{name} & " + " & ".join(vals) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def make_family_table(scores: dict, split: str) -> str:
    """Generate per-family breakdown table."""
    baselines = sorted(scores.keys())
    families = set()
    for b in baselines:
        families.update(scores[b].get("per_family", {}).keys())
    families = sorted(families)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Recall@10 by task family (" + split + r" split).}",
        r"\label{tab:per_family}",
        r"\begin{tabular}{l" + "c" * len(baselines) + "}",
        r"\toprule",
        r"Family & " + " & ".join(b.replace("_", " ").title() for b in baselines) + r" \\",
        r"\midrule",
    ]

    for family in families:
        vals = []
        for b in baselines:
            v = scores[b].get("per_family", {}).get(family, {}).get("recall@10", 0)
            vals.append(f"{v:.3f}")
        lines.append(f"{family} & " + " & ".join(vals) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate paper tables")
    parser.add_argument("--split", type=str, default="dev")
    args = parser.parse_args()

    results_dir = Path("results/baselines")
    if not results_dir.exists():
        results_dir = get_data_dir("") / ".." / "results" / "baselines"

    scores_path = results_dir / f"scores_{args.split}.json"
    if not scores_path.exists():
        print(f"Error: {scores_path} not found. Run scripts/12_score_retrieval.py first.")
        sys.exit(1)

    with open(scores_path) as f:
        scores = json.load(f)

    tables_dir = get_data_dir("") / ".." / "results" / "paper_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Main table
    main_table = make_main_table(scores, args.split)
    path = tables_dir / "main_results.tex"
    with open(path, "w") as f:
        f.write(main_table)
    print(f"Main results table: {path}")

    # Family table
    family_table = make_family_table(scores, args.split)
    path = tables_dir / "per_family.tex"
    with open(path, "w") as f:
        f.write(family_table)
    print(f"Per-family table: {path}")

    print(f"\nDone. Tables in {tables_dir}")


if __name__ == "__main__":
    main()
