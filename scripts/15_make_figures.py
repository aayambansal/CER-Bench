#!/usr/bin/env python3
"""Script 15: Generate paper-ready figures from results.

Usage:
    python scripts/15_make_figures.py [--split dev]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import get_data_dir


def plot_recall_by_family(scores: dict, split: str, output_dir: Path):
    """Bar chart: Recall@10 by family for each baseline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    baselines = sorted(scores.keys())
    families = set()
    for b in baselines:
        families.update(scores[b].get("per_family", {}).keys())
    families = sorted(families)

    x = np.arange(len(families))
    width = 0.8 / len(baselines)
    colors = plt.cm.Set2(np.linspace(0, 1, len(baselines)))

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, baseline in enumerate(baselines):
        vals = [scores[baseline].get("per_family", {}).get(f, {}).get("recall@10", 0) for f in families]
        ax.bar(x + i * width, vals, width, label=baseline.replace("_", " ").title(), color=colors[i])

    ax.set_xlabel("Task Family")
    ax.set_ylabel("Recall@10")
    ax.set_title(f"Retrieval Performance by Task Family ({split} split)")
    ax.set_xticks(x + width * (len(baselines) - 1) / 2)
    ax.set_xticklabels(families, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()

    path = output_dir / "recall_by_family.pdf"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "recall_by_family.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  recall_by_family: {path}")


def plot_recall_at_k_curve(scores: dict, split: str, output_dir: Path):
    """Line chart: Recall@k for k=5,10,20 across baselines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k_values = [5, 10, 20]
    baselines = sorted(scores.keys())
    colors = plt.cm.Set1(range(len(baselines)))

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, baseline in enumerate(baselines):
        agg = scores[baseline]["aggregate"]
        vals = [agg.get(f"recall@{k}", 0) for k in k_values]
        ax.plot(k_values, vals, "o-", label=baseline.replace("_", " ").title(), color=colors[i], linewidth=2)

    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_title(f"Recall@k Curves ({split} split)")
    ax.set_xticks(k_values)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = output_dir / "recall_at_k.pdf"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "recall_at_k.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  recall_at_k: {path}")


def plot_difficulty_breakdown(scores: dict, tasks: list[dict], split: str, output_dir: Path):
    """Bar chart: performance by difficulty level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Group tasks by difficulty
    diff_tasks = {}
    for t in tasks:
        d = t.get("difficulty", "unknown")
        if d not in diff_tasks:
            diff_tasks[d] = []
        diff_tasks[d].append(t["task_id"])

    difficulties = sorted(diff_tasks.keys())
    baselines = sorted(scores.keys())

    # We'd need per-task scores for this, so approximate from family data
    # For now, just create a placeholder
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(baselines))
    agg_vals = [scores[b]["aggregate"].get("recall@10", 0) for b in baselines]
    ax.bar(x, agg_vals, color=plt.cm.Set2(range(len(baselines))))
    ax.set_xlabel("Baseline")
    ax.set_ylabel("Recall@10")
    ax.set_title(f"Overall Recall@10 ({split} split)")
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("_", " ").title() for b in baselines], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    plt.tight_layout()

    path = output_dir / "overall_recall.pdf"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "overall_recall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  overall_recall: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
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

    # Load tasks for metadata
    split_path = get_data_dir("benchmark") / f"{args.split}.jsonl"
    tasks = []
    if split_path.exists():
        with open(split_path) as f:
            for line in f:
                tasks.append(json.loads(line))

    output_dir = get_data_dir("") / ".." / "results" / "paper_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures for {args.split} split...")
    plot_recall_by_family(scores, args.split, output_dir)
    plot_recall_at_k_curve(scores, args.split, output_dir)
    if tasks:
        plot_difficulty_breakdown(scores, tasks, args.split, output_dir)

    print(f"\nDone. Figures in {output_dir}")


if __name__ == "__main__":
    main()
