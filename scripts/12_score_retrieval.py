#!/usr/bin/env python3
"""Script 12: Score retrieval results — compute Recall@k, nDCG@k, MRR, etc.

Usage:
    python scripts/12_score_retrieval.py [--split dev]
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import get_data_dir


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    retrieved_set = set(retrieved[:k])
    return len(retrieved_set & set(gold)) / len(gold)


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k]):
        if doc in gold:
            dcg += 1.0 / math.log2(i + 2)
    # Ideal DCG
    n_relevant = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(retrieved: list[str], gold: set[str]) -> float:
    for i, doc in enumerate(retrieved):
        if doc in gold:
            return 1.0 / (i + 1)
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Score retrieval results")
    parser.add_argument("--split", type=str, default="dev")
    args = parser.parse_args()

    # Load benchmark tasks (for gold labels)
    split_path = get_data_dir("benchmark") / f"{args.split}.jsonl"
    if not split_path.exists():
        print(f"Error: {split_path} not found.")
        sys.exit(1)

    tasks = {}
    with open(split_path) as f:
        for line in f:
            task = json.loads(line)
            tasks[task["task_id"]] = task

    # Find all baseline result files
    results_dir = Path("results/baselines")
    if not results_dir.exists():
        results_dir = get_data_dir("") / ".." / "results" / "baselines"

    result_files = list(results_dir.glob(f"*_{args.split}.jsonl"))
    if not result_files:
        print(f"No result files found for split '{args.split}' in {results_dir}")
        sys.exit(1)

    k_values = [5, 10, 20]
    all_scores = {}

    for result_file in sorted(result_files):
        baseline = result_file.stem.replace(f"_{args.split}", "")
        print(f"\nScoring: {baseline}")

        results = []
        with open(result_file) as f:
            for line in f:
                results.append(json.loads(line))

        # Compute metrics
        metrics = defaultdict(list)
        family_metrics = defaultdict(lambda: defaultdict(list))

        for res in results:
            task_id = res["task_id"]
            task = tasks.get(task_id, {})
            gold_docs = set(task.get("supporting_doc_ids", []))
            retrieved = res.get("retrieved_docs", [])
            family = task.get("task_family", "unknown")

            for k in k_values:
                r = recall_at_k(retrieved, list(gold_docs), k)
                metrics[f"recall@{k}"].append(r)
                family_metrics[family][f"recall@{k}"].append(r)

            n = ndcg_at_k(retrieved, gold_docs, 10)
            metrics["ndcg@10"].append(n)
            family_metrics[family]["ndcg@10"].append(n)

            m = mrr(retrieved, gold_docs)
            metrics["mrr"].append(m)
            family_metrics[family]["mrr"].append(m)

        # Aggregate
        agg = {}
        for metric, values in metrics.items():
            agg[metric] = round(sum(values) / len(values), 4) if values else 0

        all_scores[baseline] = {
            "aggregate": agg,
            "per_family": {},
            "n_tasks": len(results),
        }

        for family, fam_metrics in family_metrics.items():
            fam_agg = {}
            for metric, values in fam_metrics.items():
                fam_agg[metric] = round(sum(values) / len(values), 4) if values else 0
            all_scores[baseline]["per_family"][family] = fam_agg

    # Save scores
    scores_path = results_dir / f"scores_{args.split}.json"
    with open(scores_path, "w") as f:
        json.dump(all_scores, f, indent=2)

    # Print table
    print(f"\n{'='*70}")
    print(f"RETRIEVAL SCORES ({args.split} split)")
    print(f"{'='*70}")
    header = f"{'Baseline':<20}"
    for k in k_values:
        header += f"{'R@'+str(k):<10}"
    header += f"{'nDCG@10':<10}{'MRR':<10}"
    print(header)
    print("-" * 70)

    for baseline, data in sorted(all_scores.items()):
        agg = data["aggregate"]
        row = f"{baseline:<20}"
        for k in k_values:
            row += f"{agg.get(f'recall@{k}', 0):<10.4f}"
        row += f"{agg.get('ndcg@10', 0):<10.4f}{agg.get('mrr', 0):<10.4f}"
        print(row)

    # Per-family breakdown for best baseline
    if all_scores:
        best = max(all_scores.items(), key=lambda x: x[1]["aggregate"].get("recall@10", 0))
        print(f"\nPer-family breakdown ({best[0]}):")
        for family, fam_agg in sorted(best[1]["per_family"].items()):
            r10 = fam_agg.get("recall@10", 0)
            print(f"  {family:<20} R@10={r10:.4f}")

    print(f"\nScores saved to {scores_path}")


if __name__ == "__main__":
    main()
