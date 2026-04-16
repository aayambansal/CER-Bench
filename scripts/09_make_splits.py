#!/usr/bin/env python3
"""Script 09: Split verified tasks into train/dev/test.

Stratified by task_family and difficulty.

Usage:
    python scripts/09_make_splits.py
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import get_data_dir, load_config


def main():
    config = load_config("generation")
    split_ratios = config["splits"]
    seed = split_ratios.get("random_seed", 42)
    random.seed(seed)

    # Load verified tasks
    tasks_path = get_data_dir("interim") / "verified_tasks.jsonl"
    if not tasks_path.exists():
        print(f"Error: {tasks_path} not found. Run scripts/08_verify_tasks.py first.")
        sys.exit(1)

    tasks = []
    with open(tasks_path) as f:
        for line in f:
            tasks.append(json.loads(line))

    print(f"Loaded {len(tasks)} verified tasks")

    # Stratify by family + difficulty
    strata = defaultdict(list)
    for task in tasks:
        key = (task.get("task_family", "unknown"), task.get("difficulty", "unknown"))
        strata[key].append(task)

    train_ratio = split_ratios["train"]
    dev_ratio = split_ratios["dev"]
    # test gets the rest

    train, dev, test = [], [], []

    for key, group in strata.items():
        random.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * train_ratio)) if n >= 3 else 0
        n_dev = max(1, round(n * dev_ratio)) if n >= 3 else 0

        # Ensure at least 1 in test if possible
        if n_train + n_dev >= n and n >= 2:
            n_train = max(0, n_train - 1)

        for i, task in enumerate(group):
            if i < n_train:
                task["split"] = "train"
                train.append(task)
            elif i < n_train + n_dev:
                task["split"] = "dev"
                dev.append(task)
            else:
                task["split"] = "test"
                test.append(task)

    # Save splits
    benchmark_dir = get_data_dir("benchmark")

    for split_name, split_data in [("train", train), ("dev", dev), ("test", test)]:
        path = benchmark_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for task in split_data:
                f.write(json.dumps(task, default=str) + "\n")

    # Save schema
    schema = {
        "dataset_name": "SynthSearch-Biomed",
        "version": "0.1.0-pilot",
        "task_families": sorted(set(t.get("task_family", "") for t in tasks)),
        "splits": {"train": len(train), "dev": len(dev), "test": len(test)},
        "total": len(tasks),
    }
    with open(benchmark_dir / "schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"  Train: {len(train)} ({100*len(train)/len(tasks):.0f}%)")
    print(f"  Dev: {len(dev)} ({100*len(dev)/len(tasks):.0f}%)")
    print(f"  Test: {len(test)} ({100*len(test)/len(tasks):.0f}%)")

    # Per-family breakdown
    for split_name, split_data in [("train", train), ("dev", dev), ("test", test)]:
        fam_counts = defaultdict(int)
        for t in split_data:
            fam_counts[t.get("task_family", "unknown")] += 1
        print(f"\n  {split_name}:")
        for fam, count in sorted(fam_counts.items()):
            print(f"    {count:>4}  {fam}")

    print(f"\n  Output: {benchmark_dir}")


if __name__ == "__main__":
    main()
