#!/usr/bin/env python3
"""Offline request plans only. Provider execution deliberately unavailable."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.budget_protocol import CONDITIONS, dry_plan, manifest, validate_resume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--index-file", type=Path, action="append", required=True)
    parser.add_argument("--model", default="openai/gpt-5.4-mini")
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--repeat", required=True)
    parser.add_argument("--condition", action="append", choices=CONDITIONS)
    parser.add_argument("--output", type=Path, default=ROOT / "results/readiness/budget/dry_run.jsonl")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/equal_budget_protocol.json")
    parser.add_argument("--dry-run", action="store_true", help="Default; no provider is constructed")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.live:
        parser.error("Live execution disabled: offline-only implementation; use an independently reviewed adapter")
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    tids = [t["task_id"] for t in tasks]
    if len(tids) != len(set(tids)):
        parser.error("Duplicate task identities")
    conditions = args.condition or list(CONDITIONS)
    if len(conditions) != len(set(conditions)):
        parser.error("Duplicate conditions")
    config = json.loads(args.config.read_text())
    config.update(conditions=conditions, mode="dry_run")
    bound = manifest(tasks=args.tasks, split=args.split, corpus=args.corpus,
                     index_files=args.index_file, source_files=[Path(__file__), ROOT / "src/evaluation/budget_protocol.py", args.config],
                     model=args.model, provider=args.provider, repeat=args.repeat,
                     config=config, task_ids=tids)
    sidecar = args.output.with_suffix(args.output.suffix + ".manifest.json")
    if args.resume:
        old = json.loads(sidecar.read_text())
        rows = [json.loads(s) for s in args.output.read_text().splitlines() if s.strip()]
        validate_resume(old, bound, rows)  # Dry plans are not completed work.
        parser.error("Dry-run resume is unnecessary; choose a fresh output")
    if args.output.exists() or sidecar.exists():
        parser.error("Output exists; refusing to overwrite existing work")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("x") as handle:
        json.dump(bound, handle, indent=2)
    with args.output.open("x") as handle:
        for task in tasks:
            for condition in conditions:
                row = dry_plan(task, condition, args.model, args.provider)
                row["provenance_id"] = bound["provenance_id"]
                handle.write(json.dumps(row) + "\n")
    print(f"Wrote {len(tasks) * len(conditions)} request plans; zero searches/model calls. {args.output}")


if __name__ == "__main__":
    main()
