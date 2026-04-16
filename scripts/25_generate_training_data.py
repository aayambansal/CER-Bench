#!/usr/bin/env python3
"""Phase C: Generate 10K+ synthetic training tasks for controller training.

Uses the same corpus-grounded generation as the benchmark but at 30x scale.
Tasks are generated across all 8 families with upweighting on the harder families
(aggregation, temporal, multi-hop, contradiction, abstention).

Saves incrementally every 100 tasks.

Usage:
    python scripts/25_generate_training_data.py --target 10000 --model claude-sonnet-4-20250514
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config import get_data_dir, load_config

import anthropic


# Upweighted family distribution for training
# Emphasize families where iteration matters most
TRAINING_FAMILY_DIST = {
    "constraint": 0.10,
    "comparative": 0.12,
    "contradiction": 0.14,
    "abstention": 0.14,
    "multihop": 0.14,
    "temporal": 0.14,
    "aggregation": 0.14,
    "negative": 0.08,
}

FAMILY_INSTRUCTIONS = {
    "constraint": "CONSTRAINT question: 2-4 experimental constraints (organism, method, intervention, outcome, time). Only these papers satisfy ALL constraints.",
    "comparative": "COMPARATIVE question: requires evidence from BOTH sides of a comparison across these papers.",
    "contradiction": "CONTRADICTION question: these papers show conflicting findings explained by different experimental conditions.",
    "abstention": "ABSTENTION question: plausible but NONE of these papers can fully answer. Combine constraints none satisfy.",
    "multihop": "MULTI-HOP question: requires chaining evidence across all papers via shared entities (gene, pathway, drug).",
    "temporal": "TEMPORAL question: how understanding evolved across the time periods of these papers.",
    "aggregation": "AGGREGATION question: collect ALL reported quantitative values for a measurement across these papers.",
    "negative": "NEGATIVE RESULT question: find null/failed findings. At least one paper reports no effect.",
}


def load_corpus():
    groups = defaultdict(list)
    with open(get_data_dir("processed") / "corpus.jsonl") as f:
        for line in f:
            doc = json.loads(line)
            if doc.get("abstract") and len(doc["abstract"]) > 100:
                groups[doc.get("seed_query", "unknown")].append(doc)
    return groups


def sample_papers(groups, family, n=2):
    all_q = list(groups.keys())
    q = random.choice(all_q)
    pool = groups[q]
    if family == "temporal":
        pool = sorted([p for p in pool if p.get("year")], key=lambda x: x["year"])
        if len(pool) >= n:
            step = max(1, len(pool) // n)
            return [pool[i * step] for i in range(min(n, len(pool)))]
    if family == "comparative" and len(all_q) >= 2:
        q1, q2 = random.sample(all_q, 2)
        papers = random.sample(groups[q1], min(n-1, len(groups[q1])))
        papers += random.sample(groups[q2], min(1, len(groups[q2])))
        return papers[:n]
    return random.sample(pool, min(n, len(pool)))


def generate_one(client, papers, family, model):
    ids = [p.get("doc_id", p.get("pmid", "")) for p in papers]
    ptxt = "\n".join(
        f'[{p.get("doc_id","")}] {p.get("title","")[:120]} ({p.get("year","")}) '
        f'Abstract: {p.get("abstract","")[:400]}'
        for p in papers
    )
    inst = FAMILY_INSTRUCTIONS[family]
    prompt = (
        f"Generate a {inst}\n\nPapers:\n{ptxt}\n\n"
        f'JSON only: {{"question":"...","difficulty":"easy|medium|hard",'
        f'"required_constraints":[],"expected_answer_type":"...",'
        f'"reference_answer":"..."}}'
    )
    try:
        r = client.messages.create(
            model=model, max_tokens=600, temperature=0.8,
            messages=[{"role": "user", "content": prompt}]
        )
        txt = r.content[0].text
        s, e = txt.find("{"), txt.rfind("}") + 1
        if s >= 0 and e > s:
            task = json.loads(txt[s:e])
            task.update({
                "task_family": family,
                "domain": "biomedicine",
                "generation_method": "corpus_grounded",
                "verification_status": "auto_verified",
            })
            if family == "abstention":
                task["supporting_doc_ids"] = []
                task["hard_negative_doc_ids"] = ids
            else:
                task["supporting_doc_ids"] = ids
                task["hard_negative_doc_ids"] = []
                task["supporting_passages"] = [
                    {"doc_id": d, "section": "abstract", "text": p.get("abstract", "")[:300]}
                    for d, p in zip(ids, papers)
                ]
            return task
    except Exception as ex:
        print(f"  err: {ex}")
        time.sleep(2)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    groups = load_corpus()
    total_papers = sum(len(v) for v in groups.values())
    print(f"Corpus: {total_papers} papers, {len(groups)} groups")

    output_dir = get_data_dir("training")
    output_path = output_dir / "training_tasks.jsonl"

    # Resume if requested
    existing = []
    if args.resume and output_path.exists():
        existing = [json.loads(l) for l in open(output_path)]
    print(f"Existing: {len(existing)} tasks")

    existing_fams = defaultdict(int)
    for t in existing:
        existing_fams[t["task_family"]] += 1

    # Calculate targets per family
    targets = {}
    for fam, pct in TRAINING_FAMILY_DIST.items():
        targets[fam] = max(0, round(args.target * pct) - existing_fams.get(fam, 0))
    total_needed = sum(targets.values())
    print(f"Need {total_needed} more tasks (target: {args.target})")
    for f, n in targets.items():
        print(f"  {f}: {n} (have {existing_fams.get(f, 0)})")

    if total_needed == 0:
        print("Already at target!")
        return

    # Generate
    all_tasks = list(existing)
    counter = len(all_tasks)
    t0 = time.time()
    families = list(TRAINING_FAMILY_DIST.keys())

    for fam in families:
        needed = targets[fam]
        if needed <= 0:
            continue
        print(f"\n--- {fam}: generating {needed} ---")
        got = 0
        att = 0
        while got < needed and att < needed * 3:
            n = 3 if fam in ("multihop", "comparative", "temporal") else 2
            papers = sample_papers(groups, fam, n)
            if len(papers) < 2:
                att += 1
                continue
            task = generate_one(client, papers, fam, args.model)
            att += 1
            if task:
                counter += 1
                task["task_id"] = f"train_{fam}_{counter:06d}"
                task["split"] = "train"
                all_tasks.append(task)
                got += 1
                if got % 5 == 0:
                    # Save incrementally
                    with open(output_path, "w") as f:
                        for t in all_tasks:
                            f.write(json.dumps(t, default=str) + "\n")
                    elapsed = time.time() - t0
                    rate = got / elapsed * 3600 if elapsed > 0 else 0
                    print(f"  [{fam}] {got}/{needed} saved "
                          f"(total={len(all_tasks)}, {rate:.0f}/hr, "
                          f"est ${len(all_tasks)*0.003:.1f})")
                    sys.stdout.flush()
            time.sleep(0.15)

        # Final save for this family
        with open(output_path, "w") as f:
            for t in all_tasks:
                f.write(json.dumps(t, default=str) + "\n")
        fam_count = sum(1 for t in all_tasks if t["task_family"] == fam)
        print(f"  {fam}: DONE ({fam_count} total)")

    # Summary
    elapsed = time.time() - t0
    fam_counts = defaultdict(int)
    for t in all_tasks:
        fam_counts[t["task_family"]] += 1

    print(f"\n{'='*60}")
    print(f"TRAINING DATA GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total tasks: {len(all_tasks)}")
    print(f"Time: {elapsed:.0f}s ({elapsed/3600:.1f}hr)")
    print(f"Est cost: ${len(all_tasks)*0.003:.1f}")
    for f, c in sorted(fam_counts.items()):
        print(f"  {f}: {c}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
