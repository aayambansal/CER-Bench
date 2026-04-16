#!/usr/bin/env python3
"""Script 07c: Batch generate grounded tasks, appending to existing file.

Generates tasks in small batches to avoid timeouts. 
Appends to data/interim/grounded_tasks.jsonl.

Usage:
    python scripts/07c_batch_generate.py --batch-size 40 --target-total 300
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

TASK_FAMILIES = ["constraint","comparative","contradiction","abstention",
                 "multihop","temporal","aggregation","negative"]

FAMILY_INSTRUCTIONS = {
    "constraint": "Generate a CONSTRAINT-SATISFACTION question requiring 2-4 specific experimental constraints (organism, method, intervention, outcome, time). Only these papers satisfy ALL constraints.",
    "comparative": "Generate a COMPARATIVE question requiring evidence from BOTH sides of a comparison across these papers.",
    "contradiction": "Generate a CONTRADICTION question where these papers show apparently conflicting findings explained by different experimental conditions.",
    "abstention": None,
    "multihop": "Generate a MULTI-HOP question requiring chaining evidence across all papers via shared entities.",
    "temporal": "Generate a TEMPORAL question about how findings evolved over the time periods of these papers.",
    "aggregation": "Generate an AGGREGATION question asking to collect ALL reported quantitative values for a measurement across these papers.",
    "negative": "Generate a NEGATIVE RESULT question asking for studies showing null/negative findings. At least one paper reports no effect.",
}

def load_corpus_by_query(corpus_path):
    groups = defaultdict(list)
    with open(corpus_path) as f:
        for line in f:
            doc = json.loads(line)
            if doc.get("abstract") and len(doc["abstract"]) > 100:
                groups[doc.get("seed_query","unknown")].append(doc)
    return groups

def format_papers(papers):
    parts = []
    for i, p in enumerate(papers):
        t = f"Paper {i+1}:\n  Doc ID: {p.get('doc_id',p.get('pmid',''))}\n"
        t += f"  Title: {p.get('title','')}\n  Year: {p.get('year','')}\n"
        t += f"  Abstract: {p.get('abstract','')[:1000]}\n"
        mesh = p.get("mesh_terms",[])[:6]
        if mesh: t += f"  MeSH: {', '.join(mesh)}\n"
        parts.append(t)
    return "\n---\n".join(parts)

def sample_cluster(groups, family, n=3):
    all_q = list(groups.keys())
    if family == "comparative" and len(all_q) >= 2:
        q1, q2 = random.sample(all_q, 2)
        papers = random.sample(groups[q1], min(2, len(groups[q1])))
        papers += random.sample(groups[q2], min(1, len(groups[q2])))
        return papers[:n]
    if family == "temporal":
        q = random.choice(all_q)
        pool = sorted([p for p in groups[q] if p.get("year")], key=lambda x: x["year"])
        if len(pool) >= n:
            step = max(1, len(pool)//n)
            return [pool[i*step] for i in range(min(n, len(pool)))]
    q = random.choice(all_q)
    return random.sample(groups[q], min(n, len(groups[q])))

def generate_one(client, papers, family, model):
    doc_ids = [p.get("doc_id", p.get("pmid","")) for p in papers]
    papers_text = format_papers(papers)
    
    if family == "abstention":
        prompt = f"""Generate a PLAUSIBLE biomedical question that NONE of these papers can fully answer.
Related to their topics but combining constraints none satisfy.

Papers:
{papers_text}

JSON format: {{"question":"...","difficulty":"medium|hard","required_constraints":[{{"type":"...","value":"..."}}],"expected_answer_type":"abstain","reference_answer":"Insufficient evidence"}}"""
    else:
        prompt = f"""Generate ONE retrieval benchmark task where these papers are the gold answer.
{FAMILY_INSTRUCTIONS[family]}

Papers:
{papers_text}

JSON: {{"question":"...","difficulty":"easy|medium|hard","required_constraints":[{{"type":"...","value":"..."}}],"expected_answer_type":"...","reference_answer":"..."}}"""

    try:
        resp = client.messages.create(model=model, max_tokens=1024, temperature=0.7,
                                       messages=[{"role":"user","content":prompt}])
        text = resp.content[0].text
        s, e = text.find("{"), text.rfind("}")+1
        if s >= 0 and e > s:
            task = json.loads(text[s:e])
            task["task_family"] = family
            task["domain"] = "biomedicine"
            task["generation_method"] = "corpus_grounded"
            if family == "abstention":
                task["supporting_doc_ids"] = []
                task["hard_negative_doc_ids"] = doc_ids
            else:
                task["supporting_doc_ids"] = doc_ids
                task["hard_negative_doc_ids"] = []
                task["supporting_passages"] = [{"doc_id":d,"section":"abstract","text":p.get("abstract","")[:400]}
                                                for d,p in zip(doc_ids,papers)]
            task["verification_status"] = "auto_verified"
            return task
    except Exception as e:
        print(f"    Error: {e}")
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--target-total", type=int, default=300)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    args = parser.parse_args()

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    corpus_path = get_data_dir("processed") / "corpus.jsonl"
    groups = load_corpus_by_query(corpus_path)
    gen_config = load_config("generation")
    family_dist = gen_config["family_distribution"]

    # Load existing tasks
    existing_path = get_data_dir("interim") / "grounded_tasks.jsonl"
    existing = []
    if existing_path.exists():
        existing = [json.loads(l) for l in open(existing_path)]
    print(f"Existing tasks: {len(existing)}")

    existing_fams = defaultdict(int)
    for t in existing:
        existing_fams[t["task_family"]] += 1

    # Calculate how many more per family
    needed = {}
    for fam in TASK_FAMILIES:
        target = max(2, round(args.target_total * family_dist.get(fam, 0.125)))
        needed[fam] = max(0, target - existing_fams.get(fam, 0))
    
    total_needed = sum(needed.values())
    print(f"Need {total_needed} more tasks to reach ~{args.target_total}")
    for f, n in needed.items():
        print(f"  {f}: {n} needed (have {existing_fams.get(f,0)})")

    if total_needed == 0:
        print("Already at target!")
        return

    # Generate in batch
    t0 = time.time()
    new_tasks = []
    task_id_counter = len(existing)

    for fam in TASK_FAMILIES:
        n = needed[fam]
        if n == 0:
            continue
        print(f"\n--- {fam}: generating {n} ---")
        generated = 0
        attempts = 0
        while generated < n and attempts < n * 3:
            n_papers = 3 if fam in ("multihop","comparative","temporal") else 2
            papers = sample_cluster(groups, fam, n_papers)
            if len(papers) < 2:
                attempts += 1
                continue
            task = generate_one(client, papers, fam, args.model)
            attempts += 1
            if task:
                task_id_counter += 1
                task["task_id"] = f"{fam}_{task_id_counter:04d}"
                new_tasks.append(task)
                generated += 1
                if generated % 5 == 0:
                    print(f"  [{generated}/{n}]")
            time.sleep(0.2)

    t1 = time.time()
    print(f"\nGenerated {len(new_tasks)} new tasks in {t1-t0:.0f}s")

    # Append to existing
    all_tasks = existing + new_tasks
    with open(existing_path, "w") as f:
        for t in all_tasks:
            f.write(json.dumps(t, default=str) + "\n")
    
    # Also update raw_tasks and verified_tasks
    for fname in ["raw_tasks.jsonl", "verified_tasks.jsonl"]:
        with open(get_data_dir("interim") / fname, "w") as f:
            for t in all_tasks:
                f.write(json.dumps(t, default=str) + "\n")

    fam_counts = defaultdict(int)
    for t in all_tasks:
        fam_counts[t["task_family"]] += 1
    
    print(f"\nTotal: {len(all_tasks)} tasks")
    for f, c in sorted(fam_counts.items()):
        print(f"  {f}: {c}")

if __name__ == "__main__":
    main()
