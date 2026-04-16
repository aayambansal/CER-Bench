#!/usr/bin/env python3
"""Generate v2 benchmark with COMPLETE gold labels.

Strategy: cluster-first, question-second.
  1. Pick a seed paper
  2. Use BM25 to find 30-50 candidate related papers
  3. LLM-judge which candidates are genuinely relevant to each other
  4. From the verified cluster (8-15 papers), generate a question
  5. Gold set = the full verified cluster

This eliminates the incomplete gold label problem by construction.

Usage:
    python scripts/30_generate_v2_benchmark.py --num-tasks 200
"""

import argparse
import json
import os
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic


def tok(t):
    return [w for w in re.sub(r"[^a-z0-9\s\-]", " ", t.lower()).split() if len(w) > 1]


def did(cid):
    sts = {"abstract", "other", "methods", "results", "discussion", "introduction",
           "conclusion", "caption", "table", "body", "results_discussion"}
    parts = cid.split("_")
    for i in range(1, len(parts)):
        if "_".join(parts[i:]).split("_")[0] in sts:
            return "_".join(parts[:i])
    return "_".join(parts[:-2]) if len(parts) >= 3 else cid


FAMILY_INSTRUCTIONS = {
    "constraint": "Generate a CONSTRAINT-SATISFACTION question requiring 2-4 experimental constraints. The question should be answerable ONLY by finding papers that satisfy ALL constraints simultaneously.",
    "comparative": "Generate a COMPARATIVE question that requires evidence from BOTH groups of papers in this cluster to answer. A system that finds only one side fails.",
    "contradiction": "Generate a CONTRADICTION question. These papers show different findings. The question should ask about the apparent contradiction and what conditions explain the differences.",
    "abstention": "Generate a question that is PLAUSIBLE and RELATED to this cluster's topic but that NONE of these papers can fully answer. Combine constraints that none satisfy.",
    "multihop": "Generate a MULTI-HOP question where answering requires chaining evidence across at least 3 of these papers. No single paper answers the full question.",
    "temporal": "Generate a TEMPORAL question about how understanding evolved. These papers span different years - the question should require papers from different time periods.",
    "aggregation": "Generate an AGGREGATION question asking to collect ALL reported values or findings across these papers for a specific measurement or outcome.",
    "negative": "Generate a NEGATIVE RESULT question. At least some of these papers report null or negative findings. The question should specifically ask about absence of effect.",
}


def find_topic_cluster(seed_doc, bm25, bm25_ids, docs, chunks_by_id, top_k=40):
    """Find papers related to a seed document using BM25."""
    # Build query from seed's title + abstract
    query = seed_doc.get("title", "") + " " + seed_doc.get("abstract", "")[:500]
    tokens = tok(query)
    scores = bm25.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k * 3]

    # Dedup to doc level
    seen = {seed_doc["doc_id"]}
    cluster_candidates = []
    for idx in top_idx:
        cid = bm25_ids[idx]
        doc_id = did(cid)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        doc = docs.get(doc_id, {})
        if doc.get("abstract") and len(doc["abstract"]) > 100:
            cluster_candidates.append(doc)
        if len(cluster_candidates) >= top_k:
            break

    return cluster_candidates


def verify_cluster(client, seed_doc, candidates, model, min_cluster=6, max_cluster=15):
    """LLM-judge which candidates are genuinely related to the seed topic."""
    seed_info = f"Seed paper: [{seed_doc.get('doc_id','')}] {seed_doc.get('title','')[:150]}\nAbstract: {seed_doc.get('abstract','')[:400]}"

    # Batch candidates for efficiency
    candidate_texts = []
    for i, c in enumerate(candidates[:25]):
        candidate_texts.append(
            f"{i+1}. [{c.get('doc_id','')}] {c.get('title','')[:120]} ({c.get('year','')})\n"
            f"   Abstract: {c.get('abstract','')[:250]}"
        )

    prompt = f"""{seed_info}

Below are candidate papers found by searching for related work. For each, respond with ONLY its number if it is genuinely relevant to the same specific research topic (not just broadly related).

Candidates:
{chr(10).join(candidate_texts)}

List ONLY the numbers of papers that are specifically relevant (same methods, organisms, conditions, or findings). Be selective - only include papers that a scientist would consider part of the same evidence cluster."""

    try:
        resp = client.messages.create(
            model=model, max_tokens=200, temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.content[0].text

        # Parse numbers from response
        numbers = set()
        for match in re.findall(r"\b(\d+)\b", text):
            n = int(match)
            if 1 <= n <= len(candidates):
                numbers.add(n - 1)  # 0-indexed

        verified = [candidates[i] for i in sorted(numbers)]
        return verified[:max_cluster]
    except Exception as e:
        print(f"  Verify error: {e}")
        return candidates[:min_cluster]


def generate_task_from_cluster(client, seed_doc, cluster, family, model, task_id):
    """Generate a task where gold set = the verified cluster."""
    all_papers = [seed_doc] + cluster
    doc_ids = [p.get("doc_id", p.get("pmid", "")) for p in all_papers]

    paper_texts = "\n---\n".join(
        f"[{p.get('doc_id','')}] {p.get('title','')[:120]} ({p.get('year','')})\n"
        f"Abstract: {p.get('abstract','')[:400]}"
        for p in all_papers[:12]  # cap prompt length
    )

    instruction = FAMILY_INSTRUCTIONS[family]

    prompt = f"""You have a cluster of {len(all_papers)} related biomedical papers. {instruction}

Papers in this cluster:
{paper_texts}

Generate ONE task in JSON format:
{{"question": "...", "difficulty": "easy|medium|hard", "required_constraints": [{{"type": "...", "value": "..."}}], "expected_answer_type": "...", "reference_answer": "..."}}

The question must be answerable using these specific papers. Use exact terms from the abstracts."""

    try:
        resp = client.messages.create(
            model=model, max_tokens=800, temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.content[0].text
        s, e = text.find("{"), text.rfind("}") + 1
        if s >= 0 and e > s:
            task = json.loads(text[s:e])
            task.update({
                "task_id": task_id,
                "task_family": family,
                "domain": "biomedicine",
                "generation_method": "cluster_first_v2",
                "verification_status": "cluster_verified",
            })
            if family == "abstention":
                task["supporting_doc_ids"] = []
                task["hard_negative_doc_ids"] = doc_ids
            else:
                task["supporting_doc_ids"] = doc_ids
                task["hard_negative_doc_ids"] = []
                task["supporting_passages"] = [
                    {"doc_id": d, "section": "abstract",
                     "text": p.get("abstract", "")[:300]}
                    for d, p in zip(doc_ids, all_papers)
                ]
            task["cluster_size"] = len(all_papers)
            return task
    except Exception as e:
        print(f"  Generate error: {e}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tasks", type=int, default=200)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--min-cluster", type=int, default=6)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    # Load BM25 index
    print("Loading BM25 index...")
    with open("data/processed/indices/bm25/bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open("data/processed/indices/bm25/chunk_ids.json") as f:
        bm25_ids = json.load(f)

    # Load corpus
    print("Loading corpus...")
    docs = {}
    docs_by_query = defaultdict(list)
    with open("data/processed/corpus.jsonl") as f:
        for line in f:
            doc = json.loads(line)
            docs[doc["doc_id"]] = doc
            if doc.get("abstract") and len(doc["abstract"]) > 100:
                docs_by_query[doc.get("seed_query", "unknown")].append(doc)

    chunks_by_id = {}
    with open("data/processed/chunks.jsonl") as f:
        for line in f:
            c = json.loads(line)
            chunks_by_id[c["chunk_id"]] = c

    print(f"Corpus: {len(docs)} docs, {len(docs_by_query)} groups")

    # Family distribution
    families = list(FAMILY_INSTRUCTIONS.keys())
    tasks_per_family = max(1, args.num_tasks // len(families))

    # Output
    output_dir = Path("data/benchmark/v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tasks_v2.jsonl"

    # Resume
    existing = []
    if output_path.exists():
        existing = [json.loads(l) for l in open(output_path)]
    existing_fams = defaultdict(int)
    for t in existing:
        existing_fams[t["task_family"]] += 1
    counter = len(existing)
    all_tasks = list(existing)

    print(f"Resuming from {len(existing)} tasks")
    print(f"Target: {args.num_tasks} tasks ({tasks_per_family} per family)")

    t0 = time.time()
    all_queries = list(docs_by_query.keys())

    for family in families:
        needed = tasks_per_family - existing_fams.get(family, 0)
        if needed <= 0:
            print(f"{family}: done ({existing_fams.get(family, 0)})")
            continue

        print(f"\n--- {family}: generating {needed} ---")
        got = 0
        attempts = 0

        while got < needed and attempts < needed * 5:
            attempts += 1

            # Pick random seed paper
            query_group = random.choice(all_queries)
            seed = random.choice(docs_by_query[query_group])

            # Find cluster
            candidates = find_topic_cluster(seed, bm25, bm25_ids, docs, chunks_by_id)
            if len(candidates) < args.min_cluster:
                continue

            # Verify cluster
            verified = verify_cluster(client, seed, candidates, args.model,
                                      min_cluster=args.min_cluster)
            if len(verified) < args.min_cluster - 1:
                continue

            # Generate task
            counter += 1
            task_id = f"v2_{family}_{counter:04d}"
            task = generate_task_from_cluster(client, seed, verified, family,
                                              args.model, task_id)
            if task:
                all_tasks.append(task)
                got += 1

                # Save incrementally
                if got % 3 == 0 or got == needed:
                    with open(output_path, "w") as f:
                        for t in all_tasks:
                            f.write(json.dumps(t, default=str) + "\n")
                    elapsed = time.time() - t0
                    cluster_size = task.get("cluster_size", 0)
                    print(f"  [{family}] {got}/{needed} "
                          f"cluster={cluster_size} "
                          f"total={len(all_tasks)} "
                          f"{elapsed:.0f}s")
                    sys.stdout.flush()

            time.sleep(0.2)

        # Final save
        with open(output_path, "w") as f:
            for t in all_tasks:
                f.write(json.dumps(t, default=str) + "\n")

    # Stats
    elapsed = time.time() - t0
    fam_counts = defaultdict(int)
    cluster_sizes = []
    for t in all_tasks:
        fam_counts[t["task_family"]] += 1
        cluster_sizes.append(t.get("cluster_size", 0))

    print(f"\n{'='*60}")
    print(f"V2 BENCHMARK GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total tasks: {len(all_tasks)}")
    print(f"Avg cluster size: {sum(cluster_sizes)/len(cluster_sizes):.1f} papers/task")
    print(f"Time: {elapsed:.0f}s")
    for f, c in sorted(fam_counts.items()):
        print(f"  {f}: {c}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
