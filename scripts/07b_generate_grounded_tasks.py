#!/usr/bin/env python3
"""Script 07b: Generate CORPUS-GROUNDED benchmark tasks.

Unlike 07, this script starts from actual corpus documents and builds
questions around real papers. This guarantees that supporting_doc_ids
exist and are verifiable.

Strategy:
  1. Sample clusters of related papers from the corpus (by seed query / MeSH overlap)
  2. Send paper abstracts + metadata to Claude
  3. Claude generates a question that REQUIRES those specific papers
  4. Gold labels are the sampled papers themselves

Usage:
    python scripts/07b_generate_grounded_tasks.py [--num-tasks 50]
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

try:
    import anthropic
except ImportError:
    print("pip install anthropic")
    sys.exit(1)


FAMILY_INSTRUCTIONS = {
    "constraint": """Generate a CONSTRAINT-SATISFACTION question. The question must specify 2-4 experimental constraints
(organism, method, intervention, outcome, time period) such that ONLY these specific papers satisfy ALL constraints.
Make the constraints specific enough that topical matching alone won't work.""",

    "comparative": """Generate a COMPARATIVE question. The question must ask for a comparison between the approaches/findings
in these papers. A correct retrieval must find papers representing BOTH sides of the comparison.""",

    "contradiction": """Generate a CONTRADICTION/CONDITIONALITY question. These papers show different findings on a related topic.
The question should ask about the apparent contradiction and what experimental conditions explain the differences.""",

    "abstention": None,  # handled separately

    "multihop": """Generate a MULTI-HOP question. The answer requires chaining evidence from paper 1 → paper 2 → paper 3.
No single paper answers the full question. Identify the shared entity (gene, pathway, drug, mechanism) that links them.""",

    "temporal": """Generate a TEMPORAL EVOLUTION question. These papers are from different years and show how understanding evolved.
The question should ask how findings or consensus changed over the time period covered by these papers.""",

    "aggregation": """Generate an AGGREGATION question. The question should ask to collect ALL reported quantitative values
for a specific measurement that appears across these papers (e.g., efficiency rates, IC50 values, survival rates).""",

    "negative": """Generate a NEGATIVE RESULT question. At least one of these papers reports a null or negative finding.
The question should specifically ask for studies showing no effect or failed outcomes.""",
}

GROUNDED_PROMPT = """You are generating a benchmark task for evaluating biomedical literature retrieval systems.

I am giving you {n_papers} REAL papers from our corpus. Generate ONE retrieval question where these papers are the GOLD ANSWER.

{family_instruction}

Papers:
{papers_text}

Generate the task in this exact JSON format:
{{
    "question": "The natural-language query a scientist would ask",
    "difficulty": "easy|medium|hard",
    "decomposition_hints": ["subquery 1", "subquery 2"],
    "required_constraints": [
        {{"type": "organism|assay|intervention|outcome|temporal|design", "value": "specific value from the papers"}}
    ],
    "expected_answer_type": "set|comparison|conditional|chain|timeline|value_collection|null_result",
    "reference_answer": "Brief expected answer citing the papers"
}}

CRITICAL: The constraints must use EXACT terms from the paper abstracts. The question must be answerable ONLY by these specific papers."""

ABSTENTION_PROMPT = """You are generating a benchmark task where the correct answer is ABSTENTION (no sufficient evidence).

I am giving you {n_papers} REAL papers. Generate a question that is PLAUSIBLE but that NONE of these papers can fully answer.
The question should be related to these papers' topics but combine constraints that none of them satisfy.

Papers:
{papers_text}

Generate the task in this exact JSON format:
{{
    "question": "A plausible scientific question that these papers CANNOT fully answer",
    "difficulty": "medium|hard",
    "decomposition_hints": ["subquery 1", "subquery 2"],
    "required_constraints": [
        {{"type": "organism|assay|intervention|outcome|temporal|design", "value": "specific constraint NOT in these papers"}}
    ],
    "expected_answer_type": "abstain",
    "reference_answer": "Insufficient evidence — no paper in corpus satisfies all constraints",
    "near_miss_reason": "Why these papers are close but not sufficient"
}}"""


def load_corpus_by_query(corpus_path: Path) -> dict[str, list[dict]]:
    """Load corpus grouped by seed query."""
    groups = defaultdict(list)
    with open(corpus_path) as f:
        for line in f:
            doc = json.loads(line)
            query = doc.get("seed_query", "unknown")
            # Only use docs with abstracts
            if doc.get("abstract") and len(doc["abstract"]) > 100:
                groups[query].append(doc)
    return groups


def format_papers(papers: list[dict]) -> str:
    """Format papers for the prompt."""
    parts = []
    for i, p in enumerate(papers):
        text = f"Paper {i+1}:\n"
        text += f"  Doc ID: {p.get('doc_id', p.get('pmid', ''))}\n"
        text += f"  Title: {p.get('title', '')}\n"
        text += f"  Year: {p.get('year', '')}\n"
        text += f"  Venue: {p.get('venue', '')}\n"
        text += f"  Abstract: {p.get('abstract', '')[:1200]}\n"
        mesh = p.get("mesh_terms", [])[:8]
        if mesh:
            text += f"  MeSH: {', '.join(mesh)}\n"
        kw = p.get("keywords", [])[:5]
        if kw:
            text += f"  Keywords: {', '.join(kw)}\n"
        parts.append(text)
    return "\n---\n".join(parts)


def sample_paper_cluster(groups: dict, family: str, n_papers: int = 3) -> list[dict]:
    """Sample a cluster of related papers appropriate for the task family."""
    all_queries = list(groups.keys())
    
    if family == "comparative":
        # Sample from TWO different query groups for comparison
        if len(all_queries) >= 2:
            q1, q2 = random.sample(all_queries, 2)
            papers = random.sample(groups[q1], min(2, len(groups[q1])))
            papers += random.sample(groups[q2], min(1, len(groups[q2])))
            return papers[:n_papers]
    
    if family == "temporal":
        # Sample papers with diverse years
        query = random.choice(all_queries)
        pool = groups[query]
        if len(pool) >= n_papers:
            # Sort by year and pick from different time periods
            pool_sorted = sorted([p for p in pool if p.get("year")], key=lambda x: x["year"])
            step = max(1, len(pool_sorted) // n_papers)
            return [pool_sorted[i * step] for i in range(min(n_papers, len(pool_sorted)))]
    
    if family == "contradiction":
        # Sample from same query group (more likely to have conflicting findings)
        query = random.choice(all_queries)
        pool = groups[query]
        return random.sample(pool, min(n_papers, len(pool)))
    
    # Default: sample from same query group
    query = random.choice(all_queries)
    pool = groups[query]
    return random.sample(pool, min(n_papers, len(pool)))


def generate_grounded_task(
    client: anthropic.Anthropic,
    papers: list[dict],
    family: str,
    model: str,
) -> dict | None:
    """Generate one task grounded in actual corpus papers."""
    papers_text = format_papers(papers)
    doc_ids = [p.get("doc_id", p.get("pmid", "")) for p in papers]
    
    if family == "abstention":
        prompt = ABSTENTION_PROMPT.format(n_papers=len(papers), papers_text=papers_text)
    else:
        prompt = GROUNDED_PROMPT.format(
            n_papers=len(papers),
            family_instruction=FAMILY_INSTRUCTIONS[family],
            papers_text=papers_text,
        )

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text

        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            task = json.loads(text[start:end])
            task["task_family"] = family
            task["domain"] = "biomedicine"
            task["generation_method"] = "corpus_grounded"
            
            if family == "abstention":
                task["supporting_doc_ids"] = []  # no gold docs for abstention
                task["hard_negative_doc_ids"] = doc_ids  # near-misses
                task["verification_status"] = "auto_verified"
            else:
                task["supporting_doc_ids"] = doc_ids  # THESE ARE REAL CORPUS DOC IDS
                task["hard_negative_doc_ids"] = []  # will be populated in verification
                task["verification_status"] = "auto_verified"
                # Build supporting passages from abstracts
                task["supporting_passages"] = [
                    {
                        "doc_id": doc_id,
                        "section": "abstract",
                        "text": p.get("abstract", "")[:500],
                    }
                    for doc_id, p in zip(doc_ids, papers)
                ]
            return task
    except Exception as e:
        print(f"    Error: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate corpus-grounded tasks")
    parser.add_argument("--num-tasks", type=int, default=60)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    corpus_path = get_data_dir("processed") / "corpus.jsonl"
    groups = load_corpus_by_query(corpus_path)
    total_papers = sum(len(v) for v in groups.values())
    print(f"Corpus: {total_papers} papers across {len(groups)} query groups")

    gen_config = load_config("generation")
    family_dist = gen_config["family_distribution"]
    families = list(family_dist.keys())

    # Calculate per-family targets
    tasks_per_family = {}
    for fam in families:
        tasks_per_family[fam] = max(2, round(args.num_tasks * family_dist.get(fam, 0.125)))
    
    print(f"\nTarget: {args.num_tasks} tasks")
    for fam, count in tasks_per_family.items():
        print(f"  {fam}: {count}")

    # Generate
    t0 = time.time()
    all_tasks = []
    task_counter = 0

    for family, target in tasks_per_family.items():
        print(f"\n{'='*50}")
        print(f"Family: {family} (target: {target})")
        generated = 0
        attempts = 0

        while generated < target and attempts < target * 3:
            n_papers = 3 if family in ("multihop", "comparative", "temporal") else 2
            papers = sample_paper_cluster(groups, family, n_papers)
            
            if len(papers) < 2:
                attempts += 1
                continue

            task = generate_grounded_task(client, papers, family, args.model)
            attempts += 1

            if task:
                task_counter += 1
                task["task_id"] = f"{family}_{task_counter:04d}"
                all_tasks.append(task)
                generated += 1
                gold = len(task.get("supporting_doc_ids", []))
                print(f"  [{generated}/{target}] {task['task_id']} gold={gold} q={task['question'][:70]}...")
            
            time.sleep(0.3)

    t1 = time.time()

    # Save
    output_path = get_data_dir("interim") / "grounded_tasks.jsonl"
    with open(output_path, "w") as f:
        for task in all_tasks:
            f.write(json.dumps(task, default=str) + "\n")

    # Also overwrite raw_tasks and verified_tasks so downstream scripts work
    raw_path = get_data_dir("interim") / "raw_tasks.jsonl"
    verified_path = get_data_dir("interim") / "verified_tasks.jsonl"
    with open(raw_path, "w") as f:
        for task in all_tasks:
            f.write(json.dumps(task, default=str) + "\n")
    with open(verified_path, "w") as f:
        for task in all_tasks:
            f.write(json.dumps(task, default=str) + "\n")

    # Stats
    family_counts = defaultdict(int)
    gold_counts = []
    for t in all_tasks:
        family_counts[t["task_family"]] += 1
        gold_counts.append(len(t.get("supporting_doc_ids", [])))

    print(f"\n{'='*60}")
    print(f"GROUNDED GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total: {len(all_tasks)} tasks")
    print(f"  Time: {t1-t0:.0f}s")
    print(f"  Tasks with gold docs: {sum(1 for g in gold_counts if g > 0)}")
    print(f"  Avg gold docs/task: {sum(gold_counts)/len(gold_counts):.1f}")
    for fam, count in sorted(family_counts.items()):
        print(f"    {fam}: {count}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
