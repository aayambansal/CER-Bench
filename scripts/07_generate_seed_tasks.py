#!/usr/bin/env python3
"""Script 07: Generate seed benchmark tasks using LLM extraction.

Three-stage pipeline:
  A) Seed extraction — sample papers, extract entities/conditions/methods
  B) Task synthesis — generate questions across 8 families
  C) Hard negative attachment — find plausible distractors

Requires: ANTHROPIC_API_KEY environment variable.

Usage:
    python scripts/07_generate_seed_tasks.py [--num-tasks 100] [--num-seeds 30]
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import get_data_dir, load_config

try:
    import anthropic
except ImportError:
    print("Install anthropic: pip install anthropic")
    sys.exit(1)


TASK_FAMILIES = [
    "constraint", "comparative", "contradiction", "abstention",
    "multihop", "temporal", "aggregation", "negative",
]

FAMILY_PROMPTS = {
    "constraint": (
        "Generate a CONSTRAINT-SATISFACTION retrieval task. The question must specify "
        "multiple experimental constraints (organism, assay, intervention, outcome, "
        "time period). A correct answer requires finding papers that satisfy ALL "
        "constraints simultaneously. The question should be specific enough that "
        "topic-matching alone won't work — the constraints must intersect narrowly."
    ),
    "comparative": (
        "Generate a COMPARATIVE retrieval task. The question must ask for a comparison "
        "across methods, organisms, conditions, or outcomes. The answer requires "
        "evidence from BOTH sides. A system that only retrieves one side fails."
    ),
    "contradiction": (
        "Generate a CONTRADICTION/CONDITIONALITY task. The question involves findings "
        "that appear contradictory across papers but are conditional on experimental "
        "setup (organism, dosage, cell type, etc.). The system must retrieve evidence "
        "showing both outcomes AND identify the conditions that explain the discrepancy."
    ),
    "abstention": (
        "Generate an ABSTENTION task. The question asks for evidence that likely does "
        "NOT exist in a biomedical corpus of ~5000 papers. The query should be plausible "
        "(not nonsensical) but combine constraints that are unlikely to intersect. "
        "The correct answer is to abstain / report insufficient evidence."
    ),
    "multihop": (
        "Generate a MULTI-HOP EVIDENCE CHAIN task. The answer requires linking evidence "
        "across 3+ papers via shared entities (a gene, pathway, compound, mechanism). "
        "No single paper answers the full question — the system must build a reasoning chain."
    ),
    "temporal": (
        "Generate a TEMPORAL EVOLUTION task. The question asks how understanding or "
        "consensus on a scientific topic changed over time (e.g., 2015 vs 2020 vs 2025). "
        "The system must retrieve papers from different eras showing the evolution."
    ),
    "aggregation": (
        "Generate an AGGREGATION task. The question asks to collect ALL reported "
        "quantitative values for a specific measurement across studies (e.g., all IC50 "
        "values, all reported efficiencies). Exhaustive recall is the key challenge."
    ),
    "negative": (
        "Generate a NEGATIVE/NULL RESULT task. The question asks specifically for "
        "studies reporting negative findings, null results, or absence of effect. "
        "The system must find papers with negative outcomes, not positive ones."
    ),
}

EXTRACTION_PROMPT = """You are a biomedical research analyst. Given these paper abstracts from a corpus, extract structured facts that can be used to generate retrieval benchmark tasks.

For each paper, extract:
1. Organisms / cell lines studied
2. Methods / assays used
3. Interventions / treatments
4. Key findings / outcomes (including negative results)
5. Experimental conditions (dosage, temperature, duration, etc.)
6. Potential comparison axes (what could be compared across papers)
7. Any apparent contradictions or condition-dependent results

Papers:
{papers}

Output a JSON array of extracted facts, one per paper. Be precise — use exact terms from the abstracts."""

TASK_GEN_PROMPT = """You are generating a benchmark task for evaluating biomedical literature retrieval systems.

CONTEXT: You have access to a corpus of ~5000 biomedical papers covering CRISPR, immunotherapy, drug repurposing, CAR-T, organoids, single-cell methods, microbiome, mRNA therapeutics, protein structure prediction, and epigenetics.

EXTRACTED FACTS from seed papers:
{facts}

TASK FAMILY: {family_instruction}

Generate exactly ONE task in this JSON format:
{{
    "task_family": "{family}",
    "difficulty": "easy|medium|hard",
    "question": "The natural-language query a scientist would ask",
    "decomposition_hints": ["subquery 1", "subquery 2"],
    "required_constraints": [
        {{"type": "organism|assay|intervention|outcome|temporal|design", "value": "specific value"}}
    ],
    "expected_answer_type": "set|comparison|conditional|abstain|chain|timeline|value_collection|null_result",
    "reference_answer": "Brief expected answer (1-2 sentences)",
    "reasoning": "Why this task is hard for retrieval systems"
}}

Make the question realistic — something a working scientist would actually ask. Make constraints specific and verifiable. For medium/hard difficulty, use 2-4 constraints."""


def load_seed_papers(corpus_path: Path, num_seeds: int, with_fulltext: bool = True) -> list[dict]:
    """Sample seed papers from corpus, preferring those with full text."""
    fulltext_papers = []
    abstract_papers = []

    with open(corpus_path) as f:
        for line in f:
            doc = json.loads(line)
            if doc.get("has_fulltext") and doc.get("abstract"):
                fulltext_papers.append(doc)
            elif doc.get("abstract"):
                abstract_papers.append(doc)

    # Prefer fulltext papers but mix in some abstract-only
    random.shuffle(fulltext_papers)
    random.shuffle(abstract_papers)

    if with_fulltext and fulltext_papers:
        seeds = fulltext_papers[:int(num_seeds * 0.7)]
        seeds += abstract_papers[:num_seeds - len(seeds)]
    else:
        seeds = (fulltext_papers + abstract_papers)[:num_seeds]

    return seeds[:num_seeds]


def extract_facts(client: anthropic.Anthropic, papers: list[dict], model: str) -> list[dict]:
    """Stage A: Extract structured facts from seed papers."""
    # Format papers for the prompt
    paper_texts = []
    for i, p in enumerate(papers):
        text = f"Paper {i+1} (PMID: {p.get('pmid', 'N/A')}):\n"
        text += f"Title: {p.get('title', 'N/A')}\n"
        text += f"Year: {p.get('year', 'N/A')}\n"
        text += f"Abstract: {p.get('abstract', 'N/A')[:1500]}\n"
        mesh = p.get("mesh_terms", [])[:10]
        if mesh:
            text += f"MeSH: {', '.join(mesh)}\n"
        paper_texts.append(text)

    prompt = EXTRACTION_PROMPT.format(papers="\n---\n".join(paper_texts))

    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text

    # Parse JSON from response
    try:
        # Try to find JSON array in response
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            facts = json.loads(text[start:end])
            return facts
    except json.JSONDecodeError:
        pass

    return [{"raw_extraction": text}]


def generate_task(
    client: anthropic.Anthropic,
    facts: list[dict],
    family: str,
    model: str,
) -> dict | None:
    """Stage B: Generate one task from extracted facts."""
    facts_text = json.dumps(facts[:5], indent=2, default=str)[:3000]
    family_instruction = FAMILY_PROMPTS[family]

    prompt = TASK_GEN_PROMPT.format(
        facts=facts_text,
        family_instruction=family_instruction,
        family=family,
    )

    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text

    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            task = json.loads(text[start:end])
            task["task_family"] = family  # ensure correct family
            return task
    except json.JSONDecodeError:
        pass

    return None


def main():
    parser = argparse.ArgumentParser(description="Generate seed benchmark tasks")
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--num-seeds", type=int, default=40)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    corpus_path = get_data_dir("processed") / "corpus.jsonl"
    if not corpus_path.exists():
        print(f"Error: {corpus_path} not found. Run scripts/04_build_corpus.py first.")
        sys.exit(1)

    gen_config = load_config("generation")
    family_dist = gen_config["family_distribution"]

    # Calculate tasks per family
    tasks_per_family = {}
    remaining = args.num_tasks
    for family in TASK_FAMILIES:
        count = max(1, round(args.num_tasks * family_dist.get(family, 0.125)))
        tasks_per_family[family] = count
        remaining -= count
    # Distribute remainder
    for family in TASK_FAMILIES:
        if remaining <= 0:
            break
        tasks_per_family[family] += 1
        remaining -= 1

    print(f"{'='*60}")
    print(f"TASK GENERATION")
    print(f"{'='*60}")
    print(f"  Target: {args.num_tasks} tasks across {len(TASK_FAMILIES)} families")
    print(f"  Model: {args.model}")
    for fam, count in tasks_per_family.items():
        print(f"    {fam}: {count}")
    print()

    # Stage A: Load seeds and extract facts
    print("Stage A: Loading seed papers...")
    seeds = load_seed_papers(corpus_path, args.num_seeds)
    print(f"  Selected {len(seeds)} seed papers")

    # Extract facts in batches of 5 papers
    print("Stage A: Extracting facts...")
    all_facts = []
    for i in range(0, len(seeds), 5):
        batch = seeds[i:i+5]
        print(f"  Extracting batch {i//5 + 1}/{(len(seeds)+4)//5}...")
        facts = extract_facts(client, batch, args.model)
        all_facts.extend(facts)
        time.sleep(1)  # Rate limit courtesy

    print(f"  Extracted {len(all_facts)} fact records")

    # Save extractions
    output_dir = get_data_dir("interim")
    extractions_path = output_dir / "seed_extractions.jsonl"
    with open(extractions_path, "w") as f:
        for fact in all_facts:
            f.write(json.dumps(fact, default=str) + "\n")

    # Stage B: Generate tasks per family
    print(f"\nStage B: Generating tasks...")
    t0 = time.time()
    all_tasks = []
    task_counter = 0

    for family, target_count in tasks_per_family.items():
        print(f"\n  Family: {family} (target: {target_count})")
        generated = 0
        attempts = 0
        max_attempts = target_count * 3  # allow retries

        while generated < target_count and attempts < max_attempts:
            # Shuffle facts for variety
            random.shuffle(all_facts)
            task = generate_task(client, all_facts, family, args.model)
            attempts += 1

            if task:
                task_counter += 1
                task["task_id"] = f"{family}_{task_counter:04d}"
                task["domain"] = "biomedicine"
                task["verification_status"] = "auto_generated"
                task["generation_method"] = "llm_extraction"
                all_tasks.append(task)
                generated += 1
                print(f"    [{generated}/{target_count}] {task['task_id']}: {task.get('question', '')[:80]}...")

            time.sleep(0.5)  # Rate limit

    t1 = time.time()

    # Save raw tasks
    tasks_path = output_dir / "raw_tasks.jsonl"
    with open(tasks_path, "w") as f:
        for task in all_tasks:
            f.write(json.dumps(task, default=str) + "\n")

    # Summary
    family_counts = {}
    difficulty_counts = {}
    for t in all_tasks:
        fam = t.get("task_family", "unknown")
        family_counts[fam] = family_counts.get(fam, 0) + 1
        diff = t.get("difficulty", "unknown")
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

    print(f"\n{'='*60}")
    print(f"GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total tasks: {len(all_tasks)}")
    print(f"  Time: {t1-t0:.0f}s")
    print(f"  Output: {tasks_path}")
    print(f"\n  By family:")
    for fam, count in sorted(family_counts.items()):
        print(f"    {count:>4}  {fam}")
    print(f"\n  By difficulty:")
    for diff, count in sorted(difficulty_counts.items()):
        print(f"    {count:>4}  {diff}")


if __name__ == "__main__":
    main()
