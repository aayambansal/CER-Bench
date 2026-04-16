#!/usr/bin/env python3
"""Script 11: Run the SynthSearch agentic retrieval controller.

Context-1-inspired agent that:
  1. Decomposes the query into subqueries
  2. Searches iteratively (BM25 + dense)
  3. Reads and evaluates evidence
  4. Prunes low-value results
  5. Refines queries for unresolved constraints
  6. Stops when evidence is sufficient or abstains

Requires: ANTHROPIC_API_KEY

Usage:
    python scripts/11_run_agent.py [--split dev] [--model claude-sonnet-4-20250514]
"""

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import get_data_dir

try:
    import anthropic
except ImportError:
    print("pip install anthropic")
    sys.exit(1)


AGENT_SYSTEM = """You are a scientific literature retrieval agent. Your job is to find the most relevant papers for a given research question.

You have these tools:
1. SEARCH: Search the corpus with a query. Returns ranked document summaries.
2. REFINE: Generate a more specific subquery based on what you've found so far.
3. STOP: Stop searching and return your final evidence set.

Process:
1. Analyze the question and identify key constraints (organism, method, condition, etc.)
2. Search with the full question first
3. Evaluate which constraints are satisfied by the results
4. If important constraints are unsatisfied, REFINE your query and search again
5. After 2-4 search rounds, STOP and return your best evidence

Respond with a JSON action:
{"action": "SEARCH", "query": "your search query"}
{"action": "REFINE", "reasoning": "what's missing", "query": "refined query"}
{"action": "STOP", "selected_docs": ["doc_id1", "doc_id2", ...], "reasoning": "why these are sufficient"}

Be concise. Focus on finding the right documents, not explaining the science."""


def tokenize_simple(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def extract_doc_id(chunk_id):
    section_types = {'abstract', 'other', 'methods', 'results', 'discussion',
                     'introduction', 'conclusion', 'caption', 'table', 'body',
                     'results_discussion'}
    parts = chunk_id.split('_')
    for i in range(1, len(parts)):
        suffix = '_'.join(parts[i:])
        for st in section_types:
            if suffix.startswith(st):
                return '_'.join(parts[:i])
    if len(parts) >= 3:
        return '_'.join(parts[:-2])
    return chunk_id


class RetrievalEnvironment:
    """Search environment wrapping BM25 + dense indices."""

    def __init__(self):
        # Load BM25
        bm25_dir = get_data_dir("processed/indices/bm25")
        with open(bm25_dir / "bm25_index.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        with open(bm25_dir / "chunk_ids.json") as f:
            self.bm25_ids = json.load(f)

        # Load corpus for doc info
        self.docs = {}
        with open(get_data_dir("processed") / "corpus.jsonl") as f:
            for line in f:
                doc = json.loads(line)
                self.docs[doc["doc_id"]] = doc

        # Load chunks
        self.chunks = {}
        with open(get_data_dir("processed") / "chunks.jsonl") as f:
            for line in f:
                chunk = json.loads(line)
                self.chunks[chunk["chunk_id"]] = chunk

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        """Search and return document summaries."""
        tokens = tokenize_simple(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k * 3]

        seen = set()
        results = []
        for idx in top_idx:
            cid = self.bm25_ids[idx]
            doc_id = extract_doc_id(cid)
            if doc_id in seen:
                continue
            seen.add(doc_id)

            doc = self.docs.get(doc_id, {})
            chunk = self.chunks.get(cid, {})

            results.append({
                "doc_id": doc_id,
                "title": doc.get("title", "")[:200],
                "year": doc.get("year"),
                "abstract_snippet": doc.get("abstract", "")[:300],
                "matched_section": chunk.get("section_type", ""),
                "matched_text": chunk.get("text", "")[:200],
                "score": float(scores[idx]),
            })

            if len(results) >= top_k:
                break

        return results


def run_agent_on_task(
    client: anthropic.Anthropic,
    env: RetrievalEnvironment,
    task: dict,
    model: str,
    max_rounds: int = 4,
) -> dict:
    """Run the agent on a single task."""
    question = task["question"]
    messages = []
    all_searched_docs = {}  # doc_id -> info
    trace = []

    # Initial prompt
    messages.append({
        "role": "user",
        "content": f"Find papers for this question:\n\n{question}\n\nStart by searching.",
    })

    for round_num in range(max_rounds):
        # Get agent action
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0.0,
            system=AGENT_SYSTEM,
            messages=messages,
        )
        agent_text = resp.content[0].text
        messages.append({"role": "assistant", "content": agent_text})

        # Parse action
        try:
            # Find JSON in response
            start = agent_text.find("{")
            end = agent_text.rfind("}") + 1
            if start >= 0 and end > start:
                action = json.loads(agent_text[start:end])
            else:
                action = {"action": "STOP", "selected_docs": list(all_searched_docs.keys())[:20]}
        except json.JSONDecodeError:
            action = {"action": "STOP", "selected_docs": list(all_searched_docs.keys())[:20]}

        trace.append({"round": round_num, "action": action})

        if action.get("action") == "STOP":
            selected = action.get("selected_docs", list(all_searched_docs.keys())[:20])
            return {
                "task_id": task["task_id"],
                "retrieved_docs": selected[:20],
                "trace": trace,
                "rounds": round_num + 1,
                "total_docs_seen": len(all_searched_docs),
            }

        # Execute search
        query = action.get("query", question)
        results = env.search(query, top_k=15)

        for r in results:
            if r["doc_id"] not in all_searched_docs:
                all_searched_docs[r["doc_id"]] = r

        # Format results for agent
        result_text = f"Search results for: '{query}'\n\n"
        for i, r in enumerate(results[:10]):
            result_text += f"{i+1}. [{r['doc_id']}] {r['title'][:150]}\n"
            result_text += f"   Year: {r['year']}, Section: {r['matched_section']}\n"
            result_text += f"   Snippet: {r['matched_text'][:150]}...\n\n"

        result_text += f"\nTotal unique docs found so far: {len(all_searched_docs)}\n"
        result_text += "What next? SEARCH again with a refined query, or STOP and select your best docs."

        messages.append({"role": "user", "content": result_text})

    # Max rounds reached
    return {
        "task_id": task["task_id"],
        "retrieved_docs": list(all_searched_docs.keys())[:20],
        "trace": trace,
        "rounds": max_rounds,
        "total_docs_seen": len(all_searched_docs),
    }


def main():
    parser = argparse.ArgumentParser(description="Run agentic retrieval controller")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--max-rounds", type=int, default=4)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load tasks
    split_path = get_data_dir("benchmark") / f"{args.split}.jsonl"
    if not split_path.exists():
        print(f"Error: {split_path} not found")
        sys.exit(1)

    tasks = [json.loads(l) for l in open(split_path)]
    print(f"Running agent on {len(tasks)} {args.split} tasks")
    print(f"Model: {args.model}")
    print(f"Max rounds: {args.max_rounds}")

    # Initialize environment
    print("Loading retrieval environment...")
    env = RetrievalEnvironment()
    print(f"  {len(env.docs)} docs, {len(env.chunks)} chunks")

    # Run agent on each task
    results = []
    t0 = time.time()

    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {task['task_id']} ({task['task_family']})")
        print(f"  Q: {task['question'][:80]}...")

        result = run_agent_on_task(client, env, task, args.model, args.max_rounds)
        results.append(result)

        print(f"  Rounds: {result['rounds']}, Docs seen: {result['total_docs_seen']}, Selected: {len(result['retrieved_docs'])}")
        time.sleep(0.5)

    t1 = time.time()

    # Save results
    output_dir = Path("results/baselines")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"agent_{args.split}.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    # Save traces separately
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(exist_ok=True)
    for r in results:
        with open(traces_dir / f"{r['task_id']}_trace.json", "w") as f:
            json.dump(r["trace"], f, indent=2, default=str)

    # Summary
    avg_rounds = sum(r["rounds"] for r in results) / len(results)
    avg_docs = sum(r["total_docs_seen"] for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"AGENT RESULTS")
    print(f"{'='*60}")
    print(f"  Tasks: {len(results)}")
    print(f"  Avg rounds: {avg_rounds:.1f}")
    print(f"  Avg docs seen: {avg_docs:.1f}")
    print(f"  Time: {t1-t0:.0f}s")
    print(f"  Output: {results_path}")


if __name__ == "__main__":
    main()
