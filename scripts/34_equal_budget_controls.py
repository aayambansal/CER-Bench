#!/usr/bin/env python3
"""Evaluate 24-document budget-matched retrieval controls.

The script never uses gold annotations to form queries or candidate sets.  It
reports candidate recall before selection and a shared original-query BM25
selector after candidate generation, which separates exposure from ranking.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.budget_protocol import CONDITIONS, file_hash, validate_manifest, heuristic_queries as protocol_heuristic_queries
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "baselines"
OUT = ROOT / "results" / "readiness" / "budget" / "scoring"
SECTION_TYPES = {
    "abstract",
    "other",
    "methods",
    "results",
    "discussion",
    "introduction",
    "conclusion",
    "caption",
    "table",
    "body",
    "results_discussion",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "these",
    "this",
    "to",
    "used",
    "using",
    "was",
    "were",
    "what",
    "when",
    "which",
    "who",
    "with",
    "would",
    "papers",
    "paper",
    "studies",
    "study",
    "evidence",
    "find",
    "identify",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def extract_doc_id(chunk_id: str) -> str:
    parts = chunk_id.split("_")
    for i in range(1, len(parts)):
        suffix = "_".join(parts[i:])
        if any(suffix.startswith(section) for section in SECTION_TYPES):
            return "_".join(parts[:i])
    return "_".join(parts[:-2]) if len(parts) >= 3 else chunk_id


def tokenize(text: str) -> list[str]:
    text = re.sub(r"[^a-z0-9\s\-]", " ", text.lower())
    return [token for token in text.split() if len(token) > 1]


class BM25Search:
    def __init__(self) -> None:
        base = DATA / "processed" / "indices" / "bm25"
        with (base / "bm25_index.pkl").open("rb") as handle:
            self.index = pickle.load(handle)
        self.chunk_ids = json.load((base / "chunk_ids.json").open())
        self._cache: dict[str, tuple[list[str], dict[str, float]]] = {}

    def ranked_docs(self, query: str) -> tuple[list[str], dict[str, float]]:
        key = " ".join(tokenize(query))
        if key in self._cache:
            return self._cache[key]
        scores = self.index.get_scores(key.split())
        order = np.argsort(-np.asarray(scores), kind="stable")
        docs: list[str] = []
        best: dict[str, float] = {}
        seen = set()
        for idx in order:
            doc_id = extract_doc_id(self.chunk_ids[int(idx)])
            score = float(scores[int(idx)])
            if doc_id not in best or score > best[doc_id]:
                best[doc_id] = score
            if doc_id not in seen:
                seen.add(doc_id)
                docs.append(doc_id)
        self._cache[key] = (docs, best)
        return docs, best

    def take_new(self, query: str, n: int, excluded: set[str]) -> list[str]:
        ranking, _ = self.ranked_docs(query)
        return [doc for doc in ranking if doc not in excluded][:n]


def heuristic_queries(question: str) -> list[str]:
    return protocol_heuristic_queries(question)


def round_robin_candidates(
    search: BM25Search, queries: list[str], per_round: int = 8
) -> tuple[list[str], list[list[str]]]:
    seen: set[str] = set()
    rounds: list[list[str]] = []
    for query in queries:
        selected = search.take_new(query, per_round, seen)
        rounds.append(selected)
        seen.update(selected)
    return [doc for row in rounds for doc in row], rounds


def rrf_rank(
    search: BM25Search, queries: list[str], candidates: list[str], k: int = 60
) -> list[str]:
    candidate_set = set(candidates)
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, tuple[int, int]] = {}
    for qi, query in enumerate(queries):
        ranking, _ = search.ranked_docs(query)
        for rank, doc in enumerate(ranking):
            if doc in candidate_set:
                scores[doc] += 1.0 / (k + rank + 1)
                first_seen.setdefault(doc, (qi, rank))
    return sorted(
        candidates, key=lambda doc: (-scores[doc], doc)
    )


def original_selector(
    search: BM25Search, question: str, candidates: list[str]
) -> list[str]:
    ranking, scores = search.ranked_docs(question)
    rank = {doc: i for i, doc in enumerate(ranking)}
    return sorted(
        candidates,
        key=lambda doc: (-scores.get(doc, float("-inf")), rank.get(doc, 10**9), doc),
    )


def recall_at(docs: list[str], gold: set[str], k: int) -> float:
    return len(set(docs[:k]) & gold) / len(gold) if gold else 0.0


def ndcg_at(docs: list[str], gold: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, doc in enumerate(docs[:k]) if doc in gold)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold), k)))
    return float(dcg / ideal) if ideal else 0.0


def mrr(docs: list[str], gold: set[str]) -> float:
    return next((1.0 / (i + 1) for i, doc in enumerate(docs) if doc in gold), 0.0)


def aggregate(rows: list[dict], gold_by_task: dict[str, set[str]], field: str) -> dict:
    valid = [row for row in rows if gold_by_task.get(row["task_id"])]
    values = {
        metric: []
        for metric in (
            "candidate_recall@24",
            "recall@5",
            "recall@10",
            "recall@20",
            "ndcg@10",
            "mrr",
        )
    }
    for row in valid:
        gold = gold_by_task[row["task_id"]]
        docs = row[field]
        candidates = row["candidate_docs"]
        values["candidate_recall@24"].append(recall_at(candidates, gold, 24))
        for k in (5, 10, 20):
            values[f"recall@{k}"].append(recall_at(docs, gold, k))
        values["ndcg@10"].append(ndcg_at(docs, gold, 10))
        values["mrr"].append(mrr(docs, gold))
    return {
        "n_tasks": len(valid),
        **{key: round(float(np.mean(val)), 4) if val else None for key, val in values.items()},
    }


def bootstrap_delta(
    condition: list[dict],
    baseline: list[dict],
    gold: dict[str, set[str]],
    field: str,
    k: int,
    seed: int = 42,
    n: int = 10000,
) -> dict:
    a = {row["task_id"]: row for row in condition}
    b = {row["task_id"]: row for row in baseline}
    tids = sorted(set(a) & set(b) & {tid for tid, docs in gold.items() if docs})
    delta = np.array(
        [
            recall_at(a[tid][field], gold[tid], k)
            - recall_at(b[tid][field], gold[tid], k)
            for tid in tids
        ]
    )
    rng = np.random.default_rng(seed)
    if not len(delta):
        return {"n_tasks": 0, "mean_delta": None, "ci95": None}
    samples = delta[rng.integers(0, len(delta), size=(n, len(delta)))].mean(axis=1)
    return {
        "n_tasks": len(delta),
        "mean_delta": round(float(delta.mean()), 4),
        "ci95": [round(float(x), 4) for x in np.quantile(samples, [0.025, 0.975])],
        "bootstrap_samples": n,
        "seed": seed,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_llm_rows(path: Path, allow_unbound=False) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    result = {}
    provenance = {(r.get("model"), r.get("provider"), r.get("provenance_id"), r.get("repeat")) for r in rows}
    if len(provenance) > 1:
        raise ValueError("Mixed model/provider/repeat/provenance input")
    for row in rows:
        key = (row["task_id"], row.get("condition", "legacy_combined"))
        if key in result:
            raise ValueError("Duplicate task/condition identity; do not merge repeats")
        if not row.get("provenance_id") and not allow_unbound:
            raise ValueError("Legacy rows are unbound; use --allow-unbound-exploratory for nonpublication diagnostics")
        result[key] = row
    return result


def matched_conditions(conditions):
    """One intersection for ALL condition summaries, qrels and paired deltas."""
    subsets = [{r["task_id"] for r in rows} for rows in conditions.values()]
    common = set.intersection(*subsets) if subsets else set()
    return {name: [r for r in rows if r["task_id"] in common] for name, rows in conditions.items()}, sorted(common)


def decision_metrics(decisions):
    # UNCERTAIN is non-answering, but is NOT a prediction of corpus absence.
    tp = sum(g and d == "ABSTAIN" for g, d in decisions)
    fn = sum(g and d != "ABSTAIN" for g, d in decisions)
    fp = sum(not g and d == "ABSTAIN" for g, d in decisions)
    tn = sum(not g and d != "ABSTAIN" for g, d in decisions)
    answered = [(g, d) for g, d in decisions if d == "ANSWER"]
    return {"n": len(decisions), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "uncertain": sum(d == "UNCERTAIN" for _, d in decisions),
            "answer_coverage": len(answered) / len(decisions) if decisions else None,
            "nonanswer_rate": sum(d != "ANSWER" for _, d in decisions) / len(decisions) if decisions else None,
            "absence_label_accuracy": (tp + tn) / len(decisions) if decisions else None,
            "abstention_recall": tp / (tp + fn) if tp + fn else None,
            "false_abstention_rate": fp / (fp + tn) if fp + tn else None,
            "selective_accuracy": None,
            "answered_answerability_proxy_accuracy": sum(not g for g, _ in answered) / len(answered) if answered else None,
            "warning": "Synthetic absence labels only. Selective answer correctness is unavailable: no scientific answers scored."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm-queries", type=Path
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--allow-unbound-exploratory", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory exists; preserve original scored outputs and choose a fresh directory")
    tasks = read_jsonl(args.tasks)
    task_by_id = {task["task_id"]: task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("Duplicate task identities")
    seed_gold = {
        tid: set(map(str, task.get("supporting_doc_ids", [])))
        for tid, task in task_by_id.items()
    }
    pooled_raw = json.load((RESULTS / "expanded_gold.json").open())
    pooled_gold = {tid: set(map(str, docs)) for tid, docs in pooled_raw.items()}
    loaded = load_llm_rows(args.llm_queries, args.allow_unbound_exploratory) if args.llm_queries else {}
    if any(tid not in task_by_id for tid, _ in loaded):
        raise ValueError("Query input has tasks outside explicit task file")
    bound_rows = [r for r in loaded.values() if r.get("provenance_id")]
    if bound_rows:
        bound = json.loads(args.llm_queries.with_suffix(args.llm_queries.suffix + ".manifest.json").read_text())
        if not validate_manifest(bound) or any(r["provenance_id"] != bound["provenance_id"] for r in bound_rows):
            raise ValueError("Invalid run manifest/row binding")
        if any((r.get("model"), r.get("provider")) != (bound["model"], bound["provider"]) for r in bound_rows):
            raise ValueError("Manifest model/provider mismatch")
        if bound["task_sha256"] != file_hash(args.tasks) or bound["split"] != args.split:
            raise ValueError("Task/split provenance mismatch")
        if bound["corpus_sha256"] != file_hash(DATA / "processed/corpus.jsonl"):
            raise ValueError("Corpus provenance mismatch")
        required_index = {str((DATA / "processed/indices/bm25" / name).resolve()) for name in ("bm25_index.pkl", "chunk_ids.json")}
        if not required_index <= bound["index_sha256"].keys() or any(file_hash(p) != h for p, h in bound["index_sha256"].items()):
            raise ValueError("Index provenance mismatch")
        if any(r.get("status") != "complete" for r in bound_rows):
            raise ValueError("Only completed rows can be scored; dry runs are not experiments")
    llm_rows = {tid: r for (tid, c), r in loaded.items() if c == "legacy_combined"}
    search = BM25Search()

    conditions: dict[str, list[dict]] = defaultdict(list)
    for row in bound_rows:
        if row.get("condition") not in CONDITIONS or row["condition"] not in bound["config"].get("conditions", CONDITIONS):
            raise ValueError("Unexpected condition")
        docs = row.get("candidate_docs", [])
        if len(docs) != 24 or any(not isinstance(d, str) or not d for d in docs) or len(set(docs)) != 24:
            raise ValueError("Invalid unique candidate budget")
        rounds = row.get("rounds", [])
        expected_sizes = [24] if row["condition"] in ("original_top24", "one_shot_rewrite") else [8, 8, 8]
        if [len(r.get("candidate_chunks", [])) for r in rounds] != expected_sizes or [c["doc_id"] for r in rounds for c in r["candidate_chunks"]] != docs:
            raise ValueError("Invalid round budgets/candidate consistency")
        if len(row.get("queries", [])) != len(expected_sizes) or any(not isinstance(q, str) or not q.strip() for q in row["queries"]):
            raise ValueError("Invalid query list")
        selected = original_selector(search, task_by_id[row["task_id"]]["question"], docs)[:20]
        if selected != row.get("common_selector_docs"):
            raise ValueError("Common original-query selector mismatch")
        conditions[row["condition"]].append({**row, "rrf_docs": rrf_rank(search, row["queries"], docs)[:20]})
    if bound_rows:
        for name in bound["config"].get("conditions", []):
            conditions[name]  # A wholly missing condition forces an empty matched subset.
    for task in tasks:
        tid, question = task["task_id"], task["question"]
        original_ranking, _ = search.ranked_docs(question)
        original_candidates = original_ranking[:24]
        conditions["Original query, top 24"].append(
            {
                "task_id": tid,
                "queries": [question],
                "candidate_docs": original_candidates,
                "rrf_docs": original_candidates[:20],
                "common_selector_docs": original_candidates[:20],
            }
        )

        repeat_candidates, repeat_rounds = round_robin_candidates(
            search, [question] * 3
        )
        conditions["Repeated original, 3x8"].append(
            {
                "task_id": tid,
                "queries": [question] * 3,
                "round_candidates": repeat_rounds,
                "candidate_docs": repeat_candidates,
                "rrf_docs": rrf_rank(search, [question] * 3, repeat_candidates)[:20],
                "common_selector_docs": original_selector(
                    search, question, repeat_candidates
                )[:20],
            }
        )

        hqueries = heuristic_queries(question)
        hcandidates, hrounds = round_robin_candidates(search, hqueries)
        conditions["Heuristic keyword windows (not RM3), 3x8"].append(
            {
                "task_id": tid,
                "queries": hqueries,
                "round_candidates": hrounds,
                "candidate_docs": hcandidates,
                "rrf_docs": rrf_rank(search, hqueries, hcandidates)[:20],
                "common_selector_docs": original_selector(
                    search, question, hcandidates
                )[:20],
            }
        )

        if tid in llm_rows:
            record = llm_rows[tid]
            one_query = record["one_shot_query"]
            one_candidates = search.ranked_docs(one_query)[0][:24]
            conditions["One-shot LLM rewrite, top 24"].append(
                {
                    "task_id": tid,
                    "queries": [one_query],
                    "candidate_docs": one_candidates,
                    "rrf_docs": one_candidates[:20],
                    "common_selector_docs": original_selector(
                        search, question, one_candidates
                    )[:20],
                }
            )
            iqueries = record["iterative_queries"]
            icandidates = [
                doc for docs in record["iterative_round_candidates"] for doc in docs
            ]
            if len(icandidates) != 24 or len(set(icandidates)) != 24 or [len(r) for r in record["iterative_round_candidates"]] != [8, 8, 8]:
                raise ValueError("Invalid legacy candidate budget/dedup")
            conditions["Iterative LLM refinement, 3x8"].append(
                {
                    "task_id": tid,
                    "queries": iqueries,
                    "round_candidates": record["iterative_round_candidates"],
                    "candidate_docs": icandidates,
                    "rrf_docs": rrf_rank(search, iqueries, icandidates)[:20],
                    "common_selector_docs": original_selector(
                        search, question, icandidates
                    )[:20],
                    "final_decision": record.get("final_decision", {}),
                }
            )

    available_counts = {name: len(rows) for name, rows in conditions.items()}
    conditions, matched_ids = matched_conditions(conditions)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in conditions.items():
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        write_jsonl(output_dir / f"{slug}.jsonl", rows)

    metrics: dict[str, dict] = {
        "protocol": {
            "split": args.split,
            "task_sha256": file_hash(args.tasks),
            "publication_ready": False,
            "legacy_unbound": bool(llm_rows),
            "available_tasks_by_condition": available_counts,
            "matched_task_ids": matched_ids,
            "compute_equality_claimed": False,
            "candidate_budget_unique_documents": 24,
            "rounds": 3,
            "new_documents_per_round": 8,
            "primary_selector": "original-query BM25 score shared across conditions",
            "sensitivity_selector": "RRF over condition queries",
            "gold_excluded_from_generation": True,
        },
        "seed_qrels": {},
        "pooled_qrels": {},
        "bootstrap_vs_original_seed_r20": {},
    }
    baseline = conditions["Original query, top 24"]
    for name, rows in conditions.items():
        metrics["seed_qrels"][name] = {
            "common_selector": aggregate(rows, seed_gold, "common_selector_docs"),
            "rrf_selector": aggregate(rows, seed_gold, "rrf_docs"),
        }
        metrics["pooled_qrels"][name] = {
            "common_selector": aggregate(rows, pooled_gold, "common_selector_docs"),
            "rrf_selector": aggregate(rows, pooled_gold, "rrf_docs"),
        }
        if name != "Original query, top 24" and len(rows) == len(baseline):
            metrics["bootstrap_vs_original_seed_r20"][name] = bootstrap_delta(
                rows, baseline, seed_gold, "common_selector_docs", 20
            )

    if llm_rows:
        decisions = []
        for task in tasks:
            if task["task_id"] not in matched_ids:
                continue
            record = llm_rows.get(task["task_id"])
            if not record:
                continue
            decision = str(
                record.get("final_decision", {}).get("decision", "UNCERTAIN")
            ).upper()
            if decision not in ("ANSWER", "ABSTAIN", "UNCERTAIN"):
                raise ValueError("Invalid decision")
            decisions.append((task["task_family"] == "abstention", decision))
        metrics["proxy_abstention"] = decision_metrics(decisions)
        metrics["api_usage"] = {
            "scope": "Entire single input run, not just matched scoring subset; legacy reported usage is unverified",
            "model": next(iter(llm_rows.values())).get("model"),
            "prompt_tokens": sum(
                row.get("usage", {}).get("prompt_tokens", 0)
                for row in llm_rows.values()
            ),
            "completion_tokens": sum(
                row.get("usage", {}).get("completion_tokens", 0)
                for row in llm_rows.values()
            ),
            "reported_cost_usd": round(
                sum(
                    row.get("usage", {}).get("cost", 0.0) or 0.0
                    for row in llm_rows.values()
                ),
                4,
            ),
        }

    output = output_dir / "equal_budget_metrics.json"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, indent=2) + "\n")
    print(f"Wrote {output}")
    for name, record in metrics["seed_qrels"].items():
        m = record["common_selector"]
        print(
            f"{name:36s} n={m['n_tasks']:3d} CandR@24={m['candidate_recall@24']} R@20={m['recall@20']}"
        )


if __name__ == "__main__":
    main()
