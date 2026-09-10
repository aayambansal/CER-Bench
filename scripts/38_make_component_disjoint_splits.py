#!/usr/bin/env python3
"""Offline, pre-publication validated splits; historical v1.1 is never a default."""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.corpus.identity import (SPLITS, annotated_ids, publish_directory,
                                read_jsonl, sha256, unique, write_json, write_jsonl)


def normalize_question(text):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def clusters(task):
    result = set()
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "evidence_cluster_id" and item is not None:
                    result.add(str(item))
                elif key == "cluster_ids":
                    if not isinstance(item, list):
                        raise ValueError("cluster_ids must be a list")
                    result.update(map(str, item))
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(task)
    return result


def relations(tasks, threshold=0.85):
    unique(tasks, "task_id")
    groups = defaultdict(list)
    for task in tasks:
        tid = task["task_id"]
        question = normalize_question(task["question"])
        if not question:
            raise ValueError(f"Empty normalized question: {tid}")
        docs, chunks = annotated_ids(task)
        for kind, values in (("documents", docs), ("chunks", chunks),
                             ("clusters", clusters(task)), ("exact_questions", [question])):
            for value in values:
                groups[(kind, value)].append(tid)
    edges = []
    for (kind, _), tids in sorted(groups.items()):
        edges.extend((tids[0], tid, kind) for tid in tids[1:])
    tokens = {t["task_id"]: set(normalize_question(t["question"]).split()) for t in tasks}
    for left, right in combinations(sorted(tokens), 2):
        a, b = tokens[left], tokens[right]
        if len(a & b) / len(a | b) >= threshold:
            edges.append((left, right, "lexical_near_duplicates"))
    return edges


def validate(rows, threshold=0.85):
    tasks = [t for s in SPLITS for t in rows[s]]
    unique(tasks, "task_id")
    placement = {}
    components = {}
    for split in SPLITS:
        for task in rows[split]:
            if task.get("split") != split:
                raise ValueError("Stale split field")
            if not task.get("component_id") or task.get("source_split") not in SPLITS:
                raise ValueError("Missing component_id or source_split")
            component = task["component_id"]
            if component in components and components[component] != split:
                raise ValueError("Component overlap")
            components[component] = split
            placement[task["task_id"]] = split
    overlap = {kind: 0 for kind in ("documents", "chunks", "clusters", "exact_questions", "lexical_near_duplicates")}
    for left, right, kind in relations(tasks, threshold):
        if placement[left] != placement[right]:
            overlap[kind] += 1
    if any(overlap.values()):
        raise ValueError(f"Split overlap remains: {overlap}")
    return overlap


def make_splits(source, output, seed=20260826, threshold=0.85):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output == ROOT / "data/benchmark/v1_1_component_disjoint":
        raise ValueError("Historical/source output path forbidden")
    if not 0 < threshold <= 1:
        raise ValueError("Threshold must be in (0, 1]")
    paths = [source / f"{s}.jsonl" for s in SPLITS]
    provenance = {str(p): sha256(p) for p in paths}
    if (source / "report.json").exists():
        provenance[str(source / "report.json")] = sha256(source / "report.json")
    tasks = []
    stale = 0
    for split, path in zip(SPLITS, paths):
        for task in read_jsonl(path):
            stale += task.get("split") != split
            tasks.append({**task, "source_split": split})
    unique(tasks, "task_id")
    if not tasks:
        raise ValueError("No tasks: no coherent split can be published")
    tasks.sort(key=lambda t: t["task_id"])
    edges = relations(tasks, threshold)
    parent = {t["task_id"]: t["task_id"] for t in tasks}
    def find(tid):
        while parent[tid] != tid:
            parent[tid] = parent[parent[tid]]
            tid = parent[tid]
        return tid
    for left, right, _ in edges:
        a, b = find(left), find(right)
        parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for task in tasks:
        groups[find(task["task_id"])].append(task)
    components = list(groups.values())
    rng = random.Random(seed)
    rng.shuffle(components)
    components.sort(key=lambda c: -len(c))
    ratios = dict(zip(SPLITS, (15 / 38, 8 / 38, 15 / 38)))
    families = Counter(t.get("task_family", "unknown") for t in tasks)
    counts = {s: Counter() for s in SPLITS}
    rows = {s: [] for s in SPLITS}
    for component in components:
        incoming = Counter(t.get("task_family", "unknown") for t in component)
        def cost(chosen):
            return sum((len(rows[s]) + (len(component) if s == chosen else 0) - len(tasks) * ratios[s]) ** 2
                       + sum((counts[s][f] + (incoming[f] if s == chosen else 0) - n * ratios[s]) ** 2
                             for f, n in families.items()) for s in SPLITS)
        chosen = min(SPLITS, key=lambda s: (cost(s), len(rows[s]), s))
        cid = "component:" + hashlib.sha256("\n".join(sorted(t["task_id"] for t in component)).encode()).hexdigest()
        rows[chosen].extend({**t, "split": chosen, "component_id": cid} for t in component)
        counts[chosen].update(incoming)
    overlap = validate(rows, threshold)
    manifest = {
        "version": "v1.2-repaired-components-candidate", "release_blocked": True,
        "seed": seed, "source_hashes": provenance, "components": len(components),
        "largest_component": max(map(len, components)), "stale_source_split_fields_corrected": stale,
        "split_sizes": {s: len(rows[s]) for s in SPLITS},
        "family_counts": {s: dict(counts[s]) for s in SPLITS},
        "cross_split_overlap": overlap, "lexical_near_duplicate_threshold": threshold,
        "grouped_relation_counts": dict(Counter(kind for _, _, kind in edges)),
        "warning": "Previously exposed synthetic proxy tasks, not a fresh test or gold. Lexical checks do not establish semantic independence. Identity repair and scientific validity remain unadjudicated.",
    }
    def build(stage):
        for split in SPLITS:
            write_jsonl(stage / f"{split}.jsonl", sorted(rows[split], key=lambda t: t["task_id"]))
        validate({s: read_jsonl(stage / f"{s}.jsonl") for s in SPLITS}, threshold)
        if any(sha256(p) != digest for p, digest in provenance.items()):
            raise ValueError("Source changed during split construction")
        manifest["output_hashes"] = {f"{s}.jsonl": sha256(stage / f"{s}.jsonl") for s in SPLITS}
        write_json(stage / "manifest.json", manifest)
        return manifest
    return publish_directory(output, build)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/processed/identity_repair_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/benchmark/v1_2_repaired_components")
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    import json
    print(json.dumps(make_splits(args.source, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
