#!/usr/bin/env python3
"""Local-only historical diagnostic. Never repairs identities or authorizes release."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
from itertools import combinations
import json
import math
from pathlib import Path
import platform
import re
import sys
from fractions import Fraction

import numpy as np
import scipy
from scipy.stats import kendalltau, spearmanr, rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.strict_metrics import METRICS, VERSION, doc_metrics, index_rows, strings

VERSION_ANALYSIS = "historical-qrel-sensitivity-v1"
POOL = ("bm25", "dense", "bge", "e5large", "medcpt", "splade", "hybrid",
        "hybrid_reranked", "bge_reranker", "agent")
METHODS = POOL + ("rm3", "agent_single_step")
FRACTIONS = (0, .25, .5, .75, 1)
SEED = 20260909
PRIMARY_METRIC = "Recall@20"
WARNING = ("release_blocked: true. Historical diagnostic only: all primary results are "
           "compromised by corpus identity failures. No repaired benchmark, publication "
           "result, validated gold, fulltext advantage, or causal agent claim. Artificial "
           "label subsampling is conditional on the original biased pool and fixed final "
           "queries/rankings; it cannot establish completeness outside that pool.")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def no_duplicate_keys(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate JSON key: {k}")
        out[k] = v
    return out


def read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=no_duplicate_keys)


def rows(path):
    result = []
    # JSON strings can contain Unicode U+2028/U+2029; only physical LF separates records.
    content = Path(path).read_text()
    physical_lines = content[:-1].split("\n") if content.endswith("\n") else content.split("\n")
    for line, text in enumerate(physical_lines, 1):
        if not text.strip():
            raise ValueError(f"{path}:{line}: blank row")
        try:
            result.append(json.loads(text, object_pairs_hook=no_duplicate_keys))
        except ValueError as e:
            raise ValueError(f"{path}:{line}: {e}") from e
    return result


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def exact_keys(actual, expected, label):
    extra, missing = sorted(set(actual) - set(expected)), sorted(set(expected) - set(actual))
    if extra or missing:
        raise ValueError(f"{label}: unknown IDs={extra}; missing rows={missing}")


def validate_docs(value, corpus, label):
    strings(value, label)
    extra = sorted(set(value) - set(corpus))
    if extra:
        raise ValueError(f"{label}: unknown doc IDs {extra}")
    return value


def verdict(value):
    if not isinstance(value, str):
        raise ValueError(f"unknown judgment string {value!r}")
    text = value.strip().upper()
    for prefix, result in (("NOT_RELEVANT", False), ("RELEVANT", True)):
        if re.match(r"^" + prefix + r"(?:\b|$)", text):
            return result
    raise ValueError(f"unknown judgment string {value!r}")


def reconstruct(seed, judgments, corpus, expected_per_task=None):
    expanded = {t: set(ds) for t, ds in seed.items()}
    seen, normalized, counts = {}, [], Counter()
    for i, row in enumerate(judgments, 1):
        t, d = row.get("task_id"), row.get("doc_id")
        if t not in seed:
            raise ValueError(f"judgment row {i}: unknown task ID {t!r}")
        validate_docs([d], corpus, f"judgment {i}/{t}")
        try:
            label = verdict(row.get("judgment"))
        except ValueError as e:
            raise ValueError(f"judgment {i}/{t}/{d}: {e}") from e
        if (t, d) in seen:
            raise ValueError(f"duplicate/contradictory relevance record {t}/{d}: {seen[t,d]} vs {label}")
        seen[t, d] = label
        if not label and d in seed[t]:
            raise ValueError(f"seed/judgment contradiction {t}/{d}: NOT_RELEVANT seed")
        if label:
            expanded[t].add(d)
        normalized.append({"task_id": t, "doc_id": d, "relevant": label})
        counts[t] += 1
    if expected_per_task is not None:
        bad = {t: {"expected": expected_per_task if ds else 0, "actual": counts[t]}
               for t, ds in seed.items() if counts[t] != (expected_per_task if ds else 0)}
        if bad:
            raise ValueError(f"missing/excess judgment rows: {bad}")
    return {t: sorted(ds) for t, ds in expanded.items()}, normalized


def rank_comparison(seed_means, expanded_means, names):
    a = np.array([seed_means[n] for n in names])
    b = np.array([expanded_means[n] for n in names])
    reversals = []
    for i, j in combinations(range(len(names)), 2):
        da, db = float(a[i] - a[j]), float(b[i] - b[j])
        if da * db < 0:
            reversals.append({"left": names[i], "right": names[j], "seed_difference": da,
                              "expanded_difference": db})
    def finite(x):
        return float(x) if np.isfinite(x) else None
    return {"n_systems": len(names), "n_pairs": len(names) * (len(names)-1)//2,
            "kendall_tau_b": finite(kendalltau(a, b, variant="b").statistic),
            "spearman": finite(spearmanr(a, b).statistic),
            "seed_ranks": dict(zip(names, rankdata(-a, method="average").tolist())),
            "expanded_ranks": dict(zip(names, rankdata(-b, method="average").tolist())),
            "seed_tied_pairs": int(sum(a[i] == a[j] for i, j in combinations(range(len(names)), 2))),
            "expanded_tied_pairs": int(sum(b[i] == b[j] for i, j in combinations(range(len(names)), 2))),
            "strict_reversals": reversals}


def component_map(tasks):
    # Reuse the audited local relation rule; do NOT read candidate corpus/splits.
    path = ROOT / "scripts/38_make_component_disjoint_splits.py"
    spec = importlib.util.spec_from_file_location("historical_component_rules", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    edges = mod.relations(tasks, threshold=.85)
    parent = {t["task_id"]: t["task_id"] for t in tasks}
    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t
    for left, right, _ in edges:
        a, b = find(left), find(right)
        parent[max(a, b)] = min(a, b)
    return {t: find(t) for t in parent}, Counter(e[2] for e in edges)


def paired_cluster(delta, components, resamples=10000, seed=SEED):
    delta = np.asarray(delta, dtype=float)
    if len(delta) != len(components) or not len(delta) or not np.all(np.isfinite(delta)):
        raise ValueError("invalid paired arrays")
    ids = sorted(set(components))
    groups = [[i for i, c in enumerate(components) if c == g] for g in ids]
    sums = np.array([delta[g].sum() for g in groups])
    sizes = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(ids), size=(resamples, len(ids)))
    boots = sums[draw].sum(axis=1) / sizes[draw].sum(axis=1)
    signs = rng.choice([-1, 1], size=(resamples, len(ids)))
    perm = (signs * sums).sum(axis=1) / len(delta)
    observed = float(delta.mean())
    # Numeric guard only for permutation equality, never for rank reversal decisions.
    exceed = int(np.count_nonzero(np.abs(perm) >= abs(observed) - 1e-14))
    return {"difference": observed, "ci95_percentile": np.quantile(boots, [.025, .975]).tolist(),
            "p_two_sided": (exceed + 1) / (resamples + 1), "exceedances": exceed,
            "resamples": resamples, "seed": seed, "n_tasks": len(delta), "n_components": len(ids),
            "component_sizes": dict(zip(ids, sizes.tolist()))}


def holm(ps):
    order = np.argsort(ps, kind="stable")
    result = np.empty(len(ps))
    running = 0.
    for k, i in enumerate(order):
        running = max(running, (len(ps)-k) * ps[i])
        result[i] = min(1., running)
    return result.tolist()


def subsample(seed, normalized, fraction, rng):
    # Uniform sample of ALL judged records, including negatives, not positive-only thinning.
    selected = rng.permutation(len(normalized))[:math.floor(fraction * len(normalized))]
    gold = {t: set(ds) for t, ds in seed.items()}
    positives = 0
    for i in selected:
        row = normalized[i]
        if row["relevant"]:
            gold[row["task_id"]].add(row["doc_id"])
            positives += 1
    return {t: sorted(ds) for t, ds in gold.items()}, len(selected), positives


def input_paths(root):
    b = root / "results/baselines"
    return ([root / f"data/benchmark/{s}.jsonl" for s in ("train", "dev", "test")]
            + [root / "data/processed/corpus.jsonl", root / "data/processed/identity_repair_v1/report.json"]
            + [b / f"{m}_test.jsonl" for m in METHODS]
            + [b / name for name in ("gold_adjudication.jsonl", "expanded_gold.json", "scores_test_adjudicated.json")]
            + [ROOT / name for name in ("scripts/44_analyze_historical_qrel_sensitivity.py",
               "tests/test_qrel_sensitivity.py", "docs/QREL_SENSITIVITY_METHODS.md",
               "src/evaluation/strict_metrics.py", "scripts/38_make_component_disjoint_splits.py",
               "src/corpus/identity.py", "scripts/31_submission_audit.py")])


def duplicate_corpus_records(corpus_rows):
    by_id = defaultdict(list)
    for line, row in enumerate(corpus_rows, 1):
        by_id[row.get("doc_id")].append((line, row))
    result = []
    for d, records in sorted(by_id.items()):
        if len(records) < 2:
            continue
        fields = sorted(set().union(*(r.keys() for _, r in records)))
        result.append({"doc_id": d, "physical_rows": [i for i, _ in records],
                       "differing_fields": [f for f in fields if len({json.dumps(r.get(f), sort_keys=True) for _, r in records}) > 1],
                       "records": [{"physical_row": i, **{k: r.get(k) for k in ("pmid", "pmcid", "title", "has_fulltext")}}
                                   for i, r in records]})
    return result


def audit_inputs(root):
    """Inventory errors without resolving duplicate identities or emitting retrieval metrics."""
    errors, audit = [], {"version": VERSION_ANALYSIS, "release_blocked": True}
    def check(label, function):
        try:
            return function()
        except (ValueError, KeyError, TypeError) as e:
            errors.append({"source": label, "error": str(e)})
            return None
    b = root / "results/baselines"
    corpus_rows = rows(root / "data/processed/corpus.jsonl")
    duplicates = duplicate_corpus_records(corpus_rows)
    # Membership only: this set never selects a record or authorizes evaluation.
    corpus_ids = {r["doc_id"] for r in corpus_rows}
    dup_ids = {r["doc_id"] for r in duplicates}
    audit["corpus"] = {"rows": len(corpus_rows), "distinct_ids": len(corpus_ids),
                       "duplicate_id_count": len(duplicates), "excess_rows": len(corpus_rows)-len(corpus_ids),
                       "duplicate_records": duplicates,
                       "note": "ID set used only to detect unknown references; no row selected or identities merged."}
    if duplicates:
        errors.append({"source": "data/processed/corpus.jsonl", "error": "duplicate corpus IDs with conflicting records",
                       "doc_ids": sorted(dup_ids), "detail": "corpus.duplicate_records gives physical rows, PMIDs, titles and differing fields"})
    splits = {s: rows(root / f"data/benchmark/{s}.jsonl") for s in ("train", "dev", "test")}
    all_tasks = sum(splits.values(), [])
    check("all tasks", lambda: index_rows(all_tasks, "all splits"))
    tasks = check("test tasks", lambda: index_rows(splits["test"], "test"))
    if tasks is None:
        audit["errors"] = errors
        return audit
    task_refs = []
    for s, ts in splits.items():
        for t in ts:
            for field in ("supporting_doc_ids", "hard_negative_doc_ids"):
                check(f"{s}/{t['task_id']}/{field}", lambda t=t, field=field: validate_docs(t.get(field), corpus_ids, f"{t['task_id']}/{field}"))
                for d in t.get(field, []):
                    if d in dup_ids:
                        task_refs.append({"split": s, "task_id": t["task_id"], "field": field, "doc_id": d})
    seed = {t: r["supporting_doc_ids"] for t, r in tasks.items()}
    supported = sorted(t for t in tasks if seed[t])
    empty = sorted(set(tasks)-set(supported))
    if len(tasks) != 125 or len(supported) != 108:
        errors.append({"source": "test tasks", "error": "wrong expected denominator", "actual": len(tasks), "supported": len(supported)})
    audit["tasks"] = {"split_sizes": {s: len(ts) for s, ts in splits.items()}, "test_count": len(tasks),
                      "supported_count": len(supported), "empty_count": len(empty), "empty_task_ids": empty,
                      "supported_task_ids": supported, "seed_qrel_pairs": sum(map(len, seed.values())),
                      "duplicate_corpus_id_references": task_refs}
    audit["runs"] = {}
    all_run_refs = []
    for m in METHODS:
        run_rows = rows(b / f"{m}_test.jsonl")
        rr = check(m, lambda: index_rows(run_rows, m))
        if rr is None:
            continue
        check(m, lambda: exact_keys(rr, tasks, m))
        refs, empty_rankings = [], []
        for t, row in rr.items():
            docs = row.get("retrieved_docs")
            check(f"{m}/{t}", lambda docs=docs, t=t: validate_docs(docs, corpus_ids, f"{m}/{t}"))
            if isinstance(docs, list):
                if not docs:
                    empty_rankings.append(t)
                refs.extend({"task_id": t, "doc_id": d, "rank": rank} for rank, d in enumerate(docs, 1) if d in dup_ids)
        all_run_refs.extend({"method": m, **ref} for ref in refs)
        audit["runs"][m] = {"row_count": len(run_rows), "missing_ids": sorted(set(tasks)-set(rr)),
                            "unknown_ids": sorted(set(rr)-set(tasks)), "empty_rankings": sorted(empty_rankings),
                            "duplicate_corpus_id_reference_count": len(refs), "affected_task_count": len({x['task_id'] for x in refs}),
                            "duplicate_corpus_id_references": refs}
    raw = rows(b / "gold_adjudication.jsonl")
    if len(raw) != 3240:
        errors.append({"source": "judgments", "error": f"expected 3240; actual {len(raw)}"})
    reconstructed = check("judgment reconstruction (ID-level audit only)", lambda: reconstruct(seed, raw, corpus_ids, 30))
    audit["judgments"] = {"raw_count": len(raw), "duplicate_corpus_id_references":
                          [{"physical_row": i, "task_id": r.get("task_id"), "doc_id": r.get("doc_id"), "judgment": r.get("judgment")}
                           for i, r in enumerate(raw, 1) if r.get("doc_id") in dup_ids]}
    if reconstructed is not None:
        expanded, normalized = reconstructed
        counts = Counter(r["relevant"] for r in normalized)
        audit["judgments"].update({"relevant": counts[True], "not_relevant": counts[False],
                                  "per_task_counts": dict(Counter(r["task_id"] for r in raw)),
                                  "expanded_qrel_pairs": sum(map(len, expanded.values())),
                                  "expanded_pairs_on_duplicate_corpus_ids": [dict(task_id=t, doc_id=d) for t, ds in expanded.items() for d in ds if d in dup_ids]})
        stored = read(b / "expanded_gold.json")
        check("stored expanded schema", lambda: exact_keys(stored, supported, "supported-only expanded schema"))
        diff = []
        for t, ds in stored.items():
            check(f"expanded/{t}", lambda ds=ds, t=t: validate_docs(ds, corpus_ids, f"expanded/{t}"))
            if t in expanded and set(ds) != set(expanded[t]):
                diff.append({"task_id": t, "stored_only": sorted(set(ds)-set(expanded[t])),
                             "reconstructed_only": sorted(set(expanded[t])-set(ds)),
                             "raw_evidence": [{"physical_row": i, **r} for i, r in enumerate(raw, 1)
                                              if r["task_id"] == t and r["doc_id"] in (set(ds) ^ set(expanded[t]))]})
        audit["judgments"]["stored_expanded_set_mismatches"] = diff
        if diff:
            errors.append({"source": "stored expanded", "error": "set mismatch", "details": diff})
    stored_scores = read(b / "scores_test_adjudicated.json")
    audit["stored_scores"] = {"methods": sorted(stored_scores), "primary_missing": sorted(set(METHODS)-set(stored_scores)),
                              "nonprimary": sorted(set(stored_scores)-set(METHODS)), "comparison": "not_performed_if_validation_errors; no recomputed scores"}
    identity = read(root / "data/processed/identity_repair_v1/report.json")
    audit["identity_candidate_scope_only"] = {k: identity[k] for k in ("version", "release_blocked", "raw_front_pmid_disagreement_records",
            "fulltext_records_removed", "original", "candidate", "quarantine", "task_quarantine_reasons")}
    audit["scope_counts"] = {"run_duplicate_identity_references": len(all_run_refs),
                             "distinct_test_tasks_with_run_duplicate_identity_references": len({r['task_id'] for r in all_run_refs}),
                             "seed_or_negative_reference_count_all_splits": len(task_refs),
                             "judgment_duplicate_identity_reference_count": len(audit['judgments']['duplicate_corpus_id_references'])}
    audit["errors"] = errors
    audit["status"] = "fail" if errors else "input_validation_passed"
    return audit


def compute(root, resamples, repeats):
    b = root / "results/baselines"
    paths = [root / f"data/benchmark/{s}.jsonl" for s in ("train", "dev", "test")]
    paths += [root / "data/processed/corpus.jsonl", root / "data/processed/identity_repair_v1/report.json"]
    paths += [b / f"{m}_test.jsonl" for m in METHODS]
    paths += [b / name for name in ("gold_adjudication.jsonl", "expanded_gold.json", "scores_test_adjudicated.json")]
    paths += [ROOT / name for name in ("scripts/44_analyze_historical_qrel_sensitivity.py",
              "tests/test_qrel_sensitivity.py", "docs/QREL_SENSITIVITY_METHODS.md",
              "src/evaluation/strict_metrics.py", "scripts/38_make_component_disjoint_splits.py",
              "src/corpus/identity.py", "scripts/31_submission_audit.py")]
    hashes = {str(p): digest(p) for p in paths}
    split_rows = {s: rows(root / f"data/benchmark/{s}.jsonl") for s in ("train", "dev", "test")}
    all_tasks = sum(split_rows.values(), [])
    index_rows(all_tasks, "all train/dev/test tasks")
    tasks = index_rows(split_rows["test"], "test")
    if len(tasks) != 125:
        raise ValueError(f"expected 125 test tasks; actual={len(tasks)} IDs={sorted(tasks)}")
    corpus_rows = rows(root / "data/processed/corpus.jsonl")
    corpus_ids = strings([r["doc_id"] for r in corpus_rows], "corpus IDs", nonempty=True)
    corpus = {r["doc_id"]: r for r in corpus_rows}
    for t in all_tasks:
        for key in ("supporting_doc_ids", "hard_negative_doc_ids"):
            validate_docs(t.get(key), corpus, f"{t['task_id']}/{key}")
    seed = {t: r["supporting_doc_ids"] for t, r in tasks.items()}
    supported = sorted(t for t in tasks if seed[t])
    empty = sorted(set(tasks) - set(supported))
    if len(supported) != 108:
        raise ValueError(f"expected 108 supported; actual={len(supported)} IDs={supported}")
    runs, coverage = {}, {}
    for m in METHODS:
        rr = index_rows(rows(b / f"{m}_test.jsonl"), m)
        exact_keys(rr, tasks, m)
        runs[m] = {t: validate_docs(rr[t].get("retrieved_docs"), corpus, f"{m}/{t}") for t in sorted(tasks)}
        coverage[m] = {"expected": 125, "actual": len(rr), "missing": [], "empty_rankings":
                       [t for t, ds in runs[m].items() if not ds],
                       "ranking_length_counts": dict(Counter(map(len, runs[m].values())))}
    raw = rows(b / "gold_adjudication.jsonl")
    if len(raw) != 3240:
        raise ValueError(f"expected 3240 judgments, got {len(raw)}")
    expanded, normalized = reconstruct(seed, raw, corpus, expected_per_task=30)
    stored = read(b / "expanded_gold.json")
    # Historical expanded file has ONLY the 108 supported keys; this is explicit, not intersection.
    exact_keys(stored, supported, "stored expanded (supported-only schema)")
    differences = []
    for t in supported:
        validate_docs(stored[t], corpus, f"stored expanded/{t}")
        if set(stored[t]) != set(expanded[t]):
            differences.append({"task_id": t, "stored_only": sorted(set(stored[t])-set(expanded[t])),
                                "reconstructed_only": sorted(set(expanded[t])-set(stored[t]))})
    if differences:
        raise ValueError(f"expanded qrel mismatch: {differences}")
    comp, edge_counts = component_map(all_tasks)
    component_members = defaultdict(list)
    placement = {t["task_id"]: s for s, ts in split_rows.items() for t in ts}
    for t, c in comp.items():
        component_members[c].append({"task_id": t, "split": placement[t], "supported_test": t in supported})
    per_task, arrays, means = [], {}, {}
    for regime, qrels in (("seed", seed), ("expanded", expanded)):
        arrays[regime], means[regime] = {}, {}
        for m in METHODS:
            records = {t: doc_metrics(runs[m][t], qrels[t]) for t in sorted(tasks)}
            arrays[regime][m] = np.array([[records[t][metric] for metric in METRICS] for t in supported])
            # math.fsum reduces order-dependent artificial ties; never round means for ranks.
            means[regime][m] = {metric: math.fsum(arrays[regime][m][:, j]) / len(supported)
                                for j, metric in enumerate(METRICS)}
            per_task.extend({"regime": regime, "method": m, "task_id": t, "output_present": True,
                             "status": "evaluated" if qrels[t] else "empty_gold", "gold_count": len(qrels[t]),
                             "retrieved_count": len(runs[m][t]), "metrics": records[t]} for t in sorted(tasks))
    score_stored = read(b / "scores_test_adjudicated.json")
    comparisons = []
    for m in METHODS:
        if m not in score_stored:
            comparisons.append({"method": m, "status": "not_present_in_stored_scores"})
            continue
        for regime, key in (("seed", "original"), ("expanded", "expanded")):
            for metric in METRICS[:3]:
                old = score_stored[m][key][metric.lower()]
                new = means[regime][m][metric]
                comparisons.append({"method": m, "regime": regime, "metric": metric, "stored": old,
                                    "recomputed": new, "delta": new-old, "matches_four_decimals": round(new, 4) == old})
    bad_scores = [x for x in comparisons if x.get("matches_four_decimals") is False]
    if bad_scores:
        raise ValueError(f"stored score mismatch (four-decimal comparison): {bad_scores}")
    rankings = {scope: {metric: rank_comparison({m: means["seed"][m][metric] for m in names},
                   {m: means["expanded"][m][metric] for m in names}, names) for metric in METRICS}
                for scope, names in (("primary_original_pool_10", POOL), ("supplemental_all_12_includes_outside_pool", METHODS))}
    infer = []
    j = METRICS.index(PRIMARY_METRIC)
    for regime in ("seed", "expanded"):
        for comparator in ("bm25", "splade", "hybrid"):
            result = paired_cluster(arrays[regime]["agent"][:, j] - arrays[regime][comparator][:, j],
                                    [comp[t] for t in supported], resamples)
            infer.append({"regime": regime, "comparator": comparator, "metric": PRIMARY_METRIC, **result})
    for r, p in zip(infer, holm([r["p_two_sided"] for r in infer])):
        r["p_holm_six"] = p
    curve_raw = []
    for repeat in range(repeats):
        for fraction in FRACTIONS:
            gold, n, positive = subsample(seed, normalized, fraction, np.random.default_rng(SEED + repeat))
            for m in METHODS:
                scores = [doc_metrics(runs[m][t], gold[t]) for t in supported]
                curve_raw.append({"repeat": repeat, "rng_seed": SEED+repeat, "fraction": fraction, "method": m,
                                  "n_judged_sampled": n, "n_positive_sampled": positive, "n_tasks": 108,
                                  "n_qrels": sum(len(gold[t]) for t in supported),
                                  "metrics": {k: math.fsum(r[k] for r in scores)/108 for k in METRICS}})
    curve = []
    for fraction in FRACTIONS:
        for m in METHODS:
            rr = [r for r in curve_raw if r["fraction"] == fraction and r["method"] == m]
            for metric in METRICS:
                values = [r["metrics"][metric] for r in rr]
                curve.append({"fraction": fraction, "method": m, "metric": metric, "repeats": repeats,
                              "mean": math.fsum(values)/repeats, "sd": float(np.std(values, ddof=1)),
                              "interval95_subsampling": np.quantile(values, [.025, .975]).tolist(),
                              "min": min(values), "max": max(values), "n_tasks": 108})
                if fraction in (0, 1) and any(v != means["seed" if fraction == 0 else "expanded"][m][metric] for v in values):
                    raise ValueError(f"curve endpoint mismatch: {fraction}/{m}/{metric}")
    strata = defaultdict(list)
    for t in supported:
        flags = [corpus[d].get("has_fulltext") for d in seed[t]]
        if any(type(f) is not bool for f in flags):
            raise ValueError(f"unrecognized historical has_fulltext: {t}/{flags}")
        strata["all" if all(flags) else "mixed" if any(flags) else "none"].append(t)
    strata_scores = []
    for stratum in ("all", "mixed", "none"):
        ids = strata[stratum]
        for regime in ("seed", "expanded"):
            for m in METHODS:
                indices = [supported.index(t) for t in ids]
                strata_scores.append({"stratum": stratum, "regime": regime, "method": m, "n_tasks": len(ids),
                                      "metrics": {k: math.fsum(arrays[regime][m][indices, i])/len(ids) if ids else None
                                                  for i, k in enumerate(METRICS)}})
    identity = read(root / "data/processed/identity_repair_v1/report.json")
    identity_scope = {k: identity[k] for k in ("version", "release_blocked", "raw_front_pmid_disagreement_records",
                      "fulltext_records_removed", "original", "candidate", "quarantine", "task_quarantine_reasons")}
    report = {"version": VERSION_ANALYSIS, "status": "historical_diagnostic_complete", "release_blocked": True,
              "warning": WARNING, "strict_metrics_version": VERSION, "primary_inference_metric": PRIMARY_METRIC,
              "denominators": {"test_universe": 125, "supported_each_regime_each_method": 108, "empty_gold": 17,
                               "seed_qrel_pairs": sum(map(len, seed.values())), "expanded_qrel_pairs": sum(map(len, expanded.values())),
                               "raw_judgments": len(raw), "judgment_counts": dict(Counter(r["relevant"] for r in normalized)),
                               "judgments_per_supported_task": 30, "empty_gold_task_ids": empty,
                               "supported_task_ids": supported, "split_sizes": {s: len(ts) for s, ts in split_rows.items()}},
              "validation": {"coverage": coverage, "expanded_qrel_mismatches": differences,
                             "stored_expanded_schema": "108 supported-only keys; 17 verified seed-empty tasks have no judgments",
                             "stored_score_comparisons": comparisons,
                             "stored_nonprimary_methods_not_evaluated": sorted(set(score_stored)-set(METHODS)),
                             "unknown_ids_duplicate_rows_ranked_docs_relevance_records": "passed; fail-closed checks"},
              "pool_systems": list(POOL), "outside_pool_supplemental": ["rm3", "agent_single_step"],
              "means": means, "rankings": rankings, "paired_inference": infer,
              "components": {"all_split_count": len(set(comp.values())), "supported_test_count": len({comp[t] for t in supported}),
                             "largest_all_split_component": max(map(len, component_members.values())),
                             "cross_split_components": sum(len({t["split"] for t in group}) > 1 for group in component_members.values()),
                             "relation_edge_counts": dict(edge_counts)},
              "historical_fulltext_strata": {k: {"n_tasks": len(strata[k]), "task_ids": strata[k]} for k in ("all", "mixed", "none")},
              "identity_candidate_scope_only": identity_scope}
    environment = {"executable": sys.executable, "python": sys.version, "platform": platform.platform(),
                   "numpy": np.__version__, "scipy": scipy.__version__, "rng": "numpy.PCG64/default_rng",
                   "numpy_file_sha256": digest(np.__file__), "scipy_file_sha256": digest(scipy.__file__)}
    manifest = {"version": VERSION_ANALYSIS, "release_blocked": True, "inputs_and_code_sha256": hashes,
                "environment": environment, "environment_sha256": hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest(),
                "parameters": {"seed": SEED, "resamples": resamples, "repeats": repeats, "fractions": FRACTIONS,
                               "component_lexical_jaccard_threshold": .85, "primary_metric": PRIMARY_METRIC},
                "command": [sys.executable, "scripts/44_analyze_historical_qrel_sensitivity.py", "--resamples", str(resamples), "--repeats", str(repeats)],
                "network_api_install_paid_jobs_git": "none"}
    changed = [p for p, h in hashes.items() if digest(p) != h]
    if changed:
        raise ValueError(f"inputs/code changed during analysis: {changed}")
    return report, manifest, per_task, expanded, normalized, dict(component_members), curve_raw, curve, strata_scores


def report_markdown(r):
    lines = ["# Historical qrel sensitivity diagnostic", "", "**release_blocked: true**", "", WARNING, "",
             "## Scope and validation", "", f"125 test rows per method; 108 supported, 17 empty-qrel tasks excluded explicitly from metric means. "
             f"Seed pairs: {r['denominators']['seed_qrel_pairs']}; expanded pairs: {r['denominators']['expanded_qrel_pairs']}. "
             "All 3,240 verdicts reconstructed; stored expanded qrels match exactly as sets. Stored available recall scores match at four decimals.", "",
             "Original-pool primary correlation uses 10 systems. The 12-system correlation is supplemental and adds RM3 and Agent T=1, which were outside the pool.", "",
             "## Unrounded-mean results (display rounded only)", "", "| Method | Seed R@20 | Expanded R@20 |", "|---|---:|---:|"]
    for m in METHODS:
        lines.append(f"| {m} | {r['means']['seed'][m]['Recall@20']:.9f} | {r['means']['expanded'][m]['Recall@20']:.9f} |")
    lines += ["", "## Rank sensitivity", "", "| Scope / metric | Kendall tau-b | Spearman | Strict reversals |", "|---|---:|---:|---:|"]
    for scope, metrics in r["rankings"].items():
        for metric, v in metrics.items():
            lines.append(f"| {scope} / {metric} | {v['kendall_tau_b']:.9f} | {v['spearman']:.9f} | {len(v['strict_reversals'])} |")
    lines += ["", "All strict reversal pairs and exact differences, tie counts, and average ranks are in `analysis.json`.", "",
              "## Paired cluster inference", "", f"Prespecified metric: {PRIMARY_METRIC}; six-test Holm family (3 comparators × 2 qrel regimes). "
              f"Components constructed over all train/dev/test: {r['components']}. Task-weighted differences use exactly the same 108 tasks.", "",
              "| Regime / agent minus | Difference | Cluster 95% percentile CI | MC p | Holm p |", "|---|---:|---|---:|---:|"]
    for v in r["paired_inference"]:
        lo, hi = v["ci95_percentile"]
        lines.append(f"| {v['regime']} / {v['comparator']} | {v['difference']:.9f} | [{lo:.9f}, {hi:.9f}] | {v['p_two_sided']:.9f} | {v['p_holm_six']:.9f} |")
    lines += ["", "## Completeness and historical metadata", "", "`curve_summary.json` contains every method × metric × fraction with 200-repeat mean, SD, central 95% subsampling interval and range (default). "
              "`curve_replicates.jsonl` retains every replicate. Sampling is uniform without replacement over all judged pooled labels, including negatives, "
              "with seeds retained; final questions and saved rankings never change. Endpoints are checked against exact seed/expanded means. "
              "Intervals describe artificial label removal, NOT uncertainty in true corpus relevance.", "",
              "Fulltext strata use original source flags for seed-support documents only; unverified historical metadata, not a valid fulltext claim: "
              + str({k: v['n_tasks'] for k, v in r['historical_fulltext_strata'].items()}), "",
              "## Identity scope and limitations", "", "Identity candidate report was read only for counts and scope; no mappings, repaired corpus, or candidate tasks were used. "
              + f"Reported PMID/XML disagreements: {r['identity_candidate_scope_only']['raw_front_pmid_disagreement_records']}. "
              "All primary historical results remain compromised. Full candidate scope counts are copied in analysis.json.", "",
              "Qrels are synthetic/automated and pool-biased, seeds are forcibly retained, unjudged documents count as nonrelevant operationally, "
              "and known component rules cannot guarantee semantic independence. Cluster sign-flip inference assumes symmetric independent component-level differences; "
              "these are conditional diagnostics, not randomized causal effects. Unequal search budgets, exposed tasks, identity failures, and unverified evidence remain unresolved. "
              "No humans, network/API calls, installations, paid jobs, git actions, paper edits, or overall-readiness gate changes were performed.", "",
              "## Source paths and reproducibility", "", "See `manifest.json` for absolute source/code paths, SHA-256 hashes, environment and parameters. "
              "Sources: `data/benchmark/{train,dev,test}.jsonl`, `data/processed/corpus.jsonl`, "
              "`data/processed/identity_repair_v1/report.json`, the 12 `results/baselines/*_test.jsonl` files, "
              "`results/baselines/{gold_adjudication.jsonl,expanded_gold.json,scores_test_adjudicated.json}`. "
              "Methods: `docs/QREL_SENSITIVITY_METHODS.md`. Strict evaluator: `src/evaluation/strict_metrics.py`.", ""]
    return "\n".join(lines)


FORENSIC_LABEL = "historical_string_label_diagnostic_not_validated_biomedical_relevance"


def forensic_envelope(**values):
    # Safety fields cannot be overridden by caller-provided status/provenance.
    return {**values, "interpretation": FORENSIC_LABEL, "release_blocked": True,
            "release_ready": False, "input_validation_status": "fail",
            "status": "forensic_token_diagnostic_only"}


def forensic_compute(root, resamples=10000):
    """Opaque string arithmetic. Never parses/indexes/dereferences corpus records."""
    frozen = root / "results/readiness/qrel_sensitivity/v1_final"
    old_manifest = read(frozen / "manifest.json")
    audit = read(frozen / "INPUT_AUDIT.json")
    failure = read(frozen / "FAILURE.json")
    if audit.get("status") != "fail" or not audit.get("errors") or failure.get("release_blocked") is not True:
        raise ValueError("forensic mode requires the preserved failed strict audit")
    for name, h in old_manifest["outputs_sha256"].items():
        if digest(frozen / name) != h:
            raise ValueError(f"strict failure artifact changed: {name}")
    historical = {p: h for p, h in old_manifest["inputs_and_code_sha256"].items()
                  if "/data/" in p or "/results/baselines/" in p}
    for p, h in historical.items():
        if digest(p) != h:
            raise ValueError(f"historical input differs from unresolved audit: {p}")
    b = root / "results/baselines"
    splits = {s: rows(root / f"data/benchmark/{s}.jsonl") for s in ("train", "dev", "test")}
    all_tasks = sum(splits.values(), [])
    index_rows(all_tasks, "all token tasks")
    tasks = index_rows(splits["test"], "token test")
    exact_keys(tasks, audit["tasks"]["supported_task_ids"] + audit["tasks"]["empty_task_ids"], "audited token universe")
    seed = {t: strings(r["supporting_doc_ids"], f"seed/{t}") for t, r in tasks.items()}
    supported = sorted(t for t in tasks if seed[t])
    exact_keys(supported, audit["tasks"]["supported_task_ids"], "supported token tasks")
    runs = {}
    for m in METHODS:
        rr = index_rows(rows(b / f"{m}_test.jsonl"), m)
        exact_keys(rr, tasks, m)
        runs[m] = {t: strings(rr[t]["retrieved_docs"], f"{m}/{t}") for t in tasks}
    raw = rows(b / "gold_adjudication.jsonl")
    # Pure token domain; membership/identity failures remain embedded and are NOT waived.
    tokens = {d for ds in seed.values() for d in ds}
    for row in raw:
        strings([row["doc_id"]], "judgment token")
        tokens.add(row["doc_id"])
    reconstructed, normalized = reconstruct(seed, raw, tokens, 30)
    stored_raw = read(b / "expanded_gold.json")
    exact_keys(stored_raw, supported, "stored supported-only token schema")
    stored = {t: strings(stored_raw[t], f"stored/{t}") if seed[t] else [] for t in tasks}
    qrels = {"seed": seed, "stored_expanded": stored, "reconstructed_expanded": reconstructed}
    per_task, means, matrix = [], {}, {}
    for regime, gold in qrels.items():
        means[regime], matrix[regime] = {}, {}
        for m in METHODS:
            scores = {t: doc_metrics(runs[m][t], gold[t]) for t in sorted(tasks)}
            matrix[regime][m] = np.array([[scores[t][metric] for metric in METRICS] for t in supported])
            means[regime][m] = {metric: math.fsum(scores[t][metric] for t in supported)/len(supported) for metric in METRICS}
            per_task.extend(forensic_envelope(regime=regime, method=m, task_id=t,
                            metric_status="evaluated" if gold[t] else "empty_gold", gold_count=len(gold[t]),
                            metrics=scores[t]) for t in sorted(tasks))
    rankings = {}
    for left, right in (("seed", "stored_expanded"), ("seed", "reconstructed_expanded"),
                        ("stored_expanded", "reconstructed_expanded")):
        key = left + "_vs_" + right
        rankings[key] = {metric: rank_comparison({m: means[left][m][metric] for m in POOL},
                                                {m: means[right][m][metric] for m in POOL}, POOL)
                         for metric in METRICS}
    diffs = [{"task_id": t, "stored_only": sorted(set(stored[t])-set(reconstructed[t])),
              "reconstructed_only": sorted(set(reconstructed[t])-set(stored[t]))}
             for t in sorted(tasks) if set(stored[t]) != set(reconstructed[t])]
    effects = []
    for diff in diffs:
        t = diff["task_id"]
        for m in METHODS:
            hits = {r: len(set(runs[m][t][:20]) & set(qrels[r][t])) for r in ("stored_expanded", "reconstructed_expanded")}
            fractions = {r: Fraction(hits[r], len(qrels[r][t])) for r in hits}
            delta = fractions["reconstructed_expanded"] - fractions["stored_expanded"]
            effects.append({"task_id": t, "method": m, "stored_hits20": hits["stored_expanded"],
                            "reconstructed_hits20": hits["reconstructed_expanded"],
                            "stored_qrels": len(stored[t]), "reconstructed_qrels": len(reconstructed[t]),
                            "added_token_ranks": {d: runs[m][t].index(d)+1 if d in runs[m][t] else None for d in diff["reconstructed_only"]},
                            "stored_task_r20_exact": str(fractions["stored_expanded"]),
                            "reconstructed_task_r20_exact": str(fractions["reconstructed_expanded"]),
                            "task_delta_exact": str(delta), "mean_delta_exact": str(delta/len(supported)),
                            "mean_delta": float(delta/len(supported))})
    comp, edges = component_map(all_tasks)
    infer = []
    for regime in ("seed", "reconstructed_expanded"):
        for comparator in ("bm25", "splade", "hybrid"):
            j = METRICS.index("Recall@20")
            result = paired_cluster(matrix[regime]["agent"][:, j]-matrix[regime][comparator][:, j],
                                    [comp[t] for t in supported], resamples)
            infer.append({"regime": regime, "comparator": comparator, "metric": "Recall@20", **result})
    for result, p in zip(infer, holm([r["p_two_sided"] for r in infer])):
        result["p_holm_six"] = p
    score_check = []
    old_scores = read(b / "scores_test_adjudicated.json")
    for m in METHODS:
        if m not in old_scores:
            score_check.append({"method": m, "comparison": "not_present_in_saved_score_file"})
            continue
        for regime, old_key in (("seed", "original"), ("stored_expanded", "expanded"), ("reconstructed_expanded", "expanded")):
            for metric in METRICS[:3]:
                old = old_scores[m][old_key][metric.lower()]
                new = means[regime][m][metric]
                score_check.append({"method": m, "regime": regime, "metric": metric, "saved_rounded": old,
                                    "token_mean": new, "agrees_at_four_decimals": round(new, 4) == old})
    report = forensic_envelope(version="forensic-token-v1", unresolved_input_audit=audit, original_failure=failure,
               corpus_record_access="none; file bytes hashed only; no corpus record indexed or dereferenced",
               denominator={"all_test_tasks": len(tasks), "supported_each_mean": len(supported),
                            "empty_gold": len(tasks)-len(supported), "split_sizes": {s: len(ts) for s, ts in splits.items()},
                            "qrel_pairs": {r: sum(map(len, qs.values())) for r, qs in qrels.items()}},
               pool_systems=POOL, supplemental_outside_pool=["rm3", "agent_single_step"], means=means,
               recall20_exact_rational_means={regime: {m: str(sum((Fraction(len(set(runs[m][t][:20]) & set(gold[t])), len(gold[t]))
                        for t in supported), Fraction())/len(supported)) for m in METHODS} for regime, gold in qrels.items()},
               primary_10_system_rank_comparisons=rankings, qrel_differences=diffs, missing_pair_effects=effects,
               paired_cluster_token_inference=infer, saved_score_comparisons=score_check,
               components={"all_split_count": len(set(comp.values())), "supported_test_count": len({comp[t] for t in supported}),
                           "all_task_to_component": comp, "relation_edge_counts": dict(edges)},
               limitations=["Opaque token equality does not establish biomedical relevance or correct document identity.",
                            "Both original strict input failures remain unresolved; this is not a repaired benchmark.",
                            "No fulltext strata computed and no fulltext claim; no corpus content dereferenced.",
                            "Pool-biased automated labels, forced-positive seeds and fixed historical rankings; no causal system claim.",
                            "Cluster tests assume symmetric independent component differences; known relations do not ensure semantic independence.",
                            "Completeness curve not run in this focused continuation; cluster inference is token-level only."])
    return report, per_task, qrels


def run_forensic(root, output, resamples):
    frozen = root / "results/readiness/qrel_sensitivity/v1_final"
    sources = input_paths(root) + sorted(frozen.iterdir())
    before = {str(p): digest(p) for p in sources}
    report, per_task, qrels = forensic_compute(root, resamples)
    if any(digest(p) != h for p, h in before.items()):
        raise ValueError("forensic source changed during arithmetic")
    write(output / "analysis.json", report)
    write(output / "qrel_versions.json", forensic_envelope(versions=qrels))
    (output / "per_task_metrics.jsonl").write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False)+"\n" for r in per_task))
    lines = ["# Historical opaque-token arithmetic", "", f"**{FORENSIC_LABEL}**", "",
             "**release_blocked: true; release_ready: false; input_validation_status: fail.**", "",
             "This is not validated biomedical relevance, a biological ranking claim, or a repaired benchmark. "
             "The strict v1_final failure artifacts remain immutable. Corpus records were not parsed, selected, indexed or dereferenced; only file bytes were hashed.", "",
             "## Token Recall@20 means", "", "All means use the same 108 supported tasks; all 125 test rows are retained in per-task outputs (17 null-metric empty-qrel rows per method/regime). "
             "Seed/stored/reconstructed pair counts are 264/1369/1370. Display rounding is not used for ranks.", "",
             "| Method | Seed | Stored expanded | Reconstructed expanded |", "|---|---:|---:|---:|"]
    for m in METHODS:
        lines.append("| " + m + " | " + " | ".join(f"{report['means'][r][m]['Recall@20']:.12f}" for r in qrels) + " |")
    lines += ["", "## Ten original-pool systems: token rank sensitivity", "",
              "RM3 and Agent T=1 means are supplemental outside-pool values; they are excluded from these correlations.", "",
              "| Comparison (R20) | Kendall tau-b | Spearman | Strict reversals |", "|---|---:|---:|---:|"]
    for name, metrics in report["primary_10_system_rank_comparisons"].items():
        r = metrics["Recall@20"]
        lines.append(f"| {name} | {r['kendall_tau_b']:.12f} | {r['spearman']:.12f} | {len(r['strict_reversals'])} |")
        lines += [""] + [f"- {name}: `{p['left']}` vs `{p['right']}`; left-minus-right differences {p['seed_difference']!r} → {p['expanded_difference']!r}."
                          for p in r["strict_reversals"]] + [""]
    lines += ["## Exact missing-pair effect", "", "Raw judgment row 1337 adds token `37307965` to `multihop_0189`: stored 9 labels → reconstructed 10. "
              "Full per-method task/mean effects and added-token ranks are in analysis.json as exact rational strings.", "",
              "| Method | Stored task R20 | Reconstructed task R20 | Exact mean delta (108 tasks) |", "|---|---:|---:|---:|"]
    for e in report["missing_pair_effects"]:
        lines.append(f"| {e['method']} | {e['stored_task_r20_exact']} | {e['reconstructed_task_r20_exact']} | {e['mean_delta_exact']} |")
    lines += ["", "## Conditional cluster token inference", "", "Recall@20, 10,000 resamples by default; paired task-weighted cluster bootstrap, "
              "two-sided cluster sign-flip with finite Monte Carlo correction, Holm across six seed/reconstructed × agent-minus-BM25/SPLADE/Hybrid contrasts. "
              "Components use all original train/dev/test tasks. These are not validated biomedical or causal inference results.", "",
              "| Regime / comparator | Difference | 95% cluster interval | MC p | Holm p |", "|---|---:|---|---:|---:|"]
    for r in report["paired_cluster_token_inference"]:
        lines.append(f"| {r['regime']} / {r['comparator']} | {r['difference']:.12f} | {r['ci95_percentile']} | {r['p_two_sided']:.12f} | {r['p_holm_six']:.12f} |")
    lines += ["", "## Unresolved provenance and scope", "", "analysis.json embeds the original failed input audit and FAILURE.json without changing either: "
              "10 colliding corpus tokens (13 excess records), and the one raw/stored qrel discrepancy. manifest.json hashes original sources and immutable strict artifacts. "
              "Matching saved scores at their rounded precision is arithmetic agreement only, not a provenance override.", ""]
    lines += ["- " + text for text in report["limitations"]]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n")
    environment = {"executable": sys.executable, "python": sys.version, "platform": platform.platform(),
                   "numpy": np.__version__, "scipy": scipy.__version__, "executable_sha256": digest(sys.executable)}
    write(output / "manifest.json", forensic_envelope(version="forensic-token-v1", inputs_and_code_sha256=before,
          environment=environment, environment_sha256=hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest(),
          parameters={"seed": SEED, "resamples": resamples, "mode": "--forensic-token-analysis", "completeness_repeats": 0},
          source_changes_during_analysis=[], outputs_sha256={p.name: digest(p) for p in sorted(output.iterdir())}))
    print(json.dumps(forensic_envelope(output=str(output)), indent=2))
    return 0  # Arithmetic completed, never release readiness or validated provenance.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--forensic-token-analysis", action="store_true", help="Opaque-token arithmetic only; all strict failures remain unresolved")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    allowed = (ROOT / "results/readiness/qrel_sensitivity").resolve()
    output = (args.output or allowed / ("v1_forensic" if args.forensic_token_analysis else "v1")).resolve()
    if allowed not in output.parents or output.exists():
        parser.error("output must be a NEW subdirectory of results/readiness/qrel_sensitivity; no overwrites")
    if not 1 <= args.resamples <= 10000 or not 2 <= args.repeats <= 200:
        parser.error("resamples must be 1..10000; repeats 2..200")
    output.mkdir(parents=True)
    try:
        if args.forensic_token_analysis:
            return run_forensic(args.root, output, args.resamples)
        paths = input_paths(args.root)
        before = {str(p): digest(p) for p in paths}
        audit = audit_inputs(args.root)
        write(output / "INPUT_AUDIT.json", audit)
        if audit["errors"]:
            changed = [p for p, h in before.items() if digest(p) != h]
            environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                           "numpy": np.__version__, "scipy": scipy.__version__,
                           "numpy_file_sha256": digest(np.__file__), "scipy_file_sha256": digest(scipy.__file__),
                           "python_executable_sha256": digest(sys.executable)}
            failure = {"version": VERSION_ANALYSIS, "status": "fail", "release_blocked": True, "errors": audit["errors"],
                       "action": "Do not choose first/last duplicate records, remap to candidate identities, intersect/drop rows, or publish historical scores. Resolve the input contract before numerical analysis.",
                       "not_executed": ["per-task retrieval metrics", "stored score comparison", "rank correlations/reversals",
                                        "cluster bootstrap/permutation/Holm", "label subsampling curve", "fulltext strata scores"]}
            write(output / "FAILURE.json", failure)
            details = audit["corpus"]
            lines = ["# Historical qrel sensitivity: input validation FAILED", "", "**release_blocked: true**", "", WARNING, "",
                     "## Blocking evidence", "", f"Historical corpus: {details['rows']} rows, {details['distinct_ids']} distinct IDs, "
                     f"{details['duplicate_id_count']} duplicated IDs, {details['excess_rows']} excess rows. "
                     "Duplicate records disagree on PMID/title and other metadata. No first/last-record selection or deduplication was performed.", "",
                     "| Ambiguous doc_id | Physical corpus rows | PMIDs |", "|---|---|---|"]
            for d in details["duplicate_records"]:
                lines.append(f"| {d['doc_id']} | {', '.join(map(str,d['physical_rows']))} | {', '.join(str(r['pmid']) for r in d['records'])} |")
            lines += ["", "### Additional qrel consistency blocker", ""]
            for mismatch in audit["judgments"].get("stored_expanded_set_mismatches", []):
                lines += [f"Task `{mismatch['task_id']}`: reconstructed-only IDs {mismatch['reconstructed_only']}; stored-only IDs {mismatch['stored_only']}."]
                for ev in mismatch["raw_evidence"]:
                    lines += [f"`results/baselines/gold_adjudication.jsonl:{ev['physical_row']}` records doc `{ev['doc_id']}` as `{ev['judgment']!r}`. "
                              "The stored expanded file does not reproduce this raw-label reconstruction."]
            lines += ["", "## Independent inventory checks (not retrieval results)", "",
                      f"Test universe: {audit['tasks']['test_count']}; supported: {audit['tasks']['supported_count']}; empty seed qrels: {audit['tasks']['empty_count']}. "
                      "All 12 primary saved runs were checked against every test ID, including empty-support tasks. "
                      "See INPUT_AUDIT.json for any additional errors and all affected task/document/rank references.", "",
                      f"Raw verdicts: {audit['judgments']['raw_count']}; positive: {audit['judgments'].get('relevant')}; "
                      f"negative: {audit['judgments'].get('not_relevant')}. Seed qrel pairs: {audit['tasks']['seed_qrel_pairs']}; "
                      f"ID-level reconstructed expanded pairs: {audit['judgments'].get('expanded_qrel_pairs')}. "
                      f"Stored-expanded set mismatches: {len(audit['judgments'].get('stored_expanded_set_mismatches', []))}. "
                      "Matching strings do not establish valid document identity or scientific relevance.", "",
                      "Affected-input counts: " + json.dumps(audit["scope_counts"], sort_keys=True), "",
                      "## Not executed", "", "Retrieval metrics, stored-score comparisons, correlations/reversals, bootstrap/sign-flip/Holm, "
                      "completeness curves, and fulltext score strata were deliberately not produced. The fail-closed input contract takes precedence over filling result tables. "
                      "The analysis implementation and formula tests are present, but the end-to-end numerical branch is unverified on these invalid inputs.", "",
                      "## Identity scope and limitations", "", f"Candidate identity report independently records {identity_count(audit)} PMID/XML mismatches. "
                      "It was read only for counts/scope, never to remap old results. All historical primary results remain compromised. "
                      "No repaired benchmark or publication claim is made. Artificial label thinning, if later run under an authorized valid input contract, "
                      "would still be conditional on the original biased pool. Historical source flags cannot support valid fulltext claims.", "",
                      "## Sources and next action", "", "`manifest.json` hashes all original input and analysis code paths; `INPUT_AUDIT.json` lists exact affected IDs and physical rows. "
                      "Corpus source: `data/processed/corpus.jsonl`; test runs and judgments: `results/baselines`; tasks: `data/benchmark/{train,dev,test}.jsonl`; "
                      "identity scope: `data/processed/identity_repair_v1/report.json`. Methods: `docs/QREL_SENSITIVITY_METHODS.md`.", "",
                      "Resolve the historical input contract before attempting numerical analysis; do not silently choose between conflicting rows or map saved results into the candidate corpus. "
                      "Paper and overall readiness gate are unchanged. Local only: no humans, network/API, install, paid jobs, or git actions.", ""]
            (output / "REPORT.md").write_text("\n".join(lines))
            manifest = {"version": VERSION_ANALYSIS, "status": "fail", "release_blocked": True,
                        "inputs_and_code_sha256": before, "source_changes_during_audit": changed,
                        "environment": environment,
                        "environment_sha256": hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest(),
                        "requested_parameters_not_executed": {"resamples": args.resamples, "repeats": args.repeats, "seed": SEED},
                        "command": [sys.executable, "scripts/44_analyze_historical_qrel_sensitivity.py", "--resamples", str(args.resamples), "--repeats", str(args.repeats)],
                        "outputs_sha256": {p.name: digest(p) for p in sorted(output.iterdir())}}
            write(output / "manifest.json", manifest)
            print(json.dumps({"status": "fail", "release_blocked": True, "errors": audit["errors"], "output": str(output)}, indent=2))
            return 1
        report, manifest, per_task, expanded, normalized, components, raw, curve, strata = compute(args.root, args.resamples, args.repeats)
        for name, value in (("analysis.json", report), ("expanded_reconstructed.json", expanded),
                            ("components.json", components), ("curve_summary.json", curve), ("historical_fulltext_strata.json", strata)):
            write(output / name, value)
        for name, records in (("per_task_metrics.jsonl", per_task), ("normalized_judgments.jsonl", normalized), ("curve_replicates.jsonl", raw)):
            (output / name).write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in records))
        (output / "REPORT.md").write_text(report_markdown(report))
        manifest["outputs_sha256"] = {p.name: digest(p) for p in sorted(output.iterdir())}
        write(output / "manifest.json", manifest)
        print(json.dumps({"status": report["status"], "release_blocked": True, "output": str(output), "components": report["components"]}, indent=2))
    except (ValueError, KeyError, TypeError) as e:
        if args.forensic_token_analysis:
            write(output / "FAILURE.json", forensic_envelope(diagnostic_completed=False, error=str(e)))
            (output / "REPORT.md").write_text(f"# Forensic arithmetic failed\n\n{FORENSIC_LABEL}\n\nrelease_blocked: true\n\n{e}\n")
            raise
        write(output / "FAILURE.json", {"status": "fail", "release_blocked": True, "error": str(e),
                                      "action": "Inspect named source IDs/rows; do not intersect/drop/repair historical inputs. Rerun in a new directory only after resolving the diagnostic."})
        (output / "REPORT.md").write_text("# Historical diagnostic FAILED\n\n**release_blocked: true**\n\n" + str(e) + "\n")
        raise


def identity_count(audit):
    return audit["identity_candidate_scope_only"]["raw_front_pmid_disagreement_records"]


if __name__ == "__main__":
    sys.exit(main())
