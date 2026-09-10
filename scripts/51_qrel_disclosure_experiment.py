#!/usr/bin/env python3
"""Paired nested disclosure of historical opaque-token judgments; offline only."""
from __future__ import annotations
import argparse
from fractions import Fraction
import importlib.util
from itertools import combinations
import math
from pathlib import Path
import platform
import sys
import time
import numpy as np
import scipy
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("historical_helpers", ROOT / "scripts/44_analyze_historical_qrel_sensitivity.py")
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
VERSION = "qrel-disclosure-v1"
POOL = h.POOL
FRACTIONS = tuple(Fraction(i, 10) for i in range(11))
SEED = 20260909
SCOPE = ("Historical opaque-token diagnostic, not validated biomedical relevance. "
         "Uniform random disclosure conditional on the already-selected biased pool; "
         "not a model of missing-not-at-random unknown relevance. No humans, live models, "
         "new qrels, corpus dereferencing, or repaired-benchmark performance claims.")


def reveal_indices(permutations, fraction):
    f = Fraction(str(fraction))
    if not 0 <= f <= 1:
        raise ValueError("fraction outside [0,1]")
    return permutations[..., :math.floor(f * permutations.shape[-1])]


def permutations_for(seed, nqueries, candidates=30):
    rng = np.random.default_rng(seed)
    return np.array([rng.permutation(candidates) for _ in range(nqueries)], dtype=np.uint8)


def rank_votes(units):
    ranks = rankdata(-units, method="average", axis=-1)
    winners = units == np.max(units, axis=-1, keepdims=True)
    return ranks, winners / winners.sum(axis=-1, keepdims=True)


def tau_from_signs(a, b):
    num = (a * b).sum(axis=-1)
    den = np.sqrt((a != 0).sum(axis=-1) * (b != 0).sum(axis=-1))
    return np.divide(num, den, out=np.full(np.shape(num), np.nan), where=den != 0)


def load_inputs():
    b = ROOT / "results/baselines"
    forensic = ROOT / "results/readiness/qrel_sensitivity/v1_forensic/analysis.json"
    paths = [ROOT / "data/benchmark/test.jsonl", forensic, b / "gold_adjudication.jsonl"]
    paths += [b / f"{m}_test.jsonl" for m in POOL]
    paths += [Path(__file__), ROOT / "scripts/44_analyze_historical_qrel_sensitivity.py",
              ROOT / "src/evaluation/strict_metrics.py", ROOT / "tests/test_qrel_disclosure.py",
              ROOT / "docs/QREL_DISCLOSURE.md"]
    hashes = {str(p.relative_to(ROOT)): h.digest(p) for p in paths}
    ref = h.read(forensic)
    assert tuple(ref["pool_systems"]) == POOL
    tasks = h.index_rows(h.rows(paths[0]), "test")
    audit = ref["unresolved_input_audit"]["tasks"]
    h.exact_keys(tasks, audit["supported_task_ids"] + audit["empty_task_ids"], "complete universe")
    seed = {t: h.strings(r["supporting_doc_ids"], t) for t, r in tasks.items()}
    ids = sorted(t for t in tasks if seed[t])
    h.exact_keys(ids, audit["supported_task_ids"], "supported universe")
    raw = h.rows(b / "gold_adjudication.jsonl")
    tokens = {d for ds in seed.values() for d in ds}
    tokens.update(r["doc_id"] for r in raw)
    full, normalized = h.reconstruct(seed, raw, tokens, 30)
    assert (len(tasks), len(ids), sum(map(len, seed.values())), len(raw), sum(map(len, full.values()))) == (125,108,264,3240,1370)
    runs = {}
    for m in POOL:
        rr = h.index_rows(h.rows(b / f"{m}_test.jsonl"), m)
        h.exact_keys(rr, tasks, m)
        runs[m] = {t: h.strings(rr[t]["retrieved_docs"], f"{m}/{t}") for t in tasks}
    grouped = [[r for r in normalized if r["task_id"] == t] for t in ids]
    # Negative labels and seed overlaps never contribute new positive tokens.
    additions = np.array([[r["relevant"] and r["doc_id"] not in seed[t] for r in rs]
                          for t, rs in zip(ids, grouped)], dtype=np.int64)
    hits = np.array([[len(set(runs[m][t][:20]) & set(seed[t])) for m in POOL] for t in ids])
    newhits = np.array([[[int(r["doc_id"] in runs[m][t][:20]) * additions[i,j] for m in POOL]
                         for j,r in enumerate(rs)] for i,(t,rs) in enumerate(zip(ids, grouped))])
    sizes = np.array([len(seed[t]) for t in ids])
    lcm = math.lcm(*range(1, max(map(len, full.values())) + 1))
    if lcm * len(ids) > np.iinfo(np.int64).max:
        raise ValueError("exact accumulator exceeds int64")
    exact = {}
    for name, gold in (("seed", seed), ("reconstructed_expanded", full)):
        exact[name] = {}
        for m in POOL:
            value = sum((Fraction(len(set(runs[m][t][:20]) & set(gold[t])), len(gold[t])) for t in ids), Fraction()) / len(ids)
            assert value == Fraction(ref["recall20_exact_rational_means"][name][m]), (name,m)
            exact[name][m] = str(value)
    return dict(hashes=hashes, ids=ids, empty=sorted(set(tasks)-set(ids)), seed=seed,
                grouped=grouped, additions=additions, hits=hits, newhits=newhits,
                sizes=sizes, lcm=lcm, exact=exact, raw_fields=sorted(set().union(*(r.keys() for r in raw))))


def score_permutation(data, permutation):
    q = np.arange(len(data["ids"]))[:, None]
    added = np.concatenate([np.zeros((len(q),1), dtype=np.int64),
                            np.cumsum(data["additions"][q,permutation], axis=1)], axis=1)
    hitadd = np.concatenate([np.zeros((len(q),1,len(POOL)), dtype=np.int64),
                             np.cumsum(data["newhits"][q,permutation], axis=1)], axis=1)
    ks = [math.floor(f * 30) for f in FRACTIONS]
    denom = data["sizes"][:,None] + added[:,ks]
    units = ((data["hits"][:,None,:] + hitadd[:,ks,:]) * (data["lcm"] // denom)[...,None]).sum(axis=0)
    return units, denom.sum(axis=0)


def describe(x):
    x = np.asarray(x)
    # Preserve invariant endpoints exactly rather than perturbing by repeated sums.
    mean = np.where(np.all(x == x[0], axis=0), x[0], np.mean(x, axis=0))
    return {"mean": mean.tolist(),
            "empirical95": np.quantile(x, [.025,.975], axis=0).tolist()}


def run(output, repeats=1000, base_seed=SEED):
    started = time.monotonic()
    data = load_inputs()
    seeds = np.arange(base_seed, base_seed + repeats, dtype=np.int64)
    perms = np.array([permutations_for(int(s),108) for s in seeds])
    result = [score_permutation(data,p) for p in perms]
    units = np.array([r[0] for r in result])
    qrel_counts = np.array([r[1] for r in result])
    divisor = data["lcm"] * 108
    for f, regime in ((0,"seed"),(-1,"reconstructed_expanded")):
        for j,m in enumerate(POOL):
            expected = Fraction(data["exact"][regime][m])
            assert Fraction(int(units[0,f,j]),divisor) == expected
            assert np.all(units[:,f,j] == units[0,f,j])
    scores = units / divisor
    ranks, votes = rank_votes(units)
    pairs = list(combinations(range(10),2))
    signs = np.stack([np.sign(units[:,:,i]-units[:,:,j]) for i,j in pairs],axis=-1)
    reversals = signs * signs[:,0:1,:] < 0
    tau = tau_from_signs(signs[:,:,None,:],signs[:,None,:,:])
    summary = {"version":VERSION,"scope":SCOPE,"status":"historical_token_disclosure_complete",
               "validated_biomedical_results":False,"systems":POOL,"replicates":repeats,
               "denominators":{"supported":108,"empty_excluded":17,"seed_pairs":264,"full_pairs":1370,"raw_judgments":3240},
               "endpoint_exact_rational_means":data["exact"],
               "interval_interpretation":"Central 95% empirical Monte Carlo sampling intervals conditional on fixed queries, rankings and biased pool; not biomedical-population confidence intervals or confidence intervals for estimated win probabilities.",
               "leave_one_system_out":{"available":False,"reason":"Raw judgments contain no system origin; cannot infer provenance from ranking overlap.","raw_fields":data["raw_fields"]},
               "fractions":[], "tau_b_all_fraction_pairs":describe(tau),
               "pair_order":[[POOL[i],POOL[j]] for i,j in pairs]}
    for k,f in enumerate(FRACTIONS):
        entry = {"fraction":float(f),"revealed_per_query":math.floor(f*30),"revealed_total":108*math.floor(f*30),
                 "qrel_count":describe(qrel_counts[:,k]),"systems":{},
                 "strict_reversal_pair_rates":reversals[:,k].mean(axis=0).tolist(),
                 "strict_reversal_fraction_all_45":describe(reversals[:,k].mean(axis=-1)),
                 "tau_b_vs_seed":describe(tau[:,0,k])}
        for j,m in enumerate(POOL):
            entry["systems"][m] = {"mean_R20":describe(scores[:,k,j]),"average_rank":describe(ranks[:,k,j]),
                                   "top1_tie_adjusted_win_rate":float(votes[:,k,j].mean()),
                                   "top1_vote_distribution":describe(votes[:,k,j])}
        summary["fractions"].append(entry)
    leaders = [POOL[int(np.argmax(scores[:,k].mean(axis=0)))] for k in range(11)]
    summary["mean_score_leader_by_fraction"] = leaders
    summary["grid_leader_changes"] = [{"from_fraction":float(FRACTIONS[k-1]),"to_fraction":float(FRACTIONS[k]),
                                       "from":leaders[k-1],"to":leaders[k]} for k in range(1,11) if leaders[k]!=leaders[k-1]]
    bge, agent = POOL.index("bge"), POOL.index("agent")
    summary["bge_minus_agent"] = [{"fraction":float(f),"difference":describe(scores[:,k,bge]-scores[:,k,agent]),
                                   "strict_exceedance_rate":float(np.mean(units[:,k,bge]>units[:,k,agent]))} for k,f in enumerate(FRACTIONS)]
    np.savez_compressed(output / "replicates.npz", seeds=seeds, permutations=perms, exact_score_units=units,
                        exact_score_divisor=np.array(divisor), mean_R20=scores, average_ranks=ranks,
                        top1_votes=votes, pair_reversals=reversals, tau_b=tau, qrel_counts=qrel_counts)
    h.write(output / "token_inputs.json", {"supported_ids":data["ids"],"empty_ids":data["empty"],
             "seed_qrels":data["seed"],"judgments_in_permutation_index_order":data["grouped"]})
    h.write(output / "summary.json", summary)
    # Availability checked before import; no installation is attempted.
    plotting = importlib.util.find_spec("matplotlib") is not None
    if plotting:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"pdf.fonttype":42,"font.size":9})
        fig, axes = plt.subplots(1,3,figsize=(13,3.8),layout="constrained")
        x = np.array([float(f) for f in FRACTIONS])
        for j,m in enumerate(POOL):
            color = plt.get_cmap("tab10")(j)
            lo,hi = np.quantile(scores[:,:,j],[.025,.975],axis=0)
            axes[0].plot(x,scores[:,:,j].mean(axis=0),label=m,color=color)
            axes[0].fill_between(x,lo,hi,color=color,alpha=.09)
            axes[1].plot(x,votes[:,:,j].mean(axis=0),color=color)
        lo,hi = np.quantile(tau[:,0,:],[.025,.975],axis=0)
        axes[2].plot(x,tau[:,0,:].mean(axis=0),color="black")
        axes[2].fill_between(x,lo,hi,color="grey",alpha=.25)
        for ax,title in zip(axes,["Mean token Recall@20","Top-1 tie-adjusted win rate","Kendall tau-b versus seed"]):
            ax.set(title=title,xlabel="Fraction of pooled judgments revealed")
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=6,ncol=2)
        fig.suptitle("Historical token-only disclosure · 108 queries · fixed biased pool\nShading: central 95% conditional Monte Carlo intervals",fontsize=10)
        fig.savefig(output / "qrel_disclosure.pdf",metadata={"CreationDate":None,"ModDate":None})
        fig.savefig(output / "qrel_disclosure.png",dpi=220)
        plt.close(fig)
    assert all(h.digest(ROOT / p)==v for p,v in data["hashes"].items()), "input changed"
    h.write(output / "manifest.json", {"version":VERSION,"scope":SCOPE,"inputs_and_code_sha256":data["hashes"],
             "parameters":{"base_seed":base_seed,"replicate_seeds":seeds.tolist(),"rng":"numpy.PCG64/default_rng",
                           "replicates":repeats,"fractions":[str(f) for f in FRACTIONS],"reveal_counts_per_query":list(range(0,31,3)),
                           "sampling":"Independent uniform per-query permutations; nested prefixes shared by all systems", "exact_score_divisor":divisor},
             "environment":{"python":sys.version,"executable":sys.executable,"platform":platform.platform(),
                            "numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__ if plotting else None},
             "elapsed_seconds":time.monotonic()-started,"corpus_access":"none, including no corpus hashing",
             "endpoint_checks":"all 20 rational endpoints equal forensic reference; invariant across replicates",
             "outputs_sha256":{p.name:h.digest(p) for p in sorted(output.iterdir())}})
    print(output / "summary.json")
    print("Mean-score grid leaders:", leaders)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replicates",type=int,default=1000)
    p.add_argument("--seed",type=int,default=SEED)
    p.add_argument("--output",type=Path,default=ROOT / "results/readiness/qrel_disclosure/v1")
    a = p.parse_args()
    allowed = ROOT / "results/readiness/qrel_disclosure"
    output = a.output.resolve()
    if allowed not in output.parents or output.exists() or not 200 <= a.replicates <= 1000:
        p.error("Require NEW owned output directory and 200..1000 replicates")
    output.mkdir(parents=True)
    run(output,a.replicates,a.seed)


if __name__ == "__main__":
    main()
