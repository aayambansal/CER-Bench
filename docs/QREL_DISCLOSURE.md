# Historical-token qrel disclosure experiment

Writer-facing final run: `results/readiness/qrel_disclosure/v2/`. The earlier `v1/` is retained for provenance; v2 preserves constant endpoint summary means without repeated-sum floating-point perturbations. Sampling, scores, ranks, and conclusions are unchanged.

This is a local empirical Monte Carlo experiment on **opaque historical tokens**, not live-model performance, human truth, newly generated qrels, or evaluation of the repaired benchmark. No conflicted corpus records are opened or even hashed. Authoritative repairs do not retroactively validate these historical labels/rankings.

## Design and estimand

Use the exact ten original-pool systems from forensic analysis: bm25, dense, bge, e5large, medcpt, splade, hybrid, hybrid_reranked, bge_reranker, agent. Validate all 125 saved task rows in every run; the fixed metric universe is the same 108 supported tasks, excluding 17 explicit seed-empty tasks. Seed support has 264 pairs; 3,240 raw judgments (30 per supported query) reconstruct 1,370 pairs, not the erroneous 1,369-pair stored expansion.

For each of 1,000 default replicates, independently permute all 30 judgments per query uniformly, including negatives. At fractions 0,.1,...,1 reveal exactly floor(30f) judgments per query. A replicate's permutations are shared across all fractions and systems, ensuring paired nested disclosure. PCG64 seeds are 20260909 through 20261908. Add only revealed RELEVANT tokens to always-retained seed labels. NOT_RELEVANT is normalized before RELEVANT, including recognized explanatory suffixes; unknown labels, duplicate pairs, and seed-negative contradictions fail closed. No negative or unrevealed positive contributes a new qrel. Saved rankings never change.

Per-query R20 is the number of seed-or-revealed-positive tokens in the first 20 saved ranked tokens divided by current qrel count; mean R20 gives every supported query weight 1/108. Unjudged tokens are operational nonrelevant, not established negatives. Integer arithmetic uses the common multiple of all possible positive qrel denominators to represent each mean exactly, avoiding artificial floating-point ties. Endpoint rational values must equal all ten seed and all ten reconstructed forensic rational means. Average ranks use exact score ties; top-1 votes divide one unit equally among tied winners. Strict reversals require opposite nonzero pairwise signs relative to seed; a broken seed tie is not a reversal. Report every pair's reversal frequency and the fraction of all 45 pairs reversed. Kendall tau-b is computed between every pair of disclosure fractions with tie correction.

Central 2.5/97.5 percentile intervals describe **conditional Monte Carlo disclosure variation**, not confidence intervals for biomedical populations, true relevance, or estimated win probabilities. Top-1 rates are Monte Carlo averages; their per-replicate vote quantiles are descriptive, not rate-estimation confidence bounds. No bootstrap or significance tests are added. Crossover brackets refer only to tested grid points, not a continuous threshold or proof of monotonicity.

This experiment is conditional on an already selected, biased pool. It does not simulate unknown missing-not-at-random relevance, establish completeness, or remove pooling bias. Leave-one-system-out pooling analysis is unavailable: raw records contain only task_id, doc_id, judgment, with no candidate-origin system. Ranking overlap is not substituted for provenance.

## Reproduction and outputs

From the analysis root, using the verified environment (no installs/network/API/secrets/Git):

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python -m pytest -p no:cacheprovider tests/test_qrel_disclosure.py
PYTHONDONTWRITEBYTECODE=1 /Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python scripts/51_qrel_disclosure_experiment.py
```

Default output is `results/readiness/qrel_disclosure/v1/`; reruns require a new owned subdirectory via `--output`. `summary.json` contains all fraction/system summaries, rational endpoints, pairwise rates, tau matrices, grid leader changes, and BGE-minus-agent differences/exceedance rates. `replicates.npz` retains seeds, full query permutations, exact score units/divisor, floating means, ranks, votes, reversals, tau matrices, and qrel counts. Axes are replicate × fraction × system (or pair); tau axes are replicate × fraction × fraction. Permutations are replicate × supported-query × raw-candidate-index; query and candidate orders are in `token_inputs.json`. Seed qrels include the explicit empty tasks. `manifest.json` records config, versions, source/code and output SHA-256 hashes, source stability checks and exact endpoint checks. PDF and PNG are produced only if matplotlib is installed, checked before importing. PDF is vector, with embedded TrueType fonts. Tests validate implementation, not biomedical truth.
