# Historical qrel sensitivity — methods v1

**release_blocked: true.** This is a local-only historical diagnostic, not publication results or a repaired benchmark. All primary historical results are compromised by the identity audit. Candidate identity data are not used to remap any run, judgment, or task. Only the candidate `report.json` is read for counts and scope.

## Contract and source validation

The 12 primary methods are those listed in `scripts/31_submission_audit.py`: BM25, SPECTER2 (`dense`), BGE, E5-large, MedCPT, SPLADE, RM3, Hybrid, Hybrid+CE, BM25+BGE-reranker, Agent T=1, and Agent T=3. Primary ranking correlations use the original pool's 10 systems; all-12 correlations are explicitly supplemental because RM3 and Agent T=1 were outside the pool. Stored nonprimary scores (e.g. partial GPT-5.4 and ColBERT) are listed but not silently admitted to the primary comparison.

All runs must contain exactly the 125 historical test task IDs, including tasks with empty supporting lists. Duplicate JSON keys, task rows, corpus IDs, ranked IDs, and judgment pairs fail. Unknown task/document IDs, unrecognized verdicts, seed-negative contradictions, missing run rows, and incomplete per-task judgments fail with actionable IDs. Seed and hard-negative lists across all splits must be valid historical corpus IDs. No intersection, ID coercion, alias repair, missing-row zero fill, or silent dropping is allowed.

The 3,240 raw verdicts must cover exactly 30 pairs for each of the 108 supported tasks and none for the 17 seed-empty tasks. Normalize by strip/uppercase and token-prefix recognition of NOT_RELEVANT **before** RELEVANT; explanatory suffixes are accepted, unknown values fail. Expanded qrels are the seed union all positive verdict pairs. Seed membership is always retained, but an explicit negative seed judgment raises a contradiction rather than silently overriding it. The stored expanded file is explicitly checked against the 108 supported-only keys; 17 seed-empty tasks remain explicit empty lists in the reconstruction. All pair sets must match. Available stored original/expanded recall scores must agree at their four-decimal precision; missing RM3 stored scores are reported as unavailable, not fabricated.

## Strict metrics and ranks

Per-task Recall@5/10/20, nDCG@10, and MRR call `src/evaluation/strict_metrics.py::doc_metrics` directly with **lists of document IDs**. Recall is relevant retrieved at k divided by qrel size. Binary DCG uses log2 rank discounts and ideal DCG is truncated to min(10, qrel count). MRR uses the entire saved ranking. Empty qrels return null metrics and are represented in per-task outputs, not scored as zero. Every supported mean has exactly 108 task observations. Means use `math.fsum` without rounding.

For each metric, SciPy Kendall tau-b corrects for ties and Spearman computes Pearson correlation on average ranks. Pair reversals require `(seed_A-seed_B)*(expanded_A-expanded_B)<0` on unrounded means; a tie becoming untied is not a strict reversal. Every reversed pair and its two differences is saved. Exact binary floating-point equality defines ties; no display rounding is used.

## Paired component inference

The single prespecified inferential endpoint is **Recall@20**, chosen as the historical saved-retrieval endpoint. The six-hypothesis family is Agent T=3 minus BM25/SPLADE/Hybrid × seed/expanded qrels. All other metrics and correlations are descriptive; no extra uncorrected hypothesis family is implied.

Connected components are constructed over **all original train/dev/test tasks**, before selecting test observations. Reuse the local relation rule in `scripts/38_make_component_disjoint_splits.py`: shared annotated support/negative documents and chunks, evidence clusters, normalized exact questions, and token-set Jaccard >=0.85 near questions. Thus train/dev tasks can bridge test tasks. Expanded pooled labels are not used to redefine dependency components. Save complete component membership with original split placement. These are known dependency proxies, not proof of semantic independence.

Let component c contain n_c supported test tasks, with sum of paired task differences S_c. The estimand is sum(S_c)/sum(n_c), **not** the unweighted mean of component means. Sample C contributing components with replacement 10,000 times and compute sum(sampled S_c)/sum(sampled n_c), allowing cluster sizes to alter each replicate's task denominator. Report percentile 2.5/97.5% intervals. Components without supported test observations determine connectivity but are not resampling units with zero outcomes.

For a two-sided cluster sign-flip test, independently multiply each S_c by ±1, keeping its tasks together and total task denominator fixed. With B=10,000 draws, p=(1+number of |permuted difference| >= |observed difference|)/(B+1). A 1e-14 tolerance is used only for numerical equality in this tail comparison. Holm step-down adjusts the six p-values jointly. Default PCG64 seed is 20260909; common random draws across contrasts/regimes improve reproducibility. This requires symmetric independent component-level differences under the null. Observational historical systems were not randomized; the test cannot justify causal or population-wide validity.

## Artificial completeness sensitivity

For each of 200 deterministic repeat seeds (20260909 through 20261108), uniformly permute **all 3,240 judged records, including negative labels**. Retain the first floor(fraction × 3240) at fractions 0, .25, .5, .75, 1; fractions are nested within a repeat. Add sampled positive pairs to always-retained seed qrels. This samples judged labels, not positive-only qrels, and never changes the final task question or saved ranking. Unknown/nonretained documents count as nonrelevant under the operational binary evaluator. Exact endpoints must recover seed and full expanded means.

Save each repeat's sampled label count, positive count, total qrel count, all five metric means for all 12 methods, plus complete curve summaries (mean, sample SD, central 95% subsampling interval, min/max). These intervals measure artificial removal of labels **conditional on the original biased pool**; they are neither confidence bounds on true relevance nor evidence that the gold is complete. Out-of-pool methods can be penalized by pool construction. Seeds are synthetic and forced to remain positive.

## Unverified historical fulltext strata

Classify supported tasks as all/mixed/none according to original corpus `has_fulltext` booleans on seed-support documents. Save task IDs, stratum denominators and both-regime metric means. No document content is revalidated. Identity errors undermine these flags; stratification cannot support a valid fulltext-access or fulltext-benefit claim.

## Reproduction and output governance

Run locally from the analysis root with the already-installed environment:

```text
/Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python -m pytest -p no:cacheprovider tests/test_qrel_sensitivity.py tests/test_strict_metrics.py
/Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python scripts/44_analyze_historical_qrel_sensitivity.py --output results/readiness/qrel_sensitivity/v1
```

No install, network/API, human annotation, paid job, git action, or historical/paper/readiness-gate modification. Output must be a new subdirectory of `results/readiness/qrel_sensitivity`; overwrites fail. The manifest includes SHA-256 for every input and local dependency source, Python executable/version/platform, NumPy/SciPy versions and module hashes, parameters and every scientific output. Source hashes are checked again after computation. No timestamp or output-directory-dependent field appears in scientific outputs, permitting byte-for-byte comparisons across fresh output directories. Failure produces `FAILURE.json` and a blocked report, not valid metric outputs. Tests validate formulas and software behavior only, never scientific claims.

### Observed fail-closed input branch

The current original corpus contains duplicate document IDs associated with conflicting records. A separate complete input inventory precedes all numerical evaluation. It preserves each duplicate record's physical row, PMID/title, and differing fields and enumerates affected seed/negative/run/judgment references. An ID-membership set is used **only for unknown-reference detection and string-level qrel reconstruction audit**, never to select a corpus record, deduplicate the evaluation universe, or authorize metric computation. With errors, the process exits 1 and saves INPUT_AUDIT.json, FAILURE.json, REPORT.md and a failure manifest. Requested bootstrap/curve parameters are recorded as **not executed**; no retrieval metrics or inferred results are emitted. Source hashes are checked before and after the inventory. The failure manifest hashes environment metadata (Python, platform, NumPy and SciPy versions), the Python executable, and NumPy/SciPy module entry files. These hashes are provenance, not a complete binary dependency lock.

## Explicit forensic continuation: opaque token arithmetic

`--forensic-token-analysis` is a separate opt-in mode, defaulting to the new `results/readiness/qrel_sensitivity/v1_forensic` directory. It never changes strict `v1_final` artifacts, and default evaluation continues to fail. Every forensic output is labeled **historical_string_label_diagnostic_not_validated_biomedical_relevance**, with `release_blocked: true`, `release_ready: false`, and `input_validation_status: fail`. Exit 0 means only that requested token arithmetic completed, not scientific provenance or release readiness.

The mode reads the frozen failed audit and verifies its artifact hashes, verifies the original historical input hashes, and embeds the original audit and failure unchanged. Corpus bytes are hashed but **no corpus record is parsed, selected, indexed, or dereferenced**. Candidate metadata are not reread for independent identity work. All old document strings are opaque tokens, including collisions; token equality does not establish biomedical identity or relevance. Task/ranking/judgment duplicate and coverage checks still apply. This interpretation does not resolve or waive the original two failures.

Compute all five strict token metrics on all 12 methods, under three distinct label versions: seed (264 pairs), stored-expanded (1,369), and honestly reconstructed raw-plus-seed (1,370). Save both expanded versions independently inside a labeled output without overwriting historical sources. All means use the same 108 supported tasks, with 17 explicit null-metric rows per method/regime. Exact rational Recall@20 means and per-task/aggregate effects of the missing pair supplement unrounded floating-point means. Kendall tau-b, average-rank Spearman and all strict pair reversals use only the ten original-pool methods, separately for seed/stored, seed/reconstructed, and stored/reconstructed comparisons. RM3 and Agent T=1 means remain outside-pool supplemental.

The bounded continuation also runs the six prespecified paired Recall@20 contrasts on **seed and reconstructed** token labels (not a nine-test family including stored labels): 10,000 cluster bootstrap draws and cluster sign-flips, finite Monte Carlo p-values, and Holm across six tests. Components use original task token references/questions over all 304 train/dev/test tasks, with no corpus dereferencing. No fulltext strata or completeness curves are produced in this focused continuation. Saved-score comparisons are reported, not required to agree: arithmetic discrepancies are diagnostic findings, never provenance overrides.

```text
PYTHONDONTWRITEBYTECODE=1 /Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python scripts/44_analyze_historical_qrel_sensitivity.py --forensic-token-analysis
```
