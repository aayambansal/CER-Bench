# Qrel disclosure: final writer handoff

Use `v2/summary.json` and `v2/qrel_disclosure.pdf` (vector) or `.png`. All paths here are relative to `results/readiness/qrel_disclosure/`. Scripts, tests and methods are in the owned `scripts/51_qrel_disclosure_experiment.py`, `tests/test_qrel_disclosure.py`, and `docs/QREL_DISCLOSURE.md` paths under the analysis root.

## Findings

1,000 paired nested replicates, 108 supported queries, ten original-pool systems. The mean-R20 leader changes from agent at f=.2 (6/30 judgments per query) to BGE at f=.3 (9/30). This is a grid bracket, not an estimated continuous crossover.

| Fraction | Agent top-1 vote rate | BGE | SPLADE | Hybrid |
|---|---:|---:|---:|---:|
| 0 | 1 | 0 | 0 | 0 |
| .1 | .8245 | .0035 | .161 | .011 |
| .2 | .400 | .188 | .368 | .044 |
| .3 | .079 | .567 | .309 | .045 |
| .4 | .007 | .802 | .170 | .021 |
| .5 | .001 | .920 | .077 | .002 |
| .6 | 0 | .979 | .021 | 0 |
| .7 | 0 | .995 | .005 | 0 |
| .8 | 0 | 1 | 0 | 0 |
| .9 | 0 | 1 | 0 | 0 |
| 1 | 0 | 1 | 0 | 0 |

Other six systems have zero observed top-1 votes. Tied winners split a unit vote. Rates are finite Monte Carlo estimates; observed 0 or 1 at intermediate fractions is not proof of impossibility/certainty.

BGE-minus-agent mean token R20 at f=.2 is -0.005432323 (central 95% empirical interval [-0.030668541, 0.020421260]); at f=.3 it is +0.011652950 ([-0.013106303, 0.035547447]). BGE strictly exceeds agent in .349 and .831 of replicates, respectively. At f=.4 the difference is +0.024970535 ([0.001644139, 0.047598148]), with strict exceedance .980.

Seed winner agent has exact R20 44/81. Full-disclosure winner BGE has exact R20 464294981094611/1113970265179200 = 0.41679297518764713. Full disclosure reverses 9/45 system pairs relative to seed; Kendall tau-b is 0.5843065474681431. All 20 system/regime rational endpoints match the forensic reference.

## Verification

- Final pytest: 11 passed in 0.57 seconds; `tests_final.xml`.
- Final 1,000-replicate run: 1.0603569583036005 seconds after plotting cache warmup (first run 13.66003829240799 seconds).
- Independent post-run verification: every final input/code and output SHA-256 matched; all 20 endpoint summary means equal float conversions of exact rational references.
- All saved replicate arrays exactly identical between v1 and v2; v2 only fixes repeated-sum floating-point perturbations in constant summary means.
- 363 independent SciPy Kendall tau-b comparisons (replicates 0, 13, 999; all 11×11 fraction pairs) passed.
- Vote sums checked; figure PNG opened and visually inspected. PDF emitted by matplotlib with vector lines/text and embedded TrueType fonts.

## Scope

Historical opaque-token disclosure conditional on an already-selected biased pool. Not human truth, new qrel generation, live-provider results, repaired benchmark evaluation, or a model of missing-not-at-random unknown relevance. Intervals describe conditional random disclosure, not biomedical-population confidence intervals. Raw records have only task_id/doc_id/judgment, so leave-one-system-out pooling provenance is unavailable. No corpus opened or hashed, no network/API/secrets/install/Git, and no paper or unowned source edits.
