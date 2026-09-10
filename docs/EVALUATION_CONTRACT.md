# CERBench strict offline evaluation contract v1

This is an additive evaluator, independent of historical scorers, seed generation,
and paper results. It uses only the Python standard library. It performs no network,
model, installation, or paid calls. Input validation rejects rather than silently
deduplicating or guessing. Existing datasets/results are never rewritten.

## Document interface and denominators

`scripts/40_evaluate_validated_runs.py --tasks TASKS --runs RUNS --output NEW.json`
accepts UTF-8 JSON arrays or JSONL objects. Corpus input is a JSON array of strings.
Duplicate JSON object keys and nonfinite JSON constants are errors. Report version:
`cerbench.evaluation.v1`. An empty task universe is allowed but all rates are null.

* Tasks: unique nonblank `task_id`, `task_family` when structural annotations are
  used, and optional `gold_doc_ids: list[str]`. This file defines the **entire
  expected evaluation universe**, not the intersection with submitted outputs.
* Runs: unique `task_id`, required `retrieved_doc_ids: list[str]` in rank order.
  Optional selective fields must be supplied on every submitted row if selective
  evaluation is requested: `decision` and `confidence` (below). No sorting by
  undocumented score fields and no legacy field-name guessing.
* Duplicate task IDs in any input table, unknown run/annotation/judgment task IDs,
  duplicate retrieved/gold IDs, blank IDs, and malformed lists are errors.
* Missing run rows fail by default. `--missing-policy zero` explicitly replaces
  missing retrievals with empty lists, penalizing eligible document and annotated
  structural metrics. Missing runs are **not** synthesized as ABSTAIN; selective
  coverage retains the full universe and reports missing outputs separately.
* Absent/null `gold_doc_ids` means missing judgments. An explicit `[]` means
  empty gold. Both yield null document metrics and exclusion from retrieval
  denominators, but are counted separately. Neither proves unsupportedness,
  correct abstention, or an explicit negative result.
* `--corpus corpus.json` activates strict referential integrity for every gold,
  retrieved, and annotated evidence document (including below-cutoff ranks).
  Corpus IDs themselves must be unique. Without it, cross-corpus IDs cannot be
  checked; this is an explicit optional validation, not an inferred corpus.

Binary qrels only. Per task with nonempty gold: Recall@5/10/20 is the fraction of
gold IDs in the ranked prefix; nDCG@10 uses gain 1 and discount `1/log2(rank+1)`,
with ideal length `min(10, |gold|)`; MRR is reciprocal rank of the first gold hit
over the entire submitted ranking. Duplicates are rejected before scoring,
preventing inflated nDCG above 1. Macro averages include every nonempty-qrel task,
including missing-output zeros under the explicit penalty policy. Every metric
reports its denominator; coverage reports expected/actual/missing counts and IDs.
Unjudged retrieved documents count as nonrelevant under these supplied binary
qrels; these scores are not claims of exhaustive corpus relevance.

## Versioned structural annotation record

All fields below are required; `constraint_policy` is optional:

```json
{
  "schema_version": "cerbench.structural.v1",
  "task_id": "SYNTHETIC-example",
  "task_family": "comparative",
  "annotation_status": "unvalidated",
  "required_units": ["side:intervention", "side:comparator"],
  "evidence_units": {"SYNTHETIC-a": ["side:intervention"], "SYNTHETIC-b": ["side:comparator"]},
  "valid_pairs": [],
  "valid_paths": [],
  "provenance": {"synthetic": true, "evidence": {"SYNTHETIC-a": "invented source span A", "SYNTHETIC-b": "invented source span B"}}
}
```

Allowed statuses: `unvalidated`, `human_adjudicated`, `expert_adjudicated`.
Evaluation requires human/expert status **and** `provenance.evidence` mapping
exactly the annotated document IDs to nonblank evidence source/locator strings.
Real annotations should retain adjudicator, protocol version, source hashes,
span locators and rationale as additional provenance fields. The evaluator
checks the declared contract, not the truth or authenticity of human review.
`--allow-proxy` admits otherwise well-formed unverified annotations only as
`descriptive_proxy`, emits a warning, and keeps proxy aggregates separate from
verified ones. Malformed annotations are errors even in proxy mode. Missing
annotations remain `not_evaluated`, denominator 0, metrics null; no seed-gold or
family-based fallback exists.

Units and document lists must be nonempty and duplicate-free. Evidence units must
belong to `required_units`; each required unit must have annotated evidence.
The record family must match the expected task family. Family names and role
namespaces below are the explicit v1 vocabulary; upstream protocols must map
their names deliberately rather than expecting implicit aliases.

| Family | Explicit units | `family_success` |
| --- | --- | --- |
| `constraint` | `constraint:<name>` | One retrieved document covers every constraint; only explicit `constraint_policy: "set_level"` permits union completion |
| `comparative` | Exactly two `side:<name>` roles | Both annotated sides covered |
| `contradiction` | Exactly two `claim:<name>` roles | At least one annotated valid pair fully retrieved |
| `multihop` | At least two `hop:<name>` units | At least one annotated valid path fully retrieved |
| `temporal` | `bin:<name>` | All required time bins covered |
| `aggregation` | `study_value:<study-and-value-id>` | All required study-value units covered |
| `negative` (the dataset spelling) or `negative_result` | Exactly one `negative_result:<target>` | Hit on a document explicitly annotated with that negative finding |

For every eligible family, common `unit_coverage` is the fraction of required
units covered by retrieved annotated documents; `set_completion` is full union
coverage. These are diagnostic and **not substitutes for `family_success`**.
Temporal unit coverage measures bins; aggregation unit coverage measures
study-values, not arbitrary document counts. All three metrics use the first
`--structural-k` documents (default 10), and their verified/proxy denominators are
reported overall and per family. With no retrieved evidence, every eligible
score is zero, never a vacuous success.

`valid_pairs` is nonempty only for contradiction; every pair contains exactly
two distinct annotated documents, each with one different claim role.
`valid_paths` is nonempty only for multihop; every path contains at least two
distinct annotated documents. Every listed pair/path must cover the required
units; duplicate groups and unannotated references are rejected. Paths are
stored in annotated semantic order; retrieval completion checks document
membership, not retrieval rank order. Validity of the relationship/hop sequence
comes from adjudication, not arbitrary unit unions. Mixed halves of two paths
fail even if their union covers all units. No claim about unseen valid paths is
made; annotation completeness remains a protocol responsibility.

## Selective evidence-set evaluation

Run decisions are exactly `ANSWER`, `ABSTAIN`, `UNCERTAIN`; confidence is a finite
number in [0,1], interpreted as candidate evidence-set success confidence, not
confidence that an abstention is appropriate. Booleans are not numeric scores.
`--selective-judgments` supplies unique task rows with human/expert status,
nonblank string `provenance`, and optional boolean/null `evidence_set_failure`
and `verified_support`. These are **external labels**, never derived from empty
gold or structural proxy scores. Evidence-set failure must judge the precise
submitted run/evidence set; use one judgments file per run and record its identity
in provenance. File hashes capture supplied inputs but cannot establish semantic
run-label alignment or authentic adjudication.

Coverage = ANSWER count / expected tasks. Selective risk = actual externally
judged failures / ANSWER count, null if no answers or any answer lacks a loss.
`observed_judged_answer_risk` is a separately labeled subset statistic, not a
replacement for full selective risk. `support_confusion` is a separate 3-by-3
decision versus supported/unsupported/unjudged table; UNCERTAIN is not collapsed
into ABSTAIN. It does not infer support from retrieval labels. No-answer and
all-empty cases retain honest null risks.

The risk-coverage curve sweeps only submitted ANSWER candidates, with expected
task count as the coverage denominator. Equal confidence values enter as whole
blocks; no favorable tie ordering, random tie breaking, or interpolated AURC is
reported. The initial answer-none point has null risk; any selected block with
missing labels makes cumulative risk null. This curve cannot recover hypothetical
answers withheld under ABSTAIN/UNCERTAIN.

Optional `--calibration DEV.json --calibration-split dev --max-risk 0.1` selects a
threshold from externally judged **all-candidate dev answers**, not a subset
already filtered by decision. Each row has `task_id`, `confidence`, verified
status/provenance and boolean `evidence_set_failure`. It maximizes dev coverage
among tie-block thresholds meeting the empirical risk target. No feasible point
gives `answer_none`, threshold null; otherwise accept confidence >= threshold.
The selector does not alter the evaluation run, and never tunes on its losses.
Train/test calibration or overlapping dev/evaluation task IDs is rejected.
Selection is explicitly **empirical calibration**, not conformal prediction,
a confidence bound, or a population risk guarantee. Full dev candidate coverage,
authentic split membership, and component/source-disjointness cannot be proved
from IDs; upstream split/protocol validation is still required. Freeze threshold
and policy before generating held-out decisions.

## Reproducibility, errors, and local smoke

Successful reports include `status: completed` (execution status, not a claim of
scientific readiness), per-component statuses, options, all raw input SHA-256s,
and SHA-256s of the CLI, three evaluator modules and package initializer. Errors
exit 2, normally with versioned error JSON on stderr; argparse usage/existing
destination errors use argparse diagnostics. No output is created for validation
errors. Output is exclusive-create; existing paths are never overwritten.

From `analysis/synthetic-science-search`:

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python -m pytest -q -p no:cacheprovider tests/test_strict_metrics.py tests/test_structural_metrics.py tests/test_selective.py

PYTHONDONTWRITEBYTECODE=1 /Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python scripts/40_evaluate_validated_runs.py --tasks examples/evaluation_contract/tasks.json --runs examples/evaluation_contract/runs.json --annotations examples/evaluation_contract/annotations.json --selective-judgments examples/evaluation_contract/selective_judgments.json --corpus examples/evaluation_contract/corpus.json --calibration examples/evaluation_contract/calibration.json --calibration-split dev --output results/readiness/evaluator_smoke/synthetic_verified_v1.json
```

Examples are **entirely synthetic**, including simulated adjudication statuses.
Expected smoke: 4/4 outputs, 2 nonempty qrels, 1 empty gold, 1 missing judgment;
Recall@5/10/20 = 0.75; two structurally eligible tasks both have unit coverage and
set completion 1 but family success 0; two missing structural annotations;
ANSWER coverage 0.5 and externally supplied risk 1; the two tied answers enter
one curve block; dev threshold 0.9. Proxy smoke replaces annotations with
`annotations_proxy.json`, adds `--allow-proxy`, and uses a different output name.
The tests include malformed/duplicate IDs, nDCG inflation, missing outputs,
all-empty judgments, invalid roles/paths, arbitrary-union traps, negative hits,
tied scores, missing losses and dev/test leakage. No historical result is
rescored or relabeled by this interface.
