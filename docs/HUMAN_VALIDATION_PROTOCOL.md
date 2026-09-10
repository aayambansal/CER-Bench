# CERBench local human-validation protocol — PROVISIONAL / NOT HUMAN DATA

## Authority and present status

This is a preparation-only workflow. **There are no human annotators. Do not recruit,
contact participants, populate judgments, use model-generated judgments, or claim human
validation.** All work here is local: no network, LLM calls, installation, paid runs,
or git operations. Existing annotation sheets and paper files are not modified.
The older `scripts/37_merge_human_annotations.py` CLI is disabled fail-closed; use
the new validator only. It deliberately has **no final human-qrel export**.

The kit is provisional until the corpus identity failures are repaired, affected
indexes/runs regenerated, and a component-disjoint split fixed and frozen. The
existing test set has already been explored. This sample is a retrospective audit,
not an untouched confirmatory test set. A random subset does not erase contamination.

## Files and distribution boundary

All paths below are relative to `analysis/synthetic-science-search`.

- `scripts/42_prepare_validation_kit.py`: standard-library-only builder. Refuses to
  overwrite either output directory, including previously filled sheets.
- `scripts/43_validate_human_labels.py`: blank-kit integrity checks or validation
  of two separately saved completed sheets; agreement/adjudication report only.
- `annotations/human_qrel_v2/annotator_A.csv` and `annotator_B.csv`: independently
  shuffled, identical pair sets; all response and provenance cells are blank.
- `annotations/human_qrel_v2/task_requirements_A.csv` and `_B.csv`: independent
  task-level constraint, unit, valid-pair/path requirement and sufficiency sheets.
- `annotations/human_qrel_v2/evidence/*.json`: complete available parsed section
  text, abstract, figure captions and table text, without snippet truncation.
- `annotations/human_qrel_v2/structural_template.json`: only `unvalidated` records
  with empty requirements, assignments, pairs, paths and provenance.
- `annotations/human_qrel_v2/coordinator/`: **not annotator material**. Contains
  selection, pool sources and identity/exclusion audit. Also withhold the manifest
  and original benchmark/qrels, reference answers, automatic labels, model outputs,
  retrieval ranks, old annotation kit and source corpus from annotator distribution.
- `annotations/human_queries_v1/human_queries_blank.csv`: 80 empty collection
  slots; target strata are a design, not observations or human-written queries.
- `annotations/human_qrel_v2/local_validation_report.json`: initial blank-kit
  verification; not annotation or empirical evidence of retrieval quality.
- `annotations/human_qrel_v2/final_local_validation_report.json`: final integrity
  check including all 110 task templates per reviewer and 80 blank query slots.
- `annotations/human_qrel_v2/software_tests.xml`: final JUnit software-test report.

If separately authorized in the future, distribute only the assigned A **or** B
pair/task sheets, the shared local evidence directory, and these instructions
(omit the coordinator-only selection/exclusion details). Do not let A and B see
one another's sheets before independent completion. Source URLs identify papers
but are not invitations to browse: all assessment in this workflow is offline.
CSV/JSON files are not access control; a coordinator must enforce that boundary.

## Predeclared selection and pool

Seed: `20260909`. Before identity filtering, sort eligible task IDs within each
of seven supported families and independently sample 14 using
`random.Random(f'{SEED}:select:{family}')`. Include every abstention task. Do not
replace excluded queries or search for favorable retained samples. This yields
98 selected supported queries plus 17 abstention queries, with 93 supported
queries and all 17 abstention queries retained after exclusions.

For each retained query, union **seed supporting papers**, original hard-negative
candidates (candidate status only), and the first 20 retrieved document IDs in
each available predeclared major-system local run. No automatic adjudication or
gold label file is read. Systems: BM25, dense, BGE, E5-large, MedCPT, SPLADE, RM3,
hybrid, hybrid-reranked, BGE-reranker, single-step agent, agent, GPT-5.4 agent,
ColBERT and Rocchio. The GPT-5.4 agent run is missing 35 retained query rows;
the other 14 systems cover the retained queries. Missing rows are recorded,
not silently interpreted as empty retrieval judgments.

Add five seeded uniform random candidates per retained query from the globally
identity-safe corpus **outside that query's union**, without replacement.
Separate random streams `SEED:tail:task_id` prevent dependence on A/B ordering.
These 550 additional pairs are distinct within query and disjoint from its pool;
the same paper may appear for another query. They are unjudged candidates, **not
verified negatives**. They do not establish complete relevance or corpus-wide
absence. Candidate source and system rank never appear on the annotator sheets.
Stable opaque pair IDs allow matching; independent A/B stream seeds control order.

### Observed size and exclusion impact

| Family | Retained queries | Candidate pairs |
|---|---:|---:|
| Constraint | 14 | 1,674 |
| Comparative | 14 | 1,543 |
| Contradiction | 13 | 1,366 |
| Multihop | 14 | 1,511 |
| Temporal | 12 | 1,306 |
| Aggregation | 14 | 1,809 |
| Negative | 12 | 1,299 |
| Abstention | 17 | 2,149 |
| **Total** | **110** | **12,657** |

There are 226 seed-containing pairs, 550 uniform out-of-pool pairs, and 3,609
unique supplied documents. Of the pairs, 3,986 have parsed full text and 8,671
have abstract-only evidence. Abstract-only means only the available abstract
can be assessed; it does not mean the complete paper has been reviewed.

Across 4,936 corpus records, 24 colliding normalized lookup keys implicate 32
records. Forty-three records with body material have no verifiable local XML at
the normalized PMCID path; seven overlap the collision group. The union of 68
records is withheld entirely, including their abstracts. No dictionary keeps an
arbitrary last record. Every alias belonging to an affected record is tainted.
There are 322 excluded candidate references among retained queries (not necessarily
322 distinct normalized papers). Whole-query exclusion takes precedence over
pair pooling, so that number does not count the candidate pools of dropped queries.

Whole-query exclusions because a seed is unsafe or absent:

| Query | Affected seed ID |
|---|---|
| `contradiction_0139` | `6286148` |
| `temporal_0229` | `8386155` |
| `temporal_0211` | `8386155` |
| `negative_0295` | `8809250` |
| `negative_0298` | `PMC10543554` |

These exclusions change the family balance and can bias the audit toward clean,
available identities. Report the selected and retained denominators together.
The full per-record reasons and affected candidate references are in
`coordinator/identity_audit.json`, indexed by zero-based corpus row.

## Identity and evidence safeguards

The audit runs over the entire corpus, not just sampled papers. It normalizes
Unicode/case/whitespace and numeric leading zeros, retains raw document-ID aliases,
and cross-indexes PMID, normalized PMCID and DOI. Multiple corpus records sharing
any such key are all quarantined, including apparent duplicate aliases; a
coordinator must resolve them upstream rather than guess which content is right.
Unresolvable candidate IDs are withheld and unsafe seed IDs exclude the query.
For body-bearing records, locally available XML must have the matching PMID and
PMCID; missing, malformed or conflicting XML identity blocks the record.

URLs come only from explicit metadata: `pmid` yields a PubMed URL; `pmcid` yields
a normalized `PMC...` article URL. A digit-only **document ID is never assumed to
be a PMID**. No URL is fetched. This is a conservative audit of supplied metadata
and local XML, not external confirmation of publication identity, bibliographic
truth, or all conceivable semantic duplicates. DOI spelling variants beyond the
normalization implemented in the script may need additional upstream curation.

Evidence JSON preserves untruncated text in section objects with stable
`section_id`s. Document provenance records the corpus input SHA-256 and 1-based
source line. The manifest hashes each consumed input file, including local XML,
and every initially built output file. Evidence is the available parsed corpus
text, not a guarantee of complete original PDF layout, supplementary materials,
all formulas or an independently re-parsed article. Missing text warrants `U`
when it prevents a defensible assessment, never a fabricated span or a negative
claim about unseen full text.

For every supplied span, provide its exact `section_id`, zero-based start and
end-exclusive offsets measured in **Unicode codepoints within that section's
text**, not bytes, serialized JSON positions or display line numbers. The
validator requires `text[start:end] == evidence_span`, checks bounds and checks
document identity. Direct relevance requires a nonempty exact span. Do not use
ellipsis, paraphrases, concatenated disjoint quotations, or text from another
document. Use notes for additional section/offset references if one primary span
cannot express all rationale; any extra quoted evidence needs manual adjudication.
JSON can be opened in a local editor; no website is needed for evidence access.

Hashes detect changes relative to the manifest, not malicious replacement of the
manifest itself; they are not signatures or proof that a person supplied a label.
Archive the frozen manifest outside annotator control before any future collection.

## Future annotation instructions — not active collection

Before inspecting candidates, each reviewer independently interprets the task on
their `task_requirements` sheet: articulate constraints, define named required
units (such as C1 or H1), and describe what would make a valid evidence pair or
ordered path. Do not adopt seed paper membership or a model answer as truth.
The supplied `required_constraints` column is an unvalidated task specification,
not a human-approved requirement; flag defects rather than silently repair it.

Judge each candidate using:

- **0**: no relevant evidence for the task in the supplied text. This is scoped
  to the available text, not proof of absence from the complete paper/corpus.
- **1**: useful context or indirect evidence, not direct support for a required unit.
- **2**: direct evidence for at least one required task unit, with exact quotation
  and role/unit assignment. It need not independently answer the entire question.
- **U**: unresolved, ambiguous, missing evidence, unclear task, or insufficient
  expertise. It is not a negative label and never enters direct-relevance kappa.

`role_json` must be an object with exactly `roles` and `units`, both lists of
unique nonempty strings. Allowed roles are `constraint`, `comparison_A`,
`comparison_B`, `finding_A`, `finding_B`, `reconciliation`, `hop`, `temporal`,
`measurement`, `explicit_null`, `negative_direction`, `failed_replication`,
`context`, `other`. Example syntax (not an annotation):
`{"roles":["hop"],"units":["H1"]}`. Use empty lists where no role/unit is
established; label 2 requires both lists to be nonempty. Units must later be
reconciled with task-level definitions, not accepted solely because they parse.

Set `independent_sufficiency` to `YES`, `NO` or `U`: can this single document,
alone, satisfy **all** task requirements? `YES` requires relevance 2, but relevance
2 does not imply `YES`. Set confidence to `LOW`, `MEDIUM` or `HIGH`. Explain
uncertainty in notes whenever relevance or sufficiency is U, or confidence is LOW.
Each completed row requires a distinct reviewer identifier, timezone-bearing ISO
timestamp and `provenance_kind=human`. Those fields must be entered only by actual
authorized reviewers. Test fixtures are synthetic software inputs, never human data.

At task level, assess collective sufficiency and record valid-pair/path requirements,
evidence limitations and abstention scope. Comparative evidence must cover the
comparison's sides and axis; contradiction needs claims and conditions rather
than merely different wording; multihop requires ordered linked units; temporal
requires the relevant times/bins; aggregation requires compatible measurements,
units and conditions; negative findings require explicit null/negative/replication
evidence. No retrieved evidence or a finite pool of 0s proves unanswerability.
Abstention assessment must distinguish “no support found in reviewed candidates”
from “corpus-wide absence,” which this design cannot establish.

## Validation, agreement and adjudication gates

Keep original blank files frozen; completed sheets must be saved separately.
The validator rejects empty sheets, duplicate IDs/headers, missing/extra pairs,
query/document/title or other metadata differences from the frozen reference,
malformed roles, missing reviewer provenance, invalid labels, and non-occurring
or mislocated spans. Two reviewers must have distinct identifiers. Matching
identity strings alone cannot establish actual independence or authenticate humans.

Four-way nominal Cohen's kappa includes 0/1/2/U for descriptive agreement.
Direct-relevance binary kappa compares 2 against 0/1 **only on pairs where neither
reviewer selected U**. Report eligible n and excluded U-pair count. Empty or
constant-marginal/degenerate kappa is `null` (undefined), never 1.0. Do not report
the blank template's agreement as an empirical result.

Every label, role/unit, span/offset, sufficiency, confidence or note disagreement
is routed to adjudication. Any U, **including U/U**, and any LOW confidence is
also routed even if reviewers agree. Equal role lists are compared semantically,
not by JSON whitespace/order. An empty adjudication queue would still not create
final qrels: agreement is not adjudicator sign-off.

A separately authorized future adjudicator must review all uncertainty and
disagreement, both independent task interpretations, exact evidence, and task
requirements. Record adjudicator identity, qualification/role, timezone timestamp,
decision, rationale, validated spans/units and the hashes of both reviewer sources
and the frozen corpus/split. Do not fill any adjudicated field automatically.
If a U cannot be resolved, withhold that pair/task and report attrition; do not
map it to 0 or quietly emit a partial final qrel set.

The planned structural worker record has exactly:

```text
task_id: str
task_family: str
annotation_status: unvalidated | human_adjudicated | expert_adjudicated
required_units: list[str]
evidence_units: dict[doc_id, list[str]]
valid_pairs: list[list[doc_id]]          # exactly two distinct documents each
valid_paths: list[list[doc_id]]          # ordered, >=2 distinct documents each
provenance: dict
```

Currently every record remains a template: `unvalidated`, empty collections and
empty provenance. The validator checks this state and basic schema, but **does
not import completed task sheets or adjudicated structural records**. A future
release implementation must validate unit membership, valid path logic,
identity-safe document membership, agreement reconciliation and authentic
annotator/adjudicator provenance against frozen sources. Until then final qrels
and adjudicated structures are blocked, even for fully completed pair sheets.

## Independent human-written query collection design

The blank collection targets 80 queries (acceptable predeclared range 50–100),
10 slots per task family including abstention. Target role, family, domain and
full-text availability fields are sampling quotas only; combinations are not a
fully crossed or population-representative design. Record actual strata separately.
Target domains are biomedicine/computational biology; roles are researcher,
clinician, information specialist and graduate researcher. The actual-role field
must reflect real qualifications, not a role assigned to an unqualified writer.
There are no collected questions, writers, consent records or payments here.

If collection is authorized later, give writers only a neutral description of
the scientific search task and target strata. Have them write genuine information
needs **before seeing seed qrels, seed papers, benchmark task text, reference
answers, retrieval outputs or model suggestions**. Do not paraphrase the existing
synthetic questions. Writers must record an independence attestation and disclose
prior CERBench/test exposure. Freeze and hash original wording before candidate
retrieval/annotation. Classify actual family/domain/full-text availability afterward
without retroactively steering writing toward known answers. Preserve original
wording and revision provenance; use separate reviewers to assess feasibility.

Record informed-consent documentation or the determination of applicability,
withdrawal/data-retention terms, compensation terms (including unpaid status only
if genuinely agreed), and ethics-review applicability and any actual institutional
reference. Do not assert IRB approval, exemption or non-applicability without an
authorized determination. Use pseudonymous writer IDs and keep identifying records
outside released research files. None of these blank columns proves compliance.

New independently written questions are not automatically a clean holdout if
writers saw the benchmark, corpus repair used test feedback, or the same paper
components cross train/dev/test. Freeze an uncontaminated component-disjoint
evaluation design before claiming prospective generalization.

## Workload and milestones (conditional planning, not an authorized study)

Actual kit size entails **25,314 pair judgments** from two independent reviewers
and **220 task-level reviews**. At an assumed 2–5 minutes per pair, pair review
alone requires **843.8–2,109.5 person-hours**. If 10–30% of pairs need adjudication
at 5–10 minutes each, allow **105.47–632.85 additional person-hours**. These are
planning assumptions, not measured timings; long full-text/path assessments may
take substantially longer. Training, task-level review, recruitment/ethics, query
writing and final audit are additional. Do not commit a completion date or cost
without actual authorized staffing and a timed pilot.

1. **Current milestone:** offline blank kit, identity quarantine, integrity checks
   and software tests only. No human participation or empirical validation.
2. **Repair gate:** corpus owner resolves aliases/missing XML provenance, regenerates
   impacted corpus/chunks/indexes/runs, freezes component-disjoint splits and source
   hashes. Rebuild into a newly authorized version, not over this frozen kit.
3. **Authority gate:** explicit authorization for actual humans, ethics/applicability
   determination, consent/compensation arrangements and independent reviewers.
4. **Writing gate:** collect and freeze 50–100 independent queries under the
   visibility restrictions above before showing seed/model outputs.
5. **Pilot gate:** timed training/pilot separate from confirmatory test; revise and
   freeze the rubric before main independent annotation. Preserve pilot provenance.
6. **Review gate:** A/B independent pair and task judgments, validation and kappa
   with all denominators; adjudicate uncertainty and disagreements.
7. **Release gate:** separately implement/review provenance-checked structural and
   final-qrel export, require no unresolved labels, signed-off corpus/split identity,
   explicit exclusion accounting and contamination disclosure. Not implemented now.

## Local commands and verification

From the analysis directory, with the verified interpreter:

```bash
PY=/Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python
"$PY" -m pytest -q -p no:cacheprovider tests/test_human_validation.py
"$PY" scripts/42_prepare_validation_kit.py
"$PY" scripts/43_validate_human_labels.py --blank-kit \
  --report annotations/human_qrel_v2/local_validation_report.json
```

The build and initial validation have already run successfully. Re-running the
builder now **must refuse overwrite**. Recheck existing material without a new report:

```bash
"$PY" scripts/43_validate_human_labels.py --blank-kit
```

The initial local check verified 544 input-source hashes and 3,617 kit-output
hashes, 12,657 blank pairs per annotator and zero human judgments. The initial
software run passed 33 tests; tests use synthetic fixtures only. The manifest
does not hash itself, nor later validation reports. Any future two-sheet check
requires separately saved human-completed sheets; the blank pair sheets correctly
fail completed-label mode. Reports may be created only inside the v2 kit and
never overwrite an existing report. No network or `rank_bm25` dependency is used.

The final regression run passed **36 tests** and repeated the blank-kit checks,
including task and query templates. Three separate CLI negative checks also
passed (each returned exit code 1): rebuild rejected with `Refusing to overwrite`,
legacy merge rejected with `Legacy merge disabled`, and the blank A/B sheets
submitted as completed labels rejected with `Invalid/missing relevance`.

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest -q -p no:cacheprovider \
  tests/test_human_validation.py \
  --junitxml=annotations/human_qrel_v2/software_tests.xml
PYTHONDONTWRITEBYTECODE=1 "$PY" scripts/43_validate_human_labels.py --blank-kit \
  --report annotations/human_qrel_v2/final_local_validation_report.json
```

These report paths already exist; omit `--report` for repeat integrity checks
and select a new authorized report name if preserving another run. The query
collection has its own CSV SHA-256 in its manifest, verified in the final check.
