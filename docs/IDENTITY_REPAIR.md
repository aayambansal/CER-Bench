# Local identity repair candidate — release blocked

This is a deterministic, deliberately lossy integrity repair candidate, **not gold,
release-ready data, or a fresh test set**. No network, API, package installation,
paid compute, secret access, or Git operation was used. Historical corpus, chunks,
tasks, indices and result files were not rewritten by this implementation.

## Observed identity/content defect

The original corpus has **4,936 records / 4,923 distinct document IDs**; its chunks
have **17,902 records / 17,770 distinct chunk IDs**. There are 304 original tasks.

Old ID `6286148` is assigned to three corpus PMIDs:

| Corpus PMID | Corpus title |
| --- | --- |
| 38308006 | CRISPR technologies for genome, epigenome and transcriptome editing. |
| 30610625 | Rapid Screening of CRISPR/Cas9-Induced Mutants Using the ACT-PCR Method. |
| 40828286 | Manipulation of a New Non-model Insect Genome Using Targeted CRISPR-Era Approaches. |

These same PMCID claims are already present in
`data/raw/metadata/pubmed_openalex_metadata.jsonl`. The parsed record keyed by
`6286148` instead has the title “A programmable dual RNA-guided DNA endonuclease
in adaptive bacterial immunity”. The article-front IDs in
`data/raw/fulltext/6286148.xml` are **PMC6286148 / PMID 22745249**, and its title
matches that parsed title. This is direct local evidence of a fulltext attachment
mismatch, not merely duplicate output names. The old builder joins parsed content
by the metadata PMCID without verifying an independent PMID.

The audit found **58 corpus records whose claimed PMCID's raw article-front PMID
disagrees with the corpus PMID**, and **124 conflicting alias keys** after combining
local corpus/metadata claims and raw article-front evidence. Alias keys are not a
count of independent papers: bare and prefixed aliases can describe the same
conflict. Of 645 raw XML files, 642 supplied a single readable article-front identity.
Files `26027431.xml`, `30207593.xml`, and `30620402.xml` failed the strict article
shape check (`Expected exactly one top-level article`). Their identity remains
unresolved; the audit does not infer it from filenames or references.

## Repair policy and retained coverage

`scripts/39_audit_repair_identity.py` uses only Python's standard library.

* PMID identifiers normalize to `PMID:<positive integer>` and PMCID identifiers
  to `PMC<positive integer>`. Explicit namespace prefixes and supported NCBI URLs
  normalize case/leading zeros. Bare numbers require an explicit namespace in
  source identity fields; old unqualified references are resolved against *all*
  observed aliases. Unknown or multiple candidates fail closed.
* Canonical PMID records are metadata/abstract-only candidates. Source metadata
  PMID uniqueness and exact title/abstract agreement are required. Duplicate
  canonical PMIDs are quarantined, not merged. Active PMCID fields are cleared;
  the normalized, unadjudicated claim survives in `identity_provenance` and mapping.
* **All fulltext fields are removed**, including apparently consistent fulltext.
  Identity agreement alone does not validate parsing or content attachment.
  Fulltext-bearing source records remain flagged in provenance.
* An old chunk is retained only if its old document ID resolves uniquely, its
  copied PMID agrees, its old chunk ID is unique, and its text exactly equals the
  source metadata abstract prefix (2,048 characters). Duplicate canonical chunk
  targets are also quarantined. The copied PMID never disambiguates an ambiguous
  old document ID. No missing candidate chunks are synthesized.
* Tasks are quarantined if any annotated supporting/negative/passage/qrels ID is
  ambiguous, unknown, unsafe, or lacks a retained candidate chunk. Tasks touching
  any originally fulltext-bearing document are excluded regardless of passage
  type, since their generation context may have been contaminated. Explicit
  supporting passages must match the retained metadata abstract. Unknown or
  quarantined chunk references also fail closed.
* Annotated references are rewritten recursively. Prose, answers and constraints
  are **not** rewritten or scientifically validated. Per-task legacy citation maps
  explain surviving old citations. Historical `verification_status` values, such
  as `auto_verified`, are preserved source annotations, **not repair validation**;
  new `identity_provenance.status` is `unadjudicated_proxy_candidate`.

| Item | Original | Candidate | Excluded/quarantined |
| --- | ---: | ---: | ---: |
| Corpus records | 4,936 | 4,936 unique canonical PMIDs | 0 whole records |
| Fulltext-bearing corpus records | 652 | 0 | 652 stripped of fulltext |
| Chunks | 17,902 | 4,856 abstract chunks | 13,046 |
| Tasks | 304 | 238 | 66 |

**80 candidate documents have no retained chunk.** Tasks using those uncovered
documents are excluded. Whole-corpus record retention must not be confused with
full retrieval/evidence coverage. The 652 fulltext-bearing records and 645 raw
files are different units; repeated/mismatched source associations exist.

Task retention by family: abstention 34/38, aggregation 28/38, comparative 26/38,
constraint 33/38, contradiction 28/38, multihop 34/38, negative 27/38, temporal 28/38.
This conservative loss changes benchmark composition and comparability.

## Component split contract

`scripts/38_make_component_disjoint_splits.py` accepts `--source`, `--output`, and
`--seed`. Its defaults are the new identity candidate and
`data/benchmark/v1_2_repaired_components`, never historical v1.1. Source/output
identity and the historical v1.1 output path are rejected. Use canonical repaired
inputs: this splitter groups annotated IDs but does not independently adjudicate
arbitrary source identity aliases or repair fulltext.

Before publication it rejects missing/duplicate task IDs and unions components
using every recursively annotated document/negative/passage/qrels ID, chunk ID,
normalized exact question, token-Jaccard question pair at **>= 0.85**, and explicit
`evidence_cluster_id`/`cluster_ids` (including nested annotations). Explicit cluster
IDs share one namespace, including transitive links across the two fields.

Every row gets an updated `split`, a stable SHA-256 membership-derived
`component_id`, and `source_split` from the actual source filename. `source_split`
means the immediately preceding source file, not an indefinitely preserved first
generation split. Seeded greedy assignment balances overall/family counts toward
15:8:15; exact family quotas and nonempty splits are not promised for arbitrary
component graphs. Empty task input is rejected.

Both in-memory rows and serialized staged files are checked for all grouped
overlaps before directory publication. Existing output directories are refused.
Publication uses a temporary sibling directory, an exclusive cooperating-writer
lock and atomic directory rename; validation failure removes staging and leaves no
published output. It is not a hostile-filesystem defense against noncooperating
writers, nor a multi-directory transaction across audit and split publication.
Manifests include source/output hashes and a second source-hash check before
publication. Manifest/report hashes are chained by consumers rather than included
as impossible self-hashes.

Observed split: **186 components**, largest **7 tasks**; **94 train / 49 dev / 95
test**. All checked cross-split overlaps are **zero**: documents, chunks, explicit
clusters, exact questions and lexical near duplicates. Seven lexical pairs and
two exact-question grouping edges were found. Actual tasks contain no explicit
cluster annotations or chunk references, so those zero-overlap claims are vacuous
for this dataset; dedicated tests exercise those paths. Semantic/template leakage
below the lexical threshold is not ruled out. The test tasks were already exposed.

## Files and provenance

All paths below are relative to `analysis/synthetic-science-search`.

* `data/processed/identity_repair_v1/report.json`: counts, policy limitations,
  source SHA-256 inventory, and hashes of all 14 non-report output files.
* Same directory: `corpus.jsonl`, `chunks.jsonl`, `train.jsonl`, `dev.jsonl`,
  `test.jsonl`; `mapping.jsonl` (source-line document decisions), `aliases.jsonl`,
  `alias_conflicts.jsonl`, `chunk_mapping.jsonl`; `raw_identity_evidence.jsonl`,
  `fulltext_identity_evidence.jsonl`; `quarantine_corpus.jsonl`,
  `quarantine_chunks.jsonl`, `quarantine_tasks.jsonl` (original records/reasons).
  Empty whole-record corpus quarantine is intentional: original fulltext remains
  in the hashed historical corpus and excluded chunks are separately quarantined.
* `data/benchmark/v1_2_repaired_components/{train,dev,test}.jsonl` and
  `manifest.json`: new component split and hash-linked identity source report.
* `results/readiness/identity/commands.json`, `pytest.xml`, `pytest.stdout.txt`,
  `pytest.stderr.txt`, `audit.stdout.txt`, `splits.stdout.txt`: exact execution
  arguments, environment overrides, return codes and logs.
* `results/readiness/identity/verification.json`: independent disk/reference checks,
  exact task payload preservation across splitting, family coverage, raw identity
  exceptions and byte-identical replay results.
* `results/readiness/identity/historical_hashes_{before,after}.json` and
  `preservation.json`: 91 historical data/result files matched across the initial
  generation run. Later replay observed one changed, separately owned file,
  `results/readiness/budget/tests.xml`; it was not modified by this phase.
  The first broad preservation assertion failed on that external result change;
  `verification_attempts.json` records it. The revised verifier still fails on
  data-source changes, but reports other-result changes without overwriting them.
  On final replay 90 snapshot files were unchanged; all audited data-source hashes
  matched. This distinction avoids claiming project-wide immutability amid workers.

Owned code changes: `scripts/04_build_corpus.py` now rejects existing historical
outputs before optional YAML imports, duplicate metadata/parsed identities,
normalized alias conflicts, and parsed merges without a matching PMID field.
This is a guard, **not a corrected/revalidated fulltext build**. The historical
builder was not run to regenerate data. Other changed/new code is limited to
`scripts/38_make_component_disjoint_splits.py`,
`scripts/39_audit_repair_identity.py`, `src/corpus/identity.py`, the two owned test
files and this identity readiness directory. No other workers' files were edited.

## Commands and checks

Working directory:
`/Users/aayambansal/Desktop/research/research-repos/CERBench/analysis/synthetic-science-search`

Python:
`/Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python`

The executed driver was:

```sh
/Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python -B results/readiness/identity/run_local.py
/Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python -B results/readiness/identity/verify_local.py
```

`commands.json` records the exact expanded subprocess commands. The driver sets
`PYTHONDONTWRITEBYTECODE=1` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, then runs:

```sh
"$PYTHON" -m pytest -q tests/test_identity_repair.py tests/test_component_splits_identity.py -p no:cacheprovider --junitxml="$ROOT/results/readiness/identity/pytest.xml"
"$PYTHON" scripts/39_audit_repair_identity.py --root "$ROOT" --output "$ROOT/data/processed/identity_repair_v1"
"$PYTHON" scripts/38_make_component_disjoint_splits.py --source "$ROOT/data/processed/identity_repair_v1" --output "$ROOT/data/benchmark/v1_2_repaired_components" --seed 20260826
```

Here `$ROOT` and `$PYTHON` denote the exact paths above. **31 tests passed** using
stdlib/pytest only. Coverage includes namespace normalization/rejection, ambiguous
collisions across namespaces, copied-PMID non-disambiguation, duplicate IDs,
qrels alias collapse, raw front IDs versus reference IDs, preservation, deterministic
audit/split output, stale split correction/rejection, each overlap relation,
transitive explicit clusters, empty sources, no-overwrite and injected staged
validation failure. Independent replay checked **663 source-hash entries and 17
output-hash entries**, and reproduced every audit/split file byte for byte.

The initial publication driver is intentionally one-shot: rerunning with existing
output paths will refuse overwrite. For verification use `verify_local.py`, which
replays into a temporary directory inside identity readiness and cleans it up. A
new candidate publication requires a separately authorized new output path, not
deleting or overwriting these artifacts.

## Remaining release blockers

Canonical PMID names are not external identity adjudication. Local metadata,
OpenAlex enrichment, synthetic answers, constraint claims and qrels remain
unverified. Unknown unstructured prose citations are not parsed into authoritative
identity assignments. Do not combine candidate tasks with historical indices,
retrieval outputs or scores. No evaluation scores were rerun or migrated.

A future, separately authorized effort would need to adjudicate source identities,
reconstruct fulltext only with independent matching evidence, regenerate affected
tasks/qrels, rebuild candidate-specific indices, and obtain scientifically validated
annotations and genuinely new held-out tasks. Until then, both reports retain
`release_blocked: true`.
