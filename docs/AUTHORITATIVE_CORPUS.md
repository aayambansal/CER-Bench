# Authoritative PubMed corpus v1

## Decision and scope

Public-source metadata verification was **executed**, not simulated, on
2026-09-09. All **4,936 unique historical corpus PMIDs** were retrieved and
parsed from PubMed EFetch XML. Metadata coverage/identity release passes;
benchmark-gold release does not. Human validation was explicitly **waived and
not done**. External authoritative identity is not biomedical relevance,
clinical correctness, or a right to redistribute article text.

The new corpus is `data/processed/authoritative_v1/corpus.jsonl`. Its canonical
`doc_id` is the PMID, including when a PMCID is available. Historical corpus,
chunks, tasks, qrels, raw metadata/full text and historical results were not
rewritten. Only the owned new raw/versioned/readiness paths and parser/client,
builder, tests and this document were written. Scripts 48+ and the main report
were not edited. No secrets were inspected or logged; no paid APIs, model calls,
remote jobs, package installs, or Git commands were used.

## Source and root cause

The historical parser traversed `PubmedData` with `.//ArticleId` (the old
`src/corpus/pubmed_client.py` lines 218–230). That includes IDs inside
`ReferenceList/Reference/ArticleIdList`, not just the current article's own IDs.
Both its PMCID extraction and DOI fallback could therefore inherit cited-paper
identifiers. A PMCID collision is not necessary for this failure.

The retained public [three-record pilot](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=38308006,30610625,40828286&retmode=xml)
demonstrates the bug directly for PMIDs `38308006`, `30610625`, and `40828286`:

- All three own `PubmedData/ArticleIdList` elements have **no PMC identifier**.
- The old descendant traversal's first PMC value is **`6286148`**, attached to a
  reference to Jinek et al., *A programmable dual-RNA-guided DNA endonuclease in
  adaptive bacterial immunity*, Science 337:816–821 (2012).
- Pilot file: `data/raw/pubmed_verified_v1/identity_pilot.xml`, **183,718 bytes**,
  SHA-256 `ce89488f124e3ba1f77ac4768a9bfd88049c13d92ef5331916ea9ebb707356a0`.
  It is evidence only, excluded from the batch union to avoid duplicate PMIDs.

The complete comparison found **226 changed historical PMC IDs** and **all 3
changed historical DOIs** in reference IDs in the newly retrieved XML. This is
direct evidence consistent with cited-ID contamination. Other differences can
also reflect PubMed updates or deliberately changed parser rules; they should
not all be attributed to this bug.

`src/corpus/pubmed_xml.py` is stdlib-only. It uses strict direct paths for
article-level IDs and `itertext()` for mixed-inline titles/abstracts. Own
ArticleIdList DOI takes priority; only direct own Article/ELocationID is a
fallback. Cited IDs never supply missing own IDs. PMC numeric values normalize
to `PMC` plus digits. Namespace-qualified PubMed XML is supported. Conflicting
own IDs, missing article identity, duplicate records, unexpected returns and
unsupported record types are explicit diagnostics. Book records are detected
and reported, not silently converted to journal articles. Repeated identical
IDs within one own list are allowed; duplicate article records are not.

`pubmed_client.py` delegates parsing to that module, lazily imports its legacy
optional Bio transport, and no longer imports yaml/configuration or reads an
API-key environment variable at import. That transport was **not used** for
this verification; all external retrieval was through WebFetch.

## Executed retrieval and provenance

There were **26 sequential successful WebFetch XML downloads**, each requesting
at most 200 IDs, well below NCBI's three-requests/second limit. They contain
**140,943,352 bytes** in total (excluding the earlier pilot). All returned
records are PubmedArticle records; none are missing, unsupported or duplicated
in the final corpus union.

`data/raw/pubmed_verified_v1/webfetch_requests.jsonl` records exact actual URLs,
requested IDs, broker-reported SHA-256 and content type. Each
`pubmed_batch_NNN.xml.manifest.json` additionally records returned/missing IDs,
actual byte size and diagnostics. UTC retrieval times use the local creation
download's mtime; this is not a server modification/publication date. Every
broker checksum was verified before transfer from scratch. Existing raw files
were guarded against overwrite. Raw XML and batch manifests are retained.

**Request anomalies, fully recovered:**

1. Batch 002 omitted `27798150` while transcribing the URL (199 requested/returned).
2. Batch 003 omitted `28770670` (199 requested/returned).
3. Batch 021 mistakenly requested `40240287758` instead of `40287758` (200
   requested, 199 returned). PubMed did not return that invalid ID.
4. Batch 026 explicitly requested and returned `27798150,28770670,40287758`.

Thus the logs contain **4,937 requested ID occurrences/unique IDs**, including
the one invalid non-corpus ID, and **4,936 returned unique corpus PMIDs**.
There were **zero HTTP/download failures**, no unchanged 404 retries, and no
remaining missing intended IDs. The request log preserves these anomalies
rather than retroactively substituting nominal planned URLs.

Every output document carries its batch URL, raw path, SHA-256, retrieval date,
transport and extraction field paths. Readiness `output_manifest.json` hashes
the derived corpus, chunks, mappings and audit outputs. The historical corpus
input SHA-256 is
`88deb174b7725b969f1eecaefd3c16121eed5349675e44a315d0f5ec18ac4668`.

## Exact results

| Quantity | Count |
|---|---:|
| Historical rows / unique PMIDs | 4,936 / 4,936 |
| Authoritative parsed / compared rows | 4,936 / 4,936 |
| Own authoritative PMC IDs | 2,641 |
| Own DOI present | 4,916 |
| Authoritative abstract chunks | 4,914 |
| Rows changed, including canonical doc_id | 3,096 |
| Rows with metadata changes excluding doc_id | 815 |
| Canonical doc_id changes | 2,840 |
| PMCID changes | 247 |
| PMCID removals / replacements / additions | 219 / 8 / 20 |
| DOI changes | 3 |
| Title / abstract changes | 118 / 22 |
| Author-list / MeSH changes | 26 / 178 |
| Publication-type changes | 277 |
| Venue / keyword changes | 1 / 1 |
| Year / venue-abbreviation changes | 0 / 0 |
| Missing/unsupported metadata records | 0 |

Counts overlap across fields. Changes are exact parsed-value comparisons, not
semantic biomedical judgments. Every historical PMID has a comparison row,
including unchanged metadata. The three removed DOIs are:

| PMID | Removed historical DOI |
|---|---|
| 26958284 | `10.1038/msb.2011.26` |
| 34355196 | `10.1016/s0939-4753(04)80048-6` |
| 38694604 | `10.1126/science.1258096` |

Optional authoritative field absences are explicit: **2,295 without PMC**,
**20 without DOI**, and **22 without an abstract**, across **2,307 distinct
records**. None lack a title or year. An absent optional field in a returned
valid source record is not a failed download. These absences are not imputed
from historical data. The 22 abstract-less documents remain in the corpus but
have no abstract chunk and are ineligible for abstract-based task generation.

## Local full text and safe task eligibility

All **645 local full-text XML files** were inspected using only article-level
JATS `front/article-meta/article-id`. Neither filenames, references, body IDs
nor subarticle IDs were used as identity. A candidate requires the own PubMed
PMCID plus local front PMID and PMCID to agree, with uniqueness across local
files and authoritative records.

- **590** documents have a unique identity-matched local raw candidate.
- **2,051** have an authoritative PMC but no matching local front.
- **2,295** have no authoritative PMC and cannot receive local full text.
- Of 645 raw files, **642** have usable front identity candidates and **3**
  have unsupported/multiple-article wrappers. All are listed individually.
- **633** raw files contain permissions XML; **587** of the 590 matched
  candidates contain it. Permissions XML retains namespace-qualified attributes
  and links. Original local XML is unchanged and linked by path/hash.

**Zero full-text bodies were restored.** The 590 matches are marked
`identity_matched_raw_candidate_text_unvalidated`. Validating a conservative
section parser is deferred rather than relabeling historical sections as
verified text. Every new corpus row has `has_fulltext=false`, empty sections,
figure captions and table texts. Historical `has_fulltext=true` occurred on 652
rows; that flag is not inherited. Index chunks contain only authoritative
abstract text—never concatenated article references or historical body text.

The per-document eligibility mapping has **4,936 rows**. It permits considering
title+abstract records for **new task generation**, not as gold. A separate
**3,242-row historical task-occurrence mapping** covers task files under
benchmark, interim and training; its rows include source path/line, task/support
IDs and candidate PMID mappings only. A legacy ID can map to multiple PMIDs;
ambiguity is retained, not resolved by arbitrary selection. No questions,
answers, support passages or labels were copied into new gold. Every historical
task is marked ineligible for automatic migration/gold; tasks and qrels require
fresh generation and evidence validation. Human validation remains waived/not
done, even if later non-human experiments succeed.

License/redistribution status is **`not_assessed_no_rights_granted`**. PubMed
access, presence in PMC, an identity match or retaining a license statement is
not itself a redistribution grant. Metadata-release pass does not approve
public full-text/abstract redistribution or certify benchmark relevance.

## Reproduction and tests

Use the verified interpreter; all rebuild/tests are local and offline:

```bash
PY=/Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python
"$PY" scripts/47_build_authoritative_corpus.py
"$PY" -m pytest -q tests/test_pubmed_authoritative.py tests/test_identity_repair.py tests/test_component_splits_identity.py --junitxml=results/readiness/authoritative_corpus/tests.junit.xml
```

The executed targeted/adjacent run passed **51 tests** (20 new authoritative
cases including parametrizations, 31 existing identity/component cases).
`tests.junit.xml` records the result. Tests cover cited-ID contamination, own
ID priority/absence, mixed-inline titles, namespaces, missing/book/nonreturned
records, duplicate requests and records, invalid responses, checksum failures,
incomplete release failure, cross-batch duplicates, misleading full-text
filenames, conflicting/multiple JATS identities, license namespaces and imports
without Bio/yaml or environment reads. The public-snapshot acceptance test
independently re-reads all 26 XML batches and checks each of the 4,936 output
PMCs/titles against direct own XML paths, plus canonical IDs and no restored
full text.

A subsequent clean offline rebuild exited 0 and reproduced all **9 manifest-
tracked derived outputs bit-for-bit**. SHA-256 checks confirmed **666 historical
input files** (corpus, chunks, local full text and task files) were unchanged
across that rebuild. `verification.json` retains those checks and the JUnit
checksum; it does not claim an unobserved before/after check for earlier work.

For a **new retrieval**, `--plan` prints deterministic <=200-ID request plans
but performs no network calls. Use WebFetch sequentially, with a new scratch
filename for each XML download. Record actual requests and broker hashes in an
explicit JSONL log before calling `--ingest SCRATCH --request-log LOG`. The log
requires filename, URL, requested_pmids, sha256 and retrieved_at_utc. Do not
ingest into existing batch filenames; the command refuses overwriting. Existing
`authoritative_v1` derived outputs can be reproduced offline from the retained
raw snapshot without retrieving again. A genuinely new snapshot should use a
new version rather than overwrite v1 source evidence.

The builder exits **2** and sets `metadata_release_pass=false` when intended
PMIDs are missing, invalid, unsupported or duplicated, checksums disagree,
unexpected IDs are returned, or required titles/unique identities fail. It exits
**0** only for complete valid metadata coverage. `benchmark_gold_release_pass`
stays false, independent of this metadata gate.

## Output inventory

Paths are relative to `analysis/synthetic-science-search/`:

- `src/corpus/pubmed_xml.py`: strict offline parser and JATS front inspector.
- `src/corpus/pubmed_client.py`: compatibility transport/parser refactor.
- `scripts/47_build_authoritative_corpus.py`: plans, guarded ingestion, rebuild,
  all-record comparison, safe mappings and release gate.
- `tests/test_pubmed_authoritative.py`: offline regression/acceptance tests.
- `data/raw/pubmed_verified_v1/`: pilot, 26 new raw batches, exact request log and
  26 per-batch manifests.
- `data/processed/authoritative_v1/`: `corpus.jsonl`, `chunks.jsonl`,
  `document_task_eligibility.jsonl`, `historical_task_eligibility.jsonl`.
- `results/readiness/authoritative_corpus/`: `summary.json`,
  `metadata_comparison.jsonl`, `metadata_absences.jsonl`,
  `missing_unsupported_records.jsonl` (empty on this successful snapshot),
  `local_fulltext_audit.jsonl`, `output_manifest.json`, `tests.junit.xml`,
  `verification.json`.

Next work should use this versioned corpus for fresh non-human retrieval/task
experiments, not promote historical qrels to gold. Validate JATS section
extraction and licensing separately before enabling the 590 raw candidates.
