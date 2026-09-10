# Verified local full-text retrieval dataset

## Outcome and worker49 handoff

The local-only restoration/integration phase is complete. The **fresh manifest**
for worker49's real index build is:

`data/processed/authoritative_fulltext_v1/dataset_manifest.json`

Paths in this document are relative to
`/Users/aayambansal/Desktop/research/research-repos/CERBench/analysis/synthetic-science-search`.

The dataset contains **4,936 documents and 10,313 chunks**, including conservative
own-article body text restored for **525 documents**. Worker49's actual
`src.retrieval.verified_bm25.validate_inputs` gate passed. No full-corpus index,
search run, task generation, or relevance evaluation was performed in this
phase. No historical task/qrel mappings were read or produced.

The manifest has schema `verified-retrieval-dataset-v1`, status
`authoritative_validated`, `identity_validated=true`, and
`canonical_ids_unique=true`. **These attest source identity/structural text
ownership only, not biomedical relevance, semantic fidelity, human validity,
publication readiness, gold labels, or redistribution rights.**

Human validation is **false**, explicitly **waived/not done**. Use scope remains
**local research only**. No additional web retrieval, paid APIs, credentials,
Git commands, package installs, remote jobs or provider calls were used.
`authoritative_v1`, its existing manifests, and original 645 local XML files were
not rewritten. No main readiness documentation or provider code was changed.

## Source identity and title policy

The builder first checks the authoritative corpus hash against its existing
output manifest and re-parses all 26 previously fetched PubMed XML batches. For
every record it checks PMID, PMCID, title, abstract, DOI and year against the
own-article PubMed metadata. The new dataset does not inherit article identity
from a filename or from references.

All **645 local XML files** are inspected again for front identity. Restoration
is restricted to the **590 previously unique identity-matched candidates**, with
their expected raw hashes checked again. A candidate requires an exact own
front PMID and canonical PMCID match, uniqueness across local files, and unique
authoritative PMCID ownership. Numeric PMC values may normalize to `PMC` plus
digits; no identifier is borrowed from a filename or cited paper.

The selected JATS root must be exactly one `article` (namespace-aware). Raw
`pmc-articleset` is accepted **only when it contains exactly one direct article
and no other elements**; this deterministic unwrapping is recorded. There must
be exactly one front/article-meta and no ambiguous multiple bodies. Nested
articles, subarticles, responses and back matter are never selected as the
article's body.

The local title combines only direct `article-title` and `subtitle` children of
the own title-group; alternate/running titles are excluded. Comparison uses
Unicode NFKC, casefolding, punctuation-to-space tokenization and collapsed
whitespace. Acceptance requires either normalized equality, or both character
SequenceMatcher ratio **>=0.90** and token-set Jaccard **>=0.75**. This permits
punctuation and minor wording differences, but is not semantic title validation.

Of 590 candidates:

| Title/body outcome | Count |
|---|---:|
| Title accepted | 588 |
| Normalized exact title matches | 581 |
| Minor differences accepted under both thresholds | 7 |
| Body restored after title acceptance | 525 |
| Accepted identity/title but no extractable body | 63 |
| Major title mismatch: local body quarantined | 2 |
| Other candidate errors | 0 |

Quarantined bodies, with authoritative metadata retained:

- **PMID:36650384**: local title begins `RETRACTED ARTICLE:`; sequence ratio
  0.8714285714285714, Jaccard 0.8181818181818182. This signal is preserved in the
  audit, not silently stripped to force a match. Retaining PubMed metadata does
  not endorse the article; downstream users may separately exclude it.
- **PMID:39343510**: local title contains duplicated/corrupted title wording;
  sequence ratio 0.6212534059945504, Jaccard 0.6666666666666666.

Both per-record comparisons, exact titles, paths, hashes, permissions and reasons
are retained in `candidate_audit.jsonl`. The two quarantines concern JATS body
restoration, not failed PubMed metadata verification.

## Conservative structural extraction

`src/corpus/verified_jats.py` is stdlib-only and performs a whitelisted structural
walk, not an arbitrary whole-article descendant-text join.

- Own front abstract is extracted and audited. If an authoritative PubMed
  abstract exists, the local front abstract is retained as nonindexed source
  blocks to avoid duplicate abstract indexing. Local front text is a fallback
  only when the authoritative abstract is absent.
- Body sections retain source sec-type, or a normalized heading when sec-type
  is missing. Direct paragraphs, nested sections, lists/list items, statements,
  quotes, definition lists, boxed text and permitted inline formatting are
  handled in source order. A paragraph split by a nested figure/list produces
  ordered source blocks without re-emitting that paragraph's descendants.
- Figure/other captions, table captions, table notes and table rows are handled
  explicitly. References, back matter, other articles, images/media and
  supplements are not flattened into body prose. Bibliographic xref markers
  are omitted; ordinary in-prose figure/table cross-reference labels can remain.
- Table rows preserve table group (`thead`, `tbody`, `tfoot` or direct rows),
  ordered cells, header status, source cell paths, column positions, rowspan and
  colspan. Rendered rows use explicit **1-based** row/column labels and tab cell
  delimiters; cell metadata coordinates are **0-based**. Rowspan occupancy is
  carried into subsequent rows. Nested cell paragraphs/lists are rendered once;
  list-item markers are explicit rendering additions. Visual table layout is
  not reconstructed as HTML or scientifically interpreted.
- Formula alternatives are not concatenated. Direct TeX is preferred when
  present; otherwise whitelisted MathML is linearized with token boundaries.
  MathML annotations/unsupported structures are omitted rather than copied as
  duplicate alternatives. Every formula path has a rendering limitation event.
  Formula images without usable text remain omitted. There is no OCR or
  mathematical equivalence validation.
- Unsupported structures receive explicit omission records. Supplement files
  and image pixels are never opened. A body section named “supplementary
  material” may retain its own heading/link-description prose; the linked or
  embedded supplementary payload remains excluded.

This is **partial, source-owned JATS restoration**, not a claim that every visual
or semantic element of a paper has been reproduced.

## Canonical retrieval schema and chunk/offset contract

Every corpus row and chunk has:

- `doc_id = "PMID:" + pmid`, e.g. `PMID:31606929`;
- `article_id` equal to that same canonical string;
- `corpus_id = "cerbench-authoritative-fulltext-v1"`.

Every chunk additionally has an opaque unique `chunk_id` and explicit `text`.
There are no cross-document/cross-article chunk assignments and no ghost
documents without chunks.

Each document begins with its authoritative **title as real indexed text**,
then its authoritative abstract when present, followed by eligible local text.
The title is not a hidden retriever metadata boost. It is an independent source
block that is normally packed with the abstract in the initial chunk; the
chunker may also pack following own-body blocks into available space. Long
abstracts or paragraphs may span chunks. Worker49 indexes only chunk `text` as
usual and need not change its scoring/tokenizer implementation.

Of the **22 documents with no authoritative abstract**, **PMID:31606929** gains
validated local text, while **21 are transparently title-only**. All 4,936 have
indexable text. `excluded_metadata.jsonl` is deliberately empty; if a future
input lacks title, abstract and eligible local text, it is explicitly excluded
there rather than becoming a zero-chunk document.

The deterministic chunker:

1. Joins indexed rendered source blocks using exactly `\n\n`.
2. Uses target and hard maximum **4,096 Unicode codepoints**.
3. Prefers block boundaries in the latter half of the target window; otherwise
   prefers a nearby whitespace boundary, then a hard codepoint split.
4. Uses at most **256 codepoints of overlap**, preferring a block-aligned next
   start where possible. Overlap never crosses documents.

Offsets are half-open **Unicode codepoint** offsets in the **rendered text**,
not byte offsets into XML. Each indexed source block has document start/end,
block start/end, section ID, XML locator, raw source path/hash and source type.
Every chunk has document start/end and exact source intersections with both
block-relative and chunk-relative start/end. Section offsets cover their indexed
blocks and nested sections. Nonindexed local front blocks/sections have null
document offsets. Original XML namespace-qualified permissions are retained;
source XML locators use local-name/index paths bound to the raw file hash.

The title, abstract labels, table delimiters/coordinates and cell-list markers
are documented rendering additions. The `\n\n` block separators are explicit
document additions. Source spans bind all nonseparator text to the appropriate
rendered source block. Build and independent post-build validation check exact span substrings, section
ownership, identity consistency, chunk bounds and complete document-character
coverage. Optimized Python (`-O`) is forbidden because release assertions must
execute.

## Exact final counts and coverage

| Quantity | Count |
|---|---:|
| Indexed documents | 4,936 |
| Chunks | 10,313 |
| Chunks containing local body-source spans | 5,899 |
| Restored body documents | 525 |
| Indexed source blocks | 61,572 |
| Retained, nonindexed front-abstract source blocks | 2,695 |
| Indexed section entries | 14,176 |
| Distinct retained source section-type/heading labels | 5,461 |
| Document characters, without chunk overlap | 26,234,327 |
| Chunk characters, including overlap | 27,148,787 |
| Document lexical token proxy, without overlap | 4,095,200 |
| Chunk lexical token proxy, including overlap | 4,243,853 |
| Maximum chunk characters | 4,096 |
| Excluded / zero-chunk documents | 0 / 0 |

The token proxy is lowercase followed by regex `[a-z0-9]+`, matching worker49's
lexical tokenizer. It is **not a BPE/model-token count or a paid-call estimate**.

Indexed block kinds:

| Block kind | Count |
|---|---:|
| Authoritative title / abstract | 4,936 / 4,914 |
| Ordinary paragraph / list-item paragraph | 18,240 / 2,236 |
| Section heading | 8,686 |
| Figure caption / table caption / other caption | 1,918 / 705 / 12 |
| Table row / table note | 14,642 / 724 |
| Display/bare formula blocks | 248 |
| Labels / definition terms / other titles | 4,078 / 220 / 13 |

Inline formula text is included in its owning paragraph/cell, so formula-block
count is not the total formula occurrence count. Section labels are retained
source labels/headings, not a harmonized biomedical section taxonomy.

For the **588 title-accepted local candidates**, paragraph accounting covers
**27,375** own front-abstract/body paragraph elements:

- **26,586 handled**, including **26,476 with rendered text**;
- **789 explicitly omitted**, under recorded excluded/unsupported structures;
- **0 unexplained paragraph gaps** and **0 duplicate paragraph visits**.

This is structural source-paragraph accounting, not a claim that all original
paragraph text is indexed: local front abstracts are deliberately deduplicated,
citation markers/media are omitted, and formulas can lose layout/semantics.

Omission/limitation events (categories can overlap on a source element):

| Event | Count |
|---|---:|
| Bibliographic citation marker excluded | 42,779 |
| Supplement/media not extracted | 683 |
| Image not extracted | 1,775 |
| Formula-image alternative not extracted | 296 |
| Formula rendering not semantically validated | 1,014 |
| MathML linearized with structure loss | 584 |
| Formula alternative not duplicated | 267 |
| Formula representation omitted / no extractable formula text | 11 / 11 |
| MathML annotation/unsupported structure omitted | 6 |
| Outside own front-abstract/body | 1,008 |
| Figure/table noncaption content omitted | 201 |
| Figure/table alternative omitted | 87 |
| Unsupported block element | 46 |

## Permissions and licensing

Each record has a full-text license object with own
`front/article-meta/permissions` scope, strings, license attributes, links and
serialized XML retaining namespace-qualified links. Exact known
`creativecommons.org` license/CC0 URLs are recognized conservatively; free-text
“open access” claims alone do not qualify. Multiple different recognized
licenses are marked conflicting rather than arbitrarily choosing one.

| Category | All 4,936 records | 525 restored bodies |
|---|---:|---:|
| License unreviewed | 4,531 | 122 |
| One machine-readable CC license recognized | 373 | 371 |
| Conflicting machine-readable licenses | 32 | 32 |

Recognition is not a permissions audit: attribution, NC/ND/SA conditions,
third-party figures/tables and different license scopes remain unadjudicated.
Even recognized licenses do not produce a blanket public redistribution claim.
Every record and manifest retains `public_redistribution_authorized=false` and
local-research-only use. PubMed title/abstract rights are not inferred from a
local article's full-text license.

## Executed builds, tests and hashes

Commands executed from the analysis directory using the verified interpreter:

```bash
PY=/Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python

"$PY" -m pytest -q tests/test_verified_jats.py tests/test_pubmed_authoritative.py tests/test_verified_bm25.py --junitxml=results/readiness/verified_fulltext/tests.junit.xml

"$PY" scripts/50_restore_verified_fulltext.py

"$PY" scripts/50_restore_verified_fulltext.py \
  --output results/readiness/verified_fulltext/independent_build \
  --compare-with data/processed/authoritative_fulltext_v1 \
  --verification results/readiness/verified_fulltext/reproducibility.json
```

**83 tests passed**, no skips or failures: 28 new JATS/integration cases, 20
authoritative metadata cases, and 35 verified retrieval cases. Tests cover
namespaces/single-root selection, exact own IDs, title/subtitle matching and
quarantine, nested lists/statements, references/other-article exclusion, ordered
paragraph splitting, captions, table coordinates/spans, formula alternatives,
MathML annotations, image/supplement omissions, license scope, Unicode chunk
offsets/bounds/coverage, title-only documents, source tampering, fresh-output
guards, two-build determinism, and worker49 schema compatibility.

The final primary build and the independently recomputed copy are
**byte-identical across all 8 dataset files, including the manifest**. The second
build did not copy or overwrite the first. Every destination is fresh-only;
repeating either command at its existing output path is intentionally rejected.
The comparison report is outside both datasets so it cannot change their hashes.

The manifest binds **673 input files**: the authoritative corpus and its prior
output manifest, 26 PubMed batches and 645 local full-text XML files. Their
hashes are rechecked after extraction to detect changes during a build.

The independent post-build `verify_sources_and_offsets.py` audit also passed:
all **673 source hashes**, **64,267 source blocks**, **15,854 section entries**
(including nonindexed front sections), and **68,828 chunk/source intersections**
were checked. It reconfirmed byte identity, code hashes, no zero-chunk documents,
zero unexplained/duplicate paragraph visits, and the 83-test JUnit result.
The report is `results/readiness/verified_fulltext/source_and_coverage_verification.json`.

| Binding | SHA-256 |
|---|---|
| `corpus.jsonl` | `1be9556874f61f2e0def5abed5812620c75b6648e2ec11f4128e9b4bcb3e359d` |
| `chunks.jsonl` | `427b6ed6c4e9db4cef87576e9f9b1e9c30c4864023eeb7e9c32efabf1ccdd300` |
| `dataset_manifest.json` | `59d7751daeb67925903629ed6c542c1e98874d17b0499e43f3a387cc1168b5ae` |
| Aggregate `sourcehash` | `ddd792a30030a2fc5f33e0a15ca7366375ae4b633df8e84e9ad6e4c894e282ab` |

`sourcehash` hashes sorted compact JSON of the source-path-to-SHA256 map; it is
not a raw concatenation of source XML. The manifest also binds extractor,
builder and PubMed parser code hashes, all output hashes, chunk policy and text
validation limits. Hash binding detects drift, not malicious replacement of
both data and attestations.

## Outputs and next action

New implementation files:

- `src/corpus/verified_jats.py`
- `scripts/50_restore_verified_fulltext.py`
- `tests/test_verified_jats.py`
- `docs/VERIFIED_FULLTEXT.md`

Primary dataset `data/processed/authoritative_fulltext_v1/` contains:

- `corpus.jsonl`, `chunks.jsonl`, `dataset_manifest.json`;
- `source_blocks.jsonl`, including indexed blocks and retained local abstracts;
- `candidate_audit.jsonl`, including title comparisons, omissions and coverage;
- `local_xml_audit.jsonl`, covering all 645 local XML files;
- `excluded_metadata.jsonl` (empty for this build);
- `summary.json`, including exact section labels/types and counts.

Readiness evidence resides in `results/readiness/verified_fulltext/`:
`tests.junit.xml`, `reproducibility.json`, `source_and_coverage_verification.json`,
the reusable `verify_sources_and_offsets.py` verifier, and the retained
byte-identical `independent_build/` dataset. Verification evidence is outside
both datasets and does not modify their contents.

Worker49 can now use the primary corpus, chunks and fresh manifest as inputs
to its `build` subcommand, selecting a **new worker49-owned index destination**.
No new qrels accompany this dataset, and historical gold must not be remapped
onto the new canonical IDs. Fresh non-human task/evidence experiments remain
distinct from the waived human biomedical validation.
