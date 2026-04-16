# Corpus Card: SynthSearch-Biomed Corpus

## Overview

| Field | Value |
|-------|-------|
| Name | SynthSearch-Biomed Corpus |
| Version | 0.1.0 (pilot) |
| Domain | Biomedical / life-science literature |
| Size (target) | 50,000-250,000 documents |
| Languages | English |
| Time range | 2015-2026 |
| License | Mixed (CC BY, CC0 per PMC OA terms; metadata is public domain) |

## Sources

| Source | What it provides | License | Access |
|--------|-----------------|---------|--------|
| PMC Open Access | Full text (XML) | CC BY / CC0 | FTP bulk download |
| PubMed | Abstracts, MeSH terms, metadata | Public domain (NLM) | E-utilities API |
| OpenAlex | Citation graph, concepts, topics | CC0 | REST API |
| Semantic Scholar | Citation graph, TLDRs, fields of study | Free API | REST API |

## Document Schema

Each document record contains:

**Required**: doc_id, title, abstract, year, source

**Metadata**: authors, venue, doi, pmid, pmcid, mesh_terms, publication_type, organisms,
keywords, topics, citation_count, reference_ids, cited_by_ids

**Content**: abstract_text, sections (list of {heading, text, section_type}),
figure_captions, table_texts, full_text_available flag

## Chunking

Documents are chunked by scientific structure:
- Abstract (max 512 tokens)
- Introduction sections (max 1024 tokens)
- Methods subsections (max 1024 tokens)
- Results subsections (max 1024 tokens)
- Discussion sections (max 1024 tokens)
- Figure captions (max 256 tokens)
- Table text (max 512 tokens)

Each chunk inherits its parent document's metadata.

## Filtering

- Excluded publication types: editorials, letters, comments, errata
- Year range: 2015-2026
- Requires at least an abstract
- English language only

## Known Limitations

- Full text available only for PMC OA subset (~3.5M articles total, subset used here)
- Figure/table extraction quality varies by XML formatting
- Citation graph completeness depends on OpenAlex/S2 coverage
- MeSH term assignment may lag for recent publications
- Organism extraction is heuristic-based (not manually verified)

## Intended Use

This corpus is built specifically for the SynthSearch-Biomed benchmark. It is designed
to support retrieval evaluation, not to be a comprehensive biomedical knowledge base.

## Maintenance

Corpus is static once frozen for benchmark evaluation. No updates during the review period.

## Statistics

_To be filled after corpus construction._

| Statistic | Value |
|-----------|-------|
| Total documents | |
| Documents with full text | |
| Total chunks | |
| Median chunks per document | |
| Unique MeSH terms | |
| Unique organisms | |
| Year distribution | |
| Venue distribution (top 20) | |
| Citation graph edges | |
