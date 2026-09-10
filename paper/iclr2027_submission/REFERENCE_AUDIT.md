# CER-Bench reference audit

Audit date: 2026-08-26

All 27 cited keys in `cerbench_iclr2027.tex` resolve to entries in `refs.bib`; there are no uncited entries, duplicate keys, or undefined citation keys. Metadata was checked against DOI resolvers, Crossref-backed publisher records, ACL Anthology, NeurIPS proceedings, arXiv, NIST TREC proceedings, NLM, OpenAlex, and the official Chroma report as applicable.

The bundled validator reports **27/27 valid entries, zero errors, and zero duplicates**. Its three residual warnings are absent page ranges for two proceedings reports (TREC 2004 RM3 and TREC 2017 Precision Medicine) and the NeurIPS 2021 Datasets and Benchmarks proceedings record for BEIR; the authoritative records do not provide conventional page spans.

## Corrections made

- Replaced the incorrect SPECTER2 citation with the archival SciRepEval paper (EMNLP 2023, DOI `10.18653/v1/2023.emnlp-main.338`).
- Corrected EpiBench's arXiv primary class from `cs.AI` to `cs.CL`.
- Corrected Robertson and Zaragoza's publisher metadata to volume 4, issues 1--2, pages 1--174 (DOI `10.1561/1500000019`).
- Distinguished website update/access dates for PubMed and PMC from publication years.
- Added traceable arXiv metadata for OpenAlex.
- Added verified primary references for incomplete judgments, assessor variation, LLM judges, and RM3.
- Removed the unused S2ORC citation after the manuscript was corrected to the three corpus sources demonstrated by saved fields.

## Source-type caveats retained in the paper

- Context-1 is an official technical report, not a peer-reviewed proceedings paper.
- EpiBench, PaperQA2, Chain of Retrieval, E5, and OpenAlex are cited through verified preprint records where no more appropriate archival record was used.
- PubMed and PMC are cited as official NLM web resources.
- The manuscript avoids claiming that the cited general-purpose LLM-judge evidence validates the paper's biomedical LLM judgments.

The bibliography was generated from verified metadata rather than model memory. The compiled final log contains no undefined citations.
