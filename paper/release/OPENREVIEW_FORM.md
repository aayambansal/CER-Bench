# OpenReview form text (ICLR 2027)

Copy each field verbatim. Upload `cerbench_openreview.pdf` as the PDF and `supplementary_material.zip` as the supplement.

## Title

Fixed Rankings, Moving Leaders: A Relevance-Label and Source-Identity Audit of Scientific Retrieval Evaluation

## Keywords

retrieval evaluation, relevance judgments, pooling bias, benchmark audit, scientific literature search, biomedical retrieval, data provenance, reproducibility, datasets and benchmarks

## TL;DR

Holding every saved ranking fixed, swapping seed for pooled relevance labels reverses 9 of 45 system orderings in a biomedical retrieval benchmark; we jointly audit that label dependence and a source-identity bug and release the audited data without a corrected leaderboard.

## Abstract

We show that a scientific retrieval benchmark can change its conclusions while every retrieval system stays fixed. In CER-Bench, a historical collection of 304 synthetic biomedical retrieval tasks over 4,936 PubMed records, replacing seed relevance labels with pooled automated judgments reverses nine of 45 pairwise system orderings on identical saved rankings (Kendall's $\tau_b = 0.584$) and moves the mean-recall leader from a three-round search agent to a dense retriever. Label incompleteness is a known problem in retrieval evaluation, but it is rarely audited together with source identity: whether an identifier in the corpus refers to the article text attached to it. We audit both. An authoritative re-fetch of all 4,936 records traces a parser error that let cited-paper identifiers replace an article's own identifiers, changing 247 PMC identifiers, 226 of which occur in reference lists. A controlled Monte Carlo experiment with 1,000 paired, nested disclosures of the existing judgment pool locates the change of leader between six and nine revealed judgments per query, with paired empirical intervals overlapping zero at both points. We release a source-identity-checked retrieval dataset (4,936 documents, 10,313 chunks, 525 documents with conservatively restored own-article text), a hash-bound BM25 substrate, all saved rankings, judgments, disclosure arrays, and audit code. We deliberately do not release a corrected leaderboard: without human relevance validation or new model runs, the defensible output is an explicit boundary between reproducible arithmetic, verified source identity, and still-unvalidated relevance.

## Primary area

datasets and benchmarks

## AI assistance (tick)

- Yes, to aid or polish writing. Details are described in the paper.
- Yes, to draft sections of the paper. Details are described in the paper.
- Yes, for research ideation or execution. Details are described in the paper.

(The paper's Ethics and AI-Use Statement discloses AI assistance for code development, audit analysis, and manuscript preparation, and that historical task generation and judging used a proprietary language model. Do not tick "generating synthetic datasets" for this submission unless you consider the historical task generation part of this paper's contribution; the paper treats it as pre-existing historical construction and says so.)

## Alternative titles considered

- Same Rankings, Different Winner: How Relevance Labels and Source Identity Govern Scientific Retrieval Conclusions
- Who Wins Depends on Who Judged: Auditing Label and Identity Dependence in Biomedical Retrieval Benchmarks
- Relevance Labels and Source Identity Can Change Scientific Retrieval Conclusions: A CER-Bench Audit (previous title)
