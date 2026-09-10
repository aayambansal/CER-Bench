# CER-Bench blinded relevance and evidence-role annotation

## Purpose

The annotation establishes a human anchor for retrieval evaluation. Automatic labels, retrieval-system identity, and candidate provenance are hidden from annotator sheets. Each pair is labeled independently by two annotators; a biomedical expert adjudicates every disagreement and every `U` judgment.

## Relevance label

Read the question, title, abstract, and—when necessary—the linked source. Enter exactly one label:

- `2`: directly supports at least one required part of the question with usable evidence;
- `1`: topically relevant or partially useful, but does not independently support a required part;
- `0`: not relevant to the question as written;
- `U`: cannot decide from accessible content or specialist uncertainty remains.

Do not infer relevance from title keywords alone. A document is relevant only under the population, intervention/method, outcome, condition, and time constraints stated in the question. Copy the shortest decisive quotation into `evidence_span`. A label of `2` requires an evidence span; use the source URL when the abstract is insufficient.

## Structural role

Enter valid JSON in `role_json`, following the row's `role_schema`.

- **Constraint:** list every task constraint directly covered by the evidence (`C1`, `C2`, …).
- **Comparative:** label evidence as side `A`, `B`, `BOTH`, or contextual and name the comparison axis.
- **Contradiction:** identify `FINDING_A`, `FINDING_B`, `RECONCILIATION`, or context; record the condition that explains disagreement when present.
- **Multi-hop:** assign one or more hop indices and the bridge entity/relation supported by the document.
- **Temporal:** assign `EARLY`, `MIDDLE`, or `LATE` relative to the question and record the evidence year.
- **Aggregation:** transcribe each distinct value, unit, and condition. Do not normalize silently.
- **Negative-result retrieval:** distinguish explicit null results, negative direction, failed replication, and background/context.
- **Abstention:** record whether the document jointly satisfies the full question (`YES`, `NO`, `UNCERTAIN`) and which constraints it violates.

Role annotation is required only for relevance `2`; it is optional for relevance `1` and should be empty for relevance `0`.

## Confidence and adjudication

Enter confidence `1` (low), `2` (moderate), or `3` (high). Do not discuss labels with the other annotator before submission. Adjudication sees both evidence spans, checks the full source, and writes a final relevance label and structural role. Inter-annotator agreement is reported for four-way labels and binary direct relevance (`2` versus all others).

## Out-of-pool and abstention audit

The main sheet includes an undisclosed sample beyond the evaluated systems' top-20 pool. Its direct-relevance prevalence estimates pool blind spots. The abstention sheet deliberately contains many near misses. An unsupported query is certified only if no reviewed candidate jointly satisfies all constraints after targeted search; low retrieval scores alone never establish impossibility.
