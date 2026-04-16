# Benchmark Specification: SynthSearch-Biomed

## Overview

SynthSearch-Biomed is a retrieval benchmark for biomedical literature that tests
condition-sensitive, multi-constraint, multi-document scientific search. It evaluates
whether retrieval systems can find the right evidence under realistic experimental
science workflows, not just match keywords to passages.

## Task Families

### Family A: Constraint-Satisfaction Retrieval

**Definition**: The query specifies multiple experimental constraints (organism, assay,
intervention, outcome, time period, etc.). The system must find papers satisfying ALL
constraints simultaneously.

**Example**:
> Find papers that use CRISPR-Cas9 in human iPSC-derived cardiomyocytes to study
> arrhythmia-related ion channel mutations published after 2020.

**Why it's hard**: Topic-matching retrieval returns many CRISPR papers and many
cardiomyocyte papers, but the intersection under all constraints is small.

**Gold label**: Set of documents satisfying all constraints + specific passages
confirming each constraint.

### Family B: Comparative Retrieval

**Definition**: The query asks for a comparison across methods, organisms, conditions,
or outcomes. The system must retrieve evidence from multiple papers that enables the
comparison.

**Example**:
> Compare the reported transfection efficiency of lipofection vs electroporation in
> primary T cells across at least three independent studies.

**Why it's hard**: The system must find papers on both sides of the comparison, not just
one. Partial retrieval (all lipofection, no electroporation) is a common failure.

**Gold label**: Paired evidence sets for each side of the comparison + passages
containing the compared values.

### Family C: Contradiction / Conditionality Retrieval

**Definition**: The query involves findings that appear contradictory across papers but
are actually conditional on experimental setup, organism, dosage, or other variables.
The system must retrieve evidence showing both the apparent contradiction and the
conditions that resolve it.

**Example**:
> Some studies report that metformin inhibits tumor growth while others report no
> effect. Retrieve papers showing both outcomes and identify the experimental conditions
> (cell type, dosage, duration) that explain the discrepancy.

**Why it's hard**: Keyword retrieval finds "metformin + tumor" easily but cannot
distinguish which papers show inhibition vs no effect, or identify the resolving
conditions.

**Gold label**: Papers on both sides + passages identifying the experimental conditions
that explain the discrepancy.

### Family D: Abstention / Unsupported Query

**Definition**: The query asks for evidence that does not exist in the corpus. The
system should recognize this and abstain rather than return marginally relevant results.

**Example**:
> Find studies that tested the effect of GLP-1 receptor agonists on Parkinson's disease
> progression in patients with co-morbid type 2 diabetes, using randomized controlled
> trial designs, published after 2023.

**Why it's hard**: There are papers on GLP-1 agonists and neurodegeneration, and papers
on GLP-1 agonists and diabetes, but the specific intersection with RCT design and
Parkinson's progression may not exist. Systems that always return something will produce
misleading evidence.

**Gold label**: Empty evidence set + flag indicating the query is unsupported +
optional "closest but insufficient" documents.

### Family E: Multi-Hop Evidence Chains

**Definition**: The query cannot be answered by any single paper. The answer requires
linking evidence across 3 or more papers via shared entities (a gene, a pathway, a
compound, or a mechanism) to build a reasoning chain.

**Example**:
> Drug X inhibits kinase Y. Kinase Y phosphorylates transcription factor Z. Transcription
> factor Z drives expression of gene W in hepatocellular carcinoma. Find the papers that
> establish each link in this chain.

**Why it's hard**: Each individual link may be well-documented, but the system must
identify that these separate findings connect into a single chain. Standard retrieval
returns papers about "Drug X" or "gene W" individually, but misses the intermediate
links (kinase Y, transcription factor Z) that bridge them. Citation expansion and
entity-aware search are critical.

**Gold label**: Ordered sequence of documents, one per link in the chain + passages
establishing each link + the shared entities connecting adjacent links.

### Family F: Temporal Evolution

**Definition**: The query asks how understanding, consensus, or reported results for a
scientific question have changed over time. The system must retrieve papers from
different time periods that show the evolution.

**Example**:
> How has the consensus on the role of gut microbiome composition in response to
> immune checkpoint inhibitors in melanoma evolved from 2015 to 2025?

**Why it's hard**: The system must find papers across a decade, not just the most
recent or most cited. It must distinguish between early speculative findings, confirmatory
studies, contradictory evidence, and current consensus. Standard retrieval biases toward
recent or highly cited papers and misses the historical arc.

**Gold label**: Time-ordered set of documents with timestamps + passages showing the
evolving claim or finding at each stage + a timeline annotation indicating the shift
(e.g., "preliminary evidence" -> "confirmed" -> "refined with conditions").

### Family G: Aggregation / Quantitative Collection

**Definition**: The query asks to collect all reported quantitative values for a specific
measurement, parameter, or outcome across studies. The system must achieve high recall
across many papers rather than finding a single best answer.

**Example**:
> Collect all reported IC50 values for sorafenib against VEGFR-2 across different
> assay conditions (cell-free kinase assay, cell-based proliferation, etc.) from
> independent studies.

**Why it's hard**: Precision-oriented retrieval stops after finding a few good hits.
This family rewards exhaustive recall. The system must find every paper that reports
the measurement, even when terminology varies (IC50 vs half-maximal inhibitory
concentration vs EC50 in some contexts). The quantitative values may be buried in
tables, supplementary data, or figure legends.

**Gold label**: Complete set of documents reporting the measurement + specific passages
or table cells containing the values + metadata on assay conditions for each value.

### Family H: Negative / Null Result Retrieval

**Definition**: The query asks specifically for studies that report a negative finding,
null result, or absence of a hypothesized effect. The system must find papers that
explicitly report failure or non-significance, not papers that report positive results.

**Example**:
> Find studies where siRNA knockdown of gene X in breast cancer cell lines showed no
> significant effect on cell proliferation or viability.

**Why it's hard**: Scientific literature has strong publication bias toward positive
results. Negative findings use hedging language ("no significant difference was
observed," "failed to replicate," "did not reach statistical significance") that is
harder to match than affirmative claims. Retrieval systems trained on positive-result
text rank positive-result papers higher even when the query explicitly asks for
negative results.

**Gold label**: Documents explicitly reporting negative/null findings + passages
containing the null result statement + the specific statistical evidence of
non-significance (p-values, confidence intervals, effect sizes).

## Task Family Summary

| Family | Code | Tests | Key Challenge | Metric Focus |
|--------|------|-------|---------------|--------------|
| A: Constraint-satisfaction | `constraint` | Multi-condition intersection | Precision under many constraints | Recall@k, constraint coverage |
| B: Comparative | `comparative` | Both-sides evidence | Balanced retrieval | Evidence set balance, paired recall |
| C: Contradiction/conditionality | `contradiction` | Condition tracking | Distinguishing condition-dependent results | Condition identification, both-sides recall |
| D: Abstention | `abstention` | Knowing what you don't know | Resisting false confidence | Abstention precision/recall |
| E: Multi-hop chains | `multihop` | Cross-paper reasoning chains | Bridging entity identification | Chain completeness, link recall |
| F: Temporal evolution | `temporal` | Tracking consensus over time | Time-distributed retrieval | Temporal coverage, era recall |
| G: Aggregation | `aggregation` | Exhaustive value collection | High recall over many papers | Recall@50+, value completeness |
| H: Negative results | `negative` | Finding null/negative findings | Overcoming positive-result bias | Negative-result precision, recall |

## Task Schema

```json
{
  "task_id": "string (unique, format: {family}_{domain}_{number})",
  "domain": "biomedicine",
  "task_family": "constraint | comparative | contradiction | abstention | multihop | temporal | aggregation | negative",
  "difficulty": "easy | medium | hard",
  "question": "string (the natural-language query)",
  "decomposition_hints": ["subquery 1", "subquery 2"],
  "required_constraints": [
    {
      "type": "organism | assay | intervention | outcome | temporal | design | other",
      "value": "string",
      "optional": false
    }
  ],
  "supporting_doc_ids": ["pmcid_1", "pmcid_2"],
  "supporting_passages": [
    {
      "doc_id": "pmcid_1",
      "section": "methods | results | abstract | caption | table",
      "text": "exact passage text",
      "constraint_satisfied": ["constraint_type_1"]
    }
  ],
  "hard_negative_doc_ids": ["pmcid_3", "pmcid_4"],
  "hard_negative_rationale": [
    {
      "doc_id": "pmcid_3",
      "reason": "matches topic but wrong organism"
    }
  ],
  "expected_answer_type": "set | comparison | conditional | abstain | chain | timeline | value_collection | null_result | boolean | freeform",
  "reference_answer": "string (brief expected answer for downstream eval)",
  "metadata_filters": {
    "year_min": null,
    "year_max": null,
    "organism": [],
    "assay": [],
    "publication_type": [],
    "mesh_terms": []
  },
  "verification_status": "auto_verified | human_verified | needs_review",
  "generation_method": "extraction_based | template_based | manual",
  "split": "train | dev | test",
  "notes": ""
}
```

## Dataset Splits

| Split | Size | Usage |
|-------|------|-------|
| train | ~40% | Controller training / few-shot examples (if fine-tuning) |
| dev | ~20% | Hyperparameter tuning, prompt iteration |
| test | ~40% | Final evaluation only, no peeking |

Test split must be locked before any evaluation begins.

## Difficulty Calibration

- **Easy**: Single constraint, answer in abstract, BM25 likely finds it.
- **Medium**: 2-3 constraints, answer may be in methods/results, requires some filtering.
- **Hard**: 4+ constraints, answer spans multiple documents, requires condition tracking
  or citation following, or requires abstention.

Target distribution: ~20% easy, ~40% medium, ~40% hard.

## Hard Negative Requirements

Every non-abstention task must include at least 2 hard negatives per gold document.
Hard negatives must:
- Match the topic broadly
- Fail on at least one specific constraint (wrong organism, wrong assay, wrong time period, etc.)
- Be from the same subfield

Hard negative selection methods:
1. Same MeSH heading, different organism/assay
2. Same intervention, different outcome measure
3. Same organism, different experimental design
4. High BM25 score but fails constraint verification

## Abstention Requirements

At least 15% of tasks should be abstention tasks.

Abstention tasks must:
- Seem plausible (not obviously nonsensical queries)
- Have "near-miss" documents in the corpus that match most but not all constraints
- Be verified by checking that no document in the corpus satisfies all constraints

## Quality Criteria

A task passes quality review if:
1. The question is unambiguous to a domain expert
2. All supporting documents genuinely answer the question
3. Supporting passages contain the actual evidence (not just topic overlap)
4. Hard negatives are plausible distractors
5. Constraints are correctly labeled
6. Abstention tasks genuinely lack sufficient evidence
7. The task cannot be trivially solved by keyword matching alone (for medium/hard)

## Croissant Metadata

The final dataset must include Croissant metadata (ML schema.org) as required by
NeurIPS E&D track. This includes:
- Dataset name, description, license
- Record set definition matching the task schema
- Distribution information (file format, download URL)
- Creator information (anonymized for review)
