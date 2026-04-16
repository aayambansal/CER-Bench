# Annotation Guidelines: SynthSearch-Biomed

## Purpose

These guidelines are for human reviewers auditing the synthetically generated benchmark
tasks. The goal is to ensure that every task in the final benchmark is scientifically
valid, unambiguous, and genuinely tests retrieval quality.

## Reviewer Qualifications

Reviewers should have:
- Graduate-level training in a biomedical or life-science field
- Familiarity with experimental methods in at least one subfield
- Ability to read methods sections critically

## Review Process

### Step 1: Read the question

- Is it clear what information the question asks for?
- Would a working scientist plausibly ask this question?
- Are the constraints specific enough to have a definite answer?

If the question is ambiguous or unrealistic, mark as **REJECT: unclear question**.

### Step 2: Check supporting documents

For each supporting document:
- Does the document actually address the question?
- Does the highlighted passage contain the claimed evidence?
- Are ALL required constraints satisfied by this document?

If a supporting document does not actually support the question, mark as
**REJECT: false positive support**.

### Step 3: Check constraint labels

- Are the listed constraints correctly typed (organism, assay, intervention, etc.)?
- Are there unlisted constraints that should be explicit?
- For constraint-satisfaction tasks: would a document matching all constraints
  genuinely answer the question?

If constraints are mislabeled or incomplete, mark as **REVISE: constraint error**.

### Step 4: Check hard negatives

For each hard negative:
- Does it match the topic broadly?
- Does it fail on a specific, identifiable constraint?
- Is the rationale for its negativity correct?

If hard negatives are too easy (obviously off-topic) or incorrectly labeled,
mark as **REVISE: weak negatives**.

### Step 5: Check abstention tasks (Family D only)

- Is the query plausible (not obviously nonsensical)?
- Have you personally verified that no document in the pilot corpus satisfies all
  constraints?
- Are there near-miss documents that match most constraints?

If the corpus actually does contain a satisfying answer, mark as
**REJECT: abstention invalid**.

### Step 6: Check difficulty label

- **Easy**: Could BM25 find the answer in the top 10? Is the answer in the abstract?
- **Medium**: Requires filtering or reading methods/results? Involves 2-3 constraints?
- **Hard**: Requires multiple documents, condition tracking, citation following, or
  abstention?

If the difficulty label is wrong, mark as **REVISE: difficulty mismatch**.

## Annotation Labels

| Label | Meaning | Action |
|-------|---------|--------|
| ACCEPT | Task is valid and well-formed | Include in benchmark |
| REVISE | Task has fixable issues | Return to generator with notes |
| REJECT | Task is fundamentally flawed | Exclude from benchmark |

## Required Fields for Each Review

```json
{
  "task_id": "string",
  "reviewer": "string (anonymized ID)",
  "label": "ACCEPT | REVISE | REJECT",
  "question_quality": "good | unclear | unrealistic",
  "evidence_quality": "correct | partial | wrong",
  "constraint_quality": "correct | incomplete | mislabeled",
  "negative_quality": "good | weak | wrong",
  "difficulty_correct": true | false,
  "notes": "free text"
}
```

## Calibration Tasks

Before the main audit, all reviewers should independently review the same 10 calibration
tasks and discuss disagreements. Calibration criteria:
- Inter-annotator agreement > 80% on ACCEPT/REJECT
- Difficulty label agreement > 70%

## Priority Review Order

1. All abstention tasks (highest error risk)
2. All contradiction/conditionality tasks (second highest)
3. Hard difficulty tasks
4. Random sample of easy/medium tasks

## Time Estimate

- ~3-5 minutes per task for experienced reviewers
- 50-task pilot audit: ~3-4 hours
- Full 300-800 task audit: split across reviewers, ~2-3 hours each

## Common Error Patterns to Watch For

1. **Topic match without constraint match**: Document discusses the same disease but
   uses a different model organism.
2. **Temporal drift**: Question specifies "after 2020" but supporting doc is from 2019.
3. **Abstraction mismatch**: Question asks about in vivo results, supporting doc only
   has in vitro data.
4. **Partial comparison**: Comparative task only has evidence for one side.
5. **False contradiction**: Papers that appear to conflict but actually measure
   different endpoints.
6. **Trivial abstention**: Query is obviously nonsensical rather than plausibly
   unanswerable.
