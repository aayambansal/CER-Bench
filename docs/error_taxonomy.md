# Error Taxonomy: SynthSearch-Biomed

This taxonomy classifies retrieval errors to support systematic error analysis in the
paper. Every failed retrieval should be categorized under one of these error types.

## E1: Entity Mismatch

The system retrieves documents about the right topic but with the wrong entity.

**Subtypes**:
- E1a: Wrong organism (e.g., mouse study instead of human)
- E1b: Wrong cell line (e.g., HeLa instead of iPSC-derived)
- E1c: Wrong gene/protein (e.g., paralog confusion)
- E1d: Wrong disease subtype (e.g., Type 1 vs Type 2 diabetes)

**Detection**: Compare extracted entities in retrieved docs against required constraints.

## E2: Condition Omission

The system matches the topic and entities but ignores a critical experimental condition.

**Subtypes**:
- E2a: Wrong dosage/concentration
- E2b: Wrong time point/duration
- E2c: Wrong temperature/pH/environment
- E2d: Wrong experimental design (in vitro vs in vivo)
- E2e: Wrong control condition

**Detection**: Check whether retrieved docs satisfy all metadata constraints.

## E3: Partial Comparison

For comparative tasks, the system retrieves evidence for one side of the comparison
but not the other.

**Subtypes**:
- E3a: Missing method A (only found method B)
- E3b: Missing condition A (only found condition B)
- E3c: Asymmetric depth (10 papers for one side, 1 for the other)

**Detection**: Check evidence set balance for comparative tasks.

## E4: False Contradiction

The system identifies papers as contradictory when they are actually measuring
different things.

**Subtypes**:
- E4a: Different endpoints (e.g., survival vs tumor size)
- E4b: Different populations (e.g., pediatric vs adult)
- E4c: Different timeframes
- E4d: Different dosing regimens

**Detection**: For contradiction tasks, verify that the "conflict" is real given
the conditions.

## E5: Missing Evidence

The system fails to find documents that exist in the corpus and match the query.

**Subtypes**:
- E5a: Vocabulary mismatch (query uses different terminology than paper)
- E5b: Evidence buried in methods/supplementary (not in abstract)
- E5c: Implicit information (paper satisfies constraint but doesn't state it explicitly)
- E5d: Retrieval depth insufficient (correct doc exists but ranked too low)

**Detection**: Compare retrieved set against gold evidence set.

## E6: Overconfident Non-Abstention

The system returns results for a query that should be marked as unanswerable.

**Subtypes**:
- E6a: Returns topically related but non-answering documents
- E6b: Returns near-miss documents (match most but not all constraints)
- E6c: Returns documents from wrong time period
- E6d: Hallucinated relevance (system claims doc answers query when it doesn't)

**Detection**: For abstention tasks, check whether system abstained.

## E7: Premature Abstention

The system abstains when sufficient evidence actually exists.

**Subtypes**:
- E7a: Evidence exists but was not found (retrieval failure + abstention)
- E7b: Evidence exists but was pruned (pruning error + abstention)
- E7c: Confidence calibration error (evidence found but confidence too low)

**Detection**: For non-abstention tasks, check whether system incorrectly abstained.

## E8: Redundancy

The system returns multiple documents containing essentially the same evidence.

**Subtypes**:
- E8a: Same study reported in multiple venues
- E8b: Review paper restating primary findings
- E8c: Highly overlapping methods sections from same lab
- E8d: Preprint and published version of same paper

**Detection**: Measure duplicate evidence ratio in retrieved set.

## E9: Temporal Error

The system retrieves documents from outside the specified time range.

**Detection**: Check publication year against temporal constraints.

## E10: Ranking Error

The system retrieves the correct documents but ranks them poorly.

**Subtypes**:
- E10a: Gold document ranked below hard negatives
- E10b: Most relevant evidence ranked below less relevant evidence
- E10c: Passage containing answer ranked below non-answer passages

**Detection**: Compare rank of gold docs against expected position.

## E11: Broken Chain (Multi-hop, Family E)

The system retrieves some links in a reasoning chain but misses intermediate links,
breaking the chain of evidence.

**Subtypes**:
- E11a: Missing bridging entity (the shared entity between links not identified)
- E11b: Endpoint-only retrieval (found first and last paper but missed intermediates)
- E11c: Wrong chain path (found a plausible but incorrect intermediate link)

**Detection**: Check chain completeness and whether bridging entities connect adjacent
retrieved documents.

## E12: Temporal Bias (Temporal, Family F)

The system over-represents one time period at the expense of others.

**Subtypes**:
- E12a: Recency bias (only retrieves recent papers, misses earlier foundational work)
- E12b: Citation bias (only retrieves highly cited older papers, misses recent updates)
- E12c: Missing era (entire time period unrepresented in retrieved evidence)

**Detection**: Compare temporal distribution of retrieved docs against gold temporal span.

## E13: Incomplete Aggregation (Aggregation, Family G)

The system finds some reported values but misses others.

**Subtypes**:
- E13a: Terminological miss (value reported using synonym not in query)
- E13b: Table/figure miss (value only reported in table or figure, not in text)
- E13c: Early stopping (found enough values and stopped searching)
- E13d: Low-citation miss (value reported in less-cited paper that ranks lower)

**Detection**: Compare count and sources of retrieved values against gold value set.

## E14: Positive-Result Contamination (Negative Results, Family H)

The system retrieves papers reporting positive results when the query specifically
asks for negative/null findings.

**Subtypes**:
- E14a: Polarity inversion (retrieved paper shows effect, query asks for no effect)
- E14b: Hedging misinterpretation (paper's hedging language misread as negative result)
- E14c: Mixed-result conflation (paper has both positive and negative results for
  different endpoints; system doesn't distinguish which endpoint was queried)

**Detection**: Verify polarity of findings in retrieved docs against query intent.

## Usage in Paper

The error analysis section should:
1. Sample ~50-80 failed retrievals from the test set (ensure coverage of all 8 families)
2. Categorize each under this taxonomy (14 error types)
3. Report distribution of error types per baseline
4. Report distribution of error types per task family
5. Identify which error types the agent fixes vs which persist
6. Present 3-4 detailed case studies of representative errors (at least one from
   the new families E-H)
7. Highlight family-specific errors: E11 for multi-hop, E12 for temporal, E13 for
   aggregation, E14 for negative results
