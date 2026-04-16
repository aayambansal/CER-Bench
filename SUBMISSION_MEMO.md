# Submission Memo: SynthSearch NeurIPS 2026 E&D

**Date**: April 12, 2026
**Deadlines**: Abstract May 4, Paper May 6 (AoE)
**Days remaining**: 22 (abstract), 24 (paper)

---

## Current State

We have a working artifact: 4,936-document corpus, 17,902 chunks, 304 tasks across
8 balanced families, BM25 baselines, and an iterative agentic controller with full
test-set results on 125 tasks. The agent shows meaningful gains on aggregation (+8.8 pts
R@10) and temporal (+5.9 pts) families while BM25 wins on top-rank precision (R@5, MRR).

The claim is now clear: we built a benchmark that exposes where single-shot retrieval
fails, and an iterative controller improves deeper evidence recall on the hardest families.

**Verdict**: Credible E&D submission candidate, but not reviewer-proof yet.

---

## MUST FIX (Blocking — paper cannot be submitted without these)

### 1. Proper dense + hybrid + reranker baselines
**Problem**: SPECTER2 base without the proximity adapter scored near-zero. The only
real non-agentic baseline in the final table is BM25. Reviewers will immediately
notice the comparison section is thin.

**Fix**:
- Load SPECTER2 with `allenai/specter2_proximity` adapter on Lambda A100
- Re-embed all 17,902 chunks with the adapter model
- Rebuild FAISS index
- Re-run dense retrieval on test set
- Add hybrid (BM25 + Dense RRF) baseline
- Add cross-encoder reranker baseline (`cross-encoder/ms-marco-MiniLM-L-12-v2`)
  on top-100 from hybrid

**Effort**: ~4 hours (2h reindex on Lambda, 2h run baselines + scoring)
**Cost**: $0 (Lambda instance already provisioned)

### 2. Separate abstention evaluation
**Problem**: Abstention tasks (17 in test) have empty gold sets. Including them in the
main R@10 table dilutes scores and confuses interpretation. Neither method can abstain,
which is a real finding but needs its own evaluation.

**Fix**:
- Report main retrieval metrics EXCLUDING abstention tasks (108 tasks)
- Add separate abstention table with:
  - Does the system attempt to return results? (always yes currently)
  - Fraction of abstention tasks where system returns high-confidence results
    (false confidence metric)
  - Fraction of non-abstention tasks where system would abstain if it could
    (false abstention, not applicable yet since neither abstains)
- Frame abstention as an explicitly unsolved problem and open challenge

**Effort**: ~2 hours (modify scoring script, add new table)
**Cost**: $0

### 3. Dev / test consistency
**Problem**: Dev results in the report are from the old 60-task pilot (11 dev tasks).
After scaling to 304 tasks, the dev split has 60 tasks but was never evaluated with
baselines + agent. The paper currently shows test results on 125 tasks but dev results
on 11 tasks from an earlier benchmark version. This inconsistency will erode reviewer
trust instantly.

**Fix**:
- Re-run BM25 + dense + hybrid + reranker + agent on the 60-task dev split
- Report dev results in appendix as development/tuning diagnostics
- Report test results in main paper as primary evaluation
- Add one sentence explaining the dev set was used for prompt iteration

**Effort**: ~3 hours (baselines fast, agent ~30 min on 60 tasks at ~$4 API)
**Cost**: ~$4 API

### 4. At least one clean ablation
**Problem**: The paper claims the agent's iterative refinement helps. But without
an ablation, reviewers can't tell if the gains come from (a) iterative search,
(b) query reformulation, (c) just seeing more documents, or (d) the LLM backbone
being smarter than BM25 at matching.

**Fix**: Run one ablation on the test set:
- **Single-step agent**: Same LLM backbone, but only 1 search round (no refinement).
  If this matches the full agent, the iterative loop adds nothing.
  If this is worse, the iterative loop matters.

**Effort**: ~2 hours (modify agent script, run on test, score)
**Cost**: ~$5 API

### 5. Human audit statistics
**Problem**: All 304 tasks are LLM-generated. The plan called for human audit of pilot
tasks, contradiction tasks, and abstention tasks. No audit counts, agreement rates, or
acceptance/rejection rates are reported.

**Fix**:
- You (lead researcher) audit at minimum:
  - All 38 abstention tasks (highest error risk)
  - Random 20 from constraint + contradiction (second highest)
  - Random 10 from remaining families
- Total: ~68 tasks, ~4-5 hours of expert review
- Report: acceptance rate, common rejection reasons, any tasks removed/revised
- This is the single most important reviewer-trust signal for an LLM-generated benchmark

**Effort**: 5 hours (human time, cannot be automated)
**Cost**: $0

---

## NICE TO HAVE (Strengthens paper but not blocking)

### 6. Run agent with Opus 4 backbone
Compare Sonnet vs Opus as agent backbone. If Opus is better, it shows the controller
benefits from stronger reasoning. If same, it shows the bottleneck is retrieval, not
reasoning.

**Effort**: 3 hours, ~$15-20 API
**Priority**: Medium — strengthens the method story but not required

### 7. 4x rollout + RRF
Context-1 showed that 4 independent rollouts merged with RRF can match much larger
models. Run 4x agent rollouts on a subset (e.g., 50 test tasks) and report RRF results.

**Effort**: 2 hours, ~$20 API
**Priority**: Medium — directly connects to Context-1 and adds a cheap performance boost

### 8. Metadata filter tool for agent
The agent currently only uses BM25 search. Adding a metadata filter tool (filter by
year, organism, MeSH term) would test whether structured filtering improves retrieval
on constraint and temporal tasks specifically.

**Effort**: 4 hours coding + 2 hours evaluation
**Priority**: Medium-high — directly tests a core thesis of the paper

### 9. Citation expansion tool for agent
Add citation graph traversal (using OpenAlex reference IDs already in the corpus).
Test whether following citations helps on multi-hop tasks.

**Effort**: 3 hours coding + 2 hours evaluation
**Priority**: Medium — tests another thesis but multihop results are already decent

### 10. Error analysis
Sample 50 failed retrievals from test set, categorize per error taxonomy (14 types),
report distribution by family and method. Include 3-4 detailed case studies.

**Effort**: 4 hours
**Priority**: Medium — strengthens the analysis section considerably

### 11. Expand to 500+ tasks
More tasks improve statistical power. Current 304 is at the low end of the planned
300-800 range.

**Effort**: ~$15 API, 2 hours
**Priority**: Low — 304 is already enough if methodology is sound

---

## CUT ENTIRELY (Do not pursue before submission)

### 12. Full RL training of the controller
The original plan considered LoRA fine-tuning a 7-8B model on agent traces. The current
prompt-based controller is sufficient for an E&D paper. RL training is future work.

### 13. Larger corpus (50K-250K papers)
4,936 papers is enough for a focused benchmark paper. Scaling corpus is lower value
than fixing baselines.

### 14. Additional domains (chemistry, materials)
Single-domain (biomedicine) is fine for submission. Multi-domain is future work.

### 15. End-to-end answer generation evaluation
The paper is about retrieval, not answer synthesis. Adding OpenScholar or PaperQA2
as downstream answer baselines would be interesting but expands scope beyond what's
needed for E&D.

### 16. Distillation to a student model
Not needed for this paper. Mention as future work.

### 17. Croissant metadata + full dataset release
Required by E&D track, but can be done in the final 2 days (May 4-6). It's a
formatting task, not a research task. Don't prioritize it over methodology fixes.

---

## Execution Priority (Time-Ordered)

| Order | Task | Days | Effort | Cost | Blocking? |
|-------|------|------|--------|------|-----------|
| 1 | SPECTER2 + adapter reindex on Lambda | Apr 13 | 2h | $0 | YES |
| 2 | Dense + hybrid + reranker baselines on test | Apr 13 | 2h | $0 | YES |
| 3 | Separate abstention evaluation | Apr 13 | 2h | $0 | YES |
| 4 | Re-run dev set (baselines + agent on 60 tasks) | Apr 14 | 3h | $4 | YES |
| 5 | Single-step ablation on test | Apr 14 | 2h | $5 | YES |
| 6 | Human audit (68 tasks) | Apr 14-15 | 5h | $0 | YES |
| 7 | Metadata filter tool + evaluation | Apr 16-17 | 6h | $5 | No |
| 8 | Error analysis (50 failures) | Apr 17 | 4h | $0 | No |
| 9 | Opus backbone comparison | Apr 18 | 3h | $15 | No |
| 10 | Paper writing | Apr 19-24 | 5 days | $0 | YES |
| 11 | Croissant metadata + anonymization | May 3-4 | 3h | $0 | YES |
| 12 | Abstract submission | May 4 | 1h | $0 | YES |
| 13 | Final paper submission | May 6 | 1h | $0 | YES |

**Total additional cost for must-fix items: ~$9**
**Total additional cost including nice-to-haves: ~$44**

---

## Paper Framing (Final)

### Title
SynthSearch-Biomed: A Condition-Sensitive Retrieval Benchmark and Iterative Search
Controller for Biomedical Literature

### Core claim
We introduce SynthSearch-Biomed, a benchmark of 304 tasks across 8 retrieval families
that tests condition-sensitive evidence collection in biomedical literature. We show
that an iterative search controller improves **deeper evidence recall** (R@10, R@20)
on aggregation and temporal families where single-shot retrieval struggles, while
BM25 remains superior for **top-rank precision** (R@5, MRR). Abstention — recognizing
when the corpus lacks sufficient evidence — is completely unsolved by all methods.

### What the paper is NOT claiming
- Not "best scientific retrieval model"
- Not "agent beats all baselines on all metrics"
- Not "solved scientific search"
- Not "this benchmark covers all of science"

### What reviewers will check
1. Is the benchmark well-constructed? → Human audit stats answer this
2. Are the baselines strong? → Dense + hybrid + reranker answer this
3. Is the agent's improvement real? → Ablation answers this
4. Is the evaluation sound? → Separate abstention metrics + per-family breakdown answer this
5. Is the contribution significant? → 8-family taxonomy + open abstention problem answer this

---

## Decision Rule

**By April 20**: If must-fix items 1-6 are complete and the results still show the
same pattern (agent helps on aggregation/temporal, BM25 wins on precision), proceed
to paper writing.

**If dense+hybrid baseline with adapter closes the gap to the agent**: The paper becomes
purely a benchmark paper. The agent is demoted to "reference baseline." This is still
a valid E&D submission.

**If dense+hybrid baseline with adapter BEATS the agent**: Same as above. The benchmark
is the contribution. The finding that a well-configured hybrid pipeline beats an
iterative agent is itself a useful result.
