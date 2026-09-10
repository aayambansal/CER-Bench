# Historical qrel sensitivity — blocked handoff

**release_blocked: true; diagnostic_status: fail; verification_status: passed.**

Use [`v1_final/REPORT.md`](v1_final/REPORT.md) and [`v1_final/INPUT_AUDIT.json`](v1_final/INPUT_AUDIT.json). [`VERIFICATION.json`](VERIFICATION.json) records local commands, expected exit statuses, hashes and reproducibility checks. Scientific numerical outputs are deliberately absent because inputs violate the requested fail-closed contract.

## Decisive findings

- All 12 primary runs contain exactly the expected 125 test rows (108 supported and 17 empty-support tasks); no missing or extra task IDs.
- Historical corpus: 4,936 records, 4,923 distinct doc IDs, 10 duplicated IDs and 13 excess rows. Duplicate records conflict on PMID/title and other fields; the detailed audit preserves each physical row rather than selecting one record.
- Duplicate identities occur at 174 ranked positions across the 12 methods and 44 distinct test tasks, in 34 raw judgment records, and in 6 seed/negative references across all original splits.
- All 3,240 judgments parse under explicit verdict normalization: 1,106 relevant and 2,134 not relevant, exactly 30 per supported task. There are 264 seed pairs and 1,370 reconstructed expanded pairs. The stored expanded file has one missing pair: task `multihop_0189`, document `37307965`, raw verdict at `results/baselines/gold_adjudication.jsonl:1337` is `RELEVANT\n\nTHE DOCUMENT DISCUSSES CELLULAR IMMUNOTHERAPIES`.
- No additional unknown-reference, duplicate task/ranked-doc/judgment-pair, seed-negative contradiction or missing-judgment errors were reported by the completed input audit. This validates string-level bookkeeping only, not scientific identities or relevance.
- The candidate identity report independently reports 58 PMID/XML disagreements. It was read for counts/scope only, not for mappings or repaired evaluation.

## Verification and limits

27 tests passed (15 new sensitivity tests plus 12 existing strict-metric tests). Two independent full input audits (`v1_final`, `v1_repeat`) each exited 1 and produced byte-identical audit/report/failure/manifest files. All manifest hashes were checked; 20 historical input files were unchanged relative to the earlier audit snapshot. Source/code hashes were also verified against the final manifest.

Retrieval metrics, stored-score comparisons, rank correlations/reversals, component bootstrap/sign-flip/Holm results, subsampling curves and fulltext score strata were **not executed**. Their implementation and formula tests are present, but the end-to-end numerical branch remains unverified on these blocked inputs. Do not treat software test success as a successful scientific analysis or release authorization.

## Attempt provenance

- `v1`: superseded initial attempt. The reader incorrectly used Python `splitlines()`, treating embedded Unicode separators inside valid JSON strings as record boundaries. Its `Unterminated string` message is **not evidence of malformed historical JSONL**. The reader was corrected to split only physical LF and a regression test was added.
- `v1_lf`: superseded attempt, correctly stopped on duplicate corpus IDs but gave only a generic error.
- `v1_audit`: first detailed inventory, found both real blockers.
- `v1_final`: authoritative final detailed report, including raw source-line evidence and hashed environment.
- `v1_repeat`: independent deterministic rerun used for verification.

No historical files were overwritten, no IDs repaired or silently deduplicated, and no paper or overall-readiness gate was changed. All work was local: no humans, network/API retrieval, installs, paid jobs, git commands, or workers.
