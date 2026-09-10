# Historical opaque-token arithmetic

**historical_string_label_diagnostic_not_validated_biomedical_relevance**

**release_blocked: true; release_ready: false; input_validation_status: fail.**

This is not validated biomedical relevance, a biological ranking claim, or a repaired benchmark. The strict v1_final failure artifacts remain immutable. Corpus records were not parsed, selected, indexed or dereferenced; only file bytes were hashed.

## Token Recall@20 means

All means use the same 108 supported tasks; all 125 test rows are retained in per-task outputs (17 null-metric empty-qrel rows per method/regime). Seed/stored/reconstructed pair counts are 264/1369/1370. Display rounding is not used for ranks.

| Method | Seed | Stored expanded | Reconstructed expanded |
|---|---:|---:|---:|
| bm25 | 0.467592592593 | 0.321392777219 | 0.320981254586 |
| dense | 0.344135802469 | 0.301470142249 | 0.301264380932 |
| bge | 0.467592592593 | 0.416072810579 | 0.416792975188 |
| e5large | 0.464506172840 | 0.362193416272 | 0.362810700222 |
| medcpt | 0.273148148148 | 0.274871771353 | 0.275591935962 |
| splade | 0.516975308642 | 0.387611195397 | 0.387405434080 |
| hybrid | 0.501543209877 | 0.376873652223 | 0.376359248931 |
| hybrid_reranked | 0.359567901235 | 0.336059505534 | 0.335647982900 |
| bge_reranker | 0.404320987654 | 0.318894177734 | 0.318482655101 |
| agent | 0.543209876543 | 0.351469156900 | 0.351057634266 |
| rm3 | 0.469135802469 | 0.317894623667 | 0.317483101033 |
| agent_single_step | 0.299382716049 | 0.198719517378 | 0.198410875403 |

## Ten original-pool systems: token rank sensitivity

RM3 and Agent T=1 means are supplemental outside-pool values; they are excluded from these correlations.

| Comparison (R20) | Kendall tau-b | Spearman | Strict reversals |
|---|---:|---:|---:|
| seed_vs_stored_expanded | 0.584306547468 | 0.735565707853 | 9 |

- seed_vs_stored_expanded: `bm25` vs `e5large`; left-minus-right differences 0.003086419753086489 → -0.04080063905234593.
- seed_vs_stored_expanded: `bm25` vs `hybrid_reranked`; left-minus-right differences 0.10802469135802467 → -0.014666728314563782.
- seed_vs_stored_expanded: `bge` vs `splade`; left-minus-right differences -0.04938271604938277 → 0.028461615181809674.
- seed_vs_stored_expanded: `bge` vs `hybrid`; left-minus-right differences -0.0339506172839506 → 0.039199158355459285.
- seed_vs_stored_expanded: `bge` vs `agent`; left-minus-right differences -0.07561728395061723 → 0.06460365367897442.
- seed_vs_stored_expanded: `e5large` vs `agent`; left-minus-right differences -0.07870370370370372 → 0.010724259372113076.
- seed_vs_stored_expanded: `splade` vs `agent`; left-minus-right differences -0.026234567901234462 → 0.03614203849716474.
- seed_vs_stored_expanded: `hybrid` vs `agent`; left-minus-right differences -0.04166666666666663 → 0.02540449532351513.
- seed_vs_stored_expanded: `hybrid_reranked` vs `bge_reranker`; left-minus-right differences -0.04475308641975301 → 0.017165327799664898.

| seed_vs_reconstructed_expanded | 0.584306547468 | 0.735565707853 | 9 |

- seed_vs_reconstructed_expanded: `bm25` vs `e5large`; left-minus-right differences 0.003086419753086489 → -0.04182944563670804.
- seed_vs_reconstructed_expanded: `bm25` vs `hybrid_reranked`; left-minus-right differences 0.10802469135802467 → -0.014666728314563726.
- seed_vs_reconstructed_expanded: `bge` vs `splade`; left-minus-right differences -0.04938271604938277 → 0.02938754110773556.
- seed_vs_reconstructed_expanded: `bge` vs `hybrid`; left-minus-right differences -0.0339506172839506 → 0.040433726256693836.
- seed_vs_reconstructed_expanded: `bge` vs `agent`; left-minus-right differences -0.07561728395061723 → 0.06573534092177274.
- seed_vs_reconstructed_expanded: `e5large` vs `agent`; left-minus-right differences -0.07870370370370372 → 0.011753065956475184.
- seed_vs_reconstructed_expanded: `splade` vs `agent`; left-minus-right differences -0.026234567901234462 → 0.036347799814037185.
- seed_vs_reconstructed_expanded: `hybrid` vs `agent`; left-minus-right differences -0.04166666666666663 → 0.02530161466507891.
- seed_vs_reconstructed_expanded: `hybrid_reranked` vs `bge_reranker`; left-minus-right differences -0.04475308641975301 → 0.017165327799664842.

| stored_expanded_vs_reconstructed_expanded | 1.000000000000 | 1.000000000000 | 0 |


## Exact missing-pair effect

Raw judgment row 1337 adds token `37307965` to `multihop_0189`: stored 9 labels → reconstructed 10. Full per-method task/mean effects and added-token ranks are in analysis.json as exact rational strings.

| Method | Stored task R20 | Reconstructed task R20 | Exact mean delta (108 tasks) |
|---|---:|---:|---:|
| bm25 | 4/9 | 2/5 | -1/2430 |
| dense | 2/9 | 1/5 | -1/4860 |
| bge | 2/9 | 3/10 | 7/9720 |
| e5large | 1/3 | 2/5 | 1/1620 |
| medcpt | 2/9 | 3/10 | 7/9720 |
| splade | 2/9 | 1/5 | -1/4860 |
| hybrid | 5/9 | 1/2 | -1/1944 |
| hybrid_reranked | 4/9 | 2/5 | -1/2430 |
| bge_reranker | 4/9 | 2/5 | -1/2430 |
| agent | 4/9 | 2/5 | -1/2430 |
| rm3 | 4/9 | 2/5 | -1/2430 |
| agent_single_step | 1/3 | 3/10 | -1/3240 |

## Conditional cluster token inference

Recall@20, 10,000 resamples by default; paired task-weighted cluster bootstrap, two-sided cluster sign-flip with finite Monte Carlo correction, Holm across six seed/reconstructed × agent-minus-BM25/SPLADE/Hybrid contrasts. Components use all original train/dev/test tasks. These are not validated biomedical or causal inference results.

| Regime / comparator | Difference | 95% cluster interval | MC p | Holm p |
|---|---:|---|---:|---:|
| seed / bm25 | 0.075617283951 | [0.031605129039253964, 0.1233974358974359] | 0.001399860014 | 0.008399160084 |
| seed / splade | 0.026234567901 | [-0.03835091899251192, 0.08636363636363636] | 0.447055294471 | 0.447055294471 |
| seed / hybrid | 0.041666666667 | [-0.006290805630428271, 0.09119874818577643] | 0.109789021098 | 0.311368863114 |
| reconstructed_expanded / bm25 | 0.030076379680 | [0.0021956344583654837, 0.059972194470957564] | 0.046595340466 | 0.212978702130 |
| reconstructed_expanded / splade | -0.036347799814 | [-0.07049901375137295, -0.0005562957973348765] | 0.042595740426 | 0.212978702130 |
| reconstructed_expanded / hybrid | -0.025301614665 | [-0.05539447306835071, 0.005166464065089841] | 0.103789621038 | 0.311368863114 |

## Unresolved provenance and scope

analysis.json embeds the original failed input audit and FAILURE.json without changing either: 10 colliding corpus tokens (13 excess records), and the one raw/stored qrel discrepancy. manifest.json hashes original sources and immutable strict artifacts. Matching saved scores at their rounded precision is arithmetic agreement only, not a provenance override.

- Opaque token equality does not establish biomedical relevance or correct document identity.
- Both original strict input failures remain unresolved; this is not a repaired benchmark.
- No fulltext strata computed and no fulltext claim; no corpus content dereferenced.
- Pool-biased automated labels, forced-positive seeds and fixed historical rankings; no causal system claim.
- Cluster tests assume symmetric independent component differences; known relations do not ensure semantic independence.
- Completeness curve not run in this focused continuation; cluster inference is token-level only.
