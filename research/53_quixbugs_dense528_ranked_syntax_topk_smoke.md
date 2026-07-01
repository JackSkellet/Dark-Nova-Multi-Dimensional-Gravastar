# QuixBugs T11c Dense-528 Ranked Syntax Top-K Smoke

Date: 2026-07-01

Result file: `results/QuixBugs_T11c_dense528_ranked_syntax_topk_smoke.json`

## Question

If T11c dense-528 top-1 likelihood ranking fails on the broader syntax-preserving mutation pool, does bounded top-k execution over the same ranking recover passing repairs?

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_quixbugs_dense_ranked_syntax_topk.py --checkpoint artifacts/T11c_dense528_adamw_fp32_50m/dense_decoder_best_model_only.pt --repo-path data/raw/public/quixbugs --program gcd --program flatten --program possible_change --program sqrt --device rocm --seed 123 --timeout-seconds 15 --max-candidates-per-program 32 --top-k 1 --top-k 2 --top-k 4 --top-k 8 --output results/QuixBugs_T11c_dense528_ranked_syntax_topk_smoke.json --experiment-id QuixBugs_T11c_dense528_ranked_syntax_topk_smoke
```

## Result

The same 63-candidate syntax-preserving AST mutation pool was ranked once by T11c dense-528 teacher-forced mean negative log likelihood. Pytest execution was then profiled cumulatively over requested top-k budgets:

| top-k per program | evaluated candidates | passing candidates | candidate pass rate | repaired programs | program repair rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 1 | 0.25 | 1 | 0.25 |
| 2 | 8 | 3 | 0.375 | 3 | 0.75 |
| 4 | 14 | 3 | 0.21428571428571427 | 3 | 0.75 |
| 8 | 26 | 4 | 0.15384615384615385 | 4 | 1.0 |

At top-8, every selected program has a passing candidate:

- `flatten`: `flatten:syntax_pool_001`
- `gcd`: `gcd:syntax_pool_001`
- `possible_change`: `possible_change:syntax_pool_003`
- `sqrt`: `sqrt:syntax_pool_001`

## Interpretation

This is positive Stage 2 pairwise evidence for `dense_ranked_repair_selection` plus `syntax_preserving_mutation_pool` plus bounded execution validation. The top-1 result remains negative at 1/4 repaired programs, but top-k execution reaches 4/4 on the same four-task smoke.

The result is not free-form model repair generation. The candidate pool is still AST-generated, source-replacement only, and evaluated on a small Python subset. It also does not prove that T11c ranking is better than a random or repair-aware non-model ordering from the same pool; that comparison remains necessary before scaling.

## Next Falsifying Test

Compare T11c top-k ranking against random order, deterministic edit order, and simple repair-aware pruning on a larger QuixBugs subset. Keep the combination only if model ranking plus bounded execution improves repair rate or lowers candidate budget versus those controls.

## Follow-Up Control

`research/54_quixbugs_syntax_pool_ordering_controls.md` runs same-pool ordering controls for this result. It preserves the useful syntax-pool plus execution-validation evidence but rejects dense likelihood as the strongest selector: deterministic pool order reaches 4/4 repaired programs by top-4, while T11c dense likelihood needs top-8.
