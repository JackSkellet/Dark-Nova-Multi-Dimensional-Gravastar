# QuixBugs Syntax-Pool Ordering Controls

Date: 2026-07-01

Result file: `results/QuixBugs_T11c_dense528_syntax_pool_ordering_controls_smoke.json`

## Question

Does T11c dense-528 likelihood ranking beat same-pool non-model ordering controls for bounded QuixBugs syntax-pool repair selection?

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_quixbugs_syntax_pool_ordering_controls.py --checkpoint artifacts/T11c_dense528_adamw_fp32_50m/dense_decoder_best_model_only.pt --repo-path data/raw/public/quixbugs --program gcd --program flatten --program possible_change --program sqrt --device rocm --seed 123 --timeout-seconds 15 --max-candidates-per-program 32 --top-k 1 --top-k 2 --top-k 4 --top-k 8 --output results/QuixBugs_T11c_dense528_syntax_pool_ordering_controls_smoke.json --experiment-id QuixBugs_T11c_dense528_syntax_pool_ordering_controls_smoke
```

## Result

All methods use the same 63-candidate syntax-preserving AST mutation pool and the same top-k budgets. The run evaluates the union of candidates selected by dense likelihood ranking, deterministic pool order, and seeded random order.

| ordering | top-1 repair rate | top-2 repair rate | top-4 repair rate | top-8 repair rate | best top-k | best repair rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T11c dense likelihood | 0.25 | 0.75 | 0.75 | 1.0 | 8 | 1.0 |
| deterministic pool order | 0.75 | 0.75 | 1.0 | 1.0 | 4 | 1.0 |
| seeded random order | 0.25 | 0.25 | 0.25 | 0.5 | 8 | 0.5 |

The deterministic pool order is the best ordering by the current tie-breaker because it reaches 4/4 repaired programs at top-4, while T11c dense likelihood needs top-8. Dense likelihood beats seeded random order but does not beat the deterministic non-model control.

## Interpretation

This is negative evidence for `dense_ranked_repair_selection` as the source of the top-k repair gain. The prior top-k run showed that bounded execution over the syntax pool can recover passing repairs; this control run shows the local dense checkpoint is not yet the best selector from that pool.

The useful surviving evidence is that the syntax-preserving pool contains passing candidates for all four selected tasks and that execution validation can find them with a bounded budget. The model-ranking component should be revised before scaling or replaced with a repair-aware ordering/localization method.

## Next Falsifying Test

Add repair-aware pruning or failure-localized candidate ordering and require it to beat deterministic pool order on the same four-task smoke before expanding to more QuixBugs programs. A larger subset should compare dense likelihood, deterministic order, random seeds, and the new repair-aware selector under identical top-k budgets.

## Follow-Up

`research/55_quixbugs_repair_aware_syntax_controls.md` adds that repair-aware static ordering. It repairs 4/4 selected programs at top-1 on the same syntax pool, beating deterministic pool order at top-4 and dense likelihood at top-8. This is still a hand-engineered selector result, not model capability evidence.
