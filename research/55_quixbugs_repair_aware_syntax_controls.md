# QuixBugs Repair-Aware Syntax Controls

Date: 2026-07-01

Result file: `results/QuixBugs_T11c_dense528_repair_aware_syntax_controls_smoke.json`

## Question

Can a non-oracle repair-aware static ordering beat deterministic pool order, dense likelihood ranking, and seeded random order on the same QuixBugs syntax-preserving mutation pool?

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_quixbugs_syntax_pool_ordering_controls.py --checkpoint artifacts/T11c_dense528_adamw_fp32_50m/dense_decoder_best_model_only.pt --repo-path data/raw/public/quixbugs --program gcd --program flatten --program possible_change --program sqrt --device rocm --seed 123 --timeout-seconds 15 --max-candidates-per-program 32 --top-k 1 --top-k 2 --top-k 4 --top-k 8 --output results/QuixBugs_T11c_dense528_repair_aware_syntax_controls_smoke.json --experiment-id QuixBugs_T11c_dense528_repair_aware_syntax_controls_smoke
```

## Result

All methods use the same 63-candidate syntax-preserving AST mutation pool and identical top-k budgets.

| ordering | top-1 repair rate | top-2 repair rate | top-4 repair rate | top-8 repair rate | best top-k | best repair rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| repair-aware static order | 1.0 | 1.0 | 1.0 | 1.0 | 1 | 1.0 |
| deterministic pool order | 0.75 | 0.75 | 1.0 | 1.0 | 4 | 1.0 |
| T11c dense likelihood | 0.25 | 0.75 | 0.75 | 1.0 | 8 | 1.0 |
| seeded random order | 0.25 | 0.25 | 0.25 | 0.5 | 8 | 0.5 |

The repair-aware selector ranks one passing candidate first for all four selected programs.

## Interpretation

This is positive evidence for a hand-engineered repair-aware ordering scaffold, not for free-form generation or model repair skill. The useful mechanism is candidate triage: simple non-oracle AST repair signals can reduce the validation budget from top-4 or top-8 to top-1 on this small smoke.

The dense likelihood component remains a revise item because it is weaker than both repair-aware static order and deterministic pool order on this broader syntax pool. The syntax pool remains useful because it contains passing candidates for all four tasks.

## Next Falsifying Test

Run the same ordering controls on a larger QuixBugs subset. Keep `repair_aware_static_ordering` only if it beats deterministic order, dense likelihood, and multiple seeded random orders under identical top-k budgets without using oracle-correct sources.
