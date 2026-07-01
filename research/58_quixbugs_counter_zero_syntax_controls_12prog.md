# QuixBugs Counter-Zero Syntax Controls

Date: 2026-07-01

Result file: `results/QuixBugs_T11c_dense528_counter_zero_syntax_controls_12prog.json`

## Question

Can one non-oracle mutation family improve the 12-program QuixBugs syntax-pool coverage after the repair-aware selector failed to generalize?

## Change

Added a generic final-counter invariant mutation:

- detect a function with an integer counter initialized to a constant and updated with `+=` or `-=`;
- when the function ends with `return True`, add a candidate that returns `counter == 0`;
- score that candidate in the repair-aware ordering with `final_counter_zero_check`.

This does not read `correct_python_programs`.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_quixbugs_syntax_pool_ordering_controls.py --checkpoint artifacts/T11c_dense528_adamw_fp32_50m/dense_decoder_best_model_only.pt --repo-path data/raw/public/quixbugs --program bitcount --program breadth_first_search --program bucketsort --program depth_first_search --program detect_cycle --program find_first_in_sorted --program find_in_sorted --program flatten --program gcd --program get_factors --program hanoi --program is_valid_parenthesization --device rocm --seed 123 --timeout-seconds 15 --max-candidates-per-program 32 --top-k 1 --top-k 2 --top-k 4 --top-k 8 --output results/QuixBugs_T11c_dense528_counter_zero_syntax_controls_12prog.json --experiment-id QuixBugs_T11c_dense528_counter_zero_syntax_controls_12prog
```

## Result

The candidate pool increased from 189 to 190 candidates. The selected top-8 union increased from 143 to 144 candidates.

| ordering | top-1 repair rate | top-2 repair rate | top-4 repair rate | top-8 repair rate | best top-k | best repair rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T11c dense likelihood | 0.0833 | 0.25 | 0.25 | 0.3333 | 8 | 0.3333 |
| repair-aware static order | 0.25 | 0.25 | 0.25 | 0.3333 | 8 | 0.3333 |
| seeded random order | 0.1667 | 0.1667 | 0.25 | 0.3333 | 8 | 0.3333 |
| deterministic pool order | 0.1667 | 0.1667 | 0.25 | 0.25 | 4 | 0.25 |

The repaired programs are now `flatten`, `gcd`, `get_factors`, and `is_valid_parenthesization`. This is an improvement from 3/12 to 4/12 over `results/QuixBugs_T11c_dense528_repair_aware_syntax_controls_12prog.json`.

## Interpretation

The useful movement is candidate coverage, not a broad selector win. Repair-aware ordering places three passing candidates in top-1, but dense likelihood, repair-aware order, and random all reach 4/12 by top-8. The result supports adding failure-localized/generic mutation families, while dense ranking and static repair-aware ordering remain revise items.

## Next Falsifying Test

Add another non-oracle mutation family for a still-uncovered failure mode and require the same 12-program gate to move beyond 4/12 without worsening the same-pool controls. The strongest near-term targets are variable substitution in calls/tuples, loop/guard insertion, and binary-search bound repair.

