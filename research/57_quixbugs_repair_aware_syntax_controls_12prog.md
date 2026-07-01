# QuixBugs 12-Program Repair-Aware Syntax Controls

Date: 2026-07-01

Result file: `results/QuixBugs_T11c_dense528_repair_aware_syntax_controls_12prog.json`

## Question

Does the four-program repair-aware static ordering advantage survive a larger, less handpicked QuixBugs Python slice?

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_quixbugs_syntax_pool_ordering_controls.py --checkpoint artifacts/T11c_dense528_adamw_fp32_50m/dense_decoder_best_model_only.pt --repo-path data/raw/public/quixbugs --program bitcount --program breadth_first_search --program bucketsort --program depth_first_search --program detect_cycle --program find_first_in_sorted --program find_in_sorted --program flatten --program gcd --program get_factors --program hanoi --program is_valid_parenthesization --device rocm --seed 123 --timeout-seconds 15 --max-candidates-per-program 32 --top-k 1 --top-k 2 --top-k 4 --top-k 8 --output results/QuixBugs_T11c_dense528_repair_aware_syntax_controls_12prog.json --experiment-id QuixBugs_T11c_dense528_repair_aware_syntax_controls_12prog
```

## Harness Fix

The first attempt failed before evaluation because the syntax-pool builder aborted when a program had no deterministic edit-baseline candidates. That was a harness bug: generic AST mutations should still be generated when deterministic templates are empty. The regression test `test_syntax_mutation_pool_keeps_generic_mutations_when_edit_baseline_empty` now covers this path.

## Result

The 12-program slice generated 189 syntax-valid candidates and evaluated the union of top-8 candidates across dense likelihood, deterministic pool order, repair-aware static order, and seeded random order.

| ordering | top-1 repair rate | top-2 repair rate | top-4 repair rate | top-8 repair rate | best top-k | best repair rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deterministic pool order | 0.1667 | 0.1667 | 0.25 | 0.25 | 4 | 0.25 |
| T11c dense likelihood | 0.0 | 0.1667 | 0.1667 | 0.25 | 8 | 0.25 |
| seeded random order | 0.1667 | 0.1667 | 0.1667 | 0.25 | 8 | 0.25 |
| repair-aware static order | 0.1667 | 0.1667 | 0.1667 | 0.25 | 8 | 0.25 |

Only `flatten`, `gcd`, and `get_factors` had passing candidates in the selected evaluation set. The broader slice therefore changes the interpretation: candidate generation coverage is now the main bottleneck, and static repair-aware ordering no longer beats deterministic order.

## Interpretation

This falsifies the four-program top-1 repair-aware selector result as a general claim. The repair-aware static order remains useful as a diagnostic scaffold, but it should move from `combine` to `revise`.

The syntax-preserving mutation pool also moves from `combine` to `revise`: it is not broad enough yet, because 9/12 programs have no passing selected candidate. Dense likelihood ranking remains `revise` because it does not beat same-pool controls on the larger slice.

## Next Falsifying Test

Improve candidate generation before spending more effort on ranking. The next useful test is a failure-localized or learned generator that increases passing-candidate coverage beyond 3/12 on this same slice, while preserving deterministic, dense, repair-aware, and multiple random same-pool controls.

