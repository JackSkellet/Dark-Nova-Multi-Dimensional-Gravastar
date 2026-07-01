# QuixBugs T11c Dense-528 Ranked Syntax-Pool Smoke

Date: 2026-07-01

Result file: `results/QuixBugs_T11c_dense528_ranked_syntax_pool_smoke.json`

## Question

Does T11c dense-528 still select correct repairs when the deterministic repair pool is broadened into a larger syntax-preserving AST mutation pool?

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_quixbugs_dense_ranked_syntax_pool.py --checkpoint artifacts/T11c_dense528_adamw_fp32_50m/dense_decoder_best_model_only.pt --repo-path data/raw/public/quixbugs --program gcd --program flatten --program possible_change --program sqrt --device rocm --seed 123 --timeout-seconds 15 --max-candidates-per-program 32 --top-candidates-per-program 1 --output results/QuixBugs_T11c_dense528_ranked_syntax_pool_smoke.json --experiment-id QuixBugs_T11c_dense528_ranked_syntax_pool_smoke
```

## Result

The syntax-preserving AST mutation pool produced 63 candidates across the four selected QuixBugs Python tasks:

- `gcd`: 10 candidates
- `flatten`: 2 candidates
- `possible_change`: 24 candidates
- `sqrt`: 27 candidates

T11c dense-528 selected one lowest-NLL candidate per program. Only one selected candidate passed:

- candidate pass rate: 0.25
- program repair rate: 0.25
- passing selected program: `sqrt`

Top selected candidates:

- `flatten`: `flatten:syntax_pool_002`, failed
- `gcd`: `gcd:syntax_pool_002`, failed
- `possible_change`: `possible_change:syntax_pool_024`, failed
- `sqrt`: `sqrt:syntax_pool_001`, passed

## Interpretation

This is a negative follow-up to `research/50_quixbugs_dense528_ranked_edit_smoke.md`. T11c can rank the passing edits in a small deterministic six-candidate pool, but top-1 likelihood ranking does not survive a broader syntax-preserving mutation pool. The model appears to prefer plausible source continuations that are not necessarily correct repairs.

## Implication

The next repair path should not simply add more syntax-preserving mutations and trust top-1 likelihood. Useful follow-ups are:

- add repair-aware pruning before model ranking;
- evaluate top-k execution instead of top-1 only;
- include test-failure-localized candidates;
- compare against random/top-baseline selection from the same pool.
