# QuixBugs T11c Dense-528 Ranked AST-Edit Smoke

Date: 2026-07-01

Result file: `results/QuixBugs_T11c_dense528_ranked_edit_smoke.json`

## Question

After free-form T11c dense-528 generation failed on the four-task QuixBugs smoke, can the checkpoint still provide useful repair signal by ranking a deterministic pool of syntax-valid AST-edit candidates?

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_quixbugs_dense_ranked_edits.py --checkpoint artifacts/T11c_dense528_adamw_fp32_50m/dense_decoder_best_model_only.pt --repo-path data/raw/public/quixbugs --program gcd --program flatten --program possible_change --program sqrt --device rocm --seed 123 --timeout-seconds 15 --max-candidates-per-program 8 --top-candidates-per-program 1 --output results/QuixBugs_T11c_dense528_ranked_edit_smoke.json --experiment-id QuixBugs_T11c_dense528_ranked_edit_smoke
```

## Result

The deterministic AST-edit pool contained 6 candidates across 4 programs. T11c dense-528 scored each candidate as a teacher-forced continuation of the repair prompt and selected one candidate per program by lowest mean negative log likelihood.

Selected candidates:

- `gcd`: `gcd:edit_01`
- `flatten`: `flatten:edit_01`
- `possible_change`: `possible_change:edit_03`
- `sqrt`: `sqrt:edit_01`

All four selected candidates passed the QuixBugs pytest harness:

- selected candidate count: 4
- candidate pass rate: 1.0
- program repair rate: 1.0

For `possible_change`, where the deterministic pool had three candidates, T11c ranked the passing edit first:

- `possible_change:edit_03`: mean NLL 1.4933, selected, passed
- `possible_change:edit_01`: mean NLL 1.5029
- `possible_change:edit_02`: mean NLL 1.5043

## Interpretation

This is model-involved repair evidence, but it is not free-form model repair generation. The candidate pool is still hand-engineered and deterministic. The useful signal is that the trained T11c dense-528 checkpoint ranks the passing syntax-valid edits above the failing edits in the selected four-task smoke.

## Implication

The next repair step should bridge away from the hand-engineered pool while preserving syntax validity. A useful follow-up would generate a broader but still syntax-constrained candidate set, then test whether T11c likelihood ranking continues to select passing repairs without oracle-correct sources.
