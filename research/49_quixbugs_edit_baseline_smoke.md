# QuixBugs Deterministic AST-Edit Baseline Smoke

Date: 2026-07-01

Result file: `results/QuixBugs_edit_baseline_repair_smoke.json`

## Question

Can the existing QuixBugs candidate-repair harness score successful non-oracle source replacements on the same four-task smoke where the T11c dense-528 model candidates failed?

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_quixbugs_edit_baseline.py --repo-path data/raw/public/quixbugs --program gcd --program flatten --program possible_change --program sqrt --timeout-seconds 15 --seed 123 --max-candidates-per-program 8 --output results/QuixBugs_edit_baseline_repair_smoke.json --experiment-id QuixBugs_edit_baseline_repair_smoke
```

## Result

The deterministic AST-edit baseline generated 6 candidate replacement sources across 4 QuixBugs Python programs. It produced 4 passing candidates and repaired all 4 selected programs:

- `gcd`: 1 candidate, best `gcd:edit_01`
- `flatten`: 1 candidate, best `flatten:edit_01`
- `possible_change`: 3 candidates, best `possible_change:edit_03`
- `sqrt`: 1 candidate, best `sqrt:edit_01`

Final metrics:

- candidate pass rate: 0.6666666666666666
- program repair rate: 1.0
- programs with passing candidate: 4

## Interpretation

This is a harness calibration result, not model capability evidence. The generator is hand-engineered, deterministic, and explicitly records `not_model_generated` and `does_not_read_oracle_correct_sources`. It shows that the four-task QuixBugs lane can distinguish failed dense-model candidates from successful replacement sources without relying on oracle-correct files.

## Implication

The immediate model-generation blocker remains: T11c dense-528 generated 0/4 repairs under greedy generation and 0/16 syntax-valid Python candidates under sampled generation. A useful next model step must first produce syntactically valid Python replacement candidates and then approach this deterministic baseline on the same bounded smoke before larger repair benchmarks are worth running.
