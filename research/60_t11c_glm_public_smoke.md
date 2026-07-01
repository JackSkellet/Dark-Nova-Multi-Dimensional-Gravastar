# T11c Dense-528 GLM Public Smoke

Date: 2026-07-01

Result file: `results/T11c_dense528_glm_public_smoke.json`

Prediction file: `results/T11c_dense528_glm_public_smoke_predictions.jsonl`

Task file: `data/glm_public_eval_tasks/glm5_2_public_smoke_tasks.jsonl`

## Question

Can the current strongest local same-budget dense checkpoint produce useful direct answers on the new GLM-5.2 public/synthetic smoke tasks before GLM outputs are scored?

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/generate_glm_local_dense_predictions.py --tasks-jsonl data/glm_public_eval_tasks/glm5_2_public_smoke_tasks.jsonl --checkpoint artifacts/T11c_dense528_adamw_fp32_50m/dense_decoder_best_model_only.pt --device rocm --seed 123 --max-new-tokens 64 --max-tasks 16 --output results/T11c_dense528_glm_public_smoke_predictions.jsonl --model-label T11c_dense528_adamw_fp32_50m

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_glm_public_tasks.py --tasks-jsonl data/glm_public_eval_tasks/glm5_2_public_smoke_tasks.jsonl --predictions-jsonl results/T11c_dense528_glm_public_smoke_predictions.jsonl --seed 123 --max-tasks 16 --baseline-category local_same_budget_baseline --output results/T11c_dense528_glm_public_smoke.json --experiment-id T11c_dense528_glm_public_smoke
```

## Result

T11c dense-528 scored 0/3 on the smoke tasks.

| Metric | Value |
| --- | ---: |
| task count | 3 |
| evaluated predictions | 3 |
| pass count | 0 |
| pass rate | 0.0 |
| complete metadata records | 3 |
| records with missing metadata | 0 |

The generated outputs are essentially newline/space continuations for all three prompts. This is negative evidence for direct instruction-style use of the current byte-level T11c checkpoint on short coding/doc tasks.

## Interpretation

This is a same-harness local baseline, not a GLM-5.2 result. It satisfies the first local-comparison requirement for the GLM track, but it does not close the gap because no saved GLM-5.2 outputs have been scored yet.

The result is consistent with prior negative free-form QuixBugs generation: T11c can provide useful teacher-forced ranking signal inside constrained candidate pools, but direct greedy byte generation is not yet a practical coding assistant behavior.

## Next

Score saved GLM-5.2 outputs on the same three tasks, then extend the task pack toward a reduced repository-completion or API-reuse slice. Keep the local comparison in the final report as a same-budget baseline with a 0/3 direct-generation score.
