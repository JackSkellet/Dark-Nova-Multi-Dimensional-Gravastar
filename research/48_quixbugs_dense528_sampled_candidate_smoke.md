# QuixBugs Dense-528 Sampled Candidate Smoke

Date: 2026-07-01

## Purpose

This run tests whether the failed T11c dense-528 greedy QuixBugs result improves with bounded sampling and syntax-aware candidate selection. It generates four candidates per program, records Python syntax validity for every generated candidate, prefers syntax-valid candidates when available, and falls back to the first generated candidate when no syntax-valid candidate exists.

## Source

- Repository: `https://github.com/jkoppel/QuixBugs`
- Local path: `data/raw/public/quixbugs`
- Pinned QuixBugs commit: `4257f44b0ff1181dedaedee6a447e133219fcebf`
- Checkpoint: `artifacts/T11c_dense528_adamw_fp32_50m/dense_decoder_best_model_only.pt`
- Checkpoint step: 195000
- Access date: 2026-07-01

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_quixbugs_dense_candidates.py --checkpoint artifacts/T11c_dense528_adamw_fp32_50m/dense_decoder_best_model_only.pt --repo-path data/raw/public/quixbugs --program gcd --program flatten --program possible_change --program sqrt --device cpu --seed 456 --max-new-tokens 192 --samples-per-program 4 --temperature 0.9 --top-k 32 --prefer-syntax-valid --timeout-seconds 15 --output results/QuixBugs_T11c_dense528_sampled_candidate_repair_smoke.json --experiment-id QuixBugs_T11c_dense528_sampled_candidate_repair_smoke
```

## Results

Aggregate metrics:

- Programs: 4.
- Generated candidates: 16.
- Syntax-valid generated candidates: 0.
- Evaluated candidates after syntax-preferred fallback: 4.
- Passing candidates: 0.
- Program repair rate: 0.0.
- End-to-end runtime: 15.16 seconds.

The sampled candidates continued to look like JavaScript, markup, comments, or malformed fragments rather than Python repair source. No sampled candidate parsed as Python.

## Interpretation

This is a stronger negative result than the greedy T11c run. The failure is not just argmax decoding: bounded top-k sampling with syntax-aware fallback still produces no syntactically valid Python candidates and no QuixBugs repairs.

The next useful step is not more blind sampling from this checkpoint. The repair path needs a stronger candidate generator, such as repair-specific prompting with a model better aligned to Python, syntax-constrained generation, supervised repair fine-tuning, or a smaller deterministic edit baseline that can be compared against local model output.

## Limitations

- Four-program smoke, not the full 40-program QuixBugs suite.
- Four samples per program is not pass@k.
- CPU inference for stability; this is not a throughput benchmark.
- Syntax filtering only checks Python parsing, not semantic quality.
- Fallback candidates are evaluated when no syntax-valid candidate exists, so pytest failures remain expected.
