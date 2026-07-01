# QuixBugs Dense-528 Model Candidate Smoke

Date: 2026-07-01

## Purpose

This run feeds the selected local dense-528 checkpoint into the QuixBugs candidate-repair lane. It is the first QuixBugs probe where replacement source candidates come from a local model checkpoint rather than the buggy source or oracle-correct reference files.

## Source

- Repository: `https://github.com/jkoppel/QuixBugs`
- Local path: `data/raw/public/quixbugs`
- Pinned QuixBugs commit: `4257f44b0ff1181dedaedee6a447e133219fcebf`
- Checkpoint: `artifacts/T11c_dense528_adamw_fp32_50m/dense_decoder_best_model_only.pt`
- Checkpoint step: 195000
- Access date: 2026-07-01

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_quixbugs_dense_candidates.py --checkpoint artifacts/T11c_dense528_adamw_fp32_50m/dense_decoder_best_model_only.pt --repo-path data/raw/public/quixbugs --program gcd --program flatten --program possible_change --program sqrt --device cpu --seed 123 --max-new-tokens 192 --timeout-seconds 15 --output results/QuixBugs_T11c_dense528_candidate_repair_smoke.json --experiment-id QuixBugs_T11c_dense528_candidate_repair_smoke
```

The result record includes the command, seed, source commit, hardware summary, checkpoint metadata, generation settings, source hashes, per-candidate pytest outcomes, aggregate metrics, and failure status.

## Results

The dense-528 checkpoint generated one greedy replacement-source candidate for each of four programs:

| Program | Candidate | Pytest result |
| --- | --- | --- |
| `gcd` | `gcd:dense_greedy` | fail during collection |
| `flatten` | `flatten:dense_greedy` | fail during collection |
| `possible_change` | `possible_change:dense_greedy` | fail during import/collection |
| `sqrt` | `sqrt:dense_greedy` | fail during import/collection |

Aggregate metrics:

- Program count: 4.
- Candidate count: 4.
- Candidate pass rate: 0.0.
- Programs with a passing model candidate: 0.
- Program repair rate: 0.0.
- End-to-end runtime: 4.31 seconds.

Generation behavior:

- `gcd` and `flatten` generated JavaScript-style documentation/API text, not Python repair source.
- `possible_change` and `sqrt` generated whitespace, which was recorded as an empty-generation placeholder.

## Interpretation

This is a negative functional repair result for the current dense-528 byte-level checkpoint under the first simple prompt. It proves the model-generated candidate lane works end-to-end, but the selected checkpoint does not produce usable QuixBugs Python repairs in this setup.

The result should not be softened into a partial success: the oracle-correct reference lane repairs all four tasks, while the local dense checkpoint repairs none. The follow-up `QuixBugs_T11c_dense528_sampled_candidate_repair_smoke` shows bounded sampling and syntax-aware fallback also repairs 0/4 tasks and finds no syntax-valid Python candidates. The next useful step is a stronger repair-oriented candidate generator, not more blind sampling from this checkpoint.

## Limitations

- Four-program smoke, not the full 40-program QuixBugs suite.
- Greedy byte-level generation only.
- Prompted replacement source from a general code language model, not a repair-specific decoder.
- No sampling, reranking, diff application, or syntax filtering.
- CPU inference for stability; this is not a throughput benchmark.
- No public leaderboard score is claimed.
