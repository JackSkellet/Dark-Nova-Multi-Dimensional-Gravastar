# D4 Executable JavaScript Syntax Probe

Date: 2026-06-21

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_dense_js_executable.py \
  --checkpoint artifacts/T11c_dense528_adamw_fp32_50m/dense_decoder_last_model_only.pt \
  --corpus-jsonl data/hf_mirror/exploratory_d3/corpus.jsonl \
  --split test \
  --device rocm \
  --seed 424242 \
  --tasks 64 \
  --prefix-chars 16 \
  --target-tokens 48 \
  --output results/T11c_dense528_adamw_fp32_50m_final_test_js_executable.json \
  --experiment-id T11c_dense528_adamw_fp32_50m_final_test_js_executable
```

The result was generated from clean commit `c49a34cd7046885d2fd0a80cee3cb1cd7ce413b8`.

## Scope

This is the first executable JavaScript checkpoint probe for the selected D4 dense-528 model. It filters held-out D4 test lines whose oracle completion passes `node --check`, asks the model for greedy byte completions from a 16-character prefix, then checks the generated completed line with the same Node parser.

This is syntax-only. It is not a unit-test benchmark, repair benchmark, API-reuse benchmark, infilling benchmark, or multilingual evaluation.

## Results

| Metric | Value |
| --- | ---: |
| Node version | v20.20.1 |
| Candidate oracle-valid lines scanned | 1,024 |
| Completed tasks | 64 |
| Oracle syntax pass rate | 1.0 |
| Generated syntax pass rate | 0.5625 |
| Exact match rate | 0.140625 |
| Token accuracy mean | 0.29024482572258886 |
| Edit similarity mean | 0.39953814128603926 |

## Interpretation

The selected D4 dense-528 checkpoint can sometimes continue short JavaScript statements into syntactically valid code, but the probe is still weak evidence. A syntax pass can be semantically wrong, and line-only context is much easier than file-aware completion, tests, repair, or API reuse.

The next evaluation step should compare dense-528 against the D5 byte/BPE pilots on this executable syntax probe, then add at least one runtime/unit-test or API-reuse task. This result should not be used as a broad functional coding claim.

## Source Artifacts

- `results/T11c_dense528_adamw_fp32_50m_final_test_js_executable.json`
- `results/research_assessment.json`
