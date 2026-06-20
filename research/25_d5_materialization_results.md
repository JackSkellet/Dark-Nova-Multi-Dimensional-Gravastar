# D5 Balanced HF Materialization Results

Date: 2026-06-21

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/materialize_hf_corpus.py --manifest results/D3_hf_corpus_manifest.json --output-jsonl data/hf_mirror/exploratory_d5_balanced/corpus.jsonl --target-train-tokens 200000000 --max-train-tokens-per-config 35000000 --max-row-bytes 256000 --min-row-bytes 20 --near-duplicate-hamming-threshold 3 --near-duplicate-min-bytes 200 --output results/D5_hf_balanced_200m_materialization.json --experiment-id D5_hf_balanced_200m_materialization --seed 123 --no-resume
```

Result: completed.

| Metric | Value |
| --- | ---: |
| Accepted rows | 27,915 |
| Train tokens | 200,000,495 |
| Validation tokens | 6,802,318 |
| Test tokens | 3,922,575 |
| Total tokens | 210,725,388 |
| Language labels | 36 |
| JSONL bytes | 246,009,237 |

Output JSONL: `data/hf_mirror/exploratory_d5_balanced/corpus.jsonl`

Output SHA256: `34dce9e9127a68362f79be53d81692bbb90b254a43fd8f86f155542e49f9680c`

## Source Balance

The per-config train cap prevented one JavaScript config from satisfying the target alone.

| Dataset/config | Train tokens |
| --- | ---: |
| `CodedotAI/code_clippy_github::JavaScript-all` | 35,000,765 |
| `CodedotAI/code_clippy_github::all-mit` | 35,017,888 |
| `CodedotAI/code_clippy_github::all-apache-2.0` | 35,127,977 |
| `codeparrot/github-code-clean::Python-all` | 35,004,266 |
| `codeparrot/github-code-clean::JavaScript-all` | 35,000,396 |
| `codeparrot/github-code-clean::all-mit` | 24,849,203 |

## Content Roles

Rows are multi-label, so role counts do not sum to accepted rows.

| Role | Rows | Tokens |
| --- | ---: | ---: |
| source | 24,079 | 187,668,879 |
| test | 3,958 | 27,166,306 |
| documentation | 1,157 | 7,494,785 |
| docstring | 8,655 | 87,689,286 |
| readme | 351 | 649,353 |
| changelog | 21 | 329,535 |

## Filters

Rejected rows:

- duplicate text: 4,206
- near-duplicate text: 3,386
- vendor/generated path: 6,323
- generated marker: 1,010
- secret/confidential pattern: 271
- too large: 316
- too small: 42
- evaluation-contamination path: 46

## Limitations

- The corpus is exploratory only, not production or redistribution approved.
- Temporal split coverage is 0 percent: these Dataset Viewer parquet rows did not expose usable timestamp columns. Repository-aware split is present; true temporal holdout still requires a timestamp-bearing source or metadata join.
- Near-duplicate filtering is SimHash-based, not full MinHash/LSH.
- D5 has not yet been used for model training.
- Tokenizer comparison and BPE-vs-byte model training remain separate follow-up steps.
