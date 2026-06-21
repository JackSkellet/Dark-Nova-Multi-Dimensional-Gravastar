# Cycle 2 Research Start Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start the next research cycle with stricter T12 uncertainty, fairer tokenizer accounting, stronger executable benchmarks, and documented cycle-2 architecture candidates.

**Architecture:** Keep reusable analysis in `src/weightlab/`, experiment runners in `scripts/`, machine-readable outputs in `results/`, and research summaries in `research/`. Preserve prior artifacts and add stricter metadata instead of overwriting conclusions with unsupported claims.

**Tech Stack:** Python 3.12, PyTorch ROCm, `uv`, `pytest`, `ruff`, JSON experiment records.

---

### Task 1: T12 Paired Uncertainty

**Files:**
- Modify: `src/weightlab/dense_training.py`
- Modify: `scripts/evaluate_dense_checkpoint.py`
- Modify: `src/weightlab/t12_analysis.py`
- Modify: `tests/test_dense_training.py`
- Modify: `tests/test_t12_analysis.py`
- Regenerate: `results/T11b_adapter528_adamw_fp32_50m_*_eval.json`
- Regenerate: `results/T11c_dense528_adamw_fp32_50m_*_eval.json`
- Regenerate: `results/T12a_dense528_seed456_adamw_fp32_50m_*_eval.json`
- Regenerate: `results/T12b_adapter528_seed456_adamw_fp32_50m_*_eval.json`
- Regenerate: `results/T12c_dense528_seed789_adamw_fp32_50m_*_eval.json`
- Regenerate: `results/T12d_adapter528_seed789_adamw_fp32_50m_*_eval.json`
- Regenerate: `results/T12_three_seed_summary.json`

- [x] **Step 1: Add failing tests**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_dense_training.py::test_evaluate_dense_checkpoint_uses_dedicated_texts_and_seed -q
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_t12_analysis.py::test_t12_three_seed_summary_reports_final_validation_winner_and_no_test_selection -q
```

Expected before implementation: evaluator rejects `include_batch_losses`; T12 summary lacks `uncertainty`.

- [x] **Step 2: Persist per-batch loss records**

Add optional `include_batch_losses` support to checkpoint evaluation. Each record contains `batch_index`, `loss`, `tokens`, and `sample_sha256`.

- [x] **Step 3: Compute paired bootstrap CIs**

Verify dense/adapter eval pairs share split, evaluation seed, batch count, sample-order hash, and per-batch sample hash. Bootstrap dense-minus-adapter loss with 2,000 deterministic resamples.

- [x] **Step 4: Regenerate T12 eval and summary records**

Run each T12 eval with `--include-batch-losses`, then run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/summarize_t12_three_seed.py
```

- [ ] **Step 5: Regenerate from a clean committed code tree**

After code and documentation are committed, rerun the 24 T12 eval commands and the T12 summary from a clean tree so new result records report `git_dirty=false`.

### Task 2: Tokenizer Raw-Byte Accounting

**Files:**
- Modify: `src/weightlab/dense_training.py`
- Modify: `src/weightlab/d5_tokenizer_training_analysis.py`
- Modify: `scripts/summarize_d5_tokenizer_training.py`
- Modify: `tests/test_d5_tokenizer_training_analysis.py`
- Regenerate: `results/D5_tokenizer_training_comparison.json`
- Modify: `research/29_d5_tokenizer_training_comparison.md`

- [x] **Step 1: Label current byte-normalized losses as estimated**

Update research prose so the current metric is not represented as exact raw-byte NLL.

- [ ] **Step 2: Add exact evaluated-byte accounting**

For each evaluated target token, record decoded byte length and token NLL. Report `total_target_nll`, `evaluated_target_bytes`, `exact_nats_per_raw_byte`, and `exact_bits_per_raw_byte`.

- [ ] **Step 3: Add tests for byte and BPE accounting**

Use a toy tokenizer fixture where token byte spans are known. Assert exact byte-normalized loss differs from split-average estimated byte loss when token lengths vary.

- [ ] **Step 4: Regenerate tokenizer comparison**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/summarize_d5_tokenizer_training.py
```

### Task 3: Repository-Balanced Executable Benchmark

**Files:**
- Create: `src/weightlab/repository_task_sampler.py`
- Create: `src/weightlab/executable_coding_eval.py`
- Create: `scripts/evaluate_repository_coding_tasks.py`
- Create: `tests/test_repository_task_sampler.py`
- Create: `tests/test_executable_coding_eval.py`
- Create: `research/34_repository_balanced_executable_benchmark.md`

- [ ] **Step 1: Implement repository-first sampling**

Sampler order must be repository first, file second, task third. Store repository id, file path, task kind, source split, seed, and sample hash for every task.

- [ ] **Step 2: Add task kinds**

Start with completion, causal infilling proxy, syntax, unit-test fixture, bug repair fixture, API reuse, relevant-file selection, source-grounded docs, stale-doc detection, and hallucinated-symbol checks.

- [ ] **Step 3: Add baselines**

Compare untrained model, n-gram baseline, dense-528, adapter-528, byte model, and BPE model where checkpoints exist. Missing checkpoints must be recorded as unavailable, not silently skipped.

### Task 4: Cycle-2 Candidate Prototypes

**Files:**
- Create: `research/idea_foundry_cycle_2.md`
- Create: `src/weightlab/if4_fast_repo_adaptation.py`
- Create: `src/weightlab/if5_graph_extraction.py`
- Create: `src/weightlab/if6_compression_successor.py`
- Create: matching tests under `tests/`

- [x] **Step 1: Document eight mechanisms**

Each mechanism includes definition, primary-source prior art, difference, predicted benefit, likely failure, falsifying experiment, ROCm execution strategy, and scaling prediction.

- [ ] **Step 2: Prototype fast repository adaptation**

Compare updated retrieval, structured symbol memory, replay adapters, fast temporary weights, fast weights plus retrieval, and periodic consolidation on a chronological public-repository fixture.

- [ ] **Step 3: Improve graph extraction before training**

Measure AST/import/package/definition/call/test/doc link coverage. Do not train a graph-conditioned model until non-heuristic extraction materially improves.

- [ ] **Step 4: Prototype successor compression**

Archive IF3 block-codebook as negative and test low-rank plus groupwise residual codebooks against BF16, INT8, groupwise INT4, and random controls with full storage/runtime accounting.
