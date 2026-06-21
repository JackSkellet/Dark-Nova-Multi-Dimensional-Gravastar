# Repository-Balanced Executable Benchmark Scaffold

Date: 2026-06-21

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_repository_coding_tasks.py \
  --corpus-jsonl data/hf_mirror/exploratory_d5_balanced/corpus.jsonl \
  --split test \
  --seed 123 \
  --max-repositories 32 \
  --files-per-repository 2 \
  --task-kind completion \
  --task-kind infilling \
  --task-kind syntax \
  --task-kind unit_tests \
  --task-kind bug_repair \
  --task-kind api_reuse \
  --task-kind relevant_file_selection \
  --task-kind source_grounded_documentation \
  --task-kind stale_document_detection \
  --task-kind hallucinated_symbols \
  --output results/D5_repository_balanced_task_sample.json \
  --experiment-id D5_repository_balanced_task_sample
```

## Scope

This is the first repository-balanced executable-benchmark scaffold. It creates a deterministic task index with repository-first, file-second, task-third ordering. It does not yet run model generations, syntax checks, unit tests, repair scoring, API-reuse scoring, source-grounded documentation scoring, stale-document detection, or hallucinated-symbol checks.

## Current Task Index

The current D5 test-split task index samples:

- 32 repositories.
- 32 files.
- 10 task kinds per selected file.
- 320 task slots total.

Task kinds:

- `completion`
- `infilling`
- `syntax`
- `unit_tests`
- `bug_repair`
- `api_reuse`
- `relevant_file_selection`
- `source_grounded_documentation`
- `stale_document_detection`
- `hallucinated_symbols`

Each task records repository id, file path, task kind, source split, seed, repository/file/task indexes, language, content roles, row hash, text hash, byte count, and sample hash.

## Interpretation

This replaces corpus-order line sampling as the benchmark indexing policy for the next executable evaluation pass. The next step is to attach concrete task constructors and scorers for untrained, n-gram, dense-528, adapter-528, byte, and BPE checkpoints. Missing checkpoints must be recorded as unavailable rather than silently skipped.

## Source Artifacts

- `src/weightlab/repository_task_sampler.py`
- `scripts/evaluate_repository_coding_tasks.py`
- `tests/test_repository_task_sampler.py`
- `results/D5_repository_balanced_task_sample.json`
