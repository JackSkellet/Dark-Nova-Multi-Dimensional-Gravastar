# IF4 Fast Repository Adaptation Probe

Date: 2026-06-21

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_if4_fast_repo_adaptation.py \
  --repo-path data/raw/public/markupsafe \
  --max-commits 30 \
  --top-k 5 \
  --output results/IF4_fast_repo_adaptation_markupsafe.json \
  --experiment-id IF4_fast_repo_adaptation_markupsafe
```

## Scope

This is the first real chronological public-repository version of the IF2 fast-weight
scratchpad idea. It uses only local Git state visible at each exposed commit to rank
files changed in the next commit. It is a changed-file retrieval and adaptation proxy,
not patch generation, unit-test repair, or language-model training.

The probe compares:

- Updated lexical retrieval.
- Structured symbol/path memory.
- Replay adapter proxy.
- Fast temporary weights.
- Fast temporary weights plus retrieval.
- Periodic consolidation.

## Result

The MarkupSafe run used 30 first-parent commits from local clone head `b2e4d9c7687b`.
The record was generated from clean project commit
`e64510b1ae7638e51540bc6aef6ccb56a90a2fbd` with `git_dirty=false`.

Top-5 future changed-file accuracy:

- Updated retrieval: 0.6667.
- Structured symbol/path memory: 0.7778.
- Replay adapter proxy: 0.7778.
- Fast temporary weights: 0.4444.
- Fast weights plus retrieval: 0.7778.
- Periodic consolidation: 0.6667.

Fast weights plus retrieval also scored 0.7778 on the deterministic paraphrase-query
proxy. Prior-task retention at the final step was 0.8571. Mean update cost was
18.99 ms, total accounted storage was 87,216 bytes, and rollback support was measured
true by removing the most recent applied fast-weight update.

## Interpretation

This promotes IF2 from a synthetic API-fact fixture to a real public Git-history probe.
The mechanism signal is mixed: fast weights alone underperform updated retrieval, while
fast weights plus retrieval ties the structured and replay controls on this MarkupSafe
sample. That is useful evidence for continuing fast temporary learning as a combined
retrieval/update mechanism, but it is not enough to promote IF4 to a training run.

## Limitations

- Changed-file retrieval proxy, not patch generation or executable repair.
- Replay adapter is a sparse commit-subject proxy, not a neural adapter.
- Fast weights are a feature-hashed associative matrix, not language-model weights.
- Paraphrase transfer uses a deterministic query rewrite proxy.
- No security, poisoning, or authorization gate is implemented.
- No ROCm kernel, packed representation, or model-training path is measured.

## Source Artifacts

- `src/weightlab/if4_fast_repo_adaptation.py`
- `scripts/run_if4_fast_repo_adaptation.py`
- `tests/test_if4_fast_repo_adaptation.py`
- `results/IF4_fast_repo_adaptation_markupsafe.json`
