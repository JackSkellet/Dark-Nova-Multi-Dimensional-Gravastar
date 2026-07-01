# IF7 Repository Linking Validation

Date: 2026-07-01

## Hypothesis

IF7 sparse Hebbian context should improve repository file-linking when a query file must rank other same-repository files above cross-repository distractors.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_if7_repository_linking.py --corpus-jsonl data/hf_mirror/exploratory_d5_balanced/corpus.jsonl --train-split train --eval-split validation --seed 123 --node-count 4096 --max-train-rows 26000 --max-eval-repositories 64 --negatives-per-query 32 --top-k 5 --output results/IF7g_repository_linking_d5_validation.json --experiment-id IF7g_repository_linking_d5_validation
```

The result record reports the resolved command, seed 123, hypothesis, git commit `1a32048fb32b1fa32dbf5c2ce4610adb217fafd5`, hardware summary, status, and metrics in `results/IF7g_repository_linking_d5_validation.json`.

## Data Scale

- Training memory rows loaded: 26,000 D5 train rows from 26,173 scanned rows.
- Evaluation rows loaded: 804 D5 validation rows.
- Eligible evaluation repositories: 58, all used.
- Evaluation tasks: 180 query files.
- Distractors: up to 32 cross-repository files per query.
- Mean candidates per query: 36.6111.

This is larger than the original 8k-row IF7 mechanism probe, but it is not a full 200M-token D5 training run. It is a repository-linking validation over the currently available multi-file validation repositories.

## Results

| Method | hit@5 | MRR | coverage@5 |
| --- | ---: | ---: | ---: |
| lexical_text_overlap | 1.0000 | 0.9713 | 0.8653 |
| path_role_overlap | 0.9556 | 0.8856 | 0.8181 |
| combined_lexical_hebbian | 0.9500 | 0.8238 | 0.7765 |
| raw_hebbian_context | 0.4389 | 0.3841 | 0.2192 |

Best method: `lexical_text_overlap`.

`hebbian_beats_lexical=false` and `combined_beats_lexical=false`.

## Interpretation

This is a negative IF7 repository-linking result. The sparse Hebbian memory still has a real associative recall signal from earlier probes, but on this same-repository file-linking benchmark it does not beat a simple lexical baseline. Adding the Hebbian score to lexical overlap also hurts the lexical baseline, so the current combination rule should not be promoted.

The result reinforces the design direction from IF7b-IF7f: dense full-score conditioning and tiny candidate reranking are not the right exploitation path. If IF7 continues, it should use a stronger task-aware retrieval model or source/test/doc linking objective where sparse co-firing proposes candidates but does not override stronger lexical/path evidence.

## Limitations

- Repository-linking proxy, not language-model generation.
- Positives are other files in the same repository, not human-labeled dependencies.
- Distractors are sampled from other validation repositories.
- Global Hebbian memory is built from the train split only.
- Validation has only 58 eligible multi-file repositories; larger held-out repository-linking needs a broader multi-file split or a task index built from the full D5 corpus.
