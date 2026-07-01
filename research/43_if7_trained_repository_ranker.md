# IF7 Trained Repository Ranker

Date: 2026-07-01

## Hypothesis

IF7 sparse Hebbian context may become useful when it is one feature inside a task-aware repository-linking ranker, rather than a dense full-score conditioning vector or a hand-combined static score.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train_if7_repository_ranker.py --corpus-jsonl data/hf_mirror/exploratory_d5_balanced/corpus.jsonl --train-split train --eval-split validation --seed 123 --node-count 4096 --max-memory-rows 26000 --max-train-repositories 512 --max-eval-repositories 64 --negatives-per-query 64 --top-k 5 --epochs 12 --batch-size 4096 --learning-rate 0.02 --device rocm --output results/IF7i_trained_repository_ranker_d5_validation.json --experiment-id IF7i_trained_repository_ranker_d5_validation
```

The result record includes the resolved command, seed 123, hypothesis, git commit `1a32048fb32b1fa32dbf5c2ce4610adb217fafd5`, hardware summary, completion status, and metrics in `results/IF7i_trained_repository_ranker_d5_validation.json`.

## Data Scale

- Hebbian memory rows: 26,000 D5 train rows from 26,173 scanned train rows.
- Training task repositories: 512.
- Training query tasks: 1,728.
- Training candidate examples: 126,560.
- Evaluation rows: 804 D5 validation rows.
- Evaluation task repositories: 58.
- Evaluation query tasks: 180.
- Negatives per query: 64.
- Mean evaluation candidates per query: 68.6111.
- Device: ROCm via Torch device `cuda`.

## Methods

The trained ranker is a seven-parameter linear model over:

- normalized path-token overlap
- normalized lexical text-token overlap
- Hebbian candidate context score
- Hebbian pair edge score, the mean learned co-activation strength between the query assembly and candidate assembly
- same file extension
- bias

The no-Hebbian ablation uses the same feature layout and training data, but the Hebbian context and pair-edge features are zeroed.

## Results

| Method | hit@5 | MRR | coverage@5 |
| --- | ---: | ---: | ---: |
| trained_no_hebbian_ranker | 0.9555555555555556 | 0.8982318215651547 | 0.806005291005291 |
| trained_task_aware_ranker | 0.95 | 0.894178530052186 | 0.7898015873015873 |
| path_role_overlap | 0.9333333333333333 | 0.8584510175902719 | 0.7808201058201057 |
| lexical_text_overlap | 0.8333333333333334 | 0.7375695787911879 | 0.6097486772486772 |
| combined_lexical_hebbian | 0.5666666666666667 | 0.44301396738328264 | 0.3183862433862434 |
| raw_hebbian_context | 0.39444444444444443 | 0.33369958797835003 | 0.17805555555555555 |

The trained ranker beats lexical and raw Hebbian static baselines:

- `trained_ranker_beats_lexical=true`
- `trained_ranker_beats_raw_hebbian=true`

It does not beat the no-Hebbian ablation:

- `trained_ranker_beats_no_hebbian=false`

## Interpretation

This is a useful product-direction result but not a Hebbian-feature win. The task-aware repository-linking ranker improves over static lexical, raw Hebbian, and hand-combined scores, but the no-Hebbian trained ablation is still the best method. Adding the pairwise Hebbian edge score makes the full Hebbian ranker worse than the previous one-feature Hebbian version and worse than the no-Hebbian ablation. The gain is therefore coming from supervised task-aware ranking over path/lexical structure, not from the current Hebbian context features.

The IF7 design should not promote the current Hebbian features into decoder integration. The better product baseline is the supervised no-Hebbian repository ranker. IF7 should only continue if the Hebbian feature becomes repository-local or edge-typed enough to beat that trained ablation.

## Limitations

- Repository-linking proxy, not code generation or repair.
- Positives are same-repository files, not human-labeled dependencies.
- Linear ranker, not a decoder language model.
- Validation split has only 58 eligible multi-file repositories.
- No executable task, unit-test repair, API-reuse, or security gate is measured here.
