# IF7 Hebbian-Conditioned Trained Model 500k Window Scale

Date: 2026-07-01

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train_if7_hebbian_model.py \
  --corpus-jsonl data/hf_mirror/exploratory_d5_balanced/corpus.jsonl \
  --split train \
  --validation-split validation \
  --node-count 2048 \
  --max-active-nodes 48 \
  --max-train-rows 26000 \
  --max-validation-rows 2048 \
  --max-train-patterns 500000 \
  --max-validation-patterns 25000 \
  --text-window-bytes 256 \
  --text-window-stride-bytes 256 \
  --epochs 1 \
  --batch-size 256 \
  --learning-rate 0.01 \
  --recall-at-k 32 \
  --device rocm \
  --output results/IF7d_hebbian_trained_model_d5_500k_windows.json \
  --experiment-id IF7d_hebbian_trained_model_d5_500k_windows \
  --seed 123
```

## Scope

This run changes the IF7 training unit from one pattern per file row to many
256-byte text-window patterns per row. It is the first IF7 trained run at a
hundreds-of-thousands pattern scale.

The run uses:

- 26,000 D5 train rows scanned from 26,173 available train rows.
- 494,403 usable train window patterns.
- 24,216 usable validation window patterns.
- 2,048 sparse assembly nodes.
- ROCm-backed Torch training.

## Result

Training and validation at recall@32:

| Method | Train loss | Validation loss | Hit@32 | Coverage@32 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw Hebbian memory | n/a | n/a | 0.8167327386851668 | 0.10158472099068659 | 0.193942667285143 |
| Trained cue-only | 0.0579814104953906 | 0.06556727285526119 | 0.8798315163528245 | 0.14248145120937647 | 0.383787117674622 |
| Trained cue-plus-Hebbian | 0.05942020124131567 | 0.07082293902317367 | 0.8619507763462174 | 0.1283229038138149 | 0.35924870462722114 |

Unlike the smaller row-level runs, cue-plus-Hebbian does not even win train loss in
this one-epoch window-pattern setting. It also loses validation loss, hit rate,
coverage, and MRR.

## Interpretation

This resolves the "too small" objection for the current IF7 integration. The simple
dense concatenation of Hebbian completion scores is not just failing because the
dataset was row-limited. At 494k real D5 window patterns, cue-only still wins.

The useful signal remains: raw Hebbian memory has real recall signal and the first
IF7 mechanism probe beat frequency/random controls. The failed part is the
conditioning design. The next IF7 design should use sparse top-k Hebbian outputs as
retrieval/reranking candidates or a gated feature path, not dense full-score
concatenation into a linear model.

## Source Artifacts

- `src/weightlab/if7_sparse_hebbian.py`
- `scripts/train_if7_hebbian_model.py`
- `results/IF7d_hebbian_trained_model_d5_500k_windows.json`
