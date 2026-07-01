# IF7 Hebbian-Conditioned Trained Model

Date: 2026-07-01

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train_if7_hebbian_model.py \
  --corpus-jsonl data/hf_mirror/exploratory_d5_balanced/corpus.jsonl \
  --split train \
  --validation-split validation \
  --node-count 2048 \
  --max-active-nodes 48 \
  --max-train-rows 8192 \
  --max-validation-rows 1024 \
  --epochs 3 \
  --batch-size 128 \
  --learning-rate 0.01 \
  --recall-at-k 32 \
  --device rocm \
  --output results/IF7b_hebbian_trained_model_d5.json \
  --experiment-id IF7b_hebbian_trained_model_d5 \
  --seed 123
```

## Scope

This is the corrective trained-model test for IF7. It keeps the sparse Hebbian
assembly from the first probe, but adds supervised learning:

- Build the Hebbian assembly from D5 train rows only.
- Train a cue-only multilabel Torch model from repository/path/language/role cue nodes.
- Train a cue-plus-Hebbian multilabel Torch model from cue nodes plus Hebbian completion
  scores.
- Evaluate both trained models on D5 validation rows.

The target is hashed identifier/import nodes, not byte tokens. This is a trained
model with Hebbian conditioning, but still not a decoder language model, executable
code generator, repair model, or documentation model.

## Result

The run used ROCm through PyTorch HIP 7.2.53211. It loaded 8,192 D5 train rows,
produced 8,190 usable train patterns, and evaluated 804 usable validation patterns.
The Hebbian memory used 2,048 nodes and accounted for 19,231,715 bytes including the
label index.

Training loss decreased for both models:

| Model | Parameters | Epoch 1 loss | Epoch 3 loss |
| --- | ---: | ---: | ---: |
| Cue-only | 4,196,352 | 0.17052208684630446 | 0.05616499613492917 |
| Cue-plus-Hebbian | 8,390,656 | 0.13597904910425562 | 0.048402638067474295 |

Validation results at recall@32:

| Method | Loss | Hit@32 | Coverage@32 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Raw Hebbian memory | n/a | 0.9490049751243781 | 0.1708374895959744 | 0.5253746872347768 |
| Trained cue-only | 0.06737370789051056 | 0.9788557213930348 | 0.24129233239018805 | 0.691614738536327 |
| Trained cue-plus-Hebbian | 0.0708797499537468 | 0.9701492537313433 | 0.2140923753660086 | 0.6678122354947751 |

The trained cue-plus-Hebbian model does not beat the trained cue-only model on
validation loss, hit rate, coverage, or MRR. The record reports
`trained_cue_only_model_not_beaten`.

## Interpretation

The first IF7 probe showed that sparse Hebbian co-activation has real associative
signal. This trained follow-up shows that simply concatenating Hebbian completion
scores into a supervised linear predictor is not enough. It overfits or adds a noisy
shortcut: train loss is lower, but validation quality is worse than cue-only.

This is useful negative evidence. The mechanism should not be promoted directly into
the dense decoder as a concatenated feature path. The next IF7 attempt should change
the integration, not just scale this one: sparse gating, regularized Hebbian top-k
features, repository-heldout training, or using the assembly only for candidate
retrieval/reranking before decoder conditioning.

## Source Artifacts

- `src/weightlab/if7_sparse_hebbian.py`
- `scripts/train_if7_hebbian_model.py`
- `tests/test_if7_sparse_hebbian.py`
- `results/IF7b_hebbian_trained_model_d5.json`
