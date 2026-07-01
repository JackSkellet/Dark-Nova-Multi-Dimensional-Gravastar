# IF7 Hebbian-Conditioned Trained Model Full-Train Scale

Date: 2026-07-01

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train_if7_hebbian_model.py \
  --corpus-jsonl data/hf_mirror/exploratory_d5_balanced/corpus.jsonl \
  --split train \
  --validation-split validation \
  --node-count 4096 \
  --max-active-nodes 48 \
  --max-train-rows 26000 \
  --max-validation-rows 2048 \
  --epochs 3 \
  --batch-size 128 \
  --learning-rate 0.01 \
  --recall-at-k 32 \
  --device rocm \
  --output results/IF7c_hebbian_trained_model_d5_fulltrain.json \
  --experiment-id IF7c_hebbian_trained_model_d5_fulltrain \
  --seed 123
```

## Scope

This is the larger-data follow-up to `IF7b_hebbian_trained_model_d5`. It raises
training coverage from 8,190 usable train patterns to 25,997 usable train patterns
and raises assembly size from 2,048 to 4,096 nodes. Validation remains the separate
D5 validation split, which currently yields 804 usable validation patterns.

## Result

The run used ROCm through PyTorch HIP 7.2.53211. It scanned 26,173 D5 train rows,
loaded 26,000 rows, and built 25,997 usable train patterns across 21,570 repositories.
The Hebbian memory accounted for 72,974,766 bytes including label-index storage.

Training loss again decreased more for the cue-plus-Hebbian model:

| Model | Parameters | Epoch 1 loss | Epoch 3 loss |
| --- | ---: | ---: | ---: |
| Cue-only | 16,781,312 | 0.0800826941346826 | 0.026130007448304446 |
| Cue-plus-Hebbian | 33,558,528 | 0.06443819338405284 | 0.024158115055253424 |

Validation at recall@32:

| Method | Loss | Hit@32 | Coverage@32 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Raw Hebbian memory | n/a | 0.9477611940298507 | 0.1718915023729845 | 0.494582516982335 |
| Trained cue-only | 0.03571857884526253 | 0.972636815920398 | 0.2508214363882148 | 0.7081591994687638 |
| Trained cue-plus-Hebbian | 0.04131173714995384 | 0.9601990049751243 | 0.20209600414695655 | 0.6307786573633818 |

The larger run therefore repeats the 8k-row conclusion: the simple dense
cue-plus-Hebbian concatenation model trains lower but validates worse than cue-only.
Scaling the row count and node count does not fix the integration.

## Interpretation

IF7 should stay alive as a sparse associative memory and retrieval/reranking
candidate, but the current trained conditioning design should be rejected. The
evidence now says the problem is not just small data. The next IF7 design should
avoid feeding dense Hebbian scores directly into a linear predictor; more plausible
paths are sparse top-k Hebbian gates, repository-heldout relevant-file reranking, or
using Hebbian assemblies to select candidate context before a decoder sees it.

## Source Artifacts

- `src/weightlab/if7_sparse_hebbian.py`
- `scripts/train_if7_hebbian_model.py`
- `results/IF7c_hebbian_trained_model_d5_fulltrain.json`
