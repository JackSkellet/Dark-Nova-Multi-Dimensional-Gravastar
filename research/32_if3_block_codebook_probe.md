# IF3 Block-Codebook Compression Probe

Date: 2026-06-21

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_if3_block_codebook_probe.py \
  --checkpoint artifacts/T11c_dense528_adamw_fp32_50m/dense_decoder_last_model_only.pt \
  --output results/IF3_block_codebook_t11c_probe.json \
  --experiment-id IF3_block_codebook_t11c_probe \
  --block-size 256 \
  --codebook-size 256 \
  --residual-fraction 0.01 \
  --seed 123
```

The result was generated from clean commit `b07353d5653c808c055f709dca4c46290d9a7e59`.

Validation-loss follow-up command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_if3_block_codebook_validation.py \
  --checkpoint artifacts/T11c_dense528_adamw_fp32_50m/dense_decoder_last_model_only.pt \
  --corpus-jsonl data/hf_mirror/exploratory_d3/corpus.jsonl \
  --split validation \
  --device rocm \
  --seed 424242 \
  --batches 512 \
  --block-size 256 \
  --codebook-size 256 \
  --residual-fraction 0.01 \
  --output results/IF3_block_codebook_t11c_validation_probe.json \
  --experiment-id IF3_block_codebook_t11c_validation_probe
```

The validation-loss result was generated from clean commit `cdd67ccea93cb590f17b54a75521e7ed6cddcc33`.

## Prototype

IF3 tests an audited block-codebook weight generator on the trained T11c dense-528 model-only checkpoint. It splits floating tensors into 256-value blocks, fits a 256-entry learned codebook, stores block ids/scales plus 1 percent FP32 sparse residuals, and compares against a random-centroid codebook with identical accounting.

The first probe was reconstruction accounting only. The follow-up reconstructs the learned and random codebook states into FP32 PyTorch tensors and evaluates language-model validation loss on the same fixed D4 validation batches as the original T11c checkpoint. Neither probe benchmarks packed kernels.

## Results

| Metric | Learned codebook | Random control |
| --- | ---: | ---: |
| MSE | 0.002059607533738017 | 0.019161246716976166 |
| Max absolute error | 0.27352720499038696 | 0.5598403215408325 |
| Encoded bytes | 1,297,648 | 1,297,648 |
| Metadata bytes | 619,352 | 619,352 |
| Runtime FP32 buffer bytes | 41,615,360 | 41,615,360 |
| Encoded plus runtime bytes | 42,913,008 | 42,913,008 |
| Effective encoded bits per padded value | 0.997822342519685 | 0.997822342519685 |
| Sparse residual values | 104,038 | 104,038 |

Validation-loss follow-up:

| Policy | Validation loss | Delta vs FP32 | Sampled tokens |
| --- | ---: | ---: | ---: |
| FP32 T11c checkpoint | 1.0335341689933557 | 0.0 | 132,096 |
| Learned block-codebook reconstruction | 5.3102278262376785 | 4.276693657244323 | 132,096 |
| Random block-codebook reconstruction | 38.73094817996025 | 37.697414010966895 | 132,096 |

All three policies used validation sample order SHA256 `c19190b30eb4b2ab4b3b4cd8296280064a47283f73d4de3423a889bdfc17e0ab`.

Checkpoint/accounting:

- Checkpoint: `artifacts/T11c_dense528_adamw_fp32_50m/dense_decoder_last_model_only.pt`.
- Checkpoint type: model-only.
- Step: 195,313.
- Floating parameters: 10,397,105.
- Blocks: 40,640.

## Interpretation

The learned block-codebook reconstruction beats the random control under identical encoded-byte accounting and validation-loss sampling, so IF3 has a real mechanism signal beyond raw MSE. It is still a negative result for deployable checkpoint compression in its current form: the learned reconstruction raises validation loss from 1.0335 to 5.3102, and the PyTorch path needs a full FP32 runtime buffer, so encoded plus runtime bytes are about 42.9 MB, close to the original model-only scale.

The current IF3 block policy should not be promoted to packed-kernel work as-is. The next useful IF3 compression step would need a less destructive reconstruction scheme, such as per-tensor/per-channel codebooks, smaller blocks, layer-sensitive protection, or a BF16/int8 packed-runtime baseline that preserves loss before chasing sub-byte codebook storage.

## Source Artifacts

- `results/IF3_block_codebook_t11c_probe.json`
- `results/IF3_block_codebook_t11c_validation_probe.json`
- `results/research_assessment.json`
