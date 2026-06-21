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

## Prototype

IF3 tests an audited block-codebook weight generator on the trained T11c dense-528 model-only checkpoint. It splits floating tensors into 256-value blocks, fits a 256-entry learned codebook, stores block ids/scales plus 1 percent FP32 sparse residuals, and compares against a random-centroid codebook with identical accounting.

This is reconstruction accounting only. It does not evaluate language-model loss and does not benchmark packed kernels.

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

Checkpoint/accounting:

- Checkpoint: `artifacts/T11c_dense528_adamw_fp32_50m/dense_decoder_last_model_only.pt`.
- Checkpoint type: model-only.
- Step: 195,313.
- Floating parameters: 10,397,105.
- Blocks: 40,640.

## Interpretation

The learned block-codebook reconstruction beats the random control under identical encoded-byte accounting, so IF3 has a positive reconstruction signal. It is not a deployment compression win yet: the current PyTorch reconstruction path needs a full FP32 runtime buffer, so encoded plus runtime bytes are about 42.9 MB, close to the original model-only scale. The next useful IF3 test is a model-loss reconstruction evaluation and then a packed-kernel/runtime measurement if loss survives.

## Source Artifacts

- `results/IF3_block_codebook_t11c_probe.json`
- `results/research_assessment.json`
