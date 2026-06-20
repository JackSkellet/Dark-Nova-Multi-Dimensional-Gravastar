# Trained Checkpoint Quantization

Date: 2026-06-20

Artifacts:

- `results/T11a_dense544_adamw_fp32_50m_quantization_test.json`
- `results/T11b_adapter528_adamw_fp32_50m_quantization_test.json`

Both probes quantized the final T11 checkpoint in memory and evaluated D4 test rows with 128 batches. The evaluator reconstructs quantized tensors into FP32 PyTorch weights for loss measurement; it does not benchmark packed kernels. `encoded_bytes` counts quantized values, scales, indexes, protected values, and sparse residual metadata. `runtime_buffer_bytes` separately counts the FP32 reconstructed tensors required by this evaluation path.

## T11a Dense 544

| Policy | Loss | Encoded bytes | Metadata bytes | Runtime buffer bytes |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 1.0825915068853647 | 44,102,020 | 0 | 44,102,020 |
| BF16 all | 1.082721871091053 | 22,051,010 | 0 | 44,102,020 |
| Uniform int8 | 1.0837366385385394 | 11,026,009 | 504 | 44,102,020 |
| Uniform int4 | 1.7007998679764569 | 5,513,257 | 504 | 44,102,020 |
| Residual-selected BF16 protected int4 | 1.694622211623937 | 6,174,685 | 441,456 | 44,102,020 |
| Residual-selected FP32 protected int4 | 1.6946575492620468 | 6,395,161 | 441,456 | 44,102,020 |
| Random BF16 protected int4 | 1.690289527643472 | 6,174,685 | 441,456 | 44,102,020 |
| Random FP32 protected int4 | 1.6901286435313523 | 6,395,161 | 441,456 | 44,102,020 |
| Sparse FP32 residual int4 | 1.6946575492620468 | 6,395,497 | 441,792 | 44,102,020 |

On T11a, BF16 and uniform int8 are close to FP32 on this 128-batch sample. Uniform int4 degrades badly. Residual-selected protected int4 does not beat matched random protection.

## T11b Residual-Adapter 528

| Policy | Loss | Encoded bytes | Metadata bytes | Runtime buffer bytes |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 1.062094802968204 | 42,419,204 | 0 | 42,419,204 |
| BF16 all | 1.062006092397496 | 21,209,602 | 0 | 42,419,204 |
| Uniform int8 | 1.0646247651893646 | 10,605,521 | 720 | 42,419,204 |
| Uniform int4 | 1.8724973928183317 | 5,303,121 | 720 | 42,419,204 |
| Residual-selected BF16 protected int4 | 1.8230411894619465 | 5,939,259 | 424,812 | 42,419,204 |
| Residual-selected FP32 protected int4 | 1.8230905253440142 | 6,151,305 | 424,812 | 42,419,204 |
| Random BF16 protected int4 | 1.8545035421848297 | 5,939,259 | 424,812 | 42,419,204 |
| Random FP32 protected int4 | 1.8545071044936776 | 6,151,305 | 424,812 | 42,419,204 |
| Sparse FP32 residual int4 | 1.8230905253440142 | 6,151,785 | 425,292 | 42,419,204 |

On T11b, BF16 and uniform int8 are close to FP32. Uniform int4 degrades badly. Residual-selected protected int4 beats matched random protection, but remains far worse than uniform int8 and still requires FP32 reconstruction buffers in the current PyTorch evaluation path.

## Interpretation

The trained checkpoints support a conservative quantization conclusion:

- Uniform int8 is the strongest simple compressed policy tested here.
- BF16 halves encoded value bytes with almost no measured loss change, but the current evaluation path still reconstructs FP32 tensors for execution.
- Uniform int4 is not acceptable at this model scale under the tested global per-tensor quantizer.
- Residual-selected protection is not consistently better than random: it loses to random on T11a and wins on T11b.
- FP32 protected values do not provide a meaningful advantage over BF16 protected values in these probes.
- Sparse FP32 residuals match FP32-protected behavior but add metadata and do not approach int8 quality.

This does not prove a deployable compression Pareto improvement. It is a trained-checkpoint falsification of the naive int4/protected-carrier path and a positive baseline for uniform int8/BF16 follow-up with real packed kernels.
