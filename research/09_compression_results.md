# Compression Results

Status: generated from `results/E2_compositional_storage.json`.

The first experiment uses synthetic matrices and therefore measures accounting mechanics, not language-model quality. A compression method is not promising unless total bytes decrease after metadata and reconstruction overhead are included and reconstruction/task error remains acceptable.

## Seed 123 Results

| Method | Total bytes | Effective bits/parameter | Reconstruction MSE | Encode/decode ms |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 16384 | 32.00 | 0.000000 | 0.0034 |
| FP16 | 8192 | 16.00 | 0.000003 | 0.0102 |
| Int8 uniform | 4108 | 8.02 | 0.004992 | 0.0364 |
| Int4 uniform | 2060 | 4.02 | 1.647585 | 0.0126 |
| Low-rank SVD rank 8 | 4140 | 8.09 | 30.645373 | 1.7277 |
| Low-rank + sparse residual | 7420 | 14.49 | 16.600148 | 0.4777 |
| Rank-1 outer product | 516 | 1.01 | 90.950989 | 0.0141 |
| Shared basis coefficients | 4108 | 8.02 | 30.645373 | 0.3851 |
| Tensorized two-block | 528 | 1.03 | 62.124198 | 0.0356 |
| Product-quantized rows | 4288 | 8.38 | 30.298383 | 3.5207 |
| Kronecker rank-1 | 544 | 1.06 | 59.978289 | 1.3948 |
| Tensor Train 4D rank 8 | 4704 | 9.19 | 30.645373 | 0.5779 |

## Interpretation

Uniform int8 dominated the tested compositional and codebook methods on this random matrix: much lower storage than FP32 with low error and low runtime. The product-quantized row baseline includes codebook and assignment bytes; it was slightly larger than int8 and had much worse reconstruction error. The Kronecker rank-1 baseline compressed aggressively but had high error, while the Tensor Train 4D baseline used more bytes than int8 and had low-rank-level error. The speculative high-compression methods reduced bytes only by destroying reconstruction quality or losing to int8 after metadata. H2 is not supported by this reduced test.
