# Importance Results

Status: generated from `results/E3_carrier_importance.json`, `results/E3b_output_error_selected_precision.json`, `results/E3c_tiny_transformer_precision.json`, `results/E3d_trained_tiny_transformer_precision.json`, `results/E3e_bf16_vs_fp32_carriers.json`, `results/E3f_internal_matrix_precision.json`, `results/E3g_trained_internal_layer_precision.json`, `results/E3h_real_open_model_matrix_precision.json`, `results/E3i_real_open_model_task_precision.json`, `results/E3j_real_open_model_natural_text_precision.json`, and `results/E3k_real_open_model_internal_tensor_precision.json`.

The first experiment has known synthetic carriers and includes a mandatory random-protection control. Causal ablation outperforming random protection is necessary but not sufficient evidence for real model carrier pathways.

## Seed 123 Results

Ranking overlap with known synthetic carriers:

- Causal ablation: 1.000.
- Gradient proxy: 1.000.
- Activation magnitude: 0.333.
- Random control: 0.000.

Precision policy MSE:

| Policy | Total bytes | Prediction MSE |
| --- | ---: | ---: |
| Full FP32 | 192 | 0.000000 |
| Uniform int4 | 36 | 0.476936 |
| Random FP32 protected | 84 | 0.387869 |
| Activation FP32 protected | 84 | 0.407582 |
| Causal FP32 protected | 84 | 0.439231 |
| Gradient FP32 protected | 84 | 0.439231 |
| Sparse FP32 residual | 84 | 0.439231 |

## Interpretation

The causal metric correctly identified the planted carriers, but causal FP32 protection did not beat the random-protection control on prediction MSE for this seed. This rejects the naive precision-allocation implementation. The likely issue is that preserving high-magnitude weights exactly is not automatically equivalent to minimizing downstream quantization error under a single global scale. The next experiment should use per-channel or groupwise quantization and optimize protected sets directly against output error.

## E3b: Output-Error-Selected Groupwise Precision

Seed 123 follow-up:

| Policy | Total bytes | Validation MSE |
| --- | ---: | ---: |
| Full FP32 | 192 | 0.000000 |
| Uniform int4 | 36 | 0.427584 |
| Groupwise int4 | 96 | 0.067606 |
| Random FP32 protected mean | 144 | 0.060586 |
| Random FP32 protected best | 144 | 0.044573 |
| Output-error FP32 protected | 144 | 0.002191 |
| Sparse FP32 residual | 144 | 0.002191 |

Selected indices: `[1, 2, 3, 4, 5, 7]`; overlap with planted carriers: 0.833.

Interpretation: on this synthetic linear task, groupwise quantization fixed much of the global-scale failure, and selecting FP32 carriers by measured output error beat random protection at the same byte budget. This supports only a narrow engineering claim: output-error selection is a better prototype precision allocator than naive causal-score protection under global int4. It does not establish real transformer carrier pathways, transfer across prompts, or a hardware-efficient layout.

## E3c: Tiny Transformer Held-Out Prompt Precision

Seed 123 follow-up:

- Architecture: deterministic random-initialized `tiny_transformer_lm`.
- Quantized tensor: output-head rows.
- Vocab size: 32.
- Sequence length: 12.
- Model width: 24.
- Calibration prompts: 64.
- Held-out prompts: 64.
- Selected protected rows: `[0, 1, 2, 3, 4, 6]`.

| Policy | Total bytes | Held-out logit MSE | Held-out KL divergence |
| --- | ---: | ---: | ---: |
| Full FP32 | 3072 | 0.000000 | 0.000000 |
| Uniform int4 | 396 | 0.081931 | 0.031628 |
| Groupwise int4 | 768 | 0.018951 | 0.009954 |
| Random FP32 protected mean | 1368 | 0.015225 | 0.008211 |
| Random FP32 protected best | 1368 | 0.009486 | 0.004597 |
| Output-error FP32 protected | 1368 | 0.007662 | 0.002143 |

Interpretation: this is the first H3 result using a transformer-style sequence model and held-out prompts. Groupwise quantization again improves over global int4, and output-error-selected FP32 rows beat both the random mean and the best random trial at the same byte budget for this seed. Evidence remains limited: the model is random initialized, only the output head is quantized, and no real language quality or hardware kernel behavior is measured.

## E3d: Trained Tiny Transformer Held-Out Prompt Precision

Seed 123 follow-up:

- Architecture: `trained_tiny_transformer_lm`.
- Trained component: output head by ridge regression.
- Target rule: next token is current token plus one modulo vocabulary size.
- Train prompts: 96.
- Held-out prompts: 64.
- Trained held-out accuracy: 0.983.
- Untrained held-out NLL: 4.417.
- Trained held-out NLL: 1.920.
- Selected protected rows: `[10, 14, 16, 24, 28, 29]`.

| Policy | Total bytes | Held-out logit MSE | Held-out KL | Held-out NLL | Held-out accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full FP32 | 3072 | 0.000000 | 0.000000 | 1.920149 | 0.983 |
| Uniform int4 | 396 | 0.017533 | 0.008146 | 1.941858 | 0.973 |
| Groupwise int4 | 768 | 0.005909 | 0.003029 | 1.914707 | 0.974 |
| Random FP32 protected mean | 1368 | 0.004833 | n/a | 1.916170 | n/a |
| Random FP32 protected best | 1368 | 0.004123 | n/a | 1.909232 | n/a |
| Output-error FP32 protected | 1368 | 0.003384 | 0.001690 | 1.910072 | 0.982 |

Interpretation: E3d is stronger than E3c because the output head is trained and evaluated on held-out targets. Output-error-selected FP32 rows beat the random mean on held-out logit MSE at the same byte budget and are close to the best random NLL draw, but NLL is mixed: the best random draw slightly beats the selected policy on NLL. This supports continued investigation, not a Pareto-improvement claim.

## E3e: BF16 Versus FP32 Carrier Rows

Seed 123 follow-up:

- Architecture: `trained_tiny_transformer_lm`.
- Trained component: output head by ridge regression.
- Quantized tensor: output-head rows.
- Protected rows: `[10, 14, 16, 17, 18, 29]`.

| Policy | Total bytes | Held-out logit MSE | Held-out KL | Held-out NLL | Held-out accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full FP32 | 3072 | 0.000000 | 0.000000 | 1.916327 | 0.983 |
| Groupwise int4 | 768 | 0.006302 | 0.003177 | 1.910919 | 0.973 |
| Random BF16 protected mean | 1080 | 0.005185 | n/a | n/a | n/a |
| Random BF16 protected best | 1080 | 0.004044 | n/a | n/a | n/a |
| Output-error BF16 protected | 1080 | 0.003354 | 0.001675 | 1.910821 | 0.977 |
| Random FP32 protected mean | 1368 | 0.005184 | n/a | n/a | n/a |
| Random FP32 protected best | 1368 | 0.004044 | n/a | n/a | n/a |
| Output-error FP32 protected | 1368 | 0.003353 | 0.001675 | 1.910719 | 0.977 |

Interpretation: selected BF16-like carrier rows beat random BF16 protected controls and groupwise int4 at the same seed, while using fewer bytes than selected FP32 carriers. FP32 carriers produced only a tiny logit-MSE improvement over BF16-like carriers in this reduced setup. This weakens the case for precision above BF16 for this output-head-only toy model; it does not answer whether FP32 matters for real transformer internals, adapter updates, optimizer state, or unstable continual updates.

## E3f: Internal MLP Matrix Precision

Seed 123 follow-up:

- Architecture: `trained_tiny_transformer_lm`.
- Trained component: output head by ridge regression.
- Quantized tensor: internal MLP output rows, `w_mlp_out`.
- Train prompts: 96.
- Held-out prompts: 64.
- Selected protected rows: `[15, 17, 20, 21, 23, 27, 35, 42]`.

| Policy | Total bytes | Held-out logit MSE | Held-out KL | Held-out NLL | Held-out accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full FP32 | 4608 | 0.000000 | 0.000000 | 1.912953 | 0.971 |
| Uniform int4 | 588 | 0.008240 | 0.004131 | 1.921588 | 0.973 |
| Groupwise int4 | 1152 | 0.002137 | 0.001051 | 1.911449 | 0.967 |
| Random FP32 protected mean | 1952 | 0.001845 | n/a | 1.911389 | n/a |
| Random FP32 protected best | 1952 | 0.001531 | n/a | 1.908083 | n/a |
| Output-error FP32 protected | 1952 | 0.001214 | 0.000598 | 1.915895 | 0.970 |

Interpretation: E3f moves the selective-precision test off the output head and onto an internal MLP matrix. Output-error-selected FP32 internal rows beat the random protected mean on held-out logit MSE at the same byte budget and improve over groupwise int4 on logit MSE, but NLL remains mixed: random protected draws and groupwise int4 have better NLL in this seed. This strengthens the evidence that output-error selection can identify sensitive internal rows in a toy transformer-style path, but it still does not establish real-model carrier pathways or a production precision policy.

## E3g: Trained Internal MLP Layer Precision

Seed 123 follow-up:

- Architecture: `trained_internal_tiny_transformer_lm`.
- Trained component: internal MLP output matrix `w_mlp_out` by ridge regression.
- Quantized tensor: trained internal MLP output rows, `trained_w_mlp_out_rows`.
- Train prompts: 96.
- Held-out prompts: 64.
- Untrained internal held-out NLL: 9.959018.
- Trained internal held-out NLL: 0.869122.
- Trained internal held-out accuracy: 0.807.
- Selected protected rows: `[1, 5, 7, 24, 28, 29, 39, 44]`.

| Policy | Total bytes | Held-out logit MSE | Held-out KL | Held-out NLL | Held-out accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full FP32 | 4608 | 0.000000 | 0.000000 | 0.869122 | 0.807 |
| Uniform int4 | 588 | 2.986089 | 0.192434 | 0.942150 | 0.803 |
| Groupwise int4 | 1152 | 0.767562 | 0.043512 | 0.868421 | 0.822 |
| Random BF16 protected mean | 1568 | 0.644747 | n/a | n/a | n/a |
| Random BF16 protected best | 1568 | 0.532743 | n/a | n/a | n/a |
| Output-error BF16 protected | 1568 | 0.450158 | 0.022425 | 0.854173 | 0.810 |
| Random FP32 protected mean | 1952 | 0.645896 | n/a | n/a | n/a |
| Random FP32 protected best | 1952 | 0.533782 | n/a | n/a | n/a |
| Output-error FP32 protected | 1952 | 0.450777 | 0.022429 | 0.854081 | 0.809 |

Interpretation: E3g is stronger than E3f because the internal matrix being quantized is itself trained on the toy next-token task instead of remaining random while only the output head is trained. Output-error-selected protected rows beat matched random BF16 and FP32 controls and improve logit MSE versus groupwise int4. BF16-like protected rows slightly beat FP32 protected rows on logit MSE while using fewer bytes, while FP32 is only marginally lower on NLL. This supports continued H3 investigation on trained internal tensors, but it still does not establish a real open-model carrier pathway or a kernel-ready precision layout.

## E3h: Real Open-Model Matrix Precision

Source checkpoint:

- Model: `sshleifer/tiny-gpt2`.
- Source: Hugging Face model repository, accessed 2026-06-20.
- Pinned commit: `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`.
- Local checkpoint path: `data/raw/public/tiny-gpt2/pytorch_model.bin`.
- Download command:

```bash
uv run python scripts/run_real_model_precision.py --allow-download --seed 123 --protected-count 128 --tensor-name transformer.wte.weight
```

Seed 123 result:

- Tensor: `transformer.wte.weight`.
- Shape: `[50257, 2]`.
- Parameters: 100514.
- Protected rows: 128.
- Measurement: matrix reconstruction MSE only, not task loss.

| Policy | Total bytes | Matrix MSE |
| --- | ---: | ---: |
| Full FP32 | 402056 | 0.000000000 |
| Uniform int4 | 50269 | 0.000015124 |
| Groupwise int4 | 653341 | 0.000000555 |
| Random BF16 protected mean | 654365 | 0.000000553 |
| Random BF16 protected best | 654365 | 0.000000553 |
| Output-error BF16 protected | 654365 | 0.000000535 |
| Random FP32 protected mean | 654877 | 0.000000553 |
| Random FP32 protected best | 654877 | 0.000000553 |
| Output-error FP32 protected | 654877 | 0.000000535 |

Interpretation: selected protected rows on a real open-model matrix beat matched random controls on reconstruction error, but this is not a Pareto improvement. The selected BF16/FP32 policies are larger than full FP32 because the tested matrix is extremely skinny and row-wise groupwise metadata dominates storage. Uniform int4 is much smaller but has higher reconstruction error. E3h therefore narrows the gap from toy tensors to a real checkpoint tensor while mostly reinforcing the storage-accounting warning: protected-row schemes can lose immediately when row metadata is large relative to row width.

## E3i: Real Open-Model Task-Loss Precision

Source checkpoint is the same pinned `sshleifer/tiny-gpt2` file used by E3h.

Command:

```bash
uv run python scripts/run_real_model_task_precision.py --seed 123 --protected-count 16 --calibration-sequences 6 --heldout-sequences 6 --sequence-length 12 --token-pool-size 256
```

Seed 123 result:

- Tensor: `transformer.wte.weight`.
- Shape: `[50257, 2]`.
- Parameters: 100514.
- Calibration sequences: 6.
- Held-out sequences: 6.
- Sequence length: 12.
- Candidate protected rows: 61.
- Protected rows: 16.
- Measurement: held-out next-token cross-entropy and KL to the FP32 model on deterministic token-id sequences, not natural-language text.

| Policy | Total bytes | Held-out KL | Held-out NLL | Held-out accuracy |
| --- | ---: | ---: | ---: | ---: |
| Full FP32 | 402056 | 0.000000000 | 10.828533 | 0.000 |
| Uniform int4 | 50269 | 0.000024287 | 10.828652 | 0.000 |
| Groupwise int4 | 653341 | 0.000000876 | 10.828637 | 0.000 |
| Random BF16 protected mean | 653469 | 0.000000842 | n/a | n/a |
| Random BF16 protected best | 653469 | 0.000000695 | n/a | n/a |
| Output-error BF16 protected | 653469 | 0.000000885 | 10.828650 | 0.000 |
| Random FP32 protected mean | 653533 | 0.000000839 | n/a | n/a |
| Random FP32 protected best | 653533 | 0.000000706 | n/a | n/a |
| Output-error FP32 protected | 653533 | 0.000000860 | 10.828650 | 0.000 |

Interpretation: E3i moves from reconstruction-only matrix MSE to an actual local forward pass and held-out next-token loss, but the result is negative. Output-error-selected protected rows did not beat the matched random protected mean on held-out KL, and the protected-row policies remained larger than full FP32 for the skinny embedding tensor. Uniform int4 was much smaller but had higher KL. The zero held-out accuracy and high NLL also show that deterministic token-id sequences are only a behavioral preservation smoke test, not useful language quality evidence. H3 therefore remains unproven on real model task behavior.

## E3j: Real Open-Model Natural-Text Task-Loss Precision

Source checkpoint is the same pinned `sshleifer/tiny-gpt2` file used by E3h/E3i. E3j additionally downloads the pinned public `vocab.json` and `merges.txt` tokenizer files from the same model commit.

Command:

```bash
uv run python scripts/run_real_model_natural_text_precision.py --seed 123 --protected-count 32 --sequence-length 24 --allow-download
```

Seed 123 result:

- Tensor: `transformer.wte.weight`.
- Shape: `[50257, 2]`.
- Parameters: 100514.
- Calibration texts: 6.
- Held-out texts: 6.
- Sequence length: 24.
- Candidate protected rows: 64.
- Protected rows: 32.
- Tokenizer: byte-level GPT-2 BPE from pinned `vocab.json` and `merges.txt`.
- Measurement: held-out next-token cross-entropy and KL to the FP32 model on natural-language repository/code prose.

| Policy | Total bytes | Held-out KL | Held-out NLL | Held-out accuracy |
| --- | ---: | ---: | ---: | ---: |
| Full FP32 | 402056 | 0.000000000 | 10.819059 | 0.000 |
| Uniform int4 | 50269 | 0.000007007 | 10.818159 | 0.000 |
| Groupwise int4 | 653341 | 0.000000977 | 10.819090 | 0.000 |
| Random BF16 protected mean | 653597 | 0.000000580 | n/a | n/a |
| Random BF16 protected best | 653597 | 0.000000166 | n/a | n/a |
| Output-error BF16 protected | 653597 | 0.000000178 | 10.819033 | 0.000 |
| Random FP32 protected mean | 653725 | 0.000000581 | n/a | n/a |
| Random FP32 protected best | 653725 | 0.000000190 | n/a | n/a |
| Output-error FP32 protected | 653725 | 0.000000214 | 10.819017 | 0.000 |

Interpretation: E3j fixes the biggest E3i benchmark weakness by using actual tokenized natural-language code/repository prose rather than random token IDs. Output-error-selected BF16/FP32 protected rows beat matched random means on held-out KL, but they do not beat the best random draws and remain larger than full FP32 because the tested embedding matrix is only two columns wide. Uniform int4 is far smaller and has the lowest NLL in this tiny run, though its KL to the FP32 model is worse than groupwise/protected layouts. This is better real-task smoke evidence than E3i, but it still does not demonstrate a storage-quality Pareto improvement or a useful-scale language result.

## E3k: Real Open-Model Internal-Tensor Natural-Text Precision

Source checkpoint and tokenizer are the same pinned `sshleifer/tiny-gpt2` artifacts used by E3j.

Command:

```bash
uv run python scripts/run_real_model_natural_text_precision.py --seed 123 --protected-count 1 --sequence-length 24 --tensor-name transformer.h.0.mlp.c_fc.weight --candidate-row-strategy all_rows --experiment-id E3k_real_open_model_internal_tensor_precision
```

Seed 123 result:

- Tensor: `transformer.h.0.mlp.c_fc.weight`.
- Shape: `[2, 8]`.
- Parameters: 16.
- Calibration texts: 6.
- Held-out texts: 6.
- Candidate row strategy: `all_rows`.
- Candidate protected rows: 2.
- Protected rows: 1.
- Measurement: held-out next-token cross-entropy and KL to the FP32 model on natural-language repository/code prose.

| Policy | Total bytes | Held-out KL | Held-out NLL | Held-out accuracy |
| --- | ---: | ---: | ---: | ---: |
| Full FP32 | 64 | 0.000000000 | 10.819059 | 0.000 |
| Uniform int4 | 20 | 0.000000000 | 10.819057 | 0.000 |
| Groupwise int4 | 32 | 0.000000000 | 10.819062 | 0.000 |
| Random BF16 protected mean | 52 | 0.000000000 | n/a | n/a |
| Random BF16 protected best | 52 | 0.000000000 | n/a | n/a |
| Output-error BF16 protected | 52 | 0.000000000 | 10.819062 | 0.000 |
| Random FP32 protected mean | 68 | 0.000000000 | n/a | n/a |
| Random FP32 protected best | 68 | 0.000000000 | n/a | n/a |
| Output-error FP32 protected | 68 | 0.000000000 | 10.819062 | 0.000 |

Interpretation: E3k addresses the E3j storage criticism by moving to a wider internal row layout, where uniform int4 and groupwise int4 are smaller than full FP32. However, this tiny model and tiny internal tensor show no measurable held-out KL difference between any precision policy after numerical roundoff is clamped to zero. Selected BF16 rows do not beat random controls, selected FP32 rows are larger than full FP32, and uniform int4 is smallest with effectively identical task behavior. This is evidence against the current selective-protected-row policy being useful on this internal tensor; it points back to needing a larger model or a tensor with real task sensitivity.
