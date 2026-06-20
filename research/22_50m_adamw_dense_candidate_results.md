# 50M AdamW Dense/Candidate Results

Date: 2026-06-20

Canonical config: `configs/d4_adamw_fp32_50m_comparison.json`

All three runs used D4 train-only rows, byte tokenizer, seed 123, validation seed 424242, test seed 424243, sequence length 128, batch size 2, 195,313 steps, 50,000,128 training tokens, AdamW FP32, learning rate 0.0001, no scheduler, `block_impl=explicit_causal`, and `attention_mask_mode=finite_causal`. Validation and test evaluation used split-specific D4 rows with 512 batches / 132,096 sampled tokens per split. Test loss was not used for checkpoint selection.

## Loss

| Run | Architecture | Final validation | Final test | Best checkpoint step | Best validation | Best test |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T11a | dense 544 | 1.1141934367478825 | 1.074529936129693 | 195000 | 1.1190997470694128 | 1.0788767997873947 |
| T11b | residual-adapter 528 | 1.0387521978118457 | 1.0450384634314105 | 195000 | 1.049464229727164 | 1.052616473985836 |
| T11c | dense 528 | 1.0335341689933557 | 1.0488805254572071 | 195000 | 1.0354822403751314 | 1.0345332396682352 |

The final checkpoints have lower validation loss than the best periodic checkpoint for all three runs because the final step is 195,313 while periodic checkpoints are every 5,000 steps. Selection remains validation-only; test losses are reported after selection.

## Runtime, Storage, And Stability

| Run | Parameters | Model-only bytes | Optimizer/training-state bytes | Tokens/s | Runtime s | Peak VRAM bytes | Max recorded grad norm | Nonfinite grad records |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T11a | 11,025,505 | 44,118,725 | 88,245,677 | 22,119.3265 | 2,260.4724 | 481,580,544 | 405,092.5625 | 0 |
| T11b | 10,604,801 | 42,442,331 | 84,895,583 | 20,532.4126 | 2,435.1804 | 469,542,400 | 18.785385131835938 | 0 |
| T11c | 10,397,105 | 41,605,125 | 83,218,477 | 22,762.2958 | 2,196.6206 | 463,684,096 | 4.899395942687988 | 0 |

T11c is the smallest and fastest of the three runs. It uses 628,400 fewer parameters than T11a and 207,696 fewer parameters than T11b, with the smallest model-only checkpoint, optimizer/training-state payload, peak allocated VRAM, and max recorded gradient norm. T11b remains the slowest run by train tokens/s.

## Functional Probes

Functional probes used deterministic greedy byte-level generation on the D4 test split. These are held-out source-local probes, not executable JavaScript tests.

| Run | Prefix token accuracy | Prefix exact match | Span token accuracy | Span exact match | Comment token accuracy | Comment exact match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T11a | 0.238125 | 0.04 | 0.286875 | 0.04 | 0.1708984375 | 0.0 |
| T11b | 0.246875 | 0.04 | 0.29875 | 0.06 | 0.248046875 | 0.03125 |
| T11c | 0.26 | 0.04 | 0.281875 | 0.04 | 0.2607421875 | 0.0625 |

The model is causal, so the span task is a left-to-right span reconstruction probe rather than suffix-conditioned infilling. D4 is JavaScript source-only and does not contain a paired documentation/source consistency benchmark; the comment-anchored source completion task is reported as a limited source-local proxy, not a documentation consistency claim.

## Interpretation

T11c changes the interpretation of the earlier T11a/T11b pair. The residual-adapter branch is not required to beat dense 544 on validation loss, storage, VRAM, throughput, or gradient smoothness. Dense 528 is the current validation/resource/throughput leader. T11b still has a small final-test-loss edge over T11c and the best causal span token accuracy, so this is not a clean dominance result.

The current assessment records `pareto_improvement_found=true` through frontier-expansion assessments, with `pareto_dominance_found=false`. T11c strongly suggests T11b's earlier gain came at least partly from width/allocation and not purely from residual adapters. This should not be read as a deployable compression win or as a final architecture result.

## Source Artifacts

- `configs/d4_adamw_fp32_50m_comparison.json`
- `results/T11a_dense544_adamw_fp32_50m.json`
- `results/T11a_dense544_adamw_fp32_50m_final_validation_eval.json`
- `results/T11a_dense544_adamw_fp32_50m_final_test_eval.json`
- `results/T11a_dense544_adamw_fp32_50m_best_validation_eval.json`
- `results/T11a_dense544_adamw_fp32_50m_best_test_eval.json`
- `results/T11a_dense544_adamw_fp32_50m_final_test_functional.json`
- `results/T11b_adapter528_adamw_fp32_50m.json`
- `results/T11b_adapter528_adamw_fp32_50m_final_validation_eval.json`
- `results/T11b_adapter528_adamw_fp32_50m_final_test_eval.json`
- `results/T11b_adapter528_adamw_fp32_50m_best_validation_eval.json`
- `results/T11b_adapter528_adamw_fp32_50m_best_test_eval.json`
- `results/T11b_adapter528_adamw_fp32_50m_final_test_functional.json`
- `results/T11c_dense528_adamw_fp32_50m.json`
- `results/T11c_dense528_adamw_fp32_50m_final_validation_eval.json`
- `results/T11c_dense528_adamw_fp32_50m_final_test_eval.json`
- `results/T11c_dense528_adamw_fp32_50m_best_validation_eval.json`
- `results/T11c_dense528_adamw_fp32_50m_best_test_eval.json`
- `results/T11c_dense528_adamw_fp32_50m_final_test_functional.json`
- `results/T11_recomputed_summary.json`
