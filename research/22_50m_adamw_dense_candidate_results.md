# 50M AdamW Dense/Candidate Results

Date: 2026-06-20

Canonical config: `configs/d4_adamw_fp32_50m_comparison.json`

Both runs used D4 train-only rows, byte tokenizer, seed 123, validation seed 424242, test seed 424243, sequence length 128, batch size 2, 195,313 steps, 50,000,128 training tokens, AdamW FP32, learning rate 0.0001, no scheduler, `block_impl=explicit_causal`, and `attention_mask_mode=finite_causal`. Validation and test evaluation used split-specific D4 rows with 512 batches / 132,096 sampled tokens per split. Test loss was not used for checkpoint selection.

## Loss

| Run | Architecture | Final validation | Final test | Best checkpoint step | Best validation | Best test |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T11a | dense 544 | 1.1141934367478825 | 1.074529936129693 | 195000 | 1.1190997470694128 | 1.0788767997873947 |
| T11b | residual-adapter 528 | 1.0387521978118457 | 1.0450384634314105 | 195000 | 1.049464229727164 | 1.052616473985836 |

The final checkpoints have lower validation loss than the best periodic checkpoint for both runs because the final step is 195,313 while periodic checkpoints are every 5,000 steps. Selection remains validation-only; test losses are reported after selection.

## Runtime, Storage, And Stability

| Run | Parameters | Model-only bytes | Optimizer/training-state bytes | Tokens/s | Runtime s | Peak VRAM bytes | Max recorded grad norm | Nonfinite grad records |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T11a | 11,025,505 | 44,118,725 | 88,245,677 | 22,119.3265 | 2,260.4724 | 481,580,544 | 405,092.5625 | 0 |
| T11b | 10,604,801 | 42,442,331 | 84,895,583 | 20,532.4126 | 2,435.1804 | 469,542,400 | 18.785385131835938 | 0 |

T11b uses 420,704 fewer trainable parameters, a 1,676,394-byte smaller model-only checkpoint, a 3,350,094-byte smaller optimizer/training-state payload, and lower peak allocated VRAM. It is about 7.2 percent slower by train tokens/s. T11a has repeated large but finite gradient-norm spikes; T11b's recorded gradient norms remain bounded in a much narrower range.

## Functional Probes

Functional probes used deterministic greedy byte-level generation on the D4 test split. These are held-out source-local probes, not executable JavaScript tests.

| Run | Prefix token accuracy | Prefix exact match | Span token accuracy | Span exact match | Comment token accuracy | Comment exact match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T11a | 0.238125 | 0.04 | 0.286875 | 0.04 | 0.1708984375 | 0.0 |
| T11b | 0.246875 | 0.04 | 0.29875 | 0.06 | 0.248046875 | 0.03125 |

The model is causal, so the span task is a left-to-right span reconstruction probe rather than suffix-conditioned infilling. D4 is JavaScript source-only and does not contain a paired documentation/source consistency benchmark; the comment-anchored source completion task is reported as a limited source-local proxy, not a documentation consistency claim.

## Interpretation

T11b is the stronger validation-loss model and has better lightweight functional scores, smaller storage, lower VRAM, and much smoother recorded gradients. T11a slightly wins final held-out test loss and throughput. The result is therefore mixed: the residual-adapter candidate improves several meaningful dimensions, but it does not dominate the dense baseline on held-out test loss or speed.

The current assessment remains `pareto_improvement_found=false` until a candidate improves a primary quality metric without giving up the key deployment/runtime dimensions, or until the project explicitly treats the validation/functionality/storage tradeoff as the target Pareto point.

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
