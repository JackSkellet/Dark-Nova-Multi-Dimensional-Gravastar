# Matched AdamW FP32 Probe Results

Date: 2026-06-20

## Protocol

Canonical config: `configs/d4_adamw_fp32_matched_comparison.json`

All three probes used D4 train-only rows, byte tokenizer, seed 123, validation seed 424242, test seed 424243, sequence length 128, batch size 2, 10,000 steps, 2,560,000 training tokens, AdamW FP32, learning rate 0.0001, no scheduler, checkpoint validation every 1,000 steps, and final/best validation/test evaluation over 512 batches / 132,096 sampled tokens per split.

The original PyTorch `TransformerEncoderLayer` causal masked path and `-inf` masking were unstable on the train-only split during bounded diagnostics. The matched probes therefore use `block_impl=explicit_causal` and `attention_mask_mode=finite_causal`, an explicit matmul/softmax causal block with a finite large-negative mask. This is not bitwise T8h reproduction; it is the first split-correct stable AdamW FP32 comparison protocol.

## Results

| Run | Architecture | Params | Train loss final | Validation loss | Test loss | Tokens/s | Peak VRAM bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T10a | Dense 544 | 11,025,505 | 1.2963 | 1.7944 | 1.6952 | 19,533 | 481,580,544 |
| T10b | Residual-adapter 528 | 10,604,801 | 1.2462 | 1.6559 | 1.5544 | 18,326 | 469,542,400 |
| T10c | Dense 528 | 10,397,105 | 1.2230 | 1.7175 | 1.6160 | 20,080 | 463,684,096 |

Best checkpoint selection used validation only. For all three probes, the best checkpoint was the final 10,000-step checkpoint.

| Run | Full checkpoint bytes | Model-only bytes | Optimizer/training-state bytes | Max recorded grad norm | Nonfinite grad-norm records |
| --- | ---: | ---: | ---: | ---: | ---: |
| T10a | 132,364,402 | 44,118,725 | 88,245,677 | 405,092.5625 | 0 |
| T10b | 127,337,850 | 42,442,331 | 84,895,519 | 18.8034 | 0 |
| T10c | 124,823,602 | 41,605,125 | 83,218,477 | 4.0932 | 0 |

Adapter contribution at the final T10b checkpoint is finite in all three layers. Adapter update-to-hidden norm ratios were approximately 0.0713, 0.0519, and 0.0783 for layers 0, 1, and 2.

## Interpretation

T10b is the best 2.56M-token probe by both held-out validation and test loss. Against T10a, it uses fewer parameters, less VRAM, and smaller checkpoints, but runs about 6.2 percent slower. Against T10c, it improves validation loss by about 0.0617 and test loss by about 0.0616, which is evidence that the residual adapter path adds value beyond the 528-width backbone alone.

This is not yet a full Pareto result. The comparison uses a revised stable causal block rather than the original T8h PyTorch encoder path, and it has no functional JavaScript completion/infilling evaluation yet. The next step is a 50,000,128-token dense AdamW run and a matched 50,000,128-token T10b-style candidate run under the same split-correct protocol, followed by held-out and functional evaluation.

## Source Artifacts

- `results/T10a_dense544_adamw_fp32_matched_probe.json`
- `results/T10b_adapter528_adamw_fp32_matched_probe.json`
- `results/T10c_dense528_adamw_fp32_width_control_probe.json`
- `results/T10a_dense544_adamw_fp32_matched_probe_final_validation_eval.json`
- `results/T10a_dense544_adamw_fp32_matched_probe_final_test_eval.json`
- `results/T10b_adapter528_adamw_fp32_matched_probe_final_validation_eval.json`
- `results/T10b_adapter528_adamw_fp32_matched_probe_final_test_eval.json`
- `results/T10c_dense528_adamw_fp32_width_control_probe_final_validation_eval.json`
- `results/T10c_dense528_adamw_fp32_width_control_probe_final_test_eval.json`
