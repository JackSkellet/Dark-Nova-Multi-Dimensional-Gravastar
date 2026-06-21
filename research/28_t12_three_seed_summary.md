# T12 Three-Seed Dense-528 Versus Adapter-528 Summary

Date: 2026-06-21

Canonical configs:

- `configs/d4_adamw_fp32_50m_comparison.json` for seed 123 T11b/T11c.
- `configs/d4_adamw_fp32_50m_second_seed_pair.json` for seed 456 T12a/T12b.
- `configs/d4_adamw_fp32_50m_third_seed_pair.json` for seed 789 T12c/T12d.

T12 compares dense-528 against residual-adapter-528 under matched AdamW FP32 D4 training. Each run uses train-only D4 rows, byte tokenizer, learning rate 0.0001, sequence length 128, batch size 2, 195,313 steps, 50,000,128 training tokens, `block_impl=explicit_causal`, and `attention_mask_mode=finite_causal`. Evaluation uses fixed D4 validation/test rows, seeds 424242/424243, and 512 batches / 132,096 sampled tokens per split.

Test loss is reported only after validation-defined selection. It is not used to choose checkpoints or architectures.

## Per-Seed Results

| Seed | Architecture | Final validation | Best validation | Final test | Best test | Tokens/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 123 | dense 528 | 1.0335341689933557 | 1.0354822403751314 | 1.0488805254572071 | 1.0345332396682352 | 22762.295803679943 |
| 123 | residual-adapter 528 | 1.0387521978118457 | 1.049464229727164 | 1.0450384634314105 | 1.052616473985836 | 20532.41263900038 |
| 456 | dense 528 | 1.0451208076847252 | 1.0458653752575628 | 1.0386690690065734 | 1.0312774028861895 | 22815.43771183141 |
| 456 | residual-adapter 528 | 1.0420977746834978 | 1.0345490074832924 | 1.0358545312774368 | 1.0275088991038501 | 20517.047508779448 |
| 789 | dense 528 | 1.039739442319842 | 1.0302509073517285 | 1.035331061517354 | 1.027727821667213 | 22420.32176303708 |
| 789 | residual-adapter 528 | 1.0431298993935343 | 1.0480714888544753 | 1.0339286656817421 | 1.0372778683668002 | 20373.82373806082 |

## Mean And Spread

| Metric | Dense mean | Dense spread | Adapter mean | Adapter spread | Winner by mean | Selection metric |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Final validation loss | 1.039464806332641 | 0.011586638691369444 | 1.0413266239629593 | 0.004377701581688598 | dense | yes |
| Best validation loss | 1.0371995076614742 | 0.015614467905834317 | 1.0440282420216438 | 0.01491522224387154 | dense | yes |
| Final test loss | 1.0409602186603781 | 0.013549463939853013 | 1.038273886796863 | 0.01110979774966836 | adapter | no |
| Best test loss | 1.0311794880738792 | 0.00680541800102219 | 1.0391344138188288 | 0.025107574881985784 | dense | no |
| Tokens/s | 22666.01842618281 | 395.11594879433324 | 20474.427961946883 | 158.58890093956143 | dense | no |
| Model-only bytes | 41605125 | 0 | 42442331 | 0 | dense | no |
| Peak VRAM bytes | 463684096 | 0 | 469542400 | 0 | dense | no |
| Max recorded gradient norm | 6.539592107137044 | 4.589032173156738 | 19.23438326517741 | 1.3020763397216797 | dense | no |

## Interpretation

Dense-528 is the three-seed validation winner. It has lower mean final validation loss, lower mean best validation loss, smaller model-only checkpoint bytes, lower peak VRAM, lower recorded max gradient norm, and higher throughput. The third seed resolves the earlier two-seed uncertainty in favor of dense-528 for validation-based architecture selection.

Residual-adapter-528 keeps a reported-only final-test-loss edge: it wins final test on all three seeds and has mean final test loss 1.038273886796863 versus dense 1.0409602186603781. This is useful diagnostic evidence, but it is not a selection criterion under the fixed protocol. Best-test mean favors dense-528.

The current decision is to treat dense-528 as the D4 AdamW FP32 validation-selected architecture for the next matched baseline. The residual-adapter branch remains preserved as a competitive but slower architecture candidate, not a parameter-efficient adapter result and not the selected architecture by validation mean.

## Source Artifacts

- `results/T11b_adapter528_adamw_fp32_50m.json`
- `results/T11c_dense528_adamw_fp32_50m.json`
- `results/T12a_dense528_seed456_adamw_fp32_50m.json`
- `results/T12b_adapter528_seed456_adamw_fp32_50m.json`
- `results/T12c_dense528_seed789_adamw_fp32_50m.json`
- `results/T12d_adapter528_seed789_adamw_fp32_50m.json`
- `results/T12c_dense528_seed789_adamw_fp32_50m_final_validation_eval.json`
- `results/T12c_dense528_seed789_adamw_fp32_50m_final_test_eval.json`
- `results/T12c_dense528_seed789_adamw_fp32_50m_best_validation_eval.json`
- `results/T12c_dense528_seed789_adamw_fp32_50m_best_test_eval.json`
- `results/T12d_adapter528_seed789_adamw_fp32_50m_final_validation_eval.json`
- `results/T12d_adapter528_seed789_adamw_fp32_50m_final_test_eval.json`
- `results/T12d_adapter528_seed789_adamw_fp32_50m_best_validation_eval.json`
- `results/T12d_adapter528_seed789_adamw_fp32_50m_best_test_eval.json`
- `results/T12_three_seed_summary.json`
