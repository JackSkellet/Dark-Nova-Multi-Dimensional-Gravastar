# T12 Second-Seed Dense-528 Versus Adapter-528 Summary

Date: 2026-06-21

Canonical config: `configs/d4_adamw_fp32_50m_second_seed_pair.json`

T12 keeps the T11 protocol fixed and changes only the training seed from 123 to 456 for the strongest pair: dense-528 and residual-adapter-528. Both T12 runs use D4 train-only rows, byte tokenizer, AdamW FP32, learning rate 0.0001, sequence length 128, batch size 2, 195,313 steps, 50,000,128 training tokens, `block_impl=explicit_causal`, and `attention_mask_mode=finite_causal`. Evaluation uses fixed D4 validation/test rows, seeds 424242/424243, and 512 batches / 132,096 sampled tokens per split.

Test loss is reported only after validation-defined selection. It is not used to select checkpoints or decide whether a third seed is needed.

## Per-Seed Results

| Seed | Run | Architecture | Final validation | Best validation | Final test | Best test | Tokens/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 123 | T11c | dense 528 | 1.0335341689933557 | 1.0354822403751314 | 1.0488805254572071 | 1.0345332396682352 | 22762.295803679943 |
| 123 | T11b | residual-adapter 528 | 1.0387521978118457 | 1.049464229727164 | 1.0450384634314105 | 1.052616473985836 | 20532.41263900038 |
| 456 | T12a | dense 528 | 1.0451208076847252 | 1.0458653752575628 | 1.0386690690065734 | 1.0312774028861895 | 22815.43771183141 |
| 456 | T12b | residual-adapter 528 | 1.0420977746834978 | 1.0345490074832924 | 1.0358545312774368 | 1.0275088991038501 | 20517.047508779448 |

## Mean And Spread

| Metric | Dense mean | Dense spread | Adapter mean | Adapter spread | Winner by mean | Selection metric |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Final validation loss | 1.0393274883390404 | 0.011586638691369444 | 1.0404249862476718 | 0.003345576871652156 | dense | yes |
| Best validation loss | 1.040673807816347 | 0.010383134882431477 | 1.0420066186052281 | 0.01491522224387154 | dense | yes |
| Final test loss | 1.0437747972318903 | 0.010211456450633705 | 1.0404464973544236 | 0.009183932153973728 | adapter | no |
| Best test loss | 1.0329053212772044 | 0.003255836782045748 | 1.040062686544843 | 0.02510757488198594 | dense | no |
| Tokens/s | 22788.866757755677 | 53.14190815146867 | 20524.730073889914 | 15.36513022093277 | dense | no |
| Model-only bytes | 41605125 | 0 | 42442331 | 0 | dense | no |
| Peak VRAM bytes | 463684096 | 0 | 469542400 | 0 | dense | no |

## Interpretation

Dense-528 remains the mean validation winner and the resource/throughput winner across two seeds. Residual-adapter-528 wins final validation on seed 456 and keeps the mean final-test edge, but test loss is not a selection criterion. The final-validation mean margin is only 0.0010974979086313397, and validation winners change by seed, so T12 is not decisive.

The next matched action is a third seed for dense-528 versus residual-adapter-528 under the same protocol before promoting either architecture further on D4.

## Source Artifacts

- `results/T11b_adapter528_adamw_fp32_50m.json`
- `results/T11c_dense528_adamw_fp32_50m.json`
- `results/T12a_dense528_seed456_adamw_fp32_50m.json`
- `results/T12b_adapter528_seed456_adamw_fp32_50m.json`
- `results/T12b_adapter528_seed456_adamw_fp32_50m_final_validation_eval.json`
- `results/T12b_adapter528_seed456_adamw_fp32_50m_final_test_eval.json`
- `results/T12b_adapter528_seed456_adamw_fp32_50m_best_validation_eval.json`
- `results/T12b_adapter528_seed456_adamw_fp32_50m_best_test_eval.json`
- `results/T12_second_seed_summary.json`
