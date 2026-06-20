# Real Training Transition Audit

Status: transition audit before useful-scale local training.

Date: 2026-06-20.

## Purpose

The project is moving from reduced falsification experiments into real model training and evidence-based architecture selection. This audit separates what was actually trained from what was simulated, records the current Git and ROCm state, and defines the immediate evidence gaps that must be closed before claiming architecture progress.

## Git State Before Snapshot

- Repository was initialized but had no commits on `master`.
- All project files were untracked before this transition snapshot.
- Existing experiment records therefore report `git_commit: uncommitted`.
- Ignore rules exclude `.venv/`, caches, raw public clones, downloaded checkpoints, processed data, and heavyweight model artifacts.
- Intended snapshot scope is source, tests, configs, research docs, and JSON/CSV experiment results. Raw clones and downloaded model files remain ignored data inputs.

## What Was Truly Trained

The current repository contains only reduced training:

- `E3d_trained_tiny_transformer_precision`: trained a tiny transformer-style output head on a synthetic next-token rule.
- `E3g_trained_internal_layer_precision`: trained a tiny internal MLP output matrix with ridge regression on a toy next-token task.
- `E5c_trainable_adapter_vs_retrieval`: trained a toy low-rank classifier adapter on synthetic API facts.

These are useful reduced controls, but they are not useful-scale language-model training, not repository-level code training, and not evidence that a production architecture has been found.

## What Used Real Model Weights Without Training Them

- `E3h_real_open_model_matrix_precision`: loaded pinned public `sshleifer/tiny-gpt2` weights and measured reconstruction error on one real tensor.
- `E3i_real_open_model_task_precision`: used the same checkpoint for a local forward-pass preservation test on deterministic token IDs.
- `E3j_real_open_model_natural_text_precision`: used pinned GPT-2 BPE files and natural-language repository/code prose for a small task-loss preservation test.
- `E3k_real_open_model_internal_tensor_precision`: moved the tiny-gpt2 natural-text precision probe to an internal tensor.

None of these updated model parameters. They are precision/preservation probes, not adaptation or training runs.

## What Was Simulated Or Reduced

- `E1`, `E1b`: routing simulations over synthetic rows/prompts.
- `E2`: compositional storage on a synthetic matrix.
- `E3`, `E3b`, `E3c`, `E3e`, `E3f`: synthetic or toy selective-precision studies.
- `E4`, `E4b`: lookup/routed-execution simulations.
- `E4c`, `E4d`: real ROCm-backed PyTorch runtime and transfer measurements, but not useful model-layer training benchmarks.
- `E5`, `E5b`, `E5d`, `E5e`: synthetic continual/update/stale-doc/patch controls.
- `E5f`, `E5g`: public historical patch replay with narrow deterministic generated candidates.
- `E6a` through `E6i`: structured-memory and documentation/code proxy tasks, mostly public-source lookup or synthetic positive controls, not trained model memory.
- `S1`, `S2`, `S2b`, `S2c`: synthetic security smoke tests.

## Supported Conclusions So Far

- No current result supports an integrated architecture or Pareto-improvement claim.
- Structured external repository memory is the strongest current alternative family for repository facts, signatures, API coverage, and deterministic scaffolding proxies.
- Retrieval and structured memory remain strong baselines for continual repository knowledge.
- Uniform/groupwise quantization remain strong baselines. Importance-selected protected rows have not shown a real trained-weight storage-quality Pareto win.
- ROCm PyTorch works locally, but current ROCm evidence is runtime/transfer smoke evidence, not stable training throughput evidence.

## Missing Required Evidence

The current repository has not yet completed any of the new real-training requirements:

- No 10-50M parameter dense decoder has been trained from random initialization.
- No matched routed/modular candidate has been trained on the same tokenizer, data, token budget, and comparable compute.
- No open coding model has received LoRA, routed-adapter, or continued-pretraining updates.
- No run has consumed 50M+ real training tokens.
- No prepared licensed corpus with license filtering, secret scanning, generated-file filtering, deduplication, repository-aware splits, token accounting, and source/license/language reporting exists.
- No checkpoint-resume training validation exists.
- No real-training loss curves, generation samples, or training throughput metrics exist.
- No trained-weight quantization comparison has been run with random controls.
- No functional coding or source-grounded documentation evaluation has been run against trained models.

## Immediate Next Evidence Gates

1. Commit and tag the current 44-record reduced-research snapshot.
2. Rerun the core experiment summary after the snapshot so new records no longer report `git_commit: uncommitted`.
3. Validate ROCm training readiness: VRAM, BF16/FP16 support, matmul throughput, optimizer step throughput, stable batch sizes, memory failure behavior, and checkpoint resume.
4. Build the real corpus preparation pipeline with license filtering, secret scanning, generated-file filtering, deduplication, repository-aware splits, and token/source reporting.
5. Build the dense decoder training pipeline with configurable model/data/tokenizer, mixed precision, gradient accumulation, checkpoints, validation, generation samples, token accounting, loss curves, and machine-readable metrics.
6. Run the smallest real dense training smoke only to validate the pipeline, then scale toward the required 50M+ real-token run.

## Current Recommendation

Use structured repository memory plus a stable dense local model as the first real architecture lane. Keep routed/modular adapters as the matched candidate lane once the dense baseline and corpus pipeline are stable. Continue importance-aware mixed precision only on trained model tensors with random controls. Defer indexed lookup and compositional storage until model/data scale makes them necessary.
