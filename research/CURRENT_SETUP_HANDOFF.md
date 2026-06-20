# Current Setup Handoff

Generated from the current workspace state. Paths are repository-relative. Private usernames, hostnames, tokens, credentials, and absolute home paths are intentionally omitted or redacted.

## 1. Repository State

- Branch: `main`
- HEAD: `3a23ce9ba9abfd7f59c0ef4cf8f848a9b8edd4b1`
- Tags at HEAD: `adamw-bf16-dense-probe-2026-06-20`
- Dirty worktree: `True`

Dirty state at capture:

```text
M research/STATUS.md
 M results/manifest.json
 M results/research_assessment.json
 M src/weightlab/dense_training.py
 M src/weightlab/metrics.py
 M tests/test_dense_training.py
?? results/T7b_rocm_adapter_decoder_10m_hf_d4_identity_init_step_debug.json
?? results/T7c_rocm_adapter_decoder_10m_hf_d4_identity_probe.json
?? results/T8h_rocm_dense_decoder_11m_hf_d4_adamw_fp32_probe.json
?? results/T8i_dense_adamw_fp32_vs_bf16_control_comparison.json
?? results/current_setup_precision_identity_diagnostic.json
?? tests/test_metrics.py
```

Relevant recent commits:

```text
3a23ce9 (HEAD -> main, tag: adamw-bf16-dense-probe-2026-06-20) Stabilize AdamW BF16 dense training path
41282c5 Add adapter decoder candidate path
349dcfa (tag: hf-50m-dense-baseline-2026-06-20) Record 50m-token dense HF baseline
ff4e0f8 Link HF corpus records and resume dense checkpoints
9cbad3b (tag: hf-50m-mirror-dense-probe-2026-06-20) Record HF mirror and instrumented dense probe
abbcf36 Flush dense training progress and checkpoints
de86b45 Add exploratory HF corpus materializer
c2fa347 (tag: exploratory-hf-corpus-2026-06-20) Record exploratory HF corpus selection
39e5247 Add exploratory HF corpus manifest builder
866d8ba Expand corpus source handling
a7f61de (tag: rocm-11m-dense-foothold-2026-06-20) Record stable 11m ROCm dense probe
7378ad6 Add dense step stability debug probe
```

Latest verification after handoff generation:

- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .`: passed, `All checks passed!`
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q`: passed to 100%; `uv run pytest --collect-only` reports 104 tests.

Important source/config/test hashes:

| Path | SHA256 |
| --- | --- |
| src/weightlab/dense_training.py | 7cfbd0cc54602364df94c68c5e53014325df26e81d8b0ac001c206b23f64b792 |
| src/weightlab/metrics.py | 1568f6d2082c682b5126d08bcef1e924f24888c25d04f92b641f704bf9a7831e |
| src/weightlab/hf_materializer.py | b409d34c049dc54738d5f5fb5afab99a56f96e359bbcea9e059d6aa51f1e8c63 |
| src/weightlab/hf_corpus.py | ac88d0f43eea9af7b074504eeac1e571fe2849be64201b8c68650cb569327838 |
| src/weightlab/assessment.py | 276739c676c24cd1c727755efca861f7fc02aea475f1fbc4061a68689ea59024 |
| scripts/train_dense_decoder.py | c09f72eda8da8a7bfea0489c7849ca7db9e424cabbfb2111b70918b069f148a5 |
| scripts/materialize_hf_corpus.py | c271c67ed0e236997a78e3844171986045f7efa8e2af52b0a3bb00a6add25c76 |
| scripts/prepare_hf_corpus_manifest.py | d3b20ae65afb1c47962f6d35abcff688d7b6bff4e59170c1af2823942eb13aec |
| scripts/debug_dense_step_stability.py | 3c25e6b6ce770dcf7abff4aaee8abb1231a2c07d929f5339250f05cceb62e19f |
| scripts/summarize_research_assessment.py | 0aede5854e402e033352a66c8ea1e05c2a96b7a80d12d06450ff84b7e87d5787 |
| configs/smoke.yaml | 4e4c5ca98e3e88cf0e27c6ce7883e204449e964b365beee7a02b9c5a7c33f096 |
| pyproject.toml | 5ecefb8d8d133280bef572a392d903eea86dcb0391428f89185009a76590d977 |
| uv.lock | 7ab8cf2a69f28afec9948451c37f2865e959ee27a59f8eb62f43ba4945d5a522 |
| tests/test_dense_training.py | f7634e2fc0162d7539c72776978e5db0132a71ed606abe65812b10de97b20abf |
| tests/test_train_dense_decoder_cli.py | 6ec45eaf013b9cf19fb3c9a6154576683a403378158ff9b8b608ff358d5d4579 |
| tests/test_metrics.py | 395258ab28fe50fc6a1a689663e2a8f8df60716f415168ca300cbb10ed834e51 |

Key code paths:

- Model/trainer/tokenizer/checkpoint/precision control: `src/weightlab/dense_training.py`
- Training CLI: `scripts/train_dense_decoder.py`
- HF manifest builder: `scripts/prepare_hf_corpus_manifest.py`
- HF materializer: `src/weightlab/hf_materializer.py`, `scripts/materialize_hf_corpus.py`
- Assessment: `src/weightlab/assessment.py`, `scripts/summarize_research_assessment.py`
- Precision identity diagnostic result: `results/current_setup_precision_identity_diagnostic.json`

## 2. Hardware And Software

- GPU: ``
- Device properties: ``
- PyTorch device string: `cpu`
- Accelerator backend: `cpu`
- ROCm runtime version reported by PyTorch: `7.2.53211`
- Torch: `2.12.1+rocm7.2`
- Python: `3.12.13`

Relevant environment variables are in `results/current_setup_snapshot.json` under `hardware_software.relevant_environment_variables` with sensitive values redacted.

Supported behavior observed:

- FP32 dense SGD and AdamW run on ROCm.
- Raw BF16 through masked transformer blocks produced nonfinite gradients in T8b/T8c.
- Bool causal masking did not fix BF16 and produced T8d failure plus GPU-hang risk.
- Current BF16-requested safe path disables autocast around masked transformer blocks; fixed-batch diagnostic proves it is numerically identical to FP32 for the dense AdamW setup.
- PyTorch exposes the ROCm device as `cuda` device type.

Unsupported/disabled kernels and fallbacks:

- The current code avoids BF16 autocast inside masked `TransformerEncoderLayer` blocks by calling `.float()` under `torch.autocast(..., enabled=False)` in `_run_transformer_block`.
- Attention score and softmax dtypes are not exposed by PyTorch public hooks in the diagnostic; the handoff records module-level dtypes and explicitly marks this limitation.

## 3. Data And Tokenizer

D4 materialization:

- Corpus path: `data/hf_mirror/exploratory_d3/corpus.jsonl` (excluded from ZIP)
- Corpus SHA256: `547b2ff1d9eedd61f4791d6cffb256ec844e8312513ec529b953952511f3a2bc`
- Corpus use: `exploratory-research-only`
- Production/redistribution approval: false / false
- Train tokens: `50102528`
- Validation tokens: `1455924`
- Test tokens: `735566`
- Accepted rows: `3415`
- Dataset config counts: `{"CodedotAI/code_clippy_github::JavaScript-all": 3415}`
- Source manifest SHA256: `3958422a783d43a2e5debd9addd73f6eb179ff3951dd56d99b0d290a17a46fc3`

Tokenizer:

- Type: byte-level UTF-8 tokenizer.
- Vocabulary: byte IDs 0-255 plus EOS 256; `vocab_size=257`.
- Special token: EOS ID 256.
- No external tokenizer file exists; checksum is the SHA256 of `src/weightlab/dense_training.py` above.

Packing and sampling:

- `load_jsonl_texts` loads every row text into memory.
- `_tokens_from_texts` concatenates all encoded documents with EOS after each document.
- `_sample_batch` samples random contiguous windows of `seq_len + 1` from the concatenated token tensor using a CPU `torch.Generator` seeded by `--seed`.
- Sampling is with replacement. There is no epoch, no dataloader worker, and no wraparound logic.
- D4 rows have split metadata, but the current trainer does not enforce split-aware train/validation separation. Training examples can enter validation because validation samples from the same loaded token tensor.
- Validation batches are sampled after training using the same CPU generator state. New records include `validation.sample_order_sha256`; older T8g does not.
- T8g and T8h have identical training loss curves and validation loss. T8h records validation `sample_order_sha256=0ff92096e61783cf57fcc5feff7e4576d7e7e980331b699b25c2bbc68b35ee0e`.

## 4. Model Architecture

Dense 11M model:

- Config: `{"adapter_dim": 0, "architecture_variant": "dense", "attention_mask_mode": "additive_causal", "batch_size": 2, "checkpoint_interval": 1000, "device": "rocm", "gradient_accumulation_steps": 1, "heads": 8, "hidden_dim": 544, "layers": 3, "learning_rate": 0.0001, "mixed_precision": "fp32", "optimizer_name": "adamw", "progress_interval": 500, "seq_len": 128, "steps": 10000, "validation_batches": 8}`
- Parameters: `11025505` total/trainable/active.
- Token embedding plus learned position embedding.
- 3 `torch.nn.TransformerEncoderLayer` blocks, batch-first, GELU, dropout 0.0, feedforward width `hidden_dim * 4`.
- Final LayerNorm then linear output head.
- No weight tying between token embedding and output head.
- Causal mask: additive float32 upper-triangular `-inf` mask by default.

Adapter candidate:

- Config: `{"adapter_dim": 64, "architecture_variant": "adapter", "attention_mask_mode": "additive_causal", "batch_size": 2, "checkpoint_interval": 1000, "device": "rocm", "gradient_accumulation_steps": 1, "heads": 8, "hidden_dim": 528, "layers": 3, "learning_rate": 0.0001, "mixed_precision": "fp32", "optimizer_name": "sgd", "progress_interval": 500, "seq_len": 128, "steps": 10000, "validation_batches": 8}`
- Parameters: `10604801` total/trainable/active.
- Same transformer backbone shape as configured, plus one residual adapter after each transformer block.
- Adapter module: LayerNorm -> down projection -> GELU -> up projection -> residual add.
- Adapter rank/dim: `64`.
- Adapter normalization: inside adapter before down projection.
- Adapter gates: no explicit learned scalar gate exists.
- Adapter initialization: down weight normal(0, 0.02), down bias zero, up weight zero, up bias zero. This makes the adapter start as exact identity.

No trained routing/residual-expert decoder candidate exists yet in this training path. Earlier routing records are proxy experiments, not trained decoder models.

## 5. Training Configuration

| Run | Status | Arch | Precision Label | Optimizer | Train Tokens | Validation Loss | Tok/s | Resume OK |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T6 | completed | None | fp32 | sgd | 50000128 | 4.0454936027526855 | 27004.015355057265 | True |
| T7a | failed | adapter | fp32 | sgd | None | None | None | None |
| T7c | completed | adapter | fp32 | sgd | 2560000 | 4.637662619352341 | 7986.957957552763 | True |
| T8g | completed | dense | bf16 | adamw | 2560000 | 1.664748802781105 | 6808.34307170223 | True |
| T8h | completed | dense | fp32 | adamw | 2560000 | 1.664748802781105 | 6642.496627869687 | True |

Exact commands are stored in each result JSON under `command` and summarized in `results/current_setup_snapshot.json` under `training_runs`.

Common resolved training behavior:

- Seed: 123 unless otherwise specified.
- RNG: `torch.manual_seed(seed)` for model init and CPU `torch.Generator(device="cpu").manual_seed(seed)` for sampling.
- Batch size: 2 for T6/T7/T8 HF runs.
- Sequence length: 128; each sampled batch has 129 IDs and predicts next-token targets.
- Gradient accumulation: 1.
- Tokens per optimizer step: `batch_size * seq_len = 256`.
- Loss: `torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))`.
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before optimizer step.
- Scheduler: none.
- Validation: only after completed training run; no periodic validation during training.
- Evaluation mode: `_validation_metrics` calls `model.eval()` with `torch.no_grad()`, then restores `model.train()`.
- Resume: checkpoint stores model, optimizer, config, step, and CPU generator state; resume restores all of these.

Optimizer details:

- SGD records use `torch.optim.SGD(model.parameters(), lr=learning_rate)` with default momentum 0 and weight_decay 0.
- AdamW records use `torch.optim.AdamW(model.parameters(), lr=learning_rate, foreach=False)`. Betas, epsilon, weight decay, and amsgrad are PyTorch defaults: betas `(0.9, 0.999)`, eps `1e-8`, weight_decay `0.01`, amsgrad false.
- No custom parameter groups are used.

Differences:

- T6: dense, 544 hidden, 3 layers, SGD FP32, 50,000,128 train tokens.
- T7a: adapter, 528 hidden, adapter dim 64, SGD FP32, two-step debug failed at step 2 gradients before identity init.
- T7c: adapter, 528 hidden, adapter dim 64, SGD FP32, identity-initialized adapter, 2.56M train tokens, stable but low quality.
- T8g: dense, AdamW, `mixed_precision=bf16`, but masked blocks forced to FP32; 2.56M train tokens.
- T8h: dense, AdamW, FP32; exact numerical control for T8g.

## 6. Actual Precision Path

Current autocast logic:

```python
with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(...)
```

Inside `DenseDecoder.forward`, masked transformer blocks call `_run_transformer_block`. If a mask is present and autocast is enabled, the code runs:

```python
with torch.autocast(device_type=device_type, enabled=False):
    return block(hidden.float(), src_mask=mask, is_causal=True)
```

Observed three-step diagnostic result:

- FP32 and BF16-requested input IDs: identical hashes.
- Logits: identical hashes for all three steps.
- Loss: exactly equal for all three steps.
- Every gradient: identical hash for all three steps.
- Parameters after each step: identical hash for all three steps.
- AdamW optimizer state after each step: identical hash for all three steps.
- Master parameters: torch.float32.
- AdamW moment buffers: torch.float32 tensors.
- Causal mask: torch.float32 additive mask.

Detailed module dtype traces are in `results/current_setup_precision_identity_diagnostic.json`. Attention score and softmax dtypes are not directly observable through public hooks in `torch.nn.MultiheadAttention`; the diagnostic records this limitation explicitly.

## 7. BF16 Versus FP32 Identity Diagnostic

Diagnostic artifact: `results/current_setup_precision_identity_diagnostic.json`.

Conclusion from collected evidence, not inference: the current BF16-requested dense AdamW path is effectively FP32 for all meaningful computation measured in this model. Exact equality is caused by the explicit fallback that disables autocast around masked transformer blocks and casts hidden states to float. Because the architecture is dominated by those blocks and parameters remain FP32, the BF16-requested and FP32 paths produce identical logits, losses, gradients, parameters, and optimizer states for three fixed steps.

The dtype option is not ignored: it enables autocast at the outer training scope. But the current implementation bypasses autocast in the masked transformer block, which removes the BF16 path responsible for prior ROCm failures. There is no evidence that logs or artifacts are accidentally shared: the diagnostic uses cloned model state, fixed in-memory batches, separate optimizers, and independent state hashes.

## 8. Results And Limitations

Scientifically comparable now:

- T8g vs T8h: comparable and exactly equal; T8h proves T8g is not an independent BF16 quality result.
- T7c vs T8h: same 2.56M token budget and D4 corpus, but optimizer differs (`SGD` vs `AdamW`) and hidden dimensions differ. It is a stability probe, not a fair final candidate comparison.
- T6 vs T8h: both dense D4 runs, but optimizer and token budget differ. T6 is the required 50M-token dense baseline; T8h is the current AdamW quality control at 2.56M tokens.

Ambiguous or inaccurate labels:

- `T8g` should be labelled `mixed_precision_requested_bf16_with_fp32_masked_transformer_blocks`, not a pure BF16 run.
- Older records lack `git_dirty`; this has been fixed for future records in `src/weightlab/metrics.py`.
- T7b/T7c/T8h/T8i were run from a dirty worktree; treat them as useful diagnostics but rerun from a clean commit before any long comparison.

Checkpoint accounting:

- T8g/T8h AdamW full checkpoint: 132,362,866 bytes.
- T8g/T8h model-only checkpoint: 44,118,165 bytes.
- AdamW optimizer/training-state overhead: 88,244,701 bytes.
- T7c adapter SGD full checkpoint: 42,446,733 bytes.
- T7c adapter model-only checkpoint: 42,441,425 bytes.

Known ROCm failures:

- Raw BF16 additive-mask gradients fail at step 1 in T8b/T8c.
- BF16 bool causal masking is not sufficient and failed in T8d, with GPU-hang risk observed during probing.
- Pre-identity adapter candidate failed at step 2 gradients in T7a.
- 512x4 and 12.9M dense paths have preserved nonfinite-gradient/loss failures in T2/T3 records.

Next smallest diagnostic:

1. Commit the current handoff and code metadata changes.
2. Rerun T7b/T7c/T8h from the clean commit so records have exact git metadata.
3. Run a fair 2.56M-token adapter-vs-dense comparison under identical AdamW FP32 settings before any 50M-token candidate.

## Claim Map

| Claim | Source Artifact |
| --- | --- |
| D4 is exploratory-research-only and exceeds 50M train tokens. | results/D4_hf_corpus_materialization.json |
| T6 is the completed 11M dense 50M-token SGD/FP32 baseline. | results/T6_rocm_dense_decoder_11m_hf_d4_50m_tokens.json |
| T7a adapter candidate failed at step_2_gradients. | results/T7a_rocm_adapter_decoder_10m_hf_d4_step_debug.json |
| T7c identity-initialized adapter trains 2.56M tokens but underperforms dense AdamW. | results/T7c_rocm_adapter_decoder_10m_hf_d4_identity_probe.json |
| T8g and T8h are exactly identical in loss, validation, sample, and checkpoint size. | results/T8i_dense_adamw_fp32_vs_bf16_control_comparison.json |
| Three-step fixed-batch diagnostic proves BF16-requested and FP32 paths are identical under current masked-block fallback. | results/current_setup_precision_identity_diagnostic.json |
| Experiment records now include dirty-worktree metadata for future runs. | src/weightlab/metrics.py |
