# ROCm RDNA4 vLLM External Notes

Status: external context for local ROCm training failures.

Date checked: 2026-06-20.

Sources:

- Reddit: <https://www.reddit.com/r/ROCm/comments/1uaedpw/2_radeon_ai_pro_r9700_rdna4gfx1201_on_vllm_0221/>
- vLLM PR 46190: <https://github.com/vllm-project/vllm/pull/46190>
- vLLM PR 46192: <https://github.com/vllm-project/vllm/pull/46192>

## What The External Report Says

The Reddit report describes 2x Radeon AI PRO R9700 RDNA4/gfx1201 serving on vLLM 0.22.1 with ROCm 7.2.x and PyTorch reporting HIP 7.2.53211. It is serving-focused, not training-focused.

The useful signal is that RDNA4/gfx1201 has active vLLM ROCm path issues around long-context attention and Gated-DeltaNet tensor-parallel startup:

- Long-context decode on the default ROCm attention path reportedly collapses badly as context grows.
- AITER unified attention is reported to remove much of that long-context serving cliff when explicitly enabled on gfx1201.
- FP8 KV cache is reported as unattractive on this path because the faster AITER unified-attention route requires bf16/fp16 KV.
- Tensor parallel serving for Gated-DeltaNet or hybrid models reportedly hangs on gfx1201 because of a Triton operand-layout change that is CUDA/Hopper-specific but mis-compiles on ROCm/RDNA.

The linked upstream PRs are still open when checked. PR 46190 is specifically about fixing a RDNA4/gfx1201 TP>=2 hang for GDN KKT kernels. PR 46192 is about enabling AITER unified attention on RDNA4/gfx1201 and reports improved long-context decode throughput.

## Relevance To This Repository

This is not direct evidence for our PyTorch dense decoder training failures. The local failures are:

- `T1_dense_decoder_training_smoke`: ROCm FP32 completed process execution but hit nonfinite loss at step 2.
- `T1a_rocm_dense_decoder_bf16_training_failure`: ROCm BF16 run exited with a GPU hang before metrics were written.
- `T1b_cpu_dense_decoder_training_smoke`: the same dense training pipeline completed on CPU.

The external evidence does make one interpretation more likely: the current ROCm failure should be treated as a runtime/kernel-path stability problem to isolate, not as evidence that the dense architecture or corpus pipeline is fundamentally invalid.

## Implications

Before scaling dense training on this RDNA4/ROCm stack, add a small ROCm transformer stability reproducer that separates:

1. Embedding-only forward/backward.
2. MLP-only forward/backward.
3. Attention with and without causal masks.
4. Transformer encoder layer with bool masks versus additive masks.
5. FP32 versus BF16 autocast.
6. Optimizer step and gradient clipping effects.
7. Sequence length, batch size, head count, and hidden size thresholds.

Only after the unstable component is isolated should the dense baseline be scaled. If the unstable path is attention-specific, try alternate attention implementations or mask representations before changing the research architecture.

## Current Conclusion

The local ROCm environment is real and usable for small runtime probes, but RDNA4/gfx1201 has enough active upstream serving-stack issues that training stability needs its own microbench gate. The next ROCm result should be an isolation record, not a larger dense baseline attempt.
