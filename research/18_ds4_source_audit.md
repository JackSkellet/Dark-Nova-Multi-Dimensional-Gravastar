# DS4 Source Audit

Status: local primary-source inspection of `data/raw/public/ds4` at commit `80ebbc396aee40eedc1d829222f3362d10fa4c6c`, matching remote `main` on 2026-06-20.

This audit exists because H3 depends on distinguishing activation-aware selective quantization from unsupported claims about semantic carrier pathways.

## Source Boundary

Repository: `https://github.com/antirez/ds4`

Pinned commit: `80ebbc396aee40eedc1d829222f3362d10fa4c6c`

Locally inspected files:

- `README.md`
- `gguf-tools/README.md`
- `gguf-tools/imatrix/README.md`
- `gguf-tools/imatrix/dataset/README.md`
- `gguf-tools/deepseek4-quantize.c`
- `gguf-tools/quants.c`
- `gguf-tools/quants.h`
- `gguf-tools/quality-testing/README.md`
- `metal/moe.metal`
- `rocm/ds4_rocm_router.cuh`
- `ds4_ssd.c`
- `ds4.c`

The clone is kept under ignored `data/raw/public/ds4`; no DS4 code is vendored into `src/weightlab`.

## What DS4 Does

DS4 is a model-specific DeepSeek V4 Flash/PRO engine, not a general GGUF runtime. The README says arbitrary DeepSeek/GGUF files will not have the expected layout, quantization mix, metadata, or optional state (`data/raw/public/ds4/README.md:93-99`). This matters because DS4 evidence should not be generalized to all MoE or GGUF models without a compatibility argument.

The 2-bit DS4 recipes are asymmetric. The routed MoE experts are the heavily quantized majority: gate/up use `IQ2_XXS`, down uses `Q2_K`, while shared experts, projections, routing, and other components are kept higher precision to protect quality (`data/raw/public/ds4/README.md:100-103`). The GGUF tooling exposes the same family controls through `--experts`, `--routed-w1`, `--routed-w2`, `--routed-w3`, `--attention-proj`, `--shared`, `--embedding`, `--output`, and per-prefix `--tensor-type` overrides (`data/raw/public/ds4/gguf-tools/deepseek4-quantize.c:1700-1722`).

The imatrix target is explicitly the routed MoE path. Flash has 43 layers and 256 routed experts per layer; Pro has 61 layers and 384 routed experts per layer. The tracked routed tensors are gate, up, and down expert weights (`data/raw/public/ds4/gguf-tools/imatrix/README.md:7-18`). The collector records squared FFN-normalized input activations for gate/up tensors and squared routed SwiGLU rows after route weighting for down tensors. That is activation usage under the DS4 inference graph, not a semantic label.

The calibration data is generated from DS4-rendered prompts. The tracked dataset covers source review, long-context, agent/tool-call, prose, extraction, translation, and benchmark prompts, with 4682 rendered prompts and a rough 2.91M-token estimate (`data/raw/public/ds4/gguf-tools/imatrix/README.md:35-48`). The collector is Metal-only because it hooks the layer-major Metal prefill graph and accumulates `sum(x[column]^2)` per routed expert without changing inference math (`data/raw/public/ds4/gguf-tools/imatrix/README.md:89-91`).

The `.dat` imatrix format is legacy llama.cpp binary format, but DS4 packs per-expert vectors into one entry per routed expert tensor. Entry length is `n_expert * n_columns`; the quantizer slices the segment for the expert currently being quantized (`data/raw/public/ds4/gguf-tools/imatrix/README.md:93-102`). The C quantizer implements that slicing in `imatrix_find`, including packed entries where `n_values == ncols * n_experts` (`data/raw/public/ds4/gguf-tools/deepseek4-quantize.c:819-855`).

The quantizer is deliberately local and narrow. It handles safetensors loading, FP8 and packed FP4 dequantization, local `Q8_0`, `Q4_K`, `Q2_K`, and `IQ2_XXS` quantization, and GGUF template metadata reuse (`data/raw/public/ds4/gguf-tools/deepseek4-quantize.c:1-20`). It maps routed expert tensor names with `blk.N.ffn_{gate,down,up}_exps.weight` and maps non-routed/shared tensors separately (`data/raw/public/ds4/gguf-tools/deepseek4-quantize.c:873-1008`).

DS4 uses imatrix weights in quantization, not only in documentation. `quants.c` marks `IQ2_XXS` as requiring an imatrix (`data/raw/public/ds4/gguf-tools/quants.c:39-55`), asserts one exists when quantizing IQ2_XXS blocks (`data/raw/public/ds4/gguf-tools/quants.c:997-1015`), and passes optional weights through Q2/Q4 weighted block quantizers (`data/raw/public/ds4/gguf-tools/quants.c:434-525`, `data/raw/public/ds4/gguf-tools/quants.c:587-660`). If no external imatrix is supplied for an imatrix-requiring target, the tooling documents and implements a synthetic fallback based on column weight energy (`data/raw/public/ds4/gguf-tools/README.md:111-123`, `data/raw/public/ds4/gguf-tools/deepseek4-quantize.c:14-20`).

Quality testing is token-likelihood based. DS4's quality tooling collects deterministic official continuations and scores local GGUF variants by target-token negative log likelihood, with comparison fields including average NLL, delta, per-prompt wins, first-token matches, and greedy longest common prefix (`data/raw/public/ds4/gguf-tools/quality-testing/README.md:1-88`). The imatrix README reports a Q4 imatrix result on 100 official Flash continuations: average NLL improved from `0.177357819` to `0.173895148`, a relative `-1.95%` change, with 54/46 case wins (`data/raw/public/ds4/gguf-tools/imatrix/README.md:166-175`).

The runtime contains real routed-expert kernel work. Metal kernels compute routed MoE activation as `silu(clamp(gate)) * clamp(up) * route_weight`, include half-precision mid storage to reduce traffic, dispatch selected expert IDs without CPU-side materialization for decode, and fuse IQ2_XXS gate/up projection with activation in a decode path (`data/raw/public/ds4/metal/moe.metal:131-180`, `data/raw/public/ds4/metal/moe.metal:819-824`, `data/raw/public/ds4/metal/moe.metal:1012-1023`). ROCm router code selects top-k experts from router logits or a hash row, then normalizes selected weights (`data/raw/public/ds4/rocm/ds4_rocm_router.cuh:1-125`).

DS4 also has capacity-oriented expert streaming. SSD streaming keeps non-routed weights resident while routed experts live in an in-memory cache loaded from GGUF on misses (`data/raw/public/ds4/README.md:180-194`). The code converts byte budgets to whole-expert counts (`data/raw/public/ds4/ds4_ssd.c:46-106`) and warns that mixed-precision routed layers can bypass the uniform expert cache slab and hurt hit rate (`data/raw/public/ds4/ds4.c:25638-25763`).

## What DS4 Does Not Prove

DS4 does not prove semantic carrier pathways. The imatrix statistics are activation/route-conditioned column usage in routed expert tensors; the units are tensor columns and routed expert segments, not concepts, repository topics, or graph-theoretic bottlenecks.

DS4 does not show that FP32 carriers are necessary. Its documented shipped recipes emphasize low-bit routed experts plus higher precision for non-routed components such as shared experts, projections, routing, and output-related tensors. H3 still needs separate BF16/FP32/low-bit comparisons for any claim about precision above BF16.

DS4 does not by itself prove a general speed or quality Pareto win for our proposed architecture. It is a specialized engine with specific GGUF templates, Metal/CUDA/ROCm kernels, model-specific tensor names, heavy model-building requirements, and external official-continuation quality data.

DS4 does not provide a continual-learning or security-governed update path. It is relevant to H3 selective precision and H4 routed expert execution/streaming, not to safe autonomous model updates.

## Generalizable Ideas For This Project

- Treat activation-aware quantization as a strong H3 baseline, not as novelty.
- Keep tensor-family policy explicit and measurable: routed expert tensors, shared experts, attention projections, embeddings, and output tensors can require different byte/quality trade-offs.
- Store and report imatrix provenance: dataset path, prompt count, token estimate, chunk count, tensor-name mapping, expert count, and whether strict coverage was required.
- Include packed per-expert metadata in storage accounting. A valid generic imatrix format still needs DS4-specific tensor-name mapping and expert slicing before it can be used correctly.
- Prefer executable quality gates over sampled-output judgment. DS4's target-token NLL comparison is a useful pattern for future small-model H3 work.
- Treat streaming expert caches as part of the architecture, not an implementation detail: whole-expert cache sizing, non-routed resident bytes, KV cache, graph scratch, activations, and cache-hit behavior all affect the Pareto frontier.

## Impact On Current Hypotheses

H3 remains only partially supported in this repository. DS4 is strong external evidence that activation-aware selective quantization can matter for routed MoE experts, but it is not evidence for semantic carrier pathways or precision above BF16.

H4 remains mixed. DS4 has real routed-expert kernels, router selection, and SSD expert streaming, but this repository has not reproduced DS4-scale GPU/ROCm measurements. Our current E4c result requests ROCm through PyTorch HIP and runs on ROCm-backed PyTorch, and E4d records ROCm transfer scaling across small payloads. These are useful local runtime measurements, but they remain small Torch benchmarks rather than DS4-scale kernel evidence.

H5 is unaffected except as a caution: any future update path that changes imatrix data, GGUF tensor policies, or expert streaming hotlists would need the same external approval, provenance, rollback, and security gates as adapter updates.
