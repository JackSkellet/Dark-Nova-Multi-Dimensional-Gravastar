# Prior Art Review

Access date: 2026-06-20.

This is a seed review, not a complete survey. `research/prior_art.csv` is the machine-readable matrix. See `research/18_ds4_source_audit.md` for the dedicated local DS4 source audit.

## Routing And Conditional Computation

Switch Transformer showed sparse top-1 MoE routing can increase parameter count while keeping per-token compute roughly fixed, but it also documents routing instability, load-balancing, and communication concerns. Expert Choice routing addresses load balance by letting experts choose tokens, but still operates inside sparse MoE constraints. Hash routing and BASE Layers show that non-semantic or assignment-based routing can be competitive, which weakens any assumption that a human hierarchy is automatically better.

## Compression And Selective Precision

GPTQ uses approximate second-order information for post-training quantization. AWQ protects salient channels based on activation statistics and avoids hardware-hostile arbitrary mixed precision by using equivalent scaling. SmoothQuant migrates activation difficulty into weights to enable quantized inference. These are strong baselines for H2 and H3.

For the first real open-model matrix smoke test, this repository uses the public Hugging Face `sshleifer/tiny-gpt2` checkpoint pinned to commit `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`. It is only a tiny GPT-2 checkpoint source for a real tensor; it is not evidence of production-scale language-model behavior.

## DS4 Source Inspection

Pinned source: `https://github.com/antirez/ds4` at commit `80ebbc396aee40eedc1d829222f3362d10fa4c6c`.

Remote `main` was checked on 2026-06-20 and still resolved to the pinned commit. A local clone is kept at `data/raw/public/ds4` for reproducible inspection; this directory is ignored and no DS4 code is vendored into `src/weightlab`.

Inspected files include `README.md`, `gguf-tools/README.md`, `gguf-tools/imatrix/README.md`, `gguf-tools/deepseek4-quantize.c`, `gguf-tools/quants.[ch]`, `gguf-tools/quality-testing/README.md`, Metal kernels, and ROCm kernel headers.

Findings:

- DS4 is a narrow DeepSeek V4 Flash/PRO inference engine, not a generic GGUF runner.
- The README states the 2-bit quantizations are asymmetric: routed MoE experts are aggressively quantized while shared experts, projections, routing, and output-related components are kept at higher precision depending on template.
- The imatrix target is the routed MoE path. It records per-routed-expert activation-derived column statistics: squared normalized FFN input for gate/up and squared routed SwiGLU rows after route weighting for down tensors.
- The quantizer can use DS4 imatrix files to weight quantization error per expert/column. If no imatrix exists for `iq2_xxs`, it falls back to a synthetic weight-energy heuristic.
- The imatrix format is compatible with legacy llama.cpp `.dat`, but DS4 packs per-expert vectors into routed tensor entries; generic tools need DS4-specific tensor-name mapping and per-expert slicing.
- Quality testing compares local GGUF variants against official DeepSeek continuations by target-token negative log likelihood; the imatrix README reports a Q4 imatrix result on 100 official Flash continuations with average NLL improving from `0.177357819` to `0.173895148`.
- Runtime code includes routed-expert Metal kernels, ROCm top-k/hash router selection, and SSD expert streaming/cache planning. These support DS4 as an implementation case study for H3/H4, but this repository has not reproduced DS4-scale GPU measurements.
- The evidence supports activation-aware selective quantization, not semantic topic choke-point claims. It does not causally prove that a sparse set of semantic carrier pathways exists.
- License note: DS4 uses MIT license and acknowledges retained/adapted GGML and llama.cpp pieces. Any copied code would require preserving license notices; this repository does not copy DS4 code.

## Continual Learning And Retrieval

RAG provides a strong non-parametric memory baseline for changing knowledge. LoRA provides a low-rank modular parameter-update baseline. ROME and related model-editing work provide targeted-edit baselines but require specificity and regression checks. Model-collapse work warns against unbounded recursive training on model-generated data.

## Security And Governance

Secure local deployment must treat model weights, adapters, retrieval indexes, training buffers, logs, and generated patches as protected artifacts. Supply-chain signing, immutable audit logs, deterministic authorization before retrieval, and rollback are part of the model architecture, not external paperwork.

## Novelty Risk

Most individual mechanisms are established: sparse MoE, activation-aware quantization, low-rank adaptation, retrieval memory, and model editing. Potential novelty, if any, is in a rigorously gated local combination with full storage/latency/security accounting. That combination may still lose to simpler RAG plus adapters.
