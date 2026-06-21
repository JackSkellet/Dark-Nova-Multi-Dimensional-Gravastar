# Idea Foundry Cycle 2

Date: 2026-06-21

Status: design candidates only. No mechanism in this document is promoted until its intended mechanism is observed in an experiment. Access date for cited web sources: 2026-06-21.

## Constraints Covered

| Requirement | Covered by |
| --- | --- |
| At least eight mechanisms | C2-1 through C2-8 |
| At least two without adapters | C2-1, C2-2, C2-3, C2-4, C2-5, C2-6, C2-7 |
| At least two without MoE/topic routing | C2-1, C2-2, C2-3, C2-4, C2-5, C2-6, C2-7 |
| Continual learning | C2-1, C2-7, C2-8 |
| Weight representation | C2-4, C2-5 |
| Code structure/repository graphs | C2-2 |
| Fast or temporary learning | C2-1, C2-7 |
| New synthesis candidate | C2-7 |

## C2-1 Chronological Fast-Weight Journal

- Definition: maintain temporary matrices `F_t = decay * F_{t-1} + eta_t * k_t v_t^T`, where `k_t` is a normalized symbol/repository key, `v_t` is a value correction in hidden space, and each update is logged as `(repo, commit, key_hash, value_hash, eta, decay, scope)`. Inference uses `h'_t = h_t + gate(q_t) * F_scope q_t`.
- Closest primary-source prior art: fast weight programmers and linear-transformer equivalence in Schlag et al., 2021, [arXiv:2102.11174](https://arxiv.org/abs/2102.11174); Titans test-time memory in Behrouz et al., 2024, [arXiv:2501.00663](https://arxiv.org/abs/2501.00663).
- Difference: the update log is explicit, scoped, reversible, and storage-accounted; it is not an unlogged recurrent hidden state.
- Predicted benefit: faster repository adaptation than checkpoint finetuning with cheap rollback and auditability.
- Likely failure mode: fast weights overfit exact API names and leak stale facts into unrelated repositories.
- Smallest falsifying experiment: chronological public-repo API rename fixture comparing updated retrieval, structured memory, fast-weight journal, and fast-weight plus retrieval on paraphrase transfer and prior-task retention.
- ROCm execution strategy: compute `F q` as batched GEMV for small scoped matrices first; later pack active scopes into grouped GEMM.
- Scaling prediction: useful if update count grows sublinearly with repository size and rollback remains O(number of updates in scope).

## C2-2 AST/Dataflow Graph Bias Decoder

- Definition: build typed edges `E = {(i, j, type)}` from AST definitions, imports, calls, tests, docs, and dataflow. Add attention bias `b_ij = w_type` when token spans map to linked nodes, so attention logits become `q_i k_j / sqrt(d) + b_ij`.
- Closest primary-source prior art: GraphCodeBERT data-flow pretraining in Guo et al., 2020, [arXiv:2009.08366](https://arxiv.org/abs/2009.08366); retrieval-augmented generation in Lewis et al., 2020, `research/references.bib`.
- Difference: this is repository-level graph conditioning for generation and file selection, not only pretraining representation learning.
- Predicted benefit: better relevant-file selection, API reuse, and multi-file reasoning than text retrieval alone.
- Likely failure mode: graph extraction remains sparse or noisy, causing bias terms to amplify wrong edges.
- Smallest falsifying experiment: rerun IF1 with AST/package resolution and require materially more non-heuristic edges before any model pilot.
- ROCm execution strategy: store edge biases as sparse block masks and add them before softmax for selected local windows.
- Scaling prediction: graph value should increase with repository size if extraction coverage rises faster than false edge rate.

## C2-3 Executable Trace Contrastive Objective

- Definition: for a code sample `x`, create positive trace `tau+` from passing syntax/unit execution and negative traces `tau-` from failing mutants. Optimize `L = L_lm + lambda * max(0, margin - s(h, tau+) + s(h, tau-))`.
- Closest primary-source prior art: code structure and executable task lines in GraphCodeBERT [arXiv:2009.08366](https://arxiv.org/abs/2009.08366); current local executable JS syntax probe in `research/33_d4_executable_javascript_probe.md`.
- Difference: supervision is execution-outcome contrastive, not static token or graph prediction.
- Predicted benefit: syntax validity and unit-test repair should improve more than plain validation loss predicts.
- Likely failure mode: cheap traces reward superficial syntax patterns instead of functional correctness.
- Smallest falsifying experiment: 200-task JavaScript fixture with syntax, unit-test, and mutant negatives; reject if syntax improves but unit-test pass rate does not.
- ROCm execution strategy: keep model training on ROCm and run Node/test execution on CPU as an offline scoring stage.
- Scaling prediction: trace cost grows with task count, so this should remain a selective auxiliary objective rather than full-corpus supervision.

## C2-4 Low-Rank Plus Groupwise Residual Codebooks

- Definition: represent each weight block as `W ~= U_r V_r + Q_g(R) + S`, where `U_r V_r` is low-rank, `Q_g` is groupwise quantized residual codebooks, and `S` is sparse high-precision corrections selected by validation loss sensitivity.
- Closest primary-source prior art: GPTQ [arXiv:2210.17323](https://arxiv.org/abs/2210.17323), AWQ [arXiv:2306.00978](https://arxiv.org/abs/2306.00978), and QLoRA NF4/double quantization [arXiv:2305.14314](https://arxiv.org/abs/2305.14314).
- Difference: this replaces the failed single block-codebook with layer-specific low-rank bases plus residual codebooks and explicit runtime-buffer accounting.
- Predicted benefit: lower validation-loss damage than IF3 at similar stored bytes.
- Likely failure mode: metadata and reconstruction buffers erase storage wins, as happened in IF3.
- Smallest falsifying experiment: apply to T11c model-only checkpoint and compare BF16, INT8, groupwise INT4, random controls, FP32 reconstruction bytes, and packed-layout bytes.
- ROCm execution strategy: first validate loss with reconstructed tensors, then only continue if an executable packed layout avoids FP32 reconstruction.
- Scaling prediction: should help matrices with low-rank residual structure; embeddings and narrow tensors may remain poor targets.

## C2-5 Ternary Base With Sparse BF16 Escapes

- Definition: train or distill a ternary base `W_t in {-a_l, 0, a_l}` per layer with sparse BF16 escape rows/columns `E`, so deployed weights are `(ternary codes, layer scales, escape index, escape values)`.
- Closest primary-source prior art: BitNet b1.58 [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) and bitnet.cpp inference work [arXiv:2410.16144](https://arxiv.org/abs/2410.16144).
- Difference: this project would test sparse escape channels for coding models rather than assuming a pure ternary model is enough.
- Predicted benefit: directly executable compact weights with fewer catastrophic rows than uniform post-training INT4.
- Likely failure mode: training from scratch at this scale is too small to learn ternary-friendly representations.
- Smallest falsifying experiment: 5-10M-token pilot against dense-528 parameter-matched baseline with full metadata and ROCm throughput accounting.
- ROCm execution strategy: prototype ternary matmul as packed int2/dequant GEMM fallback; do not claim deployment until packed execution is measured.
- Scaling prediction: improves only if trained natively; post-training ternarization is expected to fail.

## C2-6 Syntax-State Selective Mixer

- Definition: combine token embeddings with a parser-state vector `p_t`; update state by selective recurrence `s_t = A(x_t, p_t) s_{t-1} + B(x_t, p_t) x_t`, then mix with local attention every `k` layers.
- Closest primary-source prior art: Mamba selective state spaces in Gu and Dao, 2023, [arXiv:2312.00752](https://arxiv.org/abs/2312.00752).
- Difference: the selective state is conditioned on code syntax state, not only token content.
- Predicted benefit: longer effective context for braces, scopes, and imports without quadratic attention over every byte.
- Likely failure mode: parser-state extraction is brittle across languages and tokenization regimes.
- Smallest falsifying experiment: bracket/scope completion and repository file-local symbol tasks at matched parameters and tokens.
- ROCm execution strategy: start with PyTorch scan kernels; classify latency as implementation-related until a fused scan is tested.
- Scaling prediction: should improve long structured files more than short line completion.

## C2-7 Retrieval-Gated Temporary Delta Cache

- Definition: retrieval returns facts `r_i`; a temporary delta module computes `Delta W_t = sum_i alpha_i(q, r_i) u_i v_i^T` for the current request only, applies it to selected layers, and discards it after the request unless a consolidation gate approves.
- Closest primary-source prior art: RAG [arXiv:2005.11401](https://arxiv.org/abs/2005.11401), LoRA [arXiv:2106.09685](https://arxiv.org/abs/2106.09685), fast weights [arXiv:2102.11174](https://arxiv.org/abs/2102.11174), and Titans [arXiv:2501.00663](https://arxiv.org/abs/2501.00663).
- Difference: this is a new synthesis for this project: retrieval facts instantiate request-local low-rank weights, but the weights are temporary, auditable, and not a persistent adapter.
- Predicted benefit: combines retrieval freshness with parameter-space composition for API reuse and paraphrase transfer.
- Likely failure mode: deltas become an expensive, unstable alternative to simply adding retrieved text to context.
- Smallest falsifying experiment: same chronological repository fixture as C2-1, with leakage, rollback, update cost, and paraphrase transfer measured.
- ROCm execution strategy: batch low-rank deltas as rank-k outer products and fuse only if mechanism wins quality first.
- Scaling prediction: should scale with number of retrieved facts, not repository size, if gating is sparse.

## C2-8 Test-Aware Consolidation Lattice

- Definition: maintain small specialist deltas per repository cluster. Consolidate a delta into base weights only if replay tests, security checks, and stale-doc checks pass. State is `(base, specialists, replay index, rejected update log)`.
- Closest primary-source prior art: LoRA [arXiv:2106.09685](https://arxiv.org/abs/2106.09685), QLoRA [arXiv:2305.14314](https://arxiv.org/abs/2305.14314), and the project update-controller/security experiments in `research/STATUS.md`.
- Difference: consolidation is test/security-gated and lineage-preserving, not ordinary adapter accumulation.
- Predicted benefit: continual learning with lower forgetting and explicit rejection of poisoned or regressive updates.
- Likely failure mode: specialists grow without bound and retrieval remains cheaper.
- Smallest falsifying experiment: chronological public-repo sequence with update acceptance/rejection, replay retention, storage growth, and rollback.
- ROCm execution strategy: train deltas on ROCm only after CPU synthetic gates pass; use small rank and sparse activation to avoid dispatch overhead.
- Scaling prediction: viable only if consolidation reduces active specialist count over time.

## Initial Priority

1. C2-1 and C2-7 share the fast repository adaptation benchmark and should be tested first against retrieval and structured memory.
2. C2-2 must wait for better AST/package graph extraction coverage before any training.
3. C2-4 is the successor compression path because it directly addresses the IF3 block-codebook failure.
4. C2-3 should become the first stronger executable benchmark because syntax validity alone is not functional correctness.
