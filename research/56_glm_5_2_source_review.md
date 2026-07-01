# GLM-5.2 Source Review

Date: 2026-07-01

User-supplied source: <https://z.ai/blog/glm-5.2>

Expanded track: `research/59_glm_5_2_external_baseline_teacher_track.md`

Primary accessible sources:

- Z.ai / Hugging Face blog mirror, "GLM-5.2: Built for Long-Horizon Tasks", accessed 2026-07-01: <https://huggingface.co/blog/zai-org/glm-52-blog>
- Z.ai developer documentation, accessed 2026-07-01: <https://docs.z.ai/guides/llm/glm-5.2>
- Z.ai GLM-5.2 Hugging Face model card, accessed 2026-07-01: <https://huggingface.co/zai-org/GLM-5.2>
- Z.ai GLM-5 GitHub repository, accessed 2026-07-01: <https://github.com/zai-org/GLM-5>
- GLM-5 technical report, arXiv:2602.15763v2, accessed 2026-07-01: <https://arxiv.org/abs/2602.15763v2>
- MTP rejection-sampling paper cited by the GLM-5.2 blog, arXiv:2606.12370, accessed 2026-07-01: <https://arxiv.org/abs/2606.12370>

## Relevant Claims

The Z.ai page itself is a JavaScript shell in the local fetch path, but the same primary content is exposed through the Hugging Face blog mirror and model card.

GLM-5.2 is presented as a 744B-parameter, 40B-active MoE model with a 1M-token context window, MIT open weights, and deployment support through Transformers, vLLM, SGLang, KTransformers, xLLM, and related inference stacks. Z.ai reports coding benchmark results including SWE-bench Pro 62.1, NL2Repo 48.9, DeepSWE 46.2, ProgramBench 63.7, Terminal-Bench 2.1 81.0 under Terminus-2, FrontierSWE dominance 74.4, PostTrainBench 34.3, and SWE-Marathon 13.0.

The architectural ideas most relevant to Dark Nova are:

- IndexShare for sparse attention: reuse one lightweight indexer across every four sparse attention layers, reducing indexer dot-product and top-k work for long context. Z.ai claims a 2.9x per-token FLOP reduction at 1M context for this path.
- MTP with IndexShare/KVShare plus rejection sampling and end-to-end total-variation loss: improve speculative decoding acceptance length. Z.ai reports a 20 percent acceptance-length gain in coding scenarios.
- Explicit effort control: expose a higher-cost reasoning mode for hard coding tasks.
- Long-horizon benchmark emphasis: measure hour-scale and multi-step engineering work rather than only short completion accuracy.

## Dark Nova Interpretation

GLM-5.2 is not a directly runnable local baseline for the current Dark Nova machine or current first-pass resource constraints. A 744B-A40B model is far outside the current single-workstation training path, and the reported results are external vendor/model-card evidence, not reproduced local measurements.

It is still useful as frontier comparison and idea input:

- The benchmark gap should be framed against Terminal-Bench, SWE-bench Pro, FrontierSWE, PostTrainBench, SWE-Marathon, NL2Repo, and repository-completion tasks, not just validation loss or syntax checks.
- `indexshare_sparse_attention_long_context` should be tracked as a possible long-context architecture idea, but only after local structured memory and repository-completion gates show that context length is the bottleneck.
- `mtp_rejection_speculative_decoding` should be tracked as a runtime/deployment idea for local iterative repair loops, but only after a small local draft-path experiment proves wall-clock speedup without reducing executable pass rate.

## Matrix Updates

Added to `results/combination_matrix.json`:

- `indexshare_sparse_attention_long_context`
- `mtp_rejection_speculative_decoding`

These are active external ideas, not local wins. Their next falsifying tests are tiny local prototypes or surrogates with pinned repository-completion or executable-repair gates.
