# GLM-5.2 External Baseline And Teacher Track

Date: 2026-07-01

Primary sources:

- Z.ai blog, accessed 2026-07-01: <https://z.ai/blog/glm-5.2>
- Z.ai developer documentation, accessed 2026-07-01: <https://docs.z.ai/guides/llm/glm-5.2>
- Z.ai GLM-5.2 Hugging Face model card, accessed 2026-07-01: <https://huggingface.co/zai-org/GLM-5.2>
- Hugging Face blog mirror for the Z.ai release article, accessed 2026-07-01: <https://huggingface.co/blog/zai-org/glm-52-blog>

## Goal Addition

GLM-5.2 is now a serious external reference baseline for coding, long-context project understanding, and agentic engineering. It must not be treated as a same-budget local baseline unless it is actually run on the same local hardware under the same resource limits.

The project keeps four separate GLM-5.2 uses:

1. Published benchmark reference.
2. Black-box evaluation baseline when API or local serving is available.
3. Teacher and distillation baseline.
4. Architecture inspiration track.

## Baseline Categories

Maintain separate categories in all summaries and final reports:

- Local same-budget baseline: dense-528 and successors on the same hardware, token budget, dataset, and evaluation.
- Local practical baseline: best model that can run comfortably on one workstation, including quantized variants, with tokens/sec, VRAM, RAM, size, and quality measured.
- External GLM-5.2 baseline: API or separately served model, much larger compute budget, used as target, oracle, teacher, and benchmark reference.
- Published GLM-5.2 benchmark reference: official or third-party reported numbers, clearly marked as not reproduced unless this repository runs the same harness.

The final report must show the gap between the local model and GLM-5.2 on every benchmark that can be run, and must separate likely contributors: model size, dataset quality, tokenizer, long-context capability, reasoning budget, tool use, repository memory, objective, runtime limits, and architecture.

## Published Benchmark Reference

Record GLM-5.2 numbers from primary sources for coding, long-horizon, agentic, and repository-level benchmarks. The active source review already records Terminal-Bench 2.1, SWE-bench Pro, FrontierSWE, PostTrainBench, SWE-Marathon, NL2Repo, DeepSWE, and ProgramBench from Z.ai/Hugging Face sources.

These are external published numbers unless reproduced here. Do not claim Dark Nova competes with GLM-5.2 unless the same benchmark, split, scoring method, tool access, generation budget, and harness are used.

## Black-Box Evaluation Baseline

When API access or local serving is available, run GLM-5.2 on public, synthetic, or explicitly approved tasks only.

Allowed task families:

- completion;
- infilling;
- bug repair;
- unit-test generation;
- repository file selection;
- API reuse;
- minimal-diff editing;
- style preservation;
- documentation generation;
- stale-document detection;
- hallucinated symbol detection;
- multi-file reasoning;
- "do not change unrelated functions" tasks.

Every GLM-5.2 black-box result must record prompt, context length, output length, thinking effort if available, tool access, temperature, sampling settings, cost, latency, tokens in/out, pass/fail result, exact judge or test command, and failure examples.

Never send confidential company code, private prompts, private repository content, secrets, credentials, customer data, or restricted documents to a remote GLM-5.2 service.

## Teacher And Distillation Baseline

GLM-5.2 can be investigated as a teacher, but generated data must be synthetic, traceable, and filtered by executable or source-grounded validation where possible.

Possible teacher uses:

- solutions for public benchmark tasks;
- repository-structure explanations;
- minimal patches;
- test cases;
- negative examples of bad patches;
- code-review comments;
- style-preserving rewrites;
- documentation updates;
- step-by-step repair traces;
- candidate patch ranking;
- hallucinated API labels;
- repository-understanding questions.

Each teacher-generated sample must record source task, prompt, model, settings, generation date, validation result, test status, and whether it was filtered or rejected.

Teacher comparisons must include no-teacher controls and should distinguish teacher-generated solutions, tests, ranked student outputs, minimal-diff data, documentation data, traces, retrieval plus teacher data, and teacher data mixed with human/public code.

## Architecture Inspiration Track

GLM-5.2-inspired ideas should be scaled down or reinterpreted for a single-machine local model. The current matrix already tracks `indexshare_sparse_attention_long_context` and `mtp_rejection_speculative_decoding`. Additional local-scale ideas to test include:

- small project-memory index plus local model;
- context-cache-aware repository retrieval;
- long-horizon task planner plus small code model;
- reasoning-effort control that changes budget, retrieval depth, or verification depth;
- file/function dependency map injected as compact context;
- tool-call planner trained on public repair tasks;
- verification-first code generation loop;
- minimal-diff constrained decoder;
- style-preserving patch planner.

For each GLM-inspired idea, record mechanism, closest primary source, local adaptation, expected benefit, expected failure mode, smallest falsifying experiment, storage/speed implications, and ROCm implementation strategy.

## Local Deployment Check

Before downloading any large GLM-5.2 weights or derivative:

- estimate total download size;
- estimate disk, RAM, and VRAM required;
- estimate expected runtime;
- identify supported inference engines;
- check ROCm compatibility;
- identify whether multi-GPU or NVIDIA-only hardware is required.

Do not download hundreds of gigabytes without explicit approval. If local GLM-5.2 is infeasible, record that and use API or published results only.

## Success Criteria

This track is successful when:

- GLM-5.2 published coding benchmark numbers are recorded from primary sources;
- the project has a reproducible GLM-5.2 evaluation harness for public tasks;
- at least one local model is compared against GLM-5.2 on the same public tasks;
- GLM-5.2 teacher data is tested against a no-teacher control;
- GLM-inspired mechanisms are converted into at least three local-scale experiments;
- no private data is sent to any remote model;
- the final report clearly separates same-budget local results from external frontier reference results.

## Current Local Status

`GLM5_2_public_eval_harness` is the first concrete local scaffold for this track. It uses `data/glm_public_eval_tasks/glm5_2_public_smoke_tasks.jsonl`, runs through `scripts/evaluate_glm_public_tasks.py`, writes `results/GLM5_2_public_eval_harness.json`, enforces public/synthetic/approved task policies, and records that no GLM-5.2 outputs have been scored yet.

`T11c_dense528_glm_public_smoke` is the first same-harness local comparison. It uses the current dense-528 checkpoint on ROCm, records complete saved-output metadata, and scores 0/3 on the three smoke tasks.

Next required evidence: score saved GLM-5.2 outputs with complete prompt/context/output/cost/latency/token metadata and then test one teacher-data slice against a no-teacher control.
