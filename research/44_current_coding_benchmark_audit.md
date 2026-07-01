# Current Coding Benchmark Audit

Date: 2026-07-01

Access date for all linked sources: 2026-07-01. This is a planning audit only; no benchmark score is claimed here.

## Benchmark Map

| Benchmark | Primary source | Measures | Fit for this repo | First local step |
| --- | --- | --- | --- | --- |
| HumanEval | [OpenAI HumanEval paper](https://arxiv.org/abs/2107.03374) and [official harness](https://github.com/openai/human-eval) | Python function synthesis from docstrings with unit tests. | Useful as a legacy sanity check, but too small and contamination-prone to drive selection alone. | Add only after local generation harness can run pass@1 reproducibly. |
| MBPP | [Google Research MBPP README](https://github.com/google-research/google-research/blob/master/mbpp/README.md) and [paper PDF](https://arxiv.org/pdf/2108.07732) | Entry-level Python tasks from natural language with tests. | Useful as a second legacy Python function benchmark; should not replace repo-aware tasks. | Prefer EvalPlus MBPP+ variant when running public comparisons. |
| EvalPlus HumanEval+/MBPP+ | [EvalPlus repository](https://github.com/evalplus/evalplus) and [leaderboard notes](https://evalplus.github.io/leaderboard.html) | More rigorous test suites for HumanEval and MBPP plus efficiency evaluation. | Stronger first public function-synthesis gate than raw HumanEval/MBPP. | Wire HumanEval+ first, then MBPP+ if runtime is acceptable. |
| LiveCodeBench | [LiveCodeBench paper](https://arxiv.org/abs/2403.07974), [site](https://livecodebench.github.io/), and [repository](https://github.com/livecodebench/livecodebench) | Time-separated competitive-programming tasks, self-repair, code execution, and test-output prediction. | High value for contamination-resistant coding ability, but likely heavier than current tiny local model can handle. | Use a small fixed slice only after function-synthesis harness exists. |
| BigCodeBench | [BigCodeBench paper](https://arxiv.org/abs/2406.15877) and [repository](https://github.com/bigcode-project/bigcodebench) | Python tasks requiring diverse library/function calls across domains. | High value for API-use and instruction-following, directly aligned with "reuse existing functions" goals. | Add a small API-use slice before claiming coding utility from validation loss. |
| SWE-bench | [SWE-bench site](https://www.swebench.com/) and [repository](https://github.com/swe-bench/SWE-bench) | Real GitHub issue-to-patch repair with repository context and tests. | Best aligned with agentic code repair, but too heavy for the current from-scratch tiny model. | Treat as external reference and future agent benchmark, not first local model gate. |
| RepoBench | [RepoBench paper](https://arxiv.org/abs/2306.03091) and [repository](https://github.com/Leolty/repobench) | Repository-level retrieval, code completion, and pipeline tasks for Python/Java. | Strong fit for this repo's repository-understanding objective and IF7/IF1 lessons. | Use RepoBench-R/P style tasks to replace metadata-only repository-balanced scaffold. |
| MultiPL-E | [MultiPL-E paper](https://arxiv.org/abs/2208.08227), [site](https://nuprl.github.io/MultiPL-E/), and [repository](https://github.com/nuprl/MultiPL-E) | Translated unit-test-driven code generation across many languages. | Useful for multi-language breadth after Python/JavaScript gates are stable. | Defer until byte/BPE tokenizer and D5 language mix are evaluated functionally. |
| APPS | [APPS paper](https://arxiv.org/abs/2105.09938) | Python competitive/interview-style programming problems with tests. | Useful for algorithmic breadth, but older and potentially contaminated. | Lower priority than LiveCodeBench for current comparisons. |
| CodeContests | [DeepMind CodeContests repository](https://github.com/google-deepmind/code_contests) and [AlphaCode paper PDF](https://storage.googleapis.com/deepmind-media/AlphaCode/competition_level_code_generation_with_alphacode.pdf) | Competitive programming with temporal split and generated tests. | Useful as a harder algorithmic benchmark and possible data source, but expensive. | Defer behind LiveCodeBench small-slice evaluation. |
| Defects4J | [Defects4J discussion of v2.0 scale](https://arxiv.org/html/2310.19139v3) | Real Java bugs with triggering tests for repair evaluation. | Good for repair behavior but requires Java project setup and patch generation. | Future repair gate after repository patch harness exists. |
| QuixBugs | [QuixBugs repository](https://github.com/jkoppel/QuixBugs) and [paper page](https://www.jameskoppel.com/publication/quixbugs/) | Small Python/Java one-line program repair tasks. | Good cheap repair smoke test before Defects4J/SWE-bench. | Initial Python floor/ceiling, candidate source-replacement, dense-model candidate, deterministic AST-edit baseline, dense-ranked edit, and broader syntax-pool smokes added; T11c dense-528 greedy and sampled free-form candidates repair 0/4 tasks, the hand-engineered AST-edit baseline repairs 4/4 selected tasks, T11c ranks passing edits first within that constrained six-candidate pool, but broader 63-candidate syntax-pool top-1 ranking repairs only 1/4. |

## Priority For This Repository

1. Add a local generation/evaluation harness that can run small deterministic task slices and record prompts, completions, tests, seed, command, hardware, model checkpoint, and failure status.
2. First public gate: EvalPlus HumanEval+ pass@1 on a tiny local model, reported as a sanity check rather than a training-selection metric.
3. First project-aligned gate: strengthen the new D5 repository API-reuse proxy with cleaner repository-local labels, then add BigCodeBench or another reduced API-use slice for public comparison.
4. First repository-understanding gate: RepoBench-style retrieval/completion/pipeline tasks, replacing the current metadata-only D5 repository-balanced scaffold.
5. First repair gate: improve the broad syntax-valid repair pool with repair-aware pruning or top-k execution before Defects4J or SWE-bench.
6. First contamination-resistant algorithmic gate: LiveCodeBench fixed time slice after basic harness stability.

## Local Constraints

- Do not use public benchmark test loss for checkpoint selection.
- Record benchmark version, split, task IDs, prompt format, sampling parameters, pass/fail status, stdout/stderr, and exact command.
- Keep initial runs CPU-compatible or bounded ROCm inference; no benchmark should trigger a silent download from reusable code.
- Treat published frontier scores as external reference numbers only unless the same split, prompt, sampling, and scoring method are reproduced locally.
