# Combination Matrix

Date: 2026-07-01

Machine-readable source: `results/combination_matrix.json`

This is the living portfolio for Dark Nova contender systems. The project is not searching for one magic idea. Ideas are tested individually, combined only when evidence justifies it, and removed or revised when ablations show they do not add value.

## Current Stage

Current stage: Stage 2 pairwise controls, with same-pool baselines required before promotion to contender stacks.

Policy:

- Reject or revise ideas that fail their intended mechanism before scaling.
- Prefer pairwise and small-stack tests before expensive 50M, 100M, or larger token runs.
- Preserve negative results and keep machine-readable records in `results/`.
- Treat syntax-only and proxy tasks as scaffolding unless an executable or task-level benchmark proves broader utility.

## Idea Portfolio

| ID | Idea | Status | Current evidence | Strongest opposing evidence | Next falsifying test |
| --- | --- | --- | --- | --- | --- |
| `dense_528_adamw_fp32` | Dense-528 AdamW FP32 local decoder | combine | T12 selects dense-528 by validation/resource/throughput criteria; T11c has executable JavaScript syntax evidence. | Free-form QuixBugs repair generation repairs 0/4 tasks. | Run D4/D5 executable JavaScript and QuixBugs-style repair gates against T11c and a matched alternative. |
| `residual_adapter_528` | Residual-adapter-528 decoder | revise | T11b keeps a small final-test-loss edge and strong causal-span token accuracy. | T12 selects dense-528 on validation/resource/throughput criteria. | Run matched functional probes against dense-528 with identical prompts and held-out tasks. |
| `bpe_tokenizer_d5` | D5 trained byte-level BPE tokenizer | active | Reduces D5 token counts by about 3x and improves equal-compute exact nats/raw byte. | Equal-raw-byte BPE pilot is worse than byte tokens; functional quality unmeasured. | Evaluate byte-vs-BPE checkpoints on executable JS/API-reuse/repair tasks. |
| `structured_repository_memory` | Structured external repository memory | combine | E6 series passes repository symbol, signature, docs, API, and drift proxies; D5 context pairwise scores structured symbol memory at hit@5 0.8696 with proxy hallucinated API rate 0.0, still beating the best retrieval variant. | Several tasks remain exact lookup proxies, not generation. | Pair structured memory with dense generation and measure hallucinated API rate. |
| `fast_weight_scratchpad` | Signed fast-weight evolution scratchpad | revise | Synthetic IF2 and MarkupSafe IF4 show adaptation signal with rollback accounting. | Fast weights alone underperform structured/retrieval baselines on MarkupSafe chronology. | Run a second real chronology target with executable/API-reuse scoring. |
| `if7_sparse_hebbian_memory` | Sparse Hebbian repository memory | revise | Real D5 sparse assembly beats frequency and random controls on masked recall. | Cue-only/no-Hebbian ablations beat current Hebbian integration paths. | Add repository-local or typed-edge Hebbian features and require a no-Hebbian ablation win. |
| `repository_api_reuse_objective` | Repository-local API reuse objective | active | D5 API-reuse proxy records 23 validation tasks and strong symbol-name baseline. | Small validation-only task set and no executable generation yet. | Expand labels and require a method to beat symbol-name mention on held-out repositories. |
| `uniform_int8_quantization` | Uniform int8 checkpoint quantization | combine | Int8 stays close to FP32 on T11 D4 quantization samples. | No packed ROCm runtime benchmark yet. | Benchmark packed int8/BF16 runtime with functional probes. |
| `if3_block_codebook_compression` | Audited block-codebook weight generator | rejected | Learned reconstruction beats random reconstruction controls. | Validation loss degrades badly and FP32 reconstruction buffer erases deployment value. | Try less destructive low-rank plus groupwise residual compression before reconsidering. |
| `deterministic_ast_edit_baseline` | Deterministic QuixBugs AST-edit baseline | archive | Repairs 4/4 selected QuixBugs tasks with 6 candidates. | Hand-engineered and not model capability evidence. | Expand to more QuixBugs tasks only as calibration. |
| `dense_ranked_repair_selection` | Dense checkpoint repair-candidate likelihood ranking | revise | T11c ranks 4/4 passing deterministic edits first in the six-candidate pool; four-program bounded top-k syntax-pool execution repairs 4/4 at top-8; on the 12-program slice dense likelihood repairs 3/12 by top-8. | Free-form dense generation still repairs 0/4; four-program controls beat dense on candidate budget; the 12-program control does not show dense beating deterministic, repair-aware, or random same-pool controls. | Add learned or failure-localized ranking and require it to beat deterministic, repair-aware, and multiple random controls on a larger QuixBugs slice. |
| `syntax_preserving_mutation_pool` | Broader syntax-preserving AST mutation pool | revise | Four-program smoke builds 63 syntax-valid candidates and contains passing candidates for all four tasks; the 12-program slice builds 189 syntax-valid candidates and contains passing selected candidates for 3/12 programs. | The 12-program slice leaves 9/12 programs without a passing selected candidate, so candidate generation coverage is the main bottleneck. | Expand mutation families or add failure-localized generation and require passing candidates for substantially more than 3/12 programs without oracle sources. |
| `repair_aware_static_ordering` | Repair-aware static syntax-pool ordering | revise | Four-task smoke repairs 4/4 at top-1 and beats controls. | The 12-program slice falsifies the general selector claim: repair-aware order repairs 2/12 at top-1 and 3/12 by top-8, losing to deterministic order on candidate efficiency. | Replace static pattern ordering with failure-localized or learned repair-aware features and require a win over deterministic, dense-likelihood, and random controls on at least the 12-program slice. |
| `graph_conditioned_decoder` | Repository graph conditioned decoder | active | IF1 graph probe found typed edges and preserved splits. | Import resolution remains sparse and partly heuristic. | Implement AST/package-aware JS import resolution before model training. |
| `retrieval_augmented_repository_context` | Retrieval-augmented repository context | revise | RAG and RepoBench primary sources support non-parametric context and repository-level retrieval/completion gates; query-symbol-aware retrieval improves to hit@5 0.7826 with proxy hallucinated API rate 0.0. | Raw retrieved identifiers score hit@5 0.0 with hallucinated API rate 0.9826; symbol-aware retrieved snippets score hit@5 0.2174; query-symbol-aware retrieval still trails structured symbol memory at hit@5 0.8696. | Add structured-plus-retrieval context or repository-path-aware tie-breaking and require a gain over structured symbol memory before dense prompt integration. |
| `indexshare_sparse_attention_long_context` | IndexShare sparse attention for long-context coding | active | GLM-5.2 primary sources report IndexShare for 1M context, claimed 2.9x per-token FLOP reduction on the indexer path, and strong long-horizon coding benchmark results. | External vendor/model-card evidence only; current Dark Nova dense decoder does not implement DSA-style sparse attention, and KV-cache capacity remains a bottleneck. | Prototype a tiny shared-index sparse-attention or long-context surrogate and require better repository-completion cost/quality than structured memory plus dense baseline. |
| `mtp_rejection_speculative_decoding` | MTP rejection-sampling speculative decoding | active | GLM-5.2 reports up to 20 percent MTP acceptance-length gain in coding scenarios; the cited MTP paper reports up to 25 percent extra inference throughput and 1.8x async RL-training acceleration. | Not reproduced locally; current checkpoints have no MTP head or speculative decoding runtime, and ROCm overhead may erase gains at small scale. | Add a tiny local draft path and require measured wall-clock generation speedup on repair prompts with unchanged executable pass rate. |
| `low_rank_repository_adaptation_lora` | Low-rank repository adaptation | revise | LoRA primary work and implementation support frozen-base low-rank adaptation; current project adapters prove only an adjacent fully-trainable path. | T12 selects dense-528 over residual-adapter, and no frozen-base LoRA run has passed a functional coding gate here. | Train a frozen dense-528 low-rank repository adapter and require held-out API-reuse gains over structured-memory and full-finetune controls. |
| `execution_feedback_self_refinement` | Execution-feedback self-refinement loop | revise | Self-Refine and Reflexion support bounded test-time feedback; QuixBugs harness exposes executable pass/fail signals; bounded pytest execution over repair-aware syntax-pool candidates can find 4/4 repairs at top-1. | Current positive result is static repair-aware ordering plus validation, not an iterative model-generated revision loop; a feedback loop must beat a strong top-1 static baseline. | Run a bounded two-iteration feedback loop only if it can beat repair-aware static ordering on a larger subset or reduce candidate generation assumptions. |
| `agentless_localization_repair_validation` | Agentless localization-repair-validation pipeline | active | Agentless and SWE-agent primary sources support explicit software-engineering stages; Dark Nova already separates generation/ranking/pytest validation in QuixBugs. | Local dense free-form repair still repairs 0/4 selected tasks and multi-file localization is not implemented. | Build a local mini pipeline on QuixBugs or a tiny repo issue set and require a stage-level gain over direct dense completion. |
| `repobench_repository_completion_gate` | RepoBench-style repository completion gate | active | RepoBench defines repository retrieval, completion, and pipeline tasks that align with repository-understanding goals. | No pinned local RepoBench slice exists, and completion accuracy may miss test-passing patch behavior. | Create a reduced documented local slice and require at least one contender to beat lexical baselines before promotion. |
| `swe_bench_verified_primary_gate` | SWE-bench Verified as primary scale-up gate | archive | SWE-bench Verified remains useful historical context as a 500-instance human-filtered subset. | OpenAI's 2026 primary-source analysis says it is increasingly contaminated and no longer a frontier coding measure; full runs do not fit first-pass local constraints. | Do not use as the primary scale-up gate unless a small pinned local slice is affordable and a newer uncontaminated benchmark is unavailable. |

## External Primary Sources Added

The matrix now requires external paper, official-implementation, benchmark-report, and web-claim ideas to carry `primary_sources` in `results/combination_matrix.json` with title, URL, and access date. Sources added in this pass:

- RAG paper: <https://arxiv.org/abs/2005.11401>
- LoRA paper and official implementation: <https://arxiv.org/abs/2106.09685>, <https://github.com/microsoft/LoRA>
- Self-Refine and Reflexion papers/implementations: <https://arxiv.org/abs/2303.17651>, <https://github.com/madaan/self-refine>, <https://arxiv.org/abs/2303.11366>, <https://github.com/noahshinn/reflexion>
- Agentless and SWE-agent papers/implementations: <https://arxiv.org/abs/2407.01489>, <https://github.com/openautocoder/agentless>, <https://arxiv.org/abs/2405.15793>, <https://github.com/swe-agent/swe-agent>
- RepoBench paper and implementation: <https://arxiv.org/abs/2306.03091>, <https://github.com/Leolty/repobench>
- SWE-bench Verified overview and current OpenAI caveat: <https://www.swebench.com/verified.html>, <https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/>
- GLM-5.2 primary sources and technical context: <https://huggingface.co/blog/zai-org/glm-52-blog>, <https://huggingface.co/zai-org/GLM-5.2>, <https://github.com/zai-org/GLM-5>, <https://arxiv.org/abs/2602.15763v2>, <https://arxiv.org/abs/2606.12370>

## Contender Tracks

### Conservative Local Coder

Goal: strongest practical local coding system with the lowest deployment risk.

Current stack candidates:

- `dense_528_adamw_fp32`
- `bpe_tokenizer_d5`
- `structured_repository_memory`
- `retrieval_augmented_repository_context`
- `uniform_int8_quantization`
- `repository_api_reuse_objective`
- `repobench_repository_completion_gate`

Next ablation: Dense-528 byte tokenizer versus D5 BPE on executable JavaScript and API-reuse tasks, then add structured memory.

### Modular Repository Learner

Goal: best repository adaptation, rollback support, and low hallucinated API rate.

Current stack candidates:

- `dense_528_adamw_fp32`
- `structured_repository_memory`
- `retrieval_augmented_repository_context`
- `low_rank_repository_adaptation_lora`
- `fast_weight_scratchpad`
- `repository_api_reuse_objective`
- `agentless_localization_repair_validation`

Next ablation: structured memory alone versus structured memory plus fast updates on a second repository chronology target with executable or API-reuse scoring.

### Novel Mechanism Contender

Goal: strongest new architecture or training mechanism invented during this research.

Current stack candidates:

- `indexshare_sparse_attention_long_context`
- `graph_conditioned_decoder`
- `if7_sparse_hebbian_memory`
- `repository_api_reuse_objective`
- `bpe_tokenizer_d5`
- `execution_feedback_self_refinement`

Next ablation: typed-edge graph extraction and repository-local Hebbian features must beat lexical and no-Hebbian trained ranker baselines before decoder integration.

## Pairwise Test Queue

1. `structured_repository_memory` + `dense_528_adamw_fp32`: repository API-reuse prompt augmentation with hallucinated API rate.
2. `bpe_tokenizer_d5` + `dense_528_adamw_fp32`: executable JavaScript syntax/API-reuse comparison at equal raw-byte exposure.
3. `dense_ranked_repair_selection` + `syntax_preserving_mutation_pool`: `QuixBugs_T11c_dense528_repair_aware_syntax_controls_12prog` records a 12-program slice with 189 candidates. Deterministic order is best by candidate efficiency at 3/12 repairs by top-4; dense likelihood, repair-aware static order, and seeded random each reach 3/12 by top-8. Dense ranking remains revise and candidate generation coverage is the bottleneck.
4. `repair_aware_static_ordering` + `syntax_preserving_mutation_pool`: the 12-program slice falsifies the four-task top-1 selector claim. Repair-aware order repairs 2/12 at top-1 and 3/12 by top-8, losing to deterministic order on candidate efficiency.
5. `structured_repository_memory` + `fast_weight_scratchpad`: second repository chronology target with rollback and executable/API-reuse scoring.
6. `graph_conditioned_decoder` + `repository_api_reuse_objective`: AST/package-aware graph extraction followed by API-reuse scoring.
7. `retrieval_augmented_repository_context` + `structured_repository_memory`: `D5_repository_context_pairwise_validation` records structured symbol memory at hit@5 0.8696 / hallucinated API rate 0.0, raw retrieved-snippet identifiers at hit@5 0.0 / hallucinated API rate 0.9826, symbol-aware retrieved snippets at hit@5 0.2174 / hallucinated API rate 0.0, and query-symbol-aware retrieval at hit@5 0.7826 / hallucinated API rate 0.0. Retrieval remains revise: the best variant is a near miss but does not beat structured memory.
8. `execution_feedback_self_refinement` + `syntax_preserving_mutation_pool`: static repair-aware ordering no longer wins on the 12-program slice. A feedback loop should first improve candidate coverage or localization, not merely rerank the same weak pool.
9. `low_rank_repository_adaptation_lora` + `fast_weight_scratchpad`: rollback-accounted repository adaptation comparison.
10. `agentless_localization_repair_validation` + `retrieval_augmented_repository_context`: local mini Agentless pipeline with retrieval-assisted localization.
11. `indexshare_sparse_attention_long_context` + `structured_repository_memory`: tiny shared-index sparse-attention or long-context surrogate versus structured repository memory on a pinned repository-completion slice.
12. `mtp_rejection_speculative_decoding` + `execution_feedback_self_refinement`: local speculative/draft generation speedup on repair-loop prompts without reducing executable pass rate.

## Strength And Weakness Profile Fields

Every contender should eventually record:

- validation loss
- coding benchmark performance
- executable test success
- documentation accuracy
- repository understanding
- style matching
- minimal-edit behavior
- hallucinated API rate
- training speed
- inference speed
- VRAM
- RAM
- model size
- compression quality
- continual-learning ability
- forgetting
- rollback cost
- implementation complexity
- ROCm compatibility

## Next Matrix Maintenance Steps

1. Keep adding primary-source paper and official-implementation ideas with citations, then archive duplicates or already-disproven methods.
2. Convert the pairwise queue into executable result records as each test is implemented.
3. Add contender strength/weakness profiles after each stack has enough measured dimensions.
4. Promote only combinations whose ablations prove measurable value.
