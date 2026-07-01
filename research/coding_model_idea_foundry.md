# Coding Model Idea Foundry

Date: 2026-07-01

This file is the durable index for coding-model ideas. It consolidates the existing idea-foundry records instead of replacing them:

- `research/30_idea_foundry_candidates.md`
- `research/idea_foundry_cycle_2.md`
- `results/idea_foundry_candidates.json`
- `results/IF1_repository_graph_signal_probe.json`
- `results/IF2_fast_weight_continual_probe.json`
- `results/IF3_block_codebook_t11c_probe.json`
- `results/IF3_block_codebook_t11c_validation_probe.json`
- `results/IF4_fast_repo_adaptation_markupsafe.json`
- IF7 records through `results/IF7i_trained_repository_ranker_d5_validation.json`
- `results/D5_repository_api_reuse_validation.json`
- `results/QuixBugs_python_repair_smoke.json`
- `results/QuixBugs_python_candidate_repair_smoke.json`
- `results/QuixBugs_T11c_dense528_candidate_repair_smoke.json`
- `results/QuixBugs_T11c_dense528_sampled_candidate_repair_smoke.json`
- `results/QuixBugs_edit_baseline_repair_smoke.json`
- `results/QuixBugs_T11c_dense528_ranked_edit_smoke.json`
- `results/QuixBugs_T11c_dense528_ranked_syntax_pool_smoke.json`
- `research/combination_matrix.md`
- `results/combination_matrix.json`

## Current Rule

Do not promote an idea because it is novel or complex. Promote only when the intended mechanism appears in measured results and survives the right baseline or ablation.

## Active Ideas

| ID | Idea | Mechanism | Current evidence | Decision |
| --- | --- | --- | --- | --- |
| IF1 | Repository graph conditioned decoder | Typed repository edges bias attention and optionally train edge prediction. | D5 has repository/path/role fields and 325 extracted graph edges, but local import resolution remains weak. | Continue extraction only; no model pilot until AST/package resolution materially improves non-heuristic edges. |
| IF2 | Signed fast-weight evolution scratchpad | Scoped reversible fast-weight updates from verified chronological local examples. | Synthetic chronological API-fact fixture shows value beyond retrieval/structured memory, but lacks poisoning/security gates. | Continue on real repository chronology with executable/API-reuse scoring. |
| IF3 | Audited block-codebook weight generator | Learned block codebooks plus residual accounting for trained checkpoint compression. | Learned reconstruction beats random control, but validation loss degrades badly and FP32 runtime reconstruction erases deployment value. | Reject current block policy for deployment; revise toward less destructive low-rank plus groupwise residual compression. |
| IF4 | Executable trace contrastive objective / fast repository adaptation lane | Compare retrieval, structured memory, replay proxy, fast weights, and consolidation on chronological code history. | MarkupSafe probe is a real Git-history changed-file proxy with rollback support, D5 now has a small repository API-reuse symbol-selection proxy, QuixBugs has executable floor/ceiling plus candidate source-replacement smokes, T11c dense-528 free-form QuixBugs candidates repair 0/4 under both greedy and sampled syntax-aware generation, a hand-engineered deterministic AST-edit baseline repairs 4/4 selected QuixBugs tasks with 6 candidates, T11c ranks the passing edit first for each task in that constrained pool, but broader 63-candidate syntax-pool top-1 ranking repairs only 1/4. | Continue with repair-aware pruning or top-k execution for broad syntax pools, plus a second chronology target with executable or API-reuse scoring. |
| IF5 | Syntax-state recurrent mixer | Condition recurrent/state-space updates on code lexical and syntax state. | Design-only in cycle 2. | Run bracket/scope and causal-span falsification before model training. |
| IF6 | Delta-consolidated specialist lattice | Repository specialists plus consolidation gates, rollback, replay, and security checks. | Design-only in cycle 2. | Defer until retrieval and fast-update baselines are stronger. |
| C2-7 | Retrieval-gated temporary delta cache | Retrieved facts instantiate request-local low-rank deltas that are discarded unless approved. | Design-only; closest to IF2 and IF4 next-step benchmark. | Test on the same chronological repository fixture as IF2 before any persistent adapter path. |
| IF7 | Sparse Hebbian repository memory | Sparse co-activation assemblies, trained conditioning, candidate reranking, and repository-linking/ranker features. | Real-corpus associative signal exists, but cue-only, raw Hebbian, lexical, or no-Hebbian ablations beat the current integration paths. | Keep as negative/proxy evidence; continue only if repository-local or typed-edge Hebbian features beat the no-Hebbian trained ranker. |

## Required Entry Fields

Each new idea must record:

- Name and short description.
- Mechanism with equations or algorithm.
- Closest prior art with source and access date.
- What is actually new.
- Expected benefit.
- Expected failure mode.
- Smallest falsifying experiment.
- Required data.
- ROCm/local execution plan.
- Storage and speed implications.
- Results.
- Continue, revise, or reject decision.

## Next Ideas To Test

1. AST/package-aware JavaScript import resolution for IF1, measured by non-heuristic repository-local edges and split preservation.
2. A less hand-constrained QuixBugs repair candidate path that adds repair-aware pruning or top-k execution after the 1/4 broader syntax-pool result, plus a second IF4/IF2 chronological repository target with executable repair or stronger API-reuse scoring.
3. A repository-local or typed-edge IF7 feature that must beat the no-Hebbian trained ranker ablation before any decoder integration.
4. A less destructive IF3 successor using low-rank plus groupwise residual compression with packed-layout accounting.
5. A functional BPE-vs-byte evaluation on executable JavaScript, repository API reuse, or unit-test repair before scaling tokenizer pilots.
