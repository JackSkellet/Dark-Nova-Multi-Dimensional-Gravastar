# Dark Nova WeightLab

This repository investigates whether secure, local, continually adapting AI architectures can improve the quality, storage, memory, speed, adaptability, and security Pareto frontier over simpler baselines for coding and documentation work.

The initial implementation is intentionally small. It uses CPU-compatible synthetic experiments plus ROCm smoke benchmarks to falsify weak versions of the original ideas and explore alternatives before attempting larger model work:

1. Contextual and hierarchical routing.
2. Compositional or multi-axis storage.
3. Importance-aware selective precision.
4. Vector or indexed component lookup.
5. Gated continual evolution from authorized local data.

The current expanded goal also tests alternatives outside those five ideas, including structured external repository memory and a primary-source-backed combination/contender portfolio. That portfolio now includes GLM-5.2 as a separated external frontier/open-weight reference, black-box public-task baseline, possible synthetic teacher, and architecture-inspiration source; GLM-5.2 results must stay separated from same-budget local claims, and remote GLM calls may use only public, synthetic, or explicitly approved data.

## Commands

```bash
uv run pytest
uv run ruff check .
uv run python scripts/run_experiments.py --seed 123
uv run python scripts/run_experiments.py --seed 123 --config configs/smoke.yaml
```

Generated metrics are written to `results/manifest.json`, one JSON file per experiment, and `results/summary.csv`. The current manifest preserves explicit standalone runs and contains 187 records, including real-training, split-correct evaluation, held-out functional probes, trained-checkpoint quantization records, the completed exploratory D5 corpus materialization and audit, exact D5 byte-vs-BPE validation-byte accounting, the repository-balanced D5 task-index scaffold, the D5 repository API-reuse proxy, the D5 repository-context pairwise proxy, the QuixBugs Python executable repair and candidate-repair smokes, the failed T11c dense-528 QuixBugs greedy and sampled model-candidate repair smokes, the deterministic QuixBugs AST-edit baseline that repairs the same four-task smoke without being model-generated, the T11c dense-528 model-ranked AST-edit probe that selects 4/4 passing repairs from that deterministic candidate pool, the broader T11c dense-ranked syntax-pool top-1 probe that drops to 1/4 repairs, the T11c ranked syntax-pool top-k execution probe that recovers 4/4 repairs at top-8 using 26 selected candidates, the same-pool ordering-control probe showing deterministic pool order reaches 4/4 by top-4 while dense likelihood needs top-8, the four-program repair-aware static ordering probe that reaches 4/4 at top-1, the 12-program repair-aware control where deterministic pool order is best by candidate efficiency and all methods top out at 3/12 repairs, the counter-zero syntax-pool expansion that improves the same 12-program slice to 4/12 repairs, the `GLM5_2_public_eval_harness` offline public/synthetic/approved-data GLM-5.2 scoring scaffold, and the `T11c_dense528_glm_public_smoke` same-harness local dense baseline scoring 0/3. The manifest also includes the IF4 real Git-history fast repository adaptation probe, the IF7 sparse Hebbian follow-up series, and the T12 three-seed dense-528 versus residual-adapter-528 comparison. The current assessment records T11/T12 frontier expansion without Pareto dominance: dense-528 leads mean validation loss, storage, VRAM, stability, and throughput, while residual-adapter-528 keeps a reported-only mean final-test-loss edge. T12 selects dense-528 by validation mean, with paired batch bootstrap uncertainty now recorded; the final-validation interval crosses zero, best-validation favors dense, and test loss remains reported-only. IF7 records a real-corpus associative-memory signal and a useful supervised repository-ranker direction, but current Hebbian conditioning, reranking, and linking variants do not beat the relevant cue-only, raw-Hebbian, lexical, or no-Hebbian ablations strongly enough to justify decoder integration. The combination/contender program is tracked in `research/combination_matrix.md` and `results/combination_matrix.json`; it now includes primary-source-backed external ideas, archives SWE-bench Verified as a primary scale-up gate under current evidence, revises retrieval-augmented repository context after query-symbol-aware retrieval improves to hit@5 0.7826 but still trails structured symbol memory at 0.8696 on the D5 pairwise proxy, keeps dense repair ranking, syntax-pool generation, and repair-aware static ordering in revise state after the 12-program QuixBugs controls, and adds GLM-5.2-derived IndexShare long-context sparse attention plus MTP rejection-sampling speculative decoding as active external ideas, not local wins. It also records a formal GLM-5.2 external baseline and teacher track with four baseline categories, a private-data prohibition for remote calls, published-versus-reproduced benchmark separation, local-deployment preflight requirements before any large download, and future teacher-data controls against no-teacher baselines.

`configs/smoke.yaml` requests `accelerator.backend: rocm`. The project routes Linux PyTorch resolution to the ROCm 7.2 wheel index and adds `triton-rocm`; E4c and E4d map ROCm to PyTorch HIP and report the logical ROCm backend separately from PyTorch's internal `cuda` device type.

The public-repository chronology experiment is explicit because it needs a local clone:

```bash
git clone https://github.com/antirez/kilo data/raw/public/kilo
uv run python scripts/run_public_repo_chronology.py --repo-path data/raw/public/kilo --max-commits 12 --top-k 5 --include-symbol-qa --include-stale-docs
```

A larger public sample used in the current results is:

```bash
git clone https://github.com/pallets/markupsafe data/raw/public/markupsafe
uv run python scripts/run_public_repo_chronology.py --repo-path data/raw/public/markupsafe --max-commits 30 --top-k 5 --include-symbol-qa --include-stale-docs --output results/E5_markupsafe_chronology.json --symbol-output results/E5_markupsafe_symbol_qa.json --stale-doc-output results/E5_markupsafe_stale_docs.json
```

The IF4 fast repository adaptation probe compares updated retrieval, structured
memory, replay adapter proxy, temporary fast weights, fast weights plus retrieval,
and periodic consolidation on the same local public MarkupSafe history:

```bash
uv run python scripts/run_if4_fast_repo_adaptation.py --repo-path data/raw/public/markupsafe --max-commits 30 --top-k 5 --output results/IF4_fast_repo_adaptation_markupsafe.json --experiment-id IF4_fast_repo_adaptation_markupsafe
```

The current public patch task uses MarkupSafe commit `54bb00b`, which fixed proxy-object compatibility in `escape`. The exposed parent fails a small executable proxy regression; the no-op candidate fails, a deterministic generated proxy-check patch passes, and the historical future diff passes as a control:

```bash
uv run python scripts/run_public_repo_chronology.py --repo-path data/raw/public/markupsafe --max-commits 30 --top-k 5 --include-symbol-qa --include-stale-docs --include-patch-replay --patch-base-commit '54bb00b^' --patch-fix-commit 54bb00b --output results/E5_markupsafe_chronology.json --symbol-output results/E5_markupsafe_symbol_qa.json --stale-doc-output results/E5_markupsafe_stale_docs.json --patch-output results/E5_markupsafe_public_patch_replay.json --patch-test-code "import sys; sys.path.insert(0, 'src'); from markupsafe import escape; Proxy = type('Proxy', (), {'__class__': property(lambda self: str), '__str__': lambda self: '<em>'}); assert str(escape(Proxy())) == '&lt;em&gt;'"
```

## Current Scope

The first pass does not train a useful-scale language model or claim novelty. It builds a reproducible research harness, tests storage/latency accounting paths, records reduced evidence, includes explicit public Git-history changed-file retrieval, symbol-definition QA, stale-document scans, public historical patch replay, ROCm runtime checks, and reduced structured external-memory alternatives including constrained call-stub, function-skeleton, documented-skeleton, API-reference generation, API-doc coverage, and synthetic API-doc drift detection proxies. Larger experiments must be justified by component evidence and added through explicit configurations.

## Layout

- `research/`: charter, hypotheses, prior art, architecture candidates, plans, results, security, and final assessment.
- `src/weightlab/`: reusable prototype code.
- `tests/`: fast CPU tests.
- `scripts/`: documented experiment entry points.
- `configs/`: experiment configuration files.
- `results/`: machine-readable generated metrics.
- `data/`: public or synthetic experiment data only.
- `artifacts/`: generated non-source artifacts that are safe to keep locally.
