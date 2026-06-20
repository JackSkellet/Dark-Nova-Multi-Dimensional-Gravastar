# Dark Nova WeightLab

This repository investigates whether secure, local, continually adapting AI architectures can improve the quality, storage, memory, speed, adaptability, and security Pareto frontier over simpler baselines for coding and documentation work.

The initial implementation is intentionally small. It uses CPU-compatible synthetic experiments plus ROCm smoke benchmarks to falsify weak versions of the original ideas and explore alternatives before attempting larger model work:

1. Contextual and hierarchical routing.
2. Compositional or multi-axis storage.
3. Importance-aware selective precision.
4. Vector or indexed component lookup.
5. Gated continual evolution from authorized local data.

The current expanded goal also tests alternatives outside those five ideas, including structured external repository memory.

## Commands

```bash
uv run pytest
uv run ruff check .
uv run python scripts/run_experiments.py --seed 123
uv run python scripts/run_experiments.py --seed 123 --config configs/smoke.yaml
```

Generated metrics are written to `results/manifest.json`, one JSON file per experiment, and `results/summary.csv`. The current manifest preserves explicit standalone runs and contains 120 records, including real-training, split-correct evaluation, held-out functional probes, and trained-checkpoint quantization records. The current assessment records T11 frontier expansion without Pareto dominance: dense-528 T11c leads validation loss, storage, VRAM, gradient stability, and throughput, while residual-adapter T11b keeps a small final-test-loss edge.

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
