# Agent Instructions

This repository is a reproducible research project for secure local AI-weight architecture experiments.

- Keep reusable code in `src/weightlab/`, tests in `tests/`, command-line experiment runners in `scripts/`, and generated metrics in `results/`.
- Use `uv run pytest` for tests, `uv run ruff check .` for linting, and `uv run python scripts/run_experiments.py --seed 123` for the CPU smoke experiment suite.
- Every experiment must record a seed, command, hypothesis, Git commit if available, hardware summary, metrics, and failure status in JSON or CSV.
- Do not fabricate measurements. Label simulations and reduced experiments explicitly.
- Include all metadata, indexes, codebooks, scales, routing tables, caches, and reconstruction buffers in storage accounting.
- Do not place credentials, private data, raw model weights, downloaded datasets, or generated caches in Git.
- Do not make silent network calls from reusable code. Downloads must be explicit commands and documented.
- Keep initial experiments CPU-compatible and under the project resource limits: no individual model or dataset over 5 GB, no more than 20 GB of new disk, and no multi-hour first-pass jobs.
- Cite primary sources in research documents with access date and pinned repository commit when applicable.
- Definition of done for code changes: tests pass, commands are documented, generated outputs are machine-readable, and `research/STATUS.md` reflects the current state.

