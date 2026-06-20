# Initial Research Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the empty repository into a reproducible research project with first-pass evidence for H1-H5.

**Architecture:** Use a small Python package with deterministic synthetic experiments, machine-readable JSON/CSV outputs, and Markdown research reports. Keep prior-art review separate from executable prototypes.

**Tech Stack:** Python, NumPy, PyTorch as available, pytest, ruff, uv.

---

### Task 1: Bootstrap Tests and Package

**Files:**
- Create: `pyproject.toml`
- Create: `tests/test_routing.py`
- Create: `tests/test_compression.py`
- Create: `tests/test_importance.py`
- Create: `tests/test_lookup.py`
- Create: `tests/test_continual.py`
- Create: `src/weightlab/*.py`

- [x] Write failing tests for routing, compression, importance, lookup, and continual memory.
- [x] Run `uv run pytest` and verify import failures.
- [x] Implement minimal deterministic code.
- [x] Run `uv run pytest` and verify pass.

### Task 2: Research Structure

**Files:**
- Create: `AGENTS.md`, `README.md`, `ROADMAP.md`, `Makefile`
- Create: `research/*.md`, `research/prior_art.csv`, `research/references.bib`
- Create: `data/README.md`, `artifacts/README.md`, `configs/smoke.yaml`

- [x] Record environment and resource assumptions.
- [x] Formalize H1-H5 and null hypotheses.
- [x] Add first prior-art matrix and DS4 source notes.
- [x] Define candidate architectures and decision matrix.

### Task 3: Experiment Runner and Results

**Files:**
- Create: `scripts/run_experiments.py`
- Generate: `results/manifest.json`, `results/summary.csv`, `results/E*.json`
- Update: `research/08_routing_results.md` through `research/12_continual_evolution.md`
- Update: `research/17_final_assessment.md`

- [ ] Run `uv run python scripts/run_experiments.py --seed 123`.
- [ ] Inspect generated JSON.
- [ ] Update result narratives with measured values.
- [ ] Run `uv run pytest` and `uv run ruff check .`.

