# Dark Nova Goal Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the broad local coding-model research mandate into a reproducible first checkpoint that preserves existing IF7 lineage and establishes the next benchmark/corpus/idea-foundry work.

**Architecture:** Treat `results/manifest.json`, `results/research_assessment.json`, `research/STATUS.md`, and `README.md` as durable state carriers. Start from the current uncommitted IF7 experiment set, verify its records and assessment logic, then update only claims that are directly backed by machine-readable artifacts.

**Tech Stack:** Python 3.12 through `uv`, pytest, ruff, JSON experiment records, Markdown research notes.

---

### Task 1: Audit Current IF7 Lineage

**Files:**
- Read: `research/36_if7_sparse_hebbian_assembly.md`
- Read: `research/37_if7_hebbian_trained_model.md`
- Read: `research/38_if7_hebbian_trained_model_fulltrain.md`
- Read: `research/39_if7_hebbian_trained_model_500k_windows.md`
- Read: `research/40_if7_sparse_hebbian_reranker.md`
- Read: `research/41_if7_repository_linking.md`
- Read: `research/42_if7_hebbian_trained_model_2m_windows.md`
- Read: `research/43_if7_trained_repository_ranker.md`
- Read: `results/IF7_sparse_hebbian_d5_probe.json`
- Read: `results/IF7b_hebbian_trained_model_d5.json`
- Read: `results/IF7c_hebbian_trained_model_d5_fulltrain.json`
- Read: `results/IF7d_hebbian_trained_model_d5_500k_windows.json`
- Read: `results/IF7e_sparse_hebbian_reranker_d5_500k_windows.json`
- Read: `results/IF7f_sparse_hebbian_reranker_priors_d5_500k_windows.json`
- Read: `results/IF7g_repository_linking_d5_validation.json`
- Read: `results/IF7h_hebbian_trained_model_d5_2m_windows.json`
- Read: `results/IF7i_trained_repository_ranker_d5_validation.json`

- [ ] **Step 1: List the IF7 records present in the working tree**

Run: `git status --short research results scripts src tests`

Expected: IF7 research notes, IF7 result JSON files, IF7 scripts, `src/weightlab/if7_sparse_hebbian.py`, and `tests/test_if7_sparse_hebbian.py` are visible as untracked or modified files.

- [ ] **Step 2: Check each IF7 JSON record for completed status and required metadata**

Run: `uv run python - <<'PY'\nimport json\nfrom pathlib import Path\nfor path in sorted(Path('results').glob('IF7*.json')):\n    record = json.loads(path.read_text())\n    missing = [key for key in ['experiment_id', 'status', 'seed', 'command', 'hypothesis', 'hardware', 'metrics'] if key not in record]\n    print(path.name, record.get('status'), 'missing=' + ','.join(missing))\nPY`

Expected: Every IF7 record prints `completed missing=` with no missing keys.

- [ ] **Step 3: Confirm IF7 interpretation does not overclaim**

Run: `rg -n "pareto|beats|not code generation|not.*repair|not.*decoder|not.*language model|negative|ablation|cue-only|no-Hebbian" research/3*_if7*.md research/4*_if7*.md`

Expected: The notes identify proxy limits, negative follow-ups, ablation failures, and no-Hebbian ranker results where applicable.

### Task 2: Verify Assessment Integration

**Files:**
- Read: `src/weightlab/assessment.py`
- Read: `tests/test_assessment.py`
- Read: `results/research_assessment.json`
- Read: `results/manifest.json`

- [ ] **Step 1: Run the assessment unit tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_assessment.py -q`

Expected: All assessment tests pass.

- [ ] **Step 2: Run the IF7 unit tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_if7_sparse_hebbian.py -q`

Expected: All IF7 tests pass.

- [ ] **Step 3: Regenerate the assessment from the current manifest**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/summarize_research_assessment.py`

Expected: `results/research_assessment.json` is regenerated without an exception and still reports `pareto_dominance_found=false`.

- [ ] **Step 4: Confirm the manifest contains exactly the current IF7 result files**

Run: `uv run python - <<'PY'\nimport json\nfrom pathlib import Path\nmanifest = json.loads(Path('results/manifest.json').read_text())\nentries = manifest['entries'] if isinstance(manifest, dict) and 'entries' in manifest else manifest\nids = sorted(entry.get('experiment_id') for entry in entries if str(entry.get('experiment_id', '')).startswith('IF7'))\nfiles = sorted(path.stem for path in Path('results').glob('IF7*.json'))\nprint('manifest_if7=', ids)\nprint('files_if7=', files)\nprint('counts=', len(ids), len(files))\nPY`

Expected: The two IF7 lists match by experiment ID/stem, except where an older record uses a documented alternate ID.

### Task 3: Run Repository Gates

**Files:**
- Read: `pyproject.toml`
- Read: `research/STATUS.md`
- Read: `README.md`

- [ ] **Step 1: Run the focused changed-area tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_assessment.py tests/test_if7_sparse_hebbian.py -q`

Expected: The changed-area tests pass.

- [ ] **Step 2: Run the full test suite**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q`

Expected: The full suite passes.

- [ ] **Step 3: Run lint**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .`

Expected: Ruff reports `All checks passed!`.

### Task 4: Sync Durable Research State

**Files:**
- Modify if stale: `README.md`
- Modify if stale: `research/STATUS.md`
- Modify if stale: `results/manifest.json`
- Modify if stale: `results/research_assessment.json`

- [ ] **Step 1: Compare README and STATUS claims to current IF7 assessment**

Run: `rg -n "IF7|171-record|171 record|pareto_dominance|no-Hebbian|Hebbian" README.md research/STATUS.md results/research_assessment.json`

Expected: README and STATUS either already match the IF7 assessment or show exact stale lines needing edits.

- [ ] **Step 2: Edit only stale durable carriers**

Use `apply_patch` to update stale README or STATUS lines so they match the verified IF7 evidence: IF7 has useful associative and task-aware ranker signals, but current Hebbian features do not beat the no-Hebbian trained ablation and do not justify decoder integration.

- [ ] **Step 3: Re-run the focused verification after edits**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_assessment.py tests/test_if7_sparse_hebbian.py -q && UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .`

Expected: Tests and lint pass after any documentation or assessment edits.

### Task 5: Establish Next Research Work Items

**Files:**
- Create or modify if absent/stale: `research/coding_model_idea_foundry.md`
- Modify if stale: `research/STATUS.md`
- Modify if stale: `README.md`

- [ ] **Step 1: Check whether the requested idea foundry file already exists**

Run: `test -f research/coding_model_idea_foundry.md; echo $?`

Expected: Exit code `0` means the requested file exists; exit code `1` means create it from current idea-foundry records.

- [ ] **Step 2: Check current benchmark documentation coverage**

Run: `rg -n "HumanEval|MBPP|EvalPlus|SWE-bench|RepoBench|Defects4J|QuixBugs|APPS|CodeContests|MultiPL-E|BigCodeBench|LiveCodeBench" research benchmarks README.md`

Expected: Existing benchmark coverage is visible; missing current benchmark notes become the next documentation task.

- [ ] **Step 3: Record the immediate next checkpoint in STATUS**

Use `apply_patch` to add a concise next-checkpoint line to `research/STATUS.md` after verification: audit current coding benchmarks, design the stronger exploratory corpus, and convert IF7 lessons into the next falsifiable candidate.

- [ ] **Step 4: Re-run documentation grep**

Run: `rg -n "next checkpoint|coding benchmarks|exploratory corpus|IF7" research/STATUS.md README.md`

Expected: The next checkpoint is findable from durable docs.
