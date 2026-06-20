# Continual Evolution Results And Design

Status: generated from `results/E5_chronological_continual_learning.json`, `results/E5c_trainable_adapter_vs_retrieval.json`, `results/E5d_synthetic_stale_doc_positive_control.json`, `results/E5e_synthetic_patch_generation_functional.json`, public `kilo` results, public `markupsafe` results including `results/E5_markupsafe_public_patch_replay.json`, and `results/E5g_public_patch_replay_suite.json`.

## Required Architecture Boundary

The serving model may propose updates, but an external controller must approve, train, evaluate, sign, stage, and deploy them. The model must not approve or deploy its own production weights.

## Initial Prototype

The reduced E5 prototype separates:

- frozen memory,
- continuously updated retrieval memory,
- adapter-like memory without replay,
- adapter-like memory with replay,
- versioned rollback.

This is a simulation of state-management behavior, not neural continual learning.

## Seed 123 Results

Final step:

- Frozen accuracy: 0.25.
- Continuously updated retrieval accuracy: 1.00.
- Adapter-like no-replay prior accuracy: 0.667.
- Adapter-like replay prior accuracy: 1.00.
- Retrieval storage items: 4.
- Rollback restored version: 1.
- Accuracy after rollback on version-1 task set: 1.00.

## Interpretation

The simulation supports the three-timescale architecture as a control structure: retrieval updates are immediate and reversible, while replay prevents the toy adapter from forgetting old facts. It does not prove neural continual learning. The null baseline, updated retrieval, is already strong in this task.

## E5c Trainable Adapter Versus Retrieval

Status: generated from `results/E5c_trainable_adapter_vs_retrieval.json`.

E5c adds a small trainable parameter-update path. It uses a one-hot symbol classifier with a frozen base that knows only the first API fact, then trains a low-rank adapter delta over chronological API facts. The comparison includes:

- continuously updated exact retrieval,
- trainable adapter without replay,
- trainable adapter with replay,
- retained prior-task accuracy,
- new-task accuracy,
- update count and update timing,
- retrieval-table bytes versus adapter-weight bytes.

Final seed-123 metrics:

- Retrieval accuracy: 1.000.
- Adapter without replay accuracy: 0.500.
- Adapter without replay prior accuracy: 0.400.
- Adapter with replay accuracy: 1.000.
- Adapter with replay prior accuracy: 1.000.
- Retrieval storage: 235 bytes.
- Adapter storage: 384 bytes.
- Weight updates: 12.
- Null-hypothesis outcome: retrieval not beaten.

Interpretation: replay prevents forgetting in the trainable toy adapter, but the adapter does not beat continuously updated retrieval and uses more storage in this reduced task. This supports the H5 null baseline for exact API-fact memory: retrieval is simpler, immediate, smaller, and equally accurate. It still does not test real LoRA training, semantic generalization, or patch generation.

## E5d Synthetic Stale-Doc Positive Control

Status: generated from `results/E5d_synthetic_stale_doc_positive_control.json`.

E5d creates a temporary Git repository with three commits:

1. `parse_config` exists and documentation correctly mentions it.
2. Code renames `parse_config` to `load_config`, but documentation still mentions `parse_config`.
3. Documentation is updated to mention `load_config`.

Final seed-123 metrics:

- Expected stale-doc issues: 1.
- Detected stale-doc issues: 1.
- Stale symbol: `parse_config`.
- Stale file: `docs.md`.
- Retrieval stale-doc top-k accuracy: 1.000.
- Structured stale-doc top-k accuracy: 1.000.
- Post-fix stale-doc files: none.

Interpretation: the stale-doc detector has a scored positive control and verifies that a later documentation fix clears the stale state. This is synthetic evidence only; the public `kilo` and `markupsafe` scans still had zero positive stale-doc issues, so public stale-doc quality remains unmeasured.

## E5e Synthetic Patch Generation Functional Control

Status: generated from `results/E5e_synthetic_patch_generation_functional.json`.

E5e creates a temporary repository with a buggy `clamp` function and an executable `unittest` suite. It evaluates three candidate patches:

- `frozen_noop`: no patch, tests still fail.
- `retrieval_wrong_patch`: a plausible one-sided patch that fixes only the lower bound, tests still fail.
- `structured_patch`: a patch that handles low, high, and middle values, tests pass.

Final seed-123 metrics:

- Baseline tests passed: false.
- Functional success rate: 0.333.
- Best method: `structured_patch`.
- Frozen no-op passed: false.
- Retrieval wrong patch passed: false.
- Structured patch passed: true.

Interpretation: this adds a functional patch-generation grading path based on applying unified diffs and running tests, not text similarity to a reference patch. It is still synthetic and tiny. The replay harness now also includes a deterministic generated `clamp(value, low, high)` candidate for exposed public-history states that have the one-sided `return min(value, high)` pattern; that second generated heuristic is covered by repository-history regression tests but has not yet matched a real public commit in the local clones.

## E5f Public Patch Candidate Replay

Status: generated from `results/E5_markupsafe_public_patch_replay.json`.

E5f uses the local public `pallets/markupsafe` clone and evaluates commit `54bb00b`, `fix exact str check (#469)`, against its first parent `b529164`. The bug is that `escape` used `s.__class__ is str`; proxy objects can report the proxied value's class, so this incorrectly routed a proxy object into the plain-string fast path. The deterministic generated proxy candidate scans exposed Python source for `s.__class__ is str` and replaces it with `type(s) is str`. The replay harness also now has a separate deterministic clamp-bounds candidate, but this MarkupSafe state does not contain that pattern, so its public result is recorded as `null`. The historical future diff is retained as a control.

Executable regression command:

```python
import sys
sys.path.insert(0, "src")
from markupsafe import escape
Proxy = type(
    "Proxy",
    (),
    {"__class__": property(lambda self: str), "__str__": lambda self: "<em>"},
)
assert str(escape(Proxy())) == "&lt;em&gt;"
```

Final metrics:

- Base commit: `b5291646cbab`.
- Fix commit: `54bb00bfafe5`.
- Future commit not in exposed memory: true.
- Baseline tests passed: false.
- Frozen no-op passed: false.
- Generated proxy-class patch passed: true.
- Generated clamp-bounds patch passed: null.
- Historical future patch passed: true.
- Functional success rate: 0.667.
- Best method: `generated_proxy_class_patch`.

Interpretation: this remains the first public generated patch candidate with executable grading. It proves the harness can produce a small deterministic candidate from exposed public source, apply it to the exposed parent, and grade it functionally against the historical future diff. The regression command is a small hand-authored executable check rather than a full upstream test suite.

## E5g Public Patch Replay Suite

Status: generated from `results/E5g_public_patch_replay_suite.json`.

E5g aggregates two local public historical patch replay tasks:

1. `markupsafe_proxy_exact_type`: the E5f MarkupSafe `54bb00b` proxy-object fix, graded by the functional Python regression shown above.
2. `kilo_saved_hl_free`: `antirez/kilo` commit `8e9a9bb`, which adds `free(saved_hl);` to the `FIND_RESTORE_HL` macro before `saved_hl = NULL`.

The Kilo regression is executable but source-level: it reads `kilo.c`, scopes the check to the `FIND_RESTORE_HL` macro, and verifies that `free(saved_hl);` appears after the highlight `memcpy` and before `saved_hl = NULL`. It is not a Valgrind/ASAN runtime leak test and should not be treated as proof of memory safety.

Command:

```bash
uv run python scripts/run_public_patch_replay_suite.py
```

Final metrics:

- Task count: 2.
- Baseline failures: 2.
- Historical patch successes: 2.
- Generated candidate count: 2.
- Generated task successes: 2.
- Generated task success rate: 1.000.
- MarkupSafe generated method: `generated_proxy_class_patch`.
- Kilo generated method: `generated_saved_hl_free_patch`.

Interpretation: E5g broadens public patch evidence from one generated public candidate to two local public historical fixes and records task-level outcomes. It is still narrow: both generated patches are hand-authored deterministic heuristics keyed to specific source patterns, the MarkupSafe regression is a small hand-authored functional check, and the Kilo check is source-level rather than a runtime memory-leak detector. This improves replay coverage but is not broad autonomous bug fixing.

## Public Repository Chronology Attempt

Repository: `https://github.com/antirez/kilo`, cloned explicitly into ignored local data. The run used 12 first-parent commits ending at public HEAD `323d93b29bd8`, skipped empty-change steps, and evaluated 7 future changed-file steps with `top_k=5`.

Command:

```bash
uv run python scripts/run_public_repo_chronology.py --repo-path data/raw/public/kilo --max-commits 12 --top-k 5 --include-symbol-qa --include-stale-docs
```

Final metrics:

- Frozen future changed-file top-k accuracy: 0.714.
- Updated retrieval future changed-file top-k accuracy: 0.857.
- Structured memory future changed-file top-k accuracy: 1.000.
- Retrieval prior top-k accuracy at final step: 0.909.
- Structured prior top-k accuracy at final step: 1.000.
- Runtime: about 502 ms.

Interpretation: this satisfies an initial chronological public-repository attempt and prevents future-commit leakage by evaluating each future commit from prior Git state only. The result is still weak evidence: the repository is tiny, most changed files are `kilo.c`, and the task is changed-file retrieval from commit subjects rather than repository QA, patch generation, or neural continual learning.

## Public Repository Symbol-QA Attempt

The same local `antirez/kilo` clone was used for symbol-definition QA. The task asks questions of the form “which file defines symbol X?” over symbols visible at each exposed commit. Future symbols are tracked separately and are not counted as answerable before exposure.

Final metrics:

- Final frozen prior symbol top-k accuracy: 1.000.
- Final retrieval prior symbol top-k accuracy: 1.000.
- Final structured prior symbol top-k accuracy: 1.000.
- Mean retrieval prior symbol top-k accuracy: 1.000.
- Mean structured prior symbol top-k accuracy: 1.000.
- Future steps with new symbols: 1.
- Runtime: about 77 ms.

Interpretation: this satisfies a first repository-level QA attempt under chronological constraints, but the evidence is very weak because `kilo` is a tiny mostly single-file C project and top-k symbol-file questions are easy. The next E5 step should use a larger public repository and ask stale-document or API-change questions.

## Public Stale-Document Attempt

The stale-doc detector compares adjacent commits, finds symbols removed from code, and checks whether current documentation still mentions the removed symbols. E5d proves the detector finds a stale doc after `parse_config` is removed while `docs.md` still mentions it, then observes the stale state clear after a later documentation fix.

Public results:

| Repository | Commits | Removed-symbol steps | Stale-doc issues | Retrieval top-k | Structured top-k |
| --- | ---: | ---: | ---: | ---: | ---: |
| `antirez/kilo` | 12 | 1 | 0 | n/a | n/a |
| `pallets/markupsafe` | 30 | 0 | 0 | n/a | n/a |

The markupsafe command was:

```bash
uv run python scripts/run_public_repo_chronology.py --repo-path data/raw/public/markupsafe --max-commits 30 --top-k 5 --include-symbol-qa --include-stale-docs --include-patch-replay --patch-base-commit '54bb00b^' --patch-fix-commit 54bb00b --output results/E5_markupsafe_chronology.json --symbol-output results/E5_markupsafe_symbol_qa.json --stale-doc-output results/E5_markupsafe_stale_docs.json --patch-output results/E5_markupsafe_public_patch_replay.json --patch-test-code "import sys; sys.path.insert(0, 'src'); from markupsafe import escape; Proxy = type('Proxy', (), {'__class__': property(lambda self: str), '__str__': lambda self: '<em>'}); assert str(escape(Proxy())) == '&lt;em&gt;'"
```

Markupsafe changed-file and symbol-QA results:

- Changed-file frozen future top-k accuracy: 0.778.
- Changed-file updated retrieval future top-k accuracy: 0.667.
- Changed-file structured future top-k accuracy: 0.778.
- Symbol-QA final frozen prior top-k accuracy: 0.924.
- Symbol-QA final retrieval prior top-k accuracy: 0.165.
- Symbol-QA final structured prior top-k accuracy: 1.000.

Interpretation: the larger public run gives a more meaningful symbol-QA result than `kilo`, where structured symbol memory clearly beats text retrieval. The stale-document public scan is a negative result: no stale-doc issue was found in the sampled windows, so stale-doc quality cannot be estimated from those public runs yet. E5g adds a second real-public generated patch replay task, but the generated coverage remains narrow hand-written heuristic coverage rather than general bug fixing.

## E5b External Update Controller

Status: generated from `results/E5b_external_update_controller.json`.

E5b adds a small executable simulation of the Tier 2 update controller described in the goal. The language-model side may propose a repository-adapter update, but an external controller decides whether to deploy it. Each candidate carries:

- training-data lineage,
- configuration lineage,
- payload checksum,
- signature,
- external approval identity,
- new-task score versus continuously updated retrieval,
- prior-task retention score,
- security/privacy gate counts.

The run stages four candidates: one replay-backed adapter that passes all gates, one no-replay adapter rejected for prior-task regression above the 2% absolute limit, one candidate rejected for canary leakage, and one candidate rejected after payload tampering changes the checksum after signing. The accepted candidate improves new-task score by 0.06 over the retrieval baseline, then the controller rolls back to the base version.

Interpretation: this demonstrates update versioning, checksum/signature tamper detection, external approval, regression gating, security gating, immutable-style audit events, and rollback in a reduced prototype. It is still not real adapter training, real signing infrastructure, or a production deployment controller.
