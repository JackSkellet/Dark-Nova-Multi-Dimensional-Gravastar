# QuixBugs Candidate Repair Smoke

Date: 2026-07-01

## Purpose

This run adds the first candidate source-replacement lane for the QuixBugs Python repair harness. It evaluates candidate program replacements in isolated temporary checkouts and runs the normal QuixBugs pytest path against each candidate. It does not evaluate model-generated repairs yet.

## Source

- Repository: `https://github.com/jkoppel/QuixBugs`
- Local path: `data/raw/public/quixbugs`
- Pinned commit: `4257f44b0ff1181dedaedee6a447e133219fcebf`
- Access date: 2026-07-01

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_quixbugs_candidate_repairs.py --repo-path data/raw/public/quixbugs --program gcd --program flatten --program possible_change --program sqrt --include-buggy-identity --include-oracle-correct --timeout-seconds 15 --seed 123 --output results/QuixBugs_python_candidate_repair_smoke.json --experiment-id QuixBugs_python_candidate_repair_smoke
```

The result record includes the command, seed, source commit, hardware summary, per-candidate pytest outcomes, source hashes, generator labels, aggregate metrics, and failure status.

## Results

This smoke evaluates two explicit reference candidates per program:

- `buggy_identity_baseline`: unchanged buggy source.
- `oracle_correct_upper_bound`: corrected source copied from QuixBugs `correct_python_programs`.

Aggregate metrics:

- Program count: 4.
- Candidate count: 8.
- Candidate pass rate: 0.5.
- Programs with a passing candidate: 4.
- Program repair rate: 1.0.
- Runtime: 16.00 seconds.

Passing candidates:

| Program | Passing candidate |
| --- | --- |
| `gcd` | `gcd:oracle_correct` |
| `flatten` | `flatten:oracle_correct` |
| `possible_change` | `possible_change:oracle_correct` |
| `sqrt` | `sqrt:oracle_correct` |

## Interpretation

The repair harness now has a candidate-evaluation path that can score replacement source candidates without mutating the original QuixBugs checkout. This is a harness capability result, not a model-quality result: the only passing candidates in this smoke are oracle-correct upper-bound copies.

The follow-up `QuixBugs_T11c_dense528_candidate_repair_smoke` feeds local dense-528 checkpoint candidates through this lane. Those model-generated candidates repair 0/4 tasks, so the next useful step is a stronger repair-oriented candidate generator rather than broader claims about the current dense checkpoint.

## Limitations

- Oracle-correct candidates are upper-bound references, not model output.
- Replacement source only; no unified diff or multi-file patch application yet.
- Python subset only.
- Four-program smoke, not the full 40-program benchmark.
- No public leaderboard score is claimed.
- The local environment and timeout policy affect pytest outcomes.
