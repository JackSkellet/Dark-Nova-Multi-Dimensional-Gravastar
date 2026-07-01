# QuixBugs Python Repair Smoke

Date: 2026-07-01

## Purpose

This run adds the first public executable unit-test repair benchmark gate for the project. It measures a buggy floor and oracle-correct ceiling on a small QuixBugs Python subset. It does not evaluate model-generated repairs.

## Source

- Repository: `https://github.com/jkoppel/QuixBugs`
- Local path: `data/raw/public/quixbugs`
- Pinned commit: `4257f44b0ff1181dedaedee6a447e133219fcebf`
- Access date: 2026-07-01

QuixBugs describes 40 programs from the Quixey Challenge translated into Python and Java, each with a one-line defect and test cases.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_quixbugs_repair.py --repo-path data/raw/public/quixbugs --program gcd --program flatten --program possible_change --program sqrt --timeout-seconds 15 --seed 123 --output results/QuixBugs_python_repair_smoke.json --experiment-id QuixBugs_python_repair_smoke
```

The result record includes the command, seed, source commit, hardware summary, status, per-program pytest outcomes, and aggregate metrics.

## Results

| Program | Buggy pytest | Correct pytest |
| --- | --- | --- |
| `gcd` | fail | pass |
| `flatten` | fail | pass |
| `possible_change` | fail | pass |
| `sqrt` | timeout/fail | pass |

Aggregate metrics:

- Program count: 4.
- Buggy pass rate: 0.0.
- Oracle-correct pass rate: 1.0.
- Repair gap: 1.0.
- Runtime: 15.99 seconds.

## Interpretation

This is the first public executable repair gate in the repository. It proves the harness can run real unit-test repair tasks and record a machine-readable floor/ceiling. It does not prove local model repair ability because no model-generated patch is attempted. The follow-up `QuixBugs_python_candidate_repair_smoke` adds a candidate source-replacement lane with buggy identity and oracle-correct reference candidates; the next useful step is to evaluate local model-generated candidates against that lane without selecting on test loss.

## Limitations

- Python subset only.
- Four-program smoke, not the full 40-program benchmark.
- Oracle-correct programs are an upper bound, not model output.
- No public leaderboard score is claimed.
- The local environment and timeout policy affect pytest outcomes.
