from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.metrics import ExperimentRecord, write_json
from weightlab.quixbugs_edit_baseline import (
    QuixBugsEditBaselineConfig,
    evaluate_quixbugs_edit_baseline,
)


def _append_manifest(output: Path, record: dict[str, object]) -> None:
    manifest_path = output.parent / "manifest.json"
    if manifest_path.exists():
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        records = []
    experiment_id = str(record["experiment_id"])
    records = [row for row in records if row.get("experiment_id") != experiment_id]
    records.append({"experiment_id": experiment_id, "path": output.name})
    write_json(manifest_path, records)


def _command(args: argparse.Namespace) -> str:
    parts = [
        "uv",
        "run",
        "python",
        "scripts/evaluate_quixbugs_edit_baseline.py",
        "--repo-path",
        str(args.repo_path),
    ]
    for program in args.program:
        parts.extend(["--program", program])
    parts.extend(
        [
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--seed",
            str(args.seed),
            "--max-candidates-per-program",
            str(args.max_candidates_per_program),
            "--output",
            str(args.output),
            "--experiment-id",
            args.experiment_id,
        ]
    )
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--program", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-candidates-per-program", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/QuixBugs_edit_baseline_repair_smoke.json"),
    )
    parser.add_argument(
        "--experiment-id",
        default="QuixBugs_edit_baseline_repair_smoke",
    )
    args = parser.parse_args()

    config = QuixBugsEditBaselineConfig(
        programs=tuple(args.program),
        timeout_seconds=args.timeout_seconds,
        seed=args.seed,
        max_candidates_per_program=args.max_candidates_per_program,
    )
    metrics = evaluate_quixbugs_edit_baseline(args.repo_path, config)
    metrics["resolved_config"] = {
        "repo_path": str(args.repo_path),
        "programs": list(args.program),
        "timeout_seconds": args.timeout_seconds,
        "seed": args.seed,
        "max_candidates_per_program": args.max_candidates_per_program,
        "output": str(args.output),
        "experiment_id": args.experiment_id,
    }
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="deterministic_ast_edits_can_calibrate_quixbugs_repair_lane",
        seed=args.seed,
        command=_command(args),
        metrics=metrics,
        status="completed",
        notes=(
            "Runs hand-engineered deterministic AST edit candidates through the "
            "same QuixBugs Python pytest replacement-source harness. This is a "
            "baseline calibration and is not model-generated."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
