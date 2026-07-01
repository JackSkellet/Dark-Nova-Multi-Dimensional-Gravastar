from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.metrics import ExperimentRecord, write_json
from weightlab.quixbugs_repair import QuixBugsRepairConfig, evaluate_quixbugs_repair


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--program", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/QuixBugs_python_repair_smoke.json"),
    )
    parser.add_argument("--experiment-id", default="QuixBugs_python_repair_smoke")
    args = parser.parse_args()

    config = QuixBugsRepairConfig(
        programs=tuple(args.program),
        timeout_seconds=args.timeout_seconds,
        seed=args.seed,
    )
    metrics = evaluate_quixbugs_repair(args.repo_path, config)
    metrics["resolved_config"] = {
        "repo_path": str(args.repo_path),
        "programs": list(config.programs),
        "timeout_seconds": args.timeout_seconds,
        "seed": args.seed,
        "output": str(args.output),
        "experiment_id": args.experiment_id,
    }
    command = (
        "uv run python scripts/evaluate_quixbugs_repair.py "
        f"--repo-path {args.repo_path} --timeout-seconds {args.timeout_seconds} "
        f"--seed {args.seed} --output {args.output} "
        f"--experiment-id {args.experiment_id}"
    )
    for program in config.programs:
        command += f" --program {program}"
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="quixbugs_python_repair_floor_and_oracle_ceiling",
        seed=args.seed,
        command=command,
        metrics=metrics,
        status="completed",
        notes=(
            "Runs QuixBugs Python pytest tests on buggy programs and corrected oracle "
            "programs. This records executable repair floor/ceiling evidence, not "
            "model-generated repairs."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
