from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.metrics import ExperimentRecord, write_json
from weightlab.quixbugs_repair import (
    QuixBugsCandidateConfig,
    build_quixbugs_reference_candidates,
    evaluate_quixbugs_candidate_repairs,
    load_quixbugs_candidates_jsonl,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--candidates-jsonl", type=Path)
    parser.add_argument("--program", action="append", default=[])
    parser.add_argument("--include-buggy-identity", action="store_true")
    parser.add_argument("--include-oracle-correct", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/QuixBugs_python_candidate_repair_smoke.json"),
    )
    parser.add_argument(
        "--experiment-id",
        default="QuixBugs_python_candidate_repair_smoke",
    )
    args = parser.parse_args()

    config = QuixBugsCandidateConfig(
        timeout_seconds=args.timeout_seconds,
        seed=args.seed,
    )
    candidates = []
    if args.candidates_jsonl is not None:
        candidates.extend(load_quixbugs_candidates_jsonl(args.candidates_jsonl))
    if args.include_buggy_identity or args.include_oracle_correct:
        candidates.extend(
            build_quixbugs_reference_candidates(
                args.repo_path,
                tuple(args.program),
                include_buggy_identity=args.include_buggy_identity,
                include_oracle_correct=args.include_oracle_correct,
            )
        )
    metrics = evaluate_quixbugs_candidate_repairs(args.repo_path, candidates, config)
    metrics["resolved_config"] = {
        "repo_path": str(args.repo_path),
        "candidates_jsonl": (
            str(args.candidates_jsonl) if args.candidates_jsonl is not None else None
        ),
        "programs": list(args.program),
        "include_buggy_identity": args.include_buggy_identity,
        "include_oracle_correct": args.include_oracle_correct,
        "timeout_seconds": args.timeout_seconds,
        "seed": args.seed,
        "output": str(args.output),
        "experiment_id": args.experiment_id,
    }
    command = (
        "uv run python scripts/evaluate_quixbugs_candidate_repairs.py "
        f"--repo-path {args.repo_path} "
    )
    if args.candidates_jsonl is not None:
        command += f"--candidates-jsonl {args.candidates_jsonl} "
    for program in args.program:
        command += f"--program {program} "
    if args.include_buggy_identity:
        command += "--include-buggy-identity "
    if args.include_oracle_correct:
        command += "--include-oracle-correct "
    command += (
        f"--timeout-seconds {args.timeout_seconds} "
        f"--seed {args.seed} --output {args.output} "
        f"--experiment-id {args.experiment_id}"
    )
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="quixbugs_python_candidate_repair_source_replacement_probe",
        seed=args.seed,
        command=command,
        metrics=metrics,
        status="completed",
        notes=(
            "Runs QuixBugs Python pytest tests against candidate replacement "
            "sources in isolated temporary checkouts. Candidate provenance comes "
            "from the input JSONL; this is not model-generated unless that file "
            "contains model outputs."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
