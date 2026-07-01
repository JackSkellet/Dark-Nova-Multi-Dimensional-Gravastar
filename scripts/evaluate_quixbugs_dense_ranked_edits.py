from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.metrics import ExperimentRecord, write_json
from weightlab.quixbugs_model_ranked_edits import (
    DenseQuixBugsEditRankConfig,
    evaluate_dense_ranked_quixbugs_edit_candidates,
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
        "scripts/evaluate_quixbugs_dense_ranked_edits.py",
        "--checkpoint",
        str(args.checkpoint),
        "--repo-path",
        str(args.repo_path),
    ]
    for program in args.program:
        parts.extend(["--program", program])
    parts.extend(
        [
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--max-candidates-per-program",
            str(args.max_candidates_per_program),
            "--top-candidates-per-program",
            str(args.top_candidates_per_program),
            "--output",
            str(args.output),
            "--experiment-id",
            args.experiment_id,
        ]
    )
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--program", action="append", required=True)
    parser.add_argument("--device", default="rocm")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--max-candidates-per-program", type=int, default=8)
    parser.add_argument("--top-candidates-per-program", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/QuixBugs_T11c_dense528_ranked_edit_smoke.json"),
    )
    parser.add_argument(
        "--experiment-id",
        default="QuixBugs_T11c_dense528_ranked_edit_smoke",
    )
    args = parser.parse_args()

    config = DenseQuixBugsEditRankConfig(
        device=args.device,
        seed=args.seed,
        timeout_seconds=args.timeout_seconds,
        max_candidates_per_program=args.max_candidates_per_program,
        top_candidates_per_program=args.top_candidates_per_program,
    )
    metrics = evaluate_dense_ranked_quixbugs_edit_candidates(
        args.checkpoint,
        args.repo_path,
        programs=tuple(args.program),
        config=config,
    )
    metrics["resolved_config"] = {
        "checkpoint": str(args.checkpoint),
        "repo_path": str(args.repo_path),
        "programs": list(args.program),
        "device": args.device,
        "seed": args.seed,
        "timeout_seconds": args.timeout_seconds,
        "max_candidates_per_program": args.max_candidates_per_program,
        "top_candidates_per_program": args.top_candidates_per_program,
        "output": str(args.output),
        "experiment_id": args.experiment_id,
    }
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="dense_checkpoint_can_rank_deterministic_quixbugs_edit_candidates",
        seed=args.seed,
        command=_command(args),
        metrics=metrics,
        status="completed",
        notes=(
            "Evaluation-only QuixBugs repair probe. A local dense decoder checkpoint "
            "scores deterministic AST-edit candidate repairs by teacher-forced "
            "candidate likelihood; the selected candidates are then evaluated with "
            "the QuixBugs pytest harness. This is model ranking, not free-form repair "
            "generation."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
