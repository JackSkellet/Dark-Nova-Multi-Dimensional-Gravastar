from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.metrics import ExperimentRecord, write_json
from weightlab.quixbugs_syntax_mutations import (
    DenseQuixBugsSyntaxPoolRankConfig,
    evaluate_quixbugs_syntax_pool_ordering_controls,
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
        "scripts/evaluate_quixbugs_syntax_pool_ordering_controls.py",
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
        ]
    )
    for top_k in args.top_k:
        parts.extend(["--top-k", str(top_k)])
    parts.extend(
        [
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
    parser.add_argument("--max-candidates-per-program", type=int, default=32)
    parser.add_argument("--top-k", action="append", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/QuixBugs_T11c_dense528_syntax_pool_ordering_controls_smoke.json"
        ),
    )
    parser.add_argument(
        "--experiment-id",
        default="QuixBugs_T11c_dense528_syntax_pool_ordering_controls_smoke",
    )
    args = parser.parse_args()

    config = DenseQuixBugsSyntaxPoolRankConfig(
        device=args.device,
        seed=args.seed,
        timeout_seconds=args.timeout_seconds,
        max_candidates_per_program=args.max_candidates_per_program,
        top_candidates_per_program=1,
    )
    metrics = evaluate_quixbugs_syntax_pool_ordering_controls(
        args.checkpoint,
        args.repo_path,
        programs=tuple(args.program),
        top_k_values=tuple(args.top_k),
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
        "top_k": list(args.top_k),
        "output": str(args.output),
        "experiment_id": args.experiment_id,
    }
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis=(
            "repair_aware_ordering_should_beat_dense_and_same_pool_non_model_controls"
        ),
        seed=args.seed,
        command=_command(args),
        metrics=metrics,
        status="completed",
        notes=(
            "Evaluation-only QuixBugs control probe. A local dense decoder "
            "checkpoint ranks a syntax-preserving AST mutation pool, then the "
            "same top-k pytest budgets are compared against deterministic pool "
            "order, seeded random order, and a static repair-aware ordering "
            "from the same candidate pool."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
