from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.metrics import ExperimentRecord, write_json
from weightlab.quixbugs_model_candidates import (
    DenseQuixBugsCandidateConfig,
    evaluate_dense_quixbugs_candidate_repairs,
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
        "scripts/evaluate_quixbugs_dense_candidates.py",
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
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--samples-per-program",
            str(args.samples_per_program),
            "--temperature",
            str(args.temperature),
            "--top-k",
            str(args.top_k),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--output",
            str(args.output),
            "--experiment-id",
            args.experiment_id,
        ]
    )
    if args.prefer_syntax_valid:
        parts.append("--prefer-syntax-valid")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--program", action="append", required=True)
    parser.add_argument("--device", default="rocm")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--samples-per-program", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--prefer-syntax-valid", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/QuixBugs_dense528_candidate_repair_smoke.json"),
    )
    parser.add_argument(
        "--experiment-id",
        default="QuixBugs_dense528_candidate_repair_smoke",
    )
    args = parser.parse_args()

    config = DenseQuixBugsCandidateConfig(
        device=args.device,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        timeout_seconds=args.timeout_seconds,
        samples_per_program=args.samples_per_program,
        temperature=args.temperature,
        top_k=args.top_k,
        prefer_syntax_valid=args.prefer_syntax_valid,
    )
    metrics = evaluate_dense_quixbugs_candidate_repairs(
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
        "max_new_tokens": args.max_new_tokens,
        "samples_per_program": args.samples_per_program,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "prefer_syntax_valid": args.prefer_syntax_valid,
        "timeout_seconds": args.timeout_seconds,
        "output": str(args.output),
        "experiment_id": args.experiment_id,
    }
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="dense_checkpoint_generates_quixbugs_repair_candidates",
        seed=args.seed,
        command=_command(args),
        metrics=metrics,
        status="completed",
        notes=(
            "Evaluation-only QuixBugs repair candidate probe. A local dense decoder "
            "checkpoint generates replacement source candidates, which are then "
            "scored with the QuixBugs pytest harness."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
