from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.metrics import ExperimentRecord, write_json
from weightlab.repository_task_sampler import (
    RepositoryTaskSampleConfig,
    sample_repository_tasks,
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
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-repositories", type=int, default=16)
    parser.add_argument("--files-per-repository", type=int, default=2)
    parser.add_argument("--min-text-bytes", type=int, default=20)
    parser.add_argument(
        "--task-kind",
        action="append",
        default=[],
        help="Task kind to include; can be passed more than once.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/D5_repository_balanced_task_sample.json"),
    )
    parser.add_argument("--experiment-id", default="D5_repository_balanced_task_sample")
    args = parser.parse_args()

    task_kinds = tuple(args.task_kind) or ("completion", "infilling", "syntax")
    config = RepositoryTaskSampleConfig(
        split=args.split,
        seed=args.seed,
        max_repositories=args.max_repositories,
        files_per_repository=args.files_per_repository,
        task_kinds=task_kinds,
        min_text_bytes=args.min_text_bytes,
    )
    metrics = sample_repository_tasks(args.corpus_jsonl, config)
    metrics["resolved_config"] = {
        "corpus_jsonl": str(args.corpus_jsonl),
        "split": args.split,
        "seed": args.seed,
        "max_repositories": args.max_repositories,
        "files_per_repository": args.files_per_repository,
        "task_kinds": list(task_kinds),
        "min_text_bytes": args.min_text_bytes,
        "output": str(args.output),
        "experiment_id": args.experiment_id,
    }
    command = (
        "uv run python scripts/evaluate_repository_coding_tasks.py "
        f"--corpus-jsonl {args.corpus_jsonl} --split {args.split} "
        f"--seed {args.seed} --max-repositories {args.max_repositories} "
        f"--files-per-repository {args.files_per_repository} "
        f"--min-text-bytes {args.min_text_bytes} --output {args.output} "
        f"--experiment-id {args.experiment_id}"
    )
    for task_kind in task_kinds:
        command += f" --task-kind {task_kind}"
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="repository_first_file_second_task_third_sampling_for_coding_eval",
        seed=args.seed,
        command=command,
        metrics=metrics,
        status="completed",
        notes=(
            "Creates a repository-balanced task index only. It does not score model "
            "quality or functional correctness."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
