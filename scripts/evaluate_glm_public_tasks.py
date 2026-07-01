from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.glm_public_eval import GlmPublicEvalConfig, evaluate_glm_public_tasks
from weightlab.metrics import ExperimentRecord, write_json


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
    parser = argparse.ArgumentParser(
        description="Evaluate saved GLM-5.2 outputs on public/synthetic tasks."
    )
    parser.add_argument("--tasks-jsonl", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-tasks", type=int, default=256)
    parser.add_argument(
        "--baseline-category",
        choices=[
            "local_same_budget_baseline",
            "local_practical_baseline",
            "external_glm_5_2_baseline",
            "published_glm_5_2_reference",
        ],
        default="external_glm_5_2_baseline",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/GLM5_2_public_eval_harness.json"),
    )
    parser.add_argument("--experiment-id", default="GLM5_2_public_eval_harness")
    args = parser.parse_args()

    config = GlmPublicEvalConfig(
        seed=args.seed,
        predictions_jsonl=args.predictions_jsonl,
        max_tasks=args.max_tasks,
        baseline_category=args.baseline_category,
    )
    metrics = evaluate_glm_public_tasks(args.tasks_jsonl, config)
    metrics["resolved_config"] = {
        "tasks_jsonl": str(args.tasks_jsonl),
        "predictions_jsonl": (
            str(args.predictions_jsonl) if args.predictions_jsonl is not None else None
        ),
        "seed": args.seed,
        "max_tasks": args.max_tasks,
        "baseline_category": args.baseline_category,
        "output": str(args.output),
        "experiment_id": args.experiment_id,
    }
    command = (
        "uv run python scripts/evaluate_glm_public_tasks.py "
        f"--tasks-jsonl {args.tasks_jsonl} --seed {args.seed} "
        f"--max-tasks {args.max_tasks} --baseline-category {args.baseline_category} "
        f"--output {args.output} "
        f"--experiment-id {args.experiment_id}"
    )
    if args.predictions_jsonl is not None:
        command += f" --predictions-jsonl {args.predictions_jsonl}"
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="glm_5_2_external_public_task_eval_harness",
        seed=args.seed,
        command=command,
        metrics=metrics,
        status="completed",
        notes=(
            "Offline harness for saved GLM-5.2/API/local-serving outputs on "
            "public, synthetic, or explicitly approved tasks only. The script "
            "does not call a remote model."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
