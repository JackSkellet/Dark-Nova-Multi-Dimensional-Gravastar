from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.metrics import ExperimentRecord, write_json
from weightlab.repository_api_reuse import (
    RepositoryApiReuseConfig,
    evaluate_repository_context_comparison,
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
    parser.add_argument("--source-split", default="validation")
    parser.add_argument("--query-split", default="validation")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-tasks", type=int, default=256)
    parser.add_argument("--max-source-rows", type=int, default=10_000)
    parser.add_argument("--max-query-rows", type=int, default=10_000)
    parser.add_argument("--min-text-bytes", type=int, default=80)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/D5_repository_context_pairwise_validation.json"),
    )
    parser.add_argument(
        "--experiment-id",
        default="D5_repository_context_pairwise_validation",
    )
    args = parser.parse_args()

    config = RepositoryApiReuseConfig(
        source_split=args.source_split,
        query_split=args.query_split,
        seed=args.seed,
        top_k=args.top_k,
        max_tasks=args.max_tasks,
        max_source_rows=args.max_source_rows,
        max_query_rows=args.max_query_rows,
        min_text_bytes=args.min_text_bytes,
    )
    metrics = evaluate_repository_context_comparison(args.corpus_jsonl, config)
    metrics["resolved_config"] = {
        "corpus_jsonl": str(args.corpus_jsonl),
        "source_split": args.source_split,
        "query_split": args.query_split,
        "seed": args.seed,
        "top_k": args.top_k,
        "max_tasks": args.max_tasks,
        "max_source_rows": args.max_source_rows,
        "max_query_rows": args.max_query_rows,
        "min_text_bytes": args.min_text_bytes,
        "output": str(args.output),
        "experiment_id": args.experiment_id,
    }
    command = (
        "uv run python scripts/evaluate_repository_context_pairwise.py "
        f"--corpus-jsonl {args.corpus_jsonl} --source-split {args.source_split} "
        f"--query-split {args.query_split} --seed {args.seed} --top-k {args.top_k} "
        f"--max-tasks {args.max_tasks} --max-source-rows {args.max_source_rows} "
        f"--max-query-rows {args.max_query_rows} "
        f"--min-text-bytes {args.min_text_bytes} --output {args.output} "
        f"--experiment-id {args.experiment_id}"
    )
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="retrieval_context_vs_structured_memory_pairwise_probe",
        seed=args.seed,
        command=command,
        metrics=metrics,
        status="completed",
        notes=(
            "Compares structured repository-symbol memory against identifiers drawn from "
            "retrieved source snippets on repository API-reuse tasks. This is a proxy "
            "for hallucinated API rate, not code generation or executable scoring."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
