from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.if7_sparse_hebbian import run_if7_hebbian_repository_linking
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--node-count", type=int, default=2048)
    parser.add_argument("--max-train-rows", type=int, default=8192)
    parser.add_argument("--max-eval-repositories", type=int, default=64)
    parser.add_argument("--negatives-per-query", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-text-bytes", type=int, default=80)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/IF7g_repository_linking_d5_validation.json"),
    )
    parser.add_argument("--experiment-id", default="IF7g_repository_linking_d5_validation")
    args = parser.parse_args()

    metrics = run_if7_hebbian_repository_linking(
        corpus_jsonl=args.corpus_jsonl,
        train_split=args.train_split,
        eval_split=args.eval_split,
        seed=args.seed,
        node_count=args.node_count,
        max_train_rows=args.max_train_rows,
        max_eval_repositories=args.max_eval_repositories,
        negatives_per_query=args.negatives_per_query,
        top_k=args.top_k,
        min_text_bytes=args.min_text_bytes,
    )
    metrics["resolved_config"] = {
        "corpus_jsonl": str(args.corpus_jsonl),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "seed": args.seed,
        "node_count": args.node_count,
        "max_train_rows": args.max_train_rows,
        "max_eval_repositories": args.max_eval_repositories,
        "negatives_per_query": args.negatives_per_query,
        "top_k": args.top_k,
        "min_text_bytes": args.min_text_bytes,
        "output": str(args.output),
        "experiment_id": args.experiment_id,
    }
    command = (
        "uv run python scripts/run_if7_repository_linking.py "
        f"--corpus-jsonl {args.corpus_jsonl} --train-split {args.train_split} "
        f"--eval-split {args.eval_split} --seed {args.seed} "
        f"--node-count {args.node_count} --max-train-rows {args.max_train_rows} "
        f"--max-eval-repositories {args.max_eval_repositories} "
        f"--negatives-per-query {args.negatives_per_query} --top-k {args.top_k} "
        f"--min-text-bytes {args.min_text_bytes} --output {args.output} "
        f"--experiment-id {args.experiment_id}"
    )
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="if7_hebbian_context_improves_repository_file_linking",
        seed=args.seed,
        command=command,
        metrics=metrics,
        status="completed",
        notes=(
            "Evaluates IF7 Hebbian context for repository file linking: query one file, "
            "rank same-repository positives against cross-repository distractors."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
