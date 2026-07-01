from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.if7_sparse_hebbian import (
    RepositoryLinkingRankerConfig,
    run_if7_trained_repository_linking_ranker,
)
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


def build_record(
    *,
    corpus_jsonl: Path,
    config: RepositoryLinkingRankerConfig,
    output: Path,
    experiment_id: str,
) -> dict[str, object]:
    metrics = run_if7_trained_repository_linking_ranker(
        corpus_jsonl=corpus_jsonl,
        config=config,
    )
    metrics["resolved_config"] = {
        "corpus_jsonl": str(corpus_jsonl),
        "train_split": config.train_split,
        "eval_split": config.eval_split,
        "seed": config.seed,
        "node_count": config.node_count,
        "max_active_nodes": config.max_active_nodes,
        "max_memory_rows": config.max_memory_rows,
        "max_train_repositories": config.max_train_repositories,
        "max_eval_repositories": config.max_eval_repositories,
        "negatives_per_query": config.negatives_per_query,
        "top_k": config.top_k,
        "min_text_bytes": config.min_text_bytes,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "device": config.device,
        "output": str(output),
        "experiment_id": experiment_id,
    }
    command = (
        "uv run python scripts/train_if7_repository_ranker.py "
        f"--corpus-jsonl {corpus_jsonl} --train-split {config.train_split} "
        f"--eval-split {config.eval_split} --seed {config.seed} "
        f"--node-count {config.node_count} --max-active-nodes {config.max_active_nodes} "
        f"--max-memory-rows {config.max_memory_rows} "
        f"--max-train-repositories {config.max_train_repositories} "
        f"--max-eval-repositories {config.max_eval_repositories} "
        f"--negatives-per-query {config.negatives_per_query} --top-k {config.top_k} "
        f"--min-text-bytes {config.min_text_bytes} --epochs {config.epochs} "
        f"--batch-size {config.batch_size} --learning-rate {config.learning_rate} "
        f"--device {config.device} --output {output} --experiment-id {experiment_id}"
    )
    record = ExperimentRecord(
        experiment_id=experiment_id,
        hypothesis="if7_task_aware_ranker_uses_hebbian_context_for_repository_linking",
        seed=config.seed,
        command=command,
        metrics=metrics,
        status="completed",
        notes=(
            "Trains a task-aware IF7 repository-linking ranker over path, lexical, "
            "and Hebbian context features."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--node-count", type=int, default=4096)
    parser.add_argument("--max-active-nodes", type=int, default=48)
    parser.add_argument("--max-memory-rows", type=int, default=8192)
    parser.add_argument("--max-train-repositories", type=int, default=128)
    parser.add_argument("--max-eval-repositories", type=int, default=64)
    parser.add_argument("--negatives-per-query", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-text-bytes", type=int, default=80)
    parser.add_argument("--cue-node-budget", type=int, default=16)
    parser.add_argument("--target-node-budget", type=int, default=32)
    parser.add_argument("--hebbian-learning-rate", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="rocm")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/IF7i_trained_repository_ranker_d5_validation.json"),
    )
    parser.add_argument(
        "--experiment-id",
        default="IF7i_trained_repository_ranker_d5_validation",
    )
    args = parser.parse_args()

    config = RepositoryLinkingRankerConfig(
        train_split=args.train_split,
        eval_split=args.eval_split,
        seed=args.seed,
        node_count=args.node_count,
        max_active_nodes=args.max_active_nodes,
        max_memory_rows=args.max_memory_rows,
        max_train_repositories=args.max_train_repositories,
        max_eval_repositories=args.max_eval_repositories,
        negatives_per_query=args.negatives_per_query,
        top_k=args.top_k,
        min_text_bytes=args.min_text_bytes,
        cue_node_budget=args.cue_node_budget,
        target_node_budget=args.target_node_budget,
        hebbian_learning_rate=args.hebbian_learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
    )
    record = build_record(
        corpus_jsonl=args.corpus_jsonl,
        config=config,
        output=args.output,
        experiment_id=args.experiment_id,
    )
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
