from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.if7_sparse_hebbian import (
    SparseHebbianRerankerConfig,
    run_if7_hebbian_sparse_reranker,
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
    config: SparseHebbianRerankerConfig,
    output: Path,
    experiment_id: str,
) -> dict[str, object]:
    metrics = run_if7_hebbian_sparse_reranker(corpus_jsonl=corpus_jsonl, config=config)
    metrics["resolved_config"] = {
        "corpus_jsonl": str(corpus_jsonl),
        "split": config.split,
        "validation_split": config.validation_split,
        "seed": config.seed,
        "node_count": config.node_count,
        "max_train_rows": config.max_train_rows,
        "max_validation_rows": config.max_validation_rows,
        "max_train_patterns": config.max_train_patterns,
        "max_validation_patterns": config.max_validation_patterns,
        "text_window_bytes": config.text_window_bytes,
        "text_window_stride_bytes": config.text_window_stride_bytes,
        "candidate_count": config.candidate_count,
        "max_train_candidates": config.max_train_candidates,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "recall_at_k": config.recall_at_k,
        "device": config.device,
        "output": str(output),
        "experiment_id": experiment_id,
    }
    command = (
        "uv run python scripts/train_if7_sparse_reranker.py "
        f"--corpus-jsonl {corpus_jsonl} --split {config.split} "
        f"--validation-split {config.validation_split} "
        f"--node-count {config.node_count} "
        f"--max-train-rows {config.max_train_rows} "
        f"--max-validation-rows {config.max_validation_rows} "
        f"--max-train-patterns {config.max_train_patterns} "
        f"--max-validation-patterns {config.max_validation_patterns} "
        f"--text-window-bytes {config.text_window_bytes} "
        f"--text-window-stride-bytes {config.text_window_stride_bytes} "
        f"--candidate-count {config.candidate_count} "
        f"--max-train-candidates {config.max_train_candidates} "
        f"--epochs {config.epochs} --batch-size {config.batch_size} "
        f"--learning-rate {config.learning_rate} "
        f"--recall-at-k {config.recall_at_k} --device {config.device} "
        f"--output {output} --experiment-id {experiment_id} --seed {config.seed}"
    )
    record = ExperimentRecord(
        experiment_id=experiment_id,
        hypothesis="if7_sparse_hebbian_topk_reranker_improves_raw_hebbian_candidates",
        seed=config.seed,
        command=command,
        metrics=metrics,
        status="completed",
        notes=(
            "Trained IF7 sparse candidate reranker. Uses Hebbian top-k candidates and "
            "a tiny scorer instead of dense full-node Hebbian feature concatenation."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--node-count", type=int, default=2048)
    parser.add_argument("--max-active-nodes", type=int, default=48)
    parser.add_argument("--max-train-rows", type=int, default=8192)
    parser.add_argument("--max-validation-rows", type=int, default=1024)
    parser.add_argument("--max-train-patterns", type=int, default=8192)
    parser.add_argument("--max-validation-patterns", type=int, default=1024)
    parser.add_argument("--text-window-bytes", type=int, default=0)
    parser.add_argument("--text-window-stride-bytes", type=int, default=0)
    parser.add_argument("--candidate-count", type=int, default=64)
    parser.add_argument("--max-train-candidates", type=int, default=200_000)
    parser.add_argument("--recall-at-k", type=int, default=32)
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
        default=Path("results/IF7e_sparse_hebbian_reranker_d5.json"),
    )
    parser.add_argument("--experiment-id", default="IF7e_sparse_hebbian_reranker_d5")
    args = parser.parse_args()

    config = SparseHebbianRerankerConfig(
        split=args.split,
        validation_split=args.validation_split,
        seed=args.seed,
        node_count=args.node_count,
        max_active_nodes=args.max_active_nodes,
        max_train_rows=args.max_train_rows,
        max_validation_rows=args.max_validation_rows,
        max_train_patterns=args.max_train_patterns,
        max_validation_patterns=args.max_validation_patterns,
        text_window_bytes=args.text_window_bytes,
        text_window_stride_bytes=args.text_window_stride_bytes,
        candidate_count=args.candidate_count,
        max_train_candidates=args.max_train_candidates,
        recall_at_k=args.recall_at_k,
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
