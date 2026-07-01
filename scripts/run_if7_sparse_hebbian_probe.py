from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.if7_sparse_hebbian import (
    SparseHebbianConfig,
    run_if7_sparse_hebbian_probe,
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
    config: SparseHebbianConfig,
    output: Path,
    experiment_id: str,
) -> dict[str, object]:
    metrics = run_if7_sparse_hebbian_probe(corpus_jsonl=corpus_jsonl, config=config)
    metrics["resolved_config"] = {
        "corpus_jsonl": str(corpus_jsonl),
        "split": config.split,
        "seed": config.seed,
        "node_count": config.node_count,
        "max_active_nodes": config.max_active_nodes,
        "max_train_rows": config.max_train_rows,
        "max_eval_rows": config.max_eval_rows,
        "recall_at_k": config.recall_at_k,
        "output": str(output),
        "experiment_id": experiment_id,
    }
    command = (
        "uv run python scripts/run_if7_sparse_hebbian_probe.py "
        f"--corpus-jsonl {corpus_jsonl} --split {config.split} "
        f"--node-count {config.node_count} "
        f"--max-active-nodes {config.max_active_nodes} "
        f"--max-train-rows {config.max_train_rows} "
        f"--max-eval-rows {config.max_eval_rows} "
        f"--recall-at-k {config.recall_at_k} "
        f"--output {output} --experiment-id {experiment_id} --seed {config.seed}"
    )
    record = ExperimentRecord(
        experiment_id=experiment_id,
        hypothesis="if7_sparse_hebbian_assemblies_add_real_corpus_associative_signal",
        seed=config.seed,
        command=command,
        metrics=metrics,
        status="completed",
        notes=(
            "Sparse Hebbian assembly memory probe on real corpus JSONL rows. It tests "
            "whether sparse co-activation bonds complete masked node patterns beyond "
            "frequency and random sparse controls."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--node-count", type=int, default=4096)
    parser.add_argument("--max-active-nodes", type=int, default=48)
    parser.add_argument("--max-train-rows", type=int, default=8192)
    parser.add_argument("--max-eval-rows", type=int, default=512)
    parser.add_argument("--recall-at-k", type=int, default=32)
    parser.add_argument("--min-text-bytes", type=int, default=80)
    parser.add_argument("--cue-node-budget", type=int, default=16)
    parser.add_argument("--target-node-budget", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/IF7_sparse_hebbian_d5_probe.json"),
    )
    parser.add_argument("--experiment-id", default="IF7_sparse_hebbian_d5_probe")
    args = parser.parse_args()

    config = SparseHebbianConfig(
        split=args.split,
        seed=args.seed,
        node_count=args.node_count,
        max_active_nodes=args.max_active_nodes,
        max_train_rows=args.max_train_rows,
        max_eval_rows=args.max_eval_rows,
        recall_at_k=args.recall_at_k,
        min_text_bytes=args.min_text_bytes,
        cue_node_budget=args.cue_node_budget,
        target_node_budget=args.target_node_budget,
        learning_rate=args.learning_rate,
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
