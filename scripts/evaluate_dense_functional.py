from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.dense_functional_eval import (
    FunctionalEvalConfig,
    evaluate_dense_functional_checkpoint,
)
from weightlab.hf_materializer import load_jsonl_texts
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


def _command(args: argparse.Namespace) -> str:
    return " ".join(
        [
            "uv",
            "run",
            "python",
            "scripts/evaluate_dense_functional.py",
            "--checkpoint",
            str(args.checkpoint),
            "--corpus-jsonl",
            str(args.corpus_jsonl),
            "--split",
            args.split,
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--tasks-per-kind",
            str(args.tasks_per_kind),
            "--output",
            str(args.output),
            "--experiment-id",
            args.experiment_id,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--device", default="rocm")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--tasks-per-kind", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()

    texts = load_jsonl_texts(args.corpus_jsonl, split=args.split)
    if not texts:
        parser.error(f"no {args.split} rows found in {args.corpus_jsonl}")

    config = FunctionalEvalConfig(
        device=args.device,
        seed=args.seed,
        tasks_per_kind=args.tasks_per_kind,
    )
    metrics = evaluate_dense_functional_checkpoint(
        checkpoint_path=args.checkpoint,
        texts=texts,
        split_name=args.split,
        config=config,
    )
    metrics["corpus"] = {
        "source": "hf_jsonl_mirror",
        "jsonl_path": str(args.corpus_jsonl),
        "split": args.split,
        "document_count_used": len(texts),
        "split_texts_loaded_separately": True,
    }
    metrics["resolved_config"] = {
        "checkpoint": str(args.checkpoint),
        "corpus_jsonl": str(args.corpus_jsonl),
        "split": args.split,
        "device": args.device,
        "seed": args.seed,
        "tasks_per_kind": args.tasks_per_kind,
        "output": str(args.output),
        "experiment_id": args.experiment_id,
    }

    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="heldout_d4_javascript_functional_checkpoint_evaluation",
        seed=args.seed,
        command=_command(args),
        metrics=metrics,
        status="completed",
        notes=(
            "Evaluation-only deterministic functional probe against a labeled D4 split loaded "
            "separately from training rows. No optimizer step or retraining is performed."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
