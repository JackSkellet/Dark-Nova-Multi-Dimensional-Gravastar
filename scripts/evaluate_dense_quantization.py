from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.dense_checkpoint_quantization import (
    QuantizationEvalConfig,
    evaluate_checkpoint_quantization,
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
            "scripts/evaluate_dense_quantization.py",
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
            "--batches",
            str(args.batches),
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
    parser.add_argument("--seed", type=int, default=424243)
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--protected-fraction", type=float, default=0.01)
    parser.add_argument("--sparse-fraction", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()

    texts = load_jsonl_texts(args.corpus_jsonl, split=args.split)
    if not texts:
        parser.error(f"no {args.split} rows found in {args.corpus_jsonl}")

    config = QuantizationEvalConfig(
        device=args.device,
        seed=args.seed,
        batches=args.batches,
        protected_fraction=args.protected_fraction,
        sparse_fraction=args.sparse_fraction,
    )
    metrics = evaluate_checkpoint_quantization(
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
        "batches": args.batches,
        "protected_fraction": args.protected_fraction,
        "sparse_fraction": args.sparse_fraction,
        "output": str(args.output),
        "experiment_id": args.experiment_id,
    }

    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="trained_checkpoint_quantization_with_random_controls",
        seed=args.seed,
        command=_command(args),
        metrics=metrics,
        status="completed",
        notes=(
            "Evaluation-only quantization probe on a trained T11 checkpoint. Quantized weights "
            "are reconstructed into FP32 tensors for PyTorch evaluation; no packed kernel is used."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
