from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.hf_materializer import load_jsonl_texts
from weightlab.if3_block_codebook import evaluate_if3_block_codebook_checkpoint
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
            "scripts/evaluate_if3_block_codebook_validation.py",
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
            "--block-size",
            str(args.block_size),
            "--codebook-size",
            str(args.codebook_size),
            "--residual-fraction",
            str(args.residual_fraction),
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
    parser.add_argument("--batches", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--residual-fraction", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()

    texts = load_jsonl_texts(args.corpus_jsonl, split=args.split)
    if not texts:
        parser.error(f"no {args.split} rows found in {args.corpus_jsonl}")

    metrics = evaluate_if3_block_codebook_checkpoint(
        checkpoint_path=args.checkpoint,
        texts=texts,
        split_name=args.split,
        device=args.device,
        seed=args.seed,
        batches=args.batches,
        block_size=args.block_size,
        codebook_size=args.codebook_size,
        residual_fraction=args.residual_fraction,
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
        "block_size": args.block_size,
        "codebook_size": args.codebook_size,
        "residual_fraction": args.residual_fraction,
        "output": str(args.output),
        "experiment_id": args.experiment_id,
    }

    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="if3_block_codebook_reconstruction_preserves_heldout_loss_better_than_random",
        seed=args.seed,
        command=_command(args),
        metrics=metrics,
        status="completed",
        notes=(
            "IF3 validation-loss probe against a labeled held-out split. The learned and "
            "random codebook reconstructions are evaluated on the same fixed batches as the "
            "FP32 checkpoint. No packed kernel is evaluated."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
