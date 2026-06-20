from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from weightlab.hf_materializer import MaterializationConfig, materialize_hf_corpus
from weightlab.metrics import ExperimentRecord, write_json


def _append_manifest(output: Path, record: dict[str, object]) -> None:
    manifest_path = output.parent / "manifest.json"
    if manifest_path.exists():
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        records = []
    experiment_id = str(record["experiment_id"])
    records = [row for row in records if row.get("experiment_id") != experiment_id]
    records.append(record)
    write_json(manifest_path, records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a filtered exploratory HF parquet corpus mirror."
    )
    parser.add_argument("--manifest", type=Path, default=Path("results/D3_hf_corpus_manifest.json"))
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/hf_mirror/exploratory_d3/corpus.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/D4_hf_corpus_materialization.json"),
    )
    parser.add_argument("--experiment-id", default="D4_hf_corpus_materialization")
    parser.add_argument("--target-train-tokens", type=int, default=50_000_000)
    parser.add_argument("--max-train-tokens-per-config", type=int)
    parser.add_argument("--max-row-bytes", type=int, default=256_000)
    parser.add_argument("--min-row-bytes", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--child-metrics", type=Path)
    args = parser.parse_args()

    if args.child:
        _child_main(args)
        os._exit(0)

    tmp_metrics = args.output.with_suffix(".child.json")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--manifest",
        str(args.manifest),
        "--output-jsonl",
        str(args.output_jsonl),
        "--child-metrics",
        str(tmp_metrics),
        "--target-train-tokens",
        str(args.target_train_tokens),
        "--max-row-bytes",
        str(args.max_row_bytes),
        "--min-row-bytes",
        str(args.min_row_bytes),
    ]
    if args.max_train_tokens_per_config is not None:
        command.extend([
            "--max-train-tokens-per-config",
            str(args.max_train_tokens_per_config),
        ])
    if args.max_rows is not None:
        command.extend(["--max-rows", str(args.max_rows)])
    if args.no_resume:
        command.append("--no-resume")
    subprocess.run(command, check=True)

    metrics = json.loads(tmp_metrics.read_text(encoding="utf-8"))
    tmp_metrics.unlink(missing_ok=True)
    recorded_command = [
        "uv",
        "run",
        "python",
        "scripts/materialize_hf_corpus.py",
        "--manifest",
        str(args.manifest),
        "--output-jsonl",
        str(args.output_jsonl),
        "--target-train-tokens",
        str(args.target_train_tokens),
        "--max-row-bytes",
        str(args.max_row_bytes),
        "--min-row-bytes",
        str(args.min_row_bytes),
        "--output",
        str(args.output),
        "--experiment-id",
        args.experiment_id,
        "--seed",
        str(args.seed),
    ]
    if args.max_train_tokens_per_config is not None:
        recorded_command.extend([
            "--max-train-tokens-per-config",
            str(args.max_train_tokens_per_config),
        ])
    if args.max_rows is not None:
        recorded_command.extend(["--max-rows", str(args.max_rows)])
    if args.no_resume:
        recorded_command.append("--no-resume")

    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="real_training_data",
        seed=args.seed,
        command=" ".join(recorded_command),
        metrics=metrics,
        status=metrics["status"],
        notes=(
            "Filtered exploratory Hugging Face parquet corpus mirror materialized through "
            "datasets from Dataset Viewer parquet URLs. The corpus is explicitly not "
            "approved for production, redistribution, or company deployment."
        ),
    ).to_jsonable()
    write_json(args.output, record)
    _append_manifest(args.output, record)


def _child_main(args: argparse.Namespace) -> None:
    manifest_record = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = MaterializationConfig(
        target_train_tokens=args.target_train_tokens,
        max_train_tokens_per_config=args.max_train_tokens_per_config,
        max_row_bytes=args.max_row_bytes,
        min_row_bytes=args.min_row_bytes,
        max_rows=args.max_rows,
    )
    metrics = materialize_hf_corpus(
        manifest_record,
        output_jsonl=args.output_jsonl,
        row_factory=_iter_parquet_rows,
        config=config,
        resume=not args.no_resume,
    )
    if args.child_metrics is None:
        raise ValueError("--child-metrics is required in child mode")
    write_json(args.child_metrics, metrics)


def _iter_parquet_rows(
    _source: dict[str, Any],
    accepted_config: dict[str, Any],
):
    from datasets import load_dataset

    parquet_urls = [row["url"] for row in accepted_config["parquet_files"]]
    dataset = load_dataset(
        "parquet",
        data_files={"train": parquet_urls},
        split="train",
        streaming=True,
        cache_dir="data/hf_cache",
    )
    yield from dataset


if __name__ == "__main__":
    main()
