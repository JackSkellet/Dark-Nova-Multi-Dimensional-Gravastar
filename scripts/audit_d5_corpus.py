from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.d5_audit import audit_d5_corpus
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
            "scripts/audit_d5_corpus.py",
            "--corpus-jsonl",
            str(args.corpus_jsonl),
            "--near-duplicate-hamming-threshold",
            str(args.near_duplicate_hamming_threshold),
            "--near-duplicate-min-bytes",
            str(args.near_duplicate_min_bytes),
            "--output",
            str(args.output),
            "--experiment-id",
            args.experiment_id,
            "--seed",
            str(args.seed),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the materialized D5 corpus JSONL.")
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--near-duplicate-hamming-threshold", type=int, default=3)
    parser.add_argument("--near-duplicate-min-bytes", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/D5_corpus_audit.json"),
    )
    parser.add_argument("--experiment-id", default="D5_corpus_audit")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    metrics = audit_d5_corpus(
        args.corpus_jsonl,
        near_duplicate_hamming_threshold=args.near_duplicate_hamming_threshold,
        near_duplicate_min_bytes=args.near_duplicate_min_bytes,
    )
    metrics["resolved_config"] = {
        "corpus_jsonl": str(args.corpus_jsonl),
        "near_duplicate_hamming_threshold": args.near_duplicate_hamming_threshold,
        "near_duplicate_min_bytes": args.near_duplicate_min_bytes,
        "output": str(args.output),
        "experiment_id": args.experiment_id,
        "seed": args.seed,
    }
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="d5_corpus_audit",
        seed=args.seed,
        command=_command(args),
        metrics=metrics,
        status="completed",
        notes=(
            "Audit-only pass over the existing materialized D5 JSONL. It does not "
            "download data, mutate corpus rows, or make temporal-holdout claims."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
