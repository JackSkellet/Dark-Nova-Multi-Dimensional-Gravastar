from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.d5_tokenizer_training_analysis import summarize_d5_tokenizer_training
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
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/D5_tokenizer_training_comparison.json"),
    )
    parser.add_argument("--experiment-id", default="D5_tokenizer_training_comparison")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    metrics = summarize_d5_tokenizer_training(args.results_dir)
    metrics["resolved_config"] = {
        "results_dir": str(args.results_dir),
        "output": str(args.output),
        "experiment_id": args.experiment_id,
        "seed": args.seed,
    }
    command = (
        "uv run python scripts/summarize_d5_tokenizer_training.py "
        f"--results-dir {args.results_dir} --output {args.output} "
        f"--experiment-id {args.experiment_id} --seed {args.seed}"
    )
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="d5_fast_bpe_must_improve_loss_per_byte_not_only_token_count",
        seed=args.seed,
        command=command,
        metrics=metrics,
        status="completed",
        notes=(
            "Summarizes matched D5 dense-528 byte-tokenizer and fast-BPE pilots. "
            "Token reduction alone is not treated as a quality win."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
