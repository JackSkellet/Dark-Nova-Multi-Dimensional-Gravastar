from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.if3_block_codebook import run_if3_block_codebook_probe
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--residual-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    metrics = run_if3_block_codebook_probe(
        args.checkpoint,
        block_size=args.block_size,
        codebook_size=args.codebook_size,
        residual_fraction=args.residual_fraction,
        seed=args.seed,
    )
    metrics["resolved_config"] = {
        "checkpoint": str(args.checkpoint),
        "output": str(args.output),
        "experiment_id": args.experiment_id,
        "block_size": args.block_size,
        "codebook_size": args.codebook_size,
        "residual_fraction": args.residual_fraction,
        "seed": args.seed,
    }
    command = (
        "uv run python scripts/run_if3_block_codebook_probe.py "
        f"--checkpoint {args.checkpoint} --output {args.output} "
        f"--experiment-id {args.experiment_id} --block-size {args.block_size} "
        f"--codebook-size {args.codebook_size} "
        f"--residual-fraction {args.residual_fraction} --seed {args.seed}"
    )
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="if3_block_codebook_can_beat_random_control_with_accounted_metadata",
        seed=args.seed,
        command=command,
        metrics=metrics,
        status="completed",
        notes=(
            "IF3 reconstruction-only block-codebook compression proxy on a trained "
            "checkpoint. It counts metadata and runtime buffers, but does not evaluate "
            "language-model loss or packed kernels."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
