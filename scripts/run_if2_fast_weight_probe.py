from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.if2_fast_weights import run_if2_fast_weight_probe
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/IF2_fast_weight_continual_probe.json"),
    )
    parser.add_argument("--experiment-id", default="IF2_fast_weight_continual_probe")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    metrics = run_if2_fast_weight_probe(seed=args.seed)
    metrics["resolved_config"] = {
        "output": str(args.output),
        "experiment_id": args.experiment_id,
        "seed": args.seed,
    }
    command = (
        "uv run python scripts/run_if2_fast_weight_probe.py "
        f"--output {args.output} --experiment-id {args.experiment_id} "
        f"--seed {args.seed}"
    )
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="if2_fast_weights_add_value_beyond_updated_memory",
        seed=args.seed,
        command=command,
        metrics=metrics,
        status="completed",
        notes=(
            "Synthetic fast-weight continual-learning proxy. It tests whether a "
            "parameter-like update adds held-out paraphrase value beyond updated memory."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
