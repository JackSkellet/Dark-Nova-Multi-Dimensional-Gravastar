from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.metrics import ExperimentRecord, write_json
from weightlab.t12_analysis import summarize_t12_second_seed_pair


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
        default=Path("results/T12_second_seed_summary.json"),
    )
    parser.add_argument("--experiment-id", default="T12_second_seed_summary")
    args = parser.parse_args()

    metrics = summarize_t12_second_seed_pair(args.results_dir)
    metrics["resolved_config"] = {
        "results_dir": str(args.results_dir),
        "output": str(args.output),
        "experiment_id": args.experiment_id,
    }
    command = (
        "uv run python scripts/summarize_t12_second_seed.py "
        f"--results-dir {args.results_dir} --output {args.output} "
        f"--experiment-id {args.experiment_id}"
    )
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="t12_second_seed_dense528_vs_adapter528",
        seed=456,
        command=command,
        metrics=metrics,
        status="completed",
        notes=(
            "Summarizes seed 123 and seed 456 dense-528 versus residual-adapter-528 "
            "results. Validation metrics drive selection; test losses are reported only."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
