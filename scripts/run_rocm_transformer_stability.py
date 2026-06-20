from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.metrics import ExperimentRecord, write_json
from weightlab.rocm_transformer_stability import (
    StabilityCase,
    default_stability_cases,
    run_transformer_stability_microbench,
)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="rocm")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=257)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--include-large-threshold", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/R2_rocm_transformer_stability.json"),
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    cases = default_stability_cases()
    if args.include_large_threshold:
        cases.extend(
            [
                StabilityCase(
                    component="transformer",
                    dtype="fp32",
                    mask_mode="bool_causal",
                    operation_stage="optimizer",
                    batch_size=16,
                    seq_len=128,
                    hidden_dim=128,
                    heads=4,
                ),
                StabilityCase(
                    component="transformer",
                    dtype="bf16",
                    mask_mode="bool_causal",
                    operation_stage="optimizer",
                    batch_size=16,
                    seq_len=128,
                    hidden_dim=128,
                    heads=4,
                ),
            ]
        )
    metrics = run_transformer_stability_microbench(
        device=args.device,
        cases=cases,
        steps=args.steps,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        vocab_size=args.vocab_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    status = "completed" if int(metrics["failure_count"]) == 0 else "failed"
    record = ExperimentRecord(
        experiment_id="R2_rocm_transformer_stability",
        hypothesis="rocm_transformer_failure_is_component_or_kernel_path_specific",
        seed=args.seed,
        command=(
            "uv run python scripts/run_rocm_transformer_stability.py "
            f"--device {args.device} --steps {args.steps} "
            f"--hidden-dim {args.hidden_dim} --heads {args.heads} "
            f"--vocab-size {args.vocab_size} --learning-rate {args.learning_rate} "
            f"--output {args.output}"
            + (" --include-large-threshold" if args.include_large_threshold else "")
        ),
        metrics=metrics,
        status=status,
        notes=(
            "Component-isolation ROCm transformer stability microbench for embedding, "
            "MLP, attention, mask, dtype, optimizer, and size-threshold paths."
        ),
    ).to_jsonable()
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
