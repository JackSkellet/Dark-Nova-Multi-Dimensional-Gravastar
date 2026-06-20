from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.metrics import ExperimentRecord, write_json
from weightlab.rocm_validation import validate_training_runtime


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
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--steps-per-batch", type=int, default=3)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("artifacts/rocm_validation/resume.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/R1_rocm_training_validation.json"),
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    metrics = validate_training_runtime(
        device=args.device,
        batch_sizes=batch_sizes,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        vocab_size=args.vocab_size,
        steps_per_batch=args.steps_per_batch,
        checkpoint_path=args.checkpoint_path,
        seed=args.seed,
    )
    record = ExperimentRecord(
        experiment_id="R1_rocm_training_validation",
        hypothesis="real_training_readiness",
        seed=args.seed,
        command=(
            "uv run python scripts/run_rocm_training_validation.py "
            f"--device {args.device} --batch-sizes {args.batch_sizes} "
            f"--seq-len {args.seq_len} --hidden-dim {args.hidden_dim} "
            f"--vocab-size {args.vocab_size} --steps-per-batch {args.steps_per_batch} "
            f"--checkpoint-path {args.checkpoint_path} --output {args.output}"
        ),
        metrics=metrics,
        notes=(
            "Local training-readiness probe for ROCm/CPU runtime, precision support, "
            "batch throughput, stable token count, memory reporting, and checkpoint resume."
        ),
    ).to_jsonable()
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
