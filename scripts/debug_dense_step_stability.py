from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.corpus import prepare_repository_corpus
from weightlab.dense_training import DenseTrainingConfig, debug_dense_step_stability
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", action="append", type=Path, required=True)
    parser.add_argument("--device", default="rocm")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--mixed-precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--optimizer-name", choices=["adamw", "sgd"], default="sgd")
    parser.add_argument(
        "--attention-mask-mode",
        choices=["none", "bool_causal", "additive_causal"],
        default="additive_causal",
    )
    parser.add_argument("--max-documents", type=int, default=100)
    parser.add_argument("--experiment-id", default="T3_dense_step_stability_debug")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/T3_dense_step_stability_debug.json"),
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    corpus = prepare_repository_corpus(args.repo_path, min_tokens=1)
    repo_by_name = {path.name: path for path in args.repo_path}
    texts: list[str] = []
    for doc in corpus["documents"][: args.max_documents]:
        source_path = repo_by_name[doc["repo"]] / doc["relative_path"]
        texts.append(source_path.read_text(encoding="utf-8", errors="ignore"))

    config = DenseTrainingConfig(
        device=args.device,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
        batch_size=args.batch_size,
        steps=args.steps,
        validation_batches=1,
        gradient_accumulation_steps=1,
        mixed_precision=args.mixed_precision,
        learning_rate=args.learning_rate,
        attention_mask_mode=args.attention_mask_mode,
        optimizer_name=args.optimizer_name,
    )
    metrics = debug_dense_step_stability(
        texts=texts,
        config=config,
        seed=args.seed,
        steps=args.steps,
    )
    status = "completed" if metrics["first_nonfinite_phase"] is None else "failed"
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="dense_step_two_nonfinite_origin",
        seed=args.seed,
        command=(
            "uv run python scripts/debug_dense_step_stability.py "
            + " ".join(f"--repo-path {path}" for path in args.repo_path)
            + f" --device {args.device} --seq-len {args.seq_len}"
            + f" --hidden-dim {args.hidden_dim} --layers {args.layers}"
            + f" --heads {args.heads} --batch-size {args.batch_size}"
            + f" --steps {args.steps} --mixed-precision {args.mixed_precision}"
            + f" --learning-rate {args.learning_rate}"
            + f" --optimizer-name {args.optimizer_name}"
            + f" --attention-mask-mode {args.attention_mask_mode}"
            + f" --max-documents {args.max_documents}"
            + f" --experiment-id {args.experiment_id} --output {args.output}"
        ),
        metrics=metrics,
        status=status,
        notes=(
            "Dense training step-level finite-state debug probe recording inputs, logits, "
            "loss, gradients, parameters, and first nonfinite phase."
        ),
    ).to_jsonable()
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
