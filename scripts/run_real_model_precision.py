from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from weightlab.metrics import ExperimentRecord, write_json
from weightlab.real_model_precision import evaluate_real_model_matrix_precision

DEFAULT_MODEL_ID = "sshleifer/tiny-gpt2"
DEFAULT_MODEL_COMMIT = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"
DEFAULT_FILENAME = "pytorch_model.bin"


def _download_checkpoint(
    model_id: str,
    model_commit: str,
    filename: str,
    checkpoint_path: Path,
) -> None:
    url = f"https://huggingface.co/{model_id}/resolve/{model_commit}/{filename}"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, checkpoint_path)


def _append_manifest(output_dir: Path, record: dict[str, object]) -> None:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        import json

        records = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        records = []
    records = [entry for entry in records if entry.get("experiment_id") != record["experiment_id"]]
    records.append(record)
    write_json(manifest_path, records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-commit", default=DEFAULT_MODEL_COMMIT)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("data/raw/public/tiny-gpt2/pytorch_model.bin"),
    )
    parser.add_argument("--tensor-name", default="transformer.h.0.mlp.c_fc.weight")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--protected-count", type=int, default=1)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint_path.exists():
        if not args.allow_download:
            raise SystemExit(
                f"checkpoint missing at {args.checkpoint_path}; rerun with --allow-download"
            )
        _download_checkpoint(
            model_id=args.model_id,
            model_commit=args.model_commit,
            filename=args.filename,
            checkpoint_path=args.checkpoint_path,
        )

    metrics = evaluate_real_model_matrix_precision(
        checkpoint_path=args.checkpoint_path,
        tensor_name=args.tensor_name,
        model_id=args.model_id,
        model_commit=args.model_commit,
        seed=args.seed,
        protected_count=args.protected_count,
    )
    command = (
        "uv run python scripts/run_real_model_precision.py "
        f"--model-id {args.model_id} "
        f"--model-commit {args.model_commit} "
        f"--filename {args.filename} "
        f"--checkpoint-path {args.checkpoint_path} "
        f"--tensor-name {args.tensor_name} "
        f"--seed {args.seed} "
        f"--protected-count {args.protected_count}"
    )
    if args.allow_download:
        command += " --allow-download"
    record = ExperimentRecord(
        experiment_id="E3h_real_open_model_matrix_precision",
        hypothesis="H3",
        seed=args.seed,
        command=command,
        metrics=metrics,
        notes=(
            "Explicitly downloaded pinned Hugging Face tiny-gpt2 checkpoint and evaluated "
            "selective precision on a real open-model matrix. Measures reconstruction "
            "error, not task loss."
        ),
    ).to_jsonable()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "E3h_real_open_model_matrix_precision.json", record)
    _append_manifest(args.output_dir, record)


if __name__ == "__main__":
    main()
