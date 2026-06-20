from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from weightlab.metrics import ExperimentRecord, write_json
from weightlab.real_model_task_precision import evaluate_tiny_gpt2_natural_text_precision

DEFAULT_MODEL_ID = "sshleifer/tiny-gpt2"
DEFAULT_MODEL_COMMIT = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"
DEFAULT_CHECKPOINT_FILENAME = "pytorch_model.bin"
DEFAULT_VOCAB_FILENAME = "vocab.json"
DEFAULT_MERGES_FILENAME = "merges.txt"

CALIBRATION_TEXTS = [
    "The repository parser loads configuration files and returns structured options.",
    "Unit tests should run locally before a patch is accepted into the project.",
    "A stale document mentions a removed API name after the source code is changed.",
    "The audit log records authorization decisions without sending telemetry.",
    "Refactoring should preserve public behavior while simplifying internal code paths.",
    "The local assistant answers questions using repository symbols and nearby context.",
]

HELDOUT_TEXTS = [
    "Configuration documentation should match the parser implementation.",
    "Patch candidates are graded by running the local test suite.",
    "Repository question answering depends on symbols, files, and commit history.",
    "Security gates reject sensitive context before it reaches the prompt.",
    "A changelog summarizes behavior changes without exposing private data.",
    "Offline operation avoids cloud inference, cloud embeddings, and external telemetry.",
]


def _download_model_file(
    model_id: str,
    model_commit: str,
    filename: str,
    destination: Path,
) -> None:
    url = f"https://huggingface.co/{model_id}/resolve/{model_commit}/{filename}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def _require_or_download(
    path: Path,
    model_id: str,
    model_commit: str,
    filename: str,
    allow_download: bool,
) -> None:
    if path.exists():
        return
    if not allow_download:
        raise SystemExit(f"missing {filename} at {path}; rerun with --allow-download")
    _download_model_file(model_id, model_commit, filename, path)


def _append_manifest(output_dir: Path, record: dict[str, object]) -> None:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
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
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("data/raw/public/tiny-gpt2/pytorch_model.bin"),
    )
    parser.add_argument(
        "--tokenizer-vocab-path",
        type=Path,
        default=Path("data/raw/public/tiny-gpt2/vocab.json"),
    )
    parser.add_argument(
        "--tokenizer-merges-path",
        type=Path,
        default=Path("data/raw/public/tiny-gpt2/merges.txt"),
    )
    parser.add_argument("--checkpoint-filename", default=DEFAULT_CHECKPOINT_FILENAME)
    parser.add_argument("--vocab-filename", default=DEFAULT_VOCAB_FILENAME)
    parser.add_argument("--merges-filename", default=DEFAULT_MERGES_FILENAME)
    parser.add_argument("--tensor-name", default="transformer.wte.weight")
    parser.add_argument(
        "--candidate-row-strategy",
        choices=["token_ids", "all_rows"],
        default="token_ids",
    )
    parser.add_argument("--experiment-id", default="E3j_real_open_model_natural_text_precision")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--protected-count", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    _require_or_download(
        args.checkpoint_path,
        args.model_id,
        args.model_commit,
        args.checkpoint_filename,
        args.allow_download,
    )
    _require_or_download(
        args.tokenizer_vocab_path,
        args.model_id,
        args.model_commit,
        args.vocab_filename,
        args.allow_download,
    )
    _require_or_download(
        args.tokenizer_merges_path,
        args.model_id,
        args.model_commit,
        args.merges_filename,
        args.allow_download,
    )

    metrics = evaluate_tiny_gpt2_natural_text_precision(
        checkpoint_path=args.checkpoint_path,
        tensor_name=args.tensor_name,
        tokenizer_vocab_path=args.tokenizer_vocab_path,
        tokenizer_merges_path=args.tokenizer_merges_path,
        calibration_texts=CALIBRATION_TEXTS,
        heldout_texts=HELDOUT_TEXTS,
        sequence_length=args.sequence_length,
        model_id=args.model_id,
        model_commit=args.model_commit,
        seed=args.seed,
        protected_count=args.protected_count,
        n_layer=args.n_layer,
        n_head=args.n_head,
        candidate_row_strategy=args.candidate_row_strategy,
    )
    command = (
        "uv run python scripts/run_real_model_natural_text_precision.py "
        f"--model-id {args.model_id} "
        f"--model-commit {args.model_commit} "
        f"--checkpoint-path {args.checkpoint_path} "
        f"--tokenizer-vocab-path {args.tokenizer_vocab_path} "
        f"--tokenizer-merges-path {args.tokenizer_merges_path} "
        f"--checkpoint-filename {args.checkpoint_filename} "
        f"--vocab-filename {args.vocab_filename} "
        f"--merges-filename {args.merges_filename} "
        f"--tensor-name {args.tensor_name} "
        f"--candidate-row-strategy {args.candidate_row_strategy} "
        f"--experiment-id {args.experiment_id} "
        f"--seed {args.seed} "
        f"--protected-count {args.protected_count} "
        f"--sequence-length {args.sequence_length} "
        f"--n-layer {args.n_layer} "
        f"--n-head {args.n_head}"
    )
    if args.allow_download:
        command += " --allow-download"
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="H3",
        seed=args.seed,
        command=command,
        metrics=metrics,
        notes=(
            "Pinned tiny-gpt2 checkpoint evaluated with byte-level GPT-2 BPE tokenized "
            "natural-language repository/code prose. This addresses the E3i token-id "
            "smoke-test limitation but remains a tiny-model CPU experiment."
        ),
    ).to_jsonable()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"{args.experiment_id}.json", record)
    _append_manifest(args.output_dir, record)


if __name__ == "__main__":
    main()
