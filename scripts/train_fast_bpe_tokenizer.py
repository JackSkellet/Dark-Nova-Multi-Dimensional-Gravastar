from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from weightlab.fast_tokenizer import train_fast_bpe_tokenizer, write_fast_bpe_tokenizer
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_split_texts(
    path: Path,
    *,
    split: str,
    max_documents: int,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    rng = random.Random(seed)
    texts: list[str] = []
    seen = 0
    text_order_hash = hashlib.sha256()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != split:
                continue
            text = str(row["text"])
            seen += 1
            if len(texts) < max_documents:
                texts.append(text)
            else:
                index = rng.randrange(seen)
                if index < max_documents:
                    texts[index] = text
    for text in texts:
        text_order_hash.update(text.encode("utf-8", errors="ignore"))
        text_order_hash.update(b"\0")
    return texts, {
        "split": split,
        "available_document_count": seen,
        "sampled_document_count": len(texts),
        "max_train_documents": max_documents,
        "deterministic_seed": seed,
        "sample_text_order_sha256": text_order_hash.hexdigest(),
        "sample_bytes": sum(len(text.encode("utf-8", errors="ignore")) for text in texts),
    }


def _command(args: argparse.Namespace) -> str:
    return (
        "uv run python scripts/train_fast_bpe_tokenizer.py "
        f"--corpus-jsonl {args.corpus_jsonl} "
        f"--split {args.split} "
        f"--max-train-documents {args.max_train_documents} "
        f"--vocab-size {args.vocab_size} "
        f"--min-frequency {args.min_frequency} "
        f"--seed {args.seed} "
        f"--tokenizer-output {args.tokenizer_output} "
        f"--output {args.output} "
        f"--experiment-id {args.experiment_id}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-train-documents", type=int, default=4096)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--tokenizer-output",
        type=Path,
        default=Path("artifacts/D5_fast_bpe_tokenizer.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/D5_fast_bpe_tokenizer.json"),
    )
    parser.add_argument("--experiment-id", default="D5_fast_bpe_tokenizer")
    args = parser.parse_args()

    if args.max_train_documents <= 0:
        parser.error("--max-train-documents must be positive")

    texts, sampling = _sample_split_texts(
        args.corpus_jsonl,
        split=args.split,
        max_documents=args.max_train_documents,
        seed=args.seed,
    )
    if not texts:
        parser.error(f"no {args.split} rows found in {args.corpus_jsonl}")

    tokenizer = train_fast_bpe_tokenizer(
        texts,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    resolved_config = {
        "corpus_jsonl": str(args.corpus_jsonl),
        "corpus_sha256": _sha256_file(args.corpus_jsonl),
        "split": args.split,
        "max_train_documents": args.max_train_documents,
        "vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "seed": args.seed,
        "tokenizer_output": str(args.tokenizer_output),
        "output": str(args.output),
        "experiment_id": args.experiment_id,
    }
    training_config = {
        **resolved_config,
        **sampling,
    }
    write_fast_bpe_tokenizer(
        args.tokenizer_output,
        tokenizer,
        training_config=training_config,
    )
    metrics = {
        "benchmark_label": "fast_bpe_tokenizer_training",
        "corpus": {
            "jsonl_path": str(args.corpus_jsonl),
            "sha256": resolved_config["corpus_sha256"],
        },
        "sampling": sampling,
        "tokenizer": tokenizer.to_jsonable(),
        "resolved_config": resolved_config,
        "limitations": [
            "tokenizer_training_only",
            "no_model_training_with_bpe_in_this_record",
            "representative_sample_depends_on_available_d5_rows",
        ],
    }
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="fast_reproducible_code_bpe_tokenizer",
        seed=args.seed,
        command=_command(args),
        metrics=metrics,
        status="completed",
        notes=(
            "Trains and records a fast Hugging Face tokenizers byte-level BPE artifact "
            "from a deterministic D5 split sample. This does not itself establish model quality."
        ),
    ).to_jsonable()
    record["resolved_config"] = resolved_config
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
