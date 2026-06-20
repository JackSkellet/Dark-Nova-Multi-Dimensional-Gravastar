from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.hf_materializer import load_jsonl_texts
from weightlab.metrics import ExperimentRecord, write_json
from weightlab.tokenizer_compare import compare_tokenizers


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
    parser = argparse.ArgumentParser(
        description="Compare byte-tokenizer and trained byte-pair tokenizer efficiency."
    )
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--target-vocab-size", type=int, default=1024)
    parser.add_argument("--max-train-texts", type=int)
    parser.add_argument("--max-eval-documents", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/D5_tokenizer_efficiency_comparison.json"),
    )
    parser.add_argument("--experiment-id", default="D5_tokenizer_efficiency_comparison")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    train_texts = load_jsonl_texts(
        args.corpus_jsonl,
        max_documents=args.max_train_texts,
        split="train",
    )
    eval_splits = {
        split: load_jsonl_texts(
            args.corpus_jsonl,
            max_documents=args.max_eval_documents,
            split=split,
        )
        for split in ["train", "validation", "test"]
    }
    metrics = compare_tokenizers(
        train_texts,
        eval_splits,
        target_vocab_size=args.target_vocab_size,
        max_train_texts=args.max_train_texts,
    )
    metrics["corpus"] = {
        "jsonl_path": str(args.corpus_jsonl),
        "max_eval_documents": args.max_eval_documents,
    }
    metrics["resolved_config"] = {
        "corpus_jsonl": str(args.corpus_jsonl),
        "target_vocab_size": args.target_vocab_size,
        "max_train_texts": args.max_train_texts,
        "max_eval_documents": args.max_eval_documents,
        "output": str(args.output),
        "experiment_id": args.experiment_id,
        "seed": args.seed,
    }
    command = (
        "uv run python scripts/compare_tokenizers.py "
        f"--corpus-jsonl {args.corpus_jsonl} "
        f"--target-vocab-size {args.target_vocab_size} "
        f"--output {args.output} "
        f"--experiment-id {args.experiment_id} "
        f"--seed {args.seed}"
    )
    if args.max_train_texts is not None:
        command += f" --max-train-texts {args.max_train_texts}"
    if args.max_eval_documents is not None:
        command += f" --max-eval-documents {args.max_eval_documents}"
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="code_tokenizer_efficiency",
        seed=args.seed,
        command=command,
        metrics=metrics,
        status="completed",
        notes=(
            "Tokenizer efficiency comparison only. It does not train a language model "
            "with the byte-pair tokenizer yet."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
