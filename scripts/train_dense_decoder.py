from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.corpus import prepare_repository_corpus
from weightlab.dense_training import DenseTrainingConfig, train_dense_decoder
from weightlab.hf_materializer import load_jsonl_texts
from weightlab.metrics import ExperimentRecord, write_json


def _jsonl_split_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            split = str(row.get("split", "unlabeled"))
            counts[split] = counts.get(split, 0) + 1
    return counts


def _resolved_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "repo_path": [str(path) for path in args.repo_path],
        "corpus_jsonl": str(args.corpus_jsonl) if args.corpus_jsonl else "",
        "corpus_record": str(args.corpus_record) if args.corpus_record else "",
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else "",
        "device": args.device,
        "seq_len": args.seq_len,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "heads": args.heads,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "validation_batches": args.validation_batches,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "mixed_precision": args.mixed_precision,
        "learning_rate": args.learning_rate,
        "optimizer_name": args.optimizer_name,
        "attention_mask_mode": args.attention_mask_mode,
        "progress_interval": args.progress_interval,
        "checkpoint_interval": args.checkpoint_interval,
        "architecture_variant": args.architecture_variant,
        "adapter_dim": args.adapter_dim,
        "max_documents": args.max_documents,
        "max_documents_resolved": None if args.max_documents <= 0 else args.max_documents,
        "max_file_bytes": args.max_file_bytes,
        "output_dir": str(args.output_dir),
        "output": str(args.output),
        "experiment_id": args.experiment_id,
        "seed": args.seed,
    }


def _command_from_resolved(config: dict[str, object]) -> str:
    parts = ["uv", "run", "python", "scripts/train_dense_decoder.py"]
    for path in config["repo_path"]:
        parts.extend(["--repo-path", str(path)])
    optional_paths = {
        "--corpus-jsonl": config["corpus_jsonl"],
        "--corpus-record": config["corpus_record"],
        "--resume-checkpoint": config["resume_checkpoint"],
    }
    for flag, value in optional_paths.items():
        if value:
            parts.extend([flag, str(value)])
    for flag, key in [
        ("--device", "device"),
        ("--seq-len", "seq_len"),
        ("--hidden-dim", "hidden_dim"),
        ("--layers", "layers"),
        ("--heads", "heads"),
        ("--batch-size", "batch_size"),
        ("--steps", "steps"),
        ("--validation-batches", "validation_batches"),
        ("--gradient-accumulation-steps", "gradient_accumulation_steps"),
        ("--mixed-precision", "mixed_precision"),
        ("--learning-rate", "learning_rate"),
        ("--optimizer-name", "optimizer_name"),
        ("--attention-mask-mode", "attention_mask_mode"),
        ("--progress-interval", "progress_interval"),
        ("--checkpoint-interval", "checkpoint_interval"),
        ("--architecture-variant", "architecture_variant"),
        ("--adapter-dim", "adapter_dim"),
        ("--max-documents", "max_documents"),
        ("--max-file-bytes", "max_file_bytes"),
        ("--output-dir", "output_dir"),
        ("--output", "output"),
        ("--experiment-id", "experiment_id"),
        ("--seed", "seed"),
    ]:
        parts.extend([flag, str(config[key])])
    return " ".join(parts)


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
    parser.add_argument("--repo-path", action="append", type=Path, default=[])
    parser.add_argument("--corpus-jsonl", type=Path)
    parser.add_argument("--corpus-record", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--device", default="rocm")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--optimizer-name", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument(
        "--attention-mask-mode",
        choices=["none", "bool_causal", "additive_causal"],
        default="additive_causal",
    )
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--architecture-variant", choices=["dense", "adapter"], default="dense")
    parser.add_argument("--adapter-dim", type=int, default=0)
    parser.add_argument("--max-documents", type=int, default=128)
    parser.add_argument("--max-file-bytes", type=int, default=256_000)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dense_decoder_smoke"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/T1_dense_decoder_training_smoke.json"),
    )
    parser.add_argument("--experiment-id", default="T1_dense_decoder_training_smoke")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if not args.repo_path and args.corpus_jsonl is None:
        parser.error("provide at least one --repo-path or --corpus-jsonl")

    resolved_config = _resolved_config(args)
    write_json(args.output_dir / "resolved_config.json", resolved_config)
    max_documents = None if args.max_documents <= 0 else args.max_documents
    validation_texts = None
    if args.corpus_jsonl is not None:
        split_counts = _jsonl_split_counts(args.corpus_jsonl)
        texts = load_jsonl_texts(args.corpus_jsonl, max_documents=max_documents, split="train")
        validation_texts = load_jsonl_texts(args.corpus_jsonl, split="validation")
        if not texts:
            texts = load_jsonl_texts(args.corpus_jsonl, max_documents=max_documents)
        if not validation_texts:
            validation_texts = None
        corpus_record = (
            json.loads(args.corpus_record.read_text(encoding="utf-8"))
            if args.corpus_record is not None
            else {}
        )
        corpus_metrics = corpus_record.get("metrics", {})
        corpus = {
            "source": "hf_jsonl_mirror",
            "jsonl_path": str(args.corpus_jsonl),
            "document_count": len(texts),
            "total_tokens": sum(len(text.encode("utf-8", errors="ignore")) + 1 for text in texts),
            "repo_count": 0,
            "split_document_counts": split_counts,
            "train_document_count": len(texts),
            "validation_document_count": len(validation_texts or []),
            "license_counts": corpus_metrics.get("licenses", {}),
            "languages": corpus_metrics.get("languages", {}),
            "file_roles": {},
            "record": {
                "experiment_id": corpus_record.get("experiment_id", ""),
                "git_commit": corpus_record.get("git_commit", ""),
                "output_sha256": corpus_metrics.get("output", {}).get("sha256", ""),
                "corpus_use": corpus_metrics.get("corpus_use", ""),
                "source_manifest": corpus_metrics.get("source_manifest", {}),
                "dataset_config_counts": corpus_metrics.get("dataset_config_counts", {}),
                "tokens": corpus_metrics.get("tokens", {}),
            },
        }
    else:
        split_counts = {}
        corpus = prepare_repository_corpus(
            args.repo_path,
            min_tokens=1,
            max_file_bytes=args.max_file_bytes,
        )
        repo_by_name = {path.name: path for path in args.repo_path}
        texts = []
        for doc in corpus["documents"][:max_documents]:
            path = repo_by_name[doc["repo"]] / doc["relative_path"]
            texts.append(path.read_text(encoding="utf-8", errors="ignore"))

    config = DenseTrainingConfig(
        device=args.device,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
        batch_size=args.batch_size,
        steps=args.steps,
        validation_batches=args.validation_batches,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        learning_rate=args.learning_rate,
        attention_mask_mode=args.attention_mask_mode,
        optimizer_name=args.optimizer_name,
        progress_interval=args.progress_interval,
        checkpoint_interval=args.checkpoint_interval,
        architecture_variant=args.architecture_variant,
        adapter_dim=args.adapter_dim,
    )
    metrics = train_dense_decoder(
        texts,
        config,
        args.output_dir,
        seed=args.seed,
        resume_checkpoint=args.resume_checkpoint,
        validation_texts=validation_texts,
    )
    metrics["resolved_config"] = resolved_config
    metrics["corpus"] = {
        "source": corpus.get("source", "local_repositories"),
        "jsonl_path": corpus.get("jsonl_path", ""),
        "repo_count": corpus["repo_count"],
        "document_count_used": len(texts),
        "validation_document_count_used": len(validation_texts or []),
        "split_document_counts": corpus.get("split_document_counts", split_counts),
        "available_documents": corpus["document_count"],
        "available_tokens": corpus["total_tokens"],
        "licenses": corpus["license_counts"],
        "languages": corpus["languages"],
        "file_roles": corpus["file_roles"],
        "record": corpus.get("record", {}),
    }
    command = _command_from_resolved(resolved_config)
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="dense_baseline_training",
        seed=args.seed,
        command=command,
        metrics=metrics,
        status="completed" if metrics["status"] == "completed" else "failed",
        notes=(
            "Initial configurable dense decoder training smoke from random initialization "
            "on the filtered local repository corpus. This validates the pipeline only; "
            "it is not the required 10-50M parameter or 50M-token run."
        ),
    ).to_jsonable()
    record["resolved_config"] = resolved_config
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
