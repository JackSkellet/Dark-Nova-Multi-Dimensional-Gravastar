from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.corpus import prepare_repository_corpus
from weightlab.dense_training import DenseTrainingConfig, train_dense_decoder
from weightlab.hf_materializer import load_jsonl_texts
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
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
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

    max_documents = None if args.max_documents <= 0 else args.max_documents
    if args.corpus_jsonl is not None:
        texts = load_jsonl_texts(args.corpus_jsonl, max_documents=max_documents)
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
        optimizer_name=args.optimizer_name,
        progress_interval=args.progress_interval,
        checkpoint_interval=args.checkpoint_interval,
    )
    metrics = train_dense_decoder(
        texts,
        config,
        args.output_dir,
        seed=args.seed,
        resume_checkpoint=args.resume_checkpoint,
    )
    metrics["corpus"] = {
        "source": corpus.get("source", "local_repositories"),
        "jsonl_path": corpus.get("jsonl_path", ""),
        "repo_count": corpus["repo_count"],
        "document_count_used": len(texts),
        "available_documents": corpus["document_count"],
        "available_tokens": corpus["total_tokens"],
        "licenses": corpus["license_counts"],
        "languages": corpus["languages"],
        "file_roles": corpus["file_roles"],
        "record": corpus.get("record", {}),
    }
    command_repos = " ".join(f"--repo-path {path}" for path in args.repo_path)
    command_corpus = f"--corpus-jsonl {args.corpus_jsonl} " if args.corpus_jsonl else ""
    command_corpus_record = f"--corpus-record {args.corpus_record} " if args.corpus_record else ""
    command_resume = (
        f"--resume-checkpoint {args.resume_checkpoint} " if args.resume_checkpoint else ""
    )
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="dense_baseline_training",
        seed=args.seed,
        command=(
            "uv run python scripts/train_dense_decoder.py "
            f"{command_corpus}{command_corpus_record}{command_resume}{command_repos} "
            f"--device {args.device} --seq-len {args.seq_len} "
            f"--hidden-dim {args.hidden_dim} --layers {args.layers} --heads {args.heads} "
            f"--batch-size {args.batch_size} --steps {args.steps} "
            f"--validation-batches {args.validation_batches} "
            f"--gradient-accumulation-steps {args.gradient_accumulation_steps} "
            f"--max-file-bytes {args.max_file_bytes} "
            f"--mixed-precision {args.mixed_precision} --output-dir {args.output_dir} "
            f"--progress-interval {args.progress_interval} "
            f"--checkpoint-interval {args.checkpoint_interval} "
            f"--output {args.output}"
        ),
        metrics=metrics,
        status="completed" if metrics["status"] == "completed" else "failed",
        notes=(
            "Initial configurable dense decoder training smoke from random initialization "
            "on the filtered local repository corpus. This validates the pipeline only; "
            "it is not the required 10-50M parameter or 50M-token run."
        ),
    ).to_jsonable()
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
