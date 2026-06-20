from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.corpus import prepare_repository_corpus
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
    parser.add_argument(
        "--repo-path",
        action="append",
        type=Path,
        required=True,
        help="Local public or approved repository path. Repeat for multiple repos.",
    )
    parser.add_argument("--min-tokens", type=int, default=50_000_000)
    parser.add_argument("--max-file-bytes", type=int, default=256_000)
    parser.add_argument("--output", type=Path, default=Path("results/D1_corpus_preparation.json"))
    parser.add_argument("--experiment-id", default="D1_corpus_preparation")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    metrics = prepare_repository_corpus(
        repo_paths=args.repo_path,
        min_tokens=args.min_tokens,
        max_file_bytes=args.max_file_bytes,
    )
    command_repos = " ".join(f"--repo-path {path}" for path in args.repo_path)
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="real_training_data",
        seed=args.seed,
        command=(
            "uv run python scripts/prepare_training_corpus.py "
            f"{command_repos} --min-tokens {args.min_tokens} "
            f"--max-file-bytes {args.max_file_bytes} "
            f"--experiment-id {args.experiment_id} --output {args.output}"
        ),
        metrics=metrics,
        status=metrics["status"],
        notes=(
            "Licensed local repository corpus preparation report with license filtering, "
            "secret scanning, generated-file filtering, deduplication, repository-aware "
            "splits, and approximate token/source accounting."
        ),
    ).to_jsonable()
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
