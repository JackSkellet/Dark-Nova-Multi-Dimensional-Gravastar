from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.hf_corpus import HFSourceSpec, build_hf_corpus_manifest
from weightlab.metrics import ExperimentRecord, write_json

DEFAULT_SOURCES = [
    HFSourceSpec(
        dataset="CodedotAI/code_clippy_github",
        revision="cf9f33dec640f45228f8f3f5e6a7899a37f5f83e",
        configs=["JavaScript-all", "all-mit", "all-apache-2.0"],
        card_review=(
            "HF card reports MIT license and public GitHub BigQuery provenance. "
            "Dataset configs include language and license partitions. This exploratory "
            "manifest intentionally includes mixed-license metadata where normally "
            "accessible, and records it as not approved for production or redistribution. "
            "Preview shows vendored paths such as node_modules, so streaming materialization "
            "must apply vendor/generated/secret filters."
        ),
    ),
    HFSourceSpec(
        dataset="codeparrot/github-code-clean",
        revision="c48d40f9e70f0196f8236901ee35807f7d6c44c0",
        configs=["Python-all", "JavaScript-all", "all-mit", "all-apache-2.0"],
        card_review=(
            "HF card reports Apache-2.0 for the cleaned GitHub-code dataset. "
            "Dataset configs include language and license partitions. This exploratory "
            "manifest records stated license metadata and restrictions, while deferring "
            "strict legal/company-policy filtering to a later production approval pass."
        ),
    ),
]


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
    parser = argparse.ArgumentParser(
        description="Build a reviewed, revision-pinned Hugging Face corpus manifest."
    )
    parser.add_argument("--output", type=Path, default=Path("results/D3_hf_corpus_manifest.json"))
    parser.add_argument("--experiment-id", default="D3_hf_corpus_manifest")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    metrics = build_hf_corpus_manifest(DEFAULT_SOURCES)
    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        hypothesis="real_training_data",
        seed=args.seed,
        command=(
            "uv run python scripts/prepare_hf_corpus_manifest.py "
            f"--experiment-id {args.experiment_id} --output {args.output}"
        ),
        metrics=metrics,
        status=metrics["status"],
        notes=(
            "Reviewed Hugging Face Dataset Viewer manifest with immutable revisions, "
            "exploratory-research-only license metadata recording, parquet shard metadata, "
            "datasets streaming policy, and local mirror requirements. Raw rows are not "
            "downloaded here."
        ),
    ).to_jsonable()
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
