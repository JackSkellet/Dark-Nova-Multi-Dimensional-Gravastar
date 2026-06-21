from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.if4_fast_repo_adaptation import run_if4_fast_repo_adaptation_probe
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


def build_record(
    *,
    repo_path: Path,
    max_commits: int,
    top_k: int,
    output: Path,
    experiment_id: str,
    consolidation_interval: int = 3,
) -> dict[str, object]:
    metrics = run_if4_fast_repo_adaptation_probe(
        repo_path=repo_path,
        max_commits=max_commits,
        top_k=top_k,
        consolidation_interval=consolidation_interval,
    )
    metrics["resolved_config"] = {
        "repo_path": str(repo_path),
        "max_commits": max_commits,
        "top_k": top_k,
        "output": str(output),
        "experiment_id": experiment_id,
        "consolidation_interval": consolidation_interval,
    }
    command = (
        "uv run python scripts/run_if4_fast_repo_adaptation.py "
        f"--repo-path {repo_path} --max-commits {max_commits} "
        f"--top-k {top_k} --output {output} --experiment-id {experiment_id} "
        f"--consolidation-interval {consolidation_interval}"
    )
    record = ExperimentRecord(
        experiment_id=experiment_id,
        hypothesis="if4_fast_temporary_weights_improve_chronological_repository_adaptation",
        seed=0,
        command=command,
        metrics=metrics,
        status="completed",
        notes=(
            "Chronological public-repository fast adaptation probe. It compares updated "
            "retrieval, structured memory, replay adapter proxy, fast temporary weights, "
            "fast weights plus retrieval, and periodic consolidation."
        ),
    ).to_jsonable()
    record["resolved_config"] = metrics["resolved_config"]
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--max-commits", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--consolidation-interval", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/IF4_fast_repo_adaptation_markupsafe.json"),
    )
    parser.add_argument("--experiment-id", default="IF4_fast_repo_adaptation_markupsafe")
    args = parser.parse_args()

    record = build_record(
        repo_path=args.repo_path,
        max_commits=args.max_commits,
        top_k=args.top_k,
        output=args.output,
        experiment_id=args.experiment_id,
        consolidation_interval=args.consolidation_interval,
    )
    write_json(args.output, record)
    _append_manifest(args.output, record)


if __name__ == "__main__":
    main()
