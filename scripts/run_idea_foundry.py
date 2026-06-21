from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.idea_foundry import (
    IDEA_FOUNDRY_CANDIDATES,
    run_repository_graph_signal_probe,
    summarize_candidate_constraints,
)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument(
        "--candidates-output",
        type=Path,
        default=Path("results/idea_foundry_candidates.json"),
    )
    parser.add_argument(
        "--probe-output",
        type=Path,
        default=Path("results/IF1_repository_graph_signal_probe.json"),
    )
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--experiment-id", default="idea_foundry")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    command = (
        "uv run python scripts/run_idea_foundry.py "
        f"--corpus-jsonl {args.corpus_jsonl} "
        f"--candidates-output {args.candidates_output} "
        f"--probe-output {args.probe_output} "
        f"--experiment-id {args.experiment_id} "
        f"--seed {args.seed}"
    )
    if args.max_documents is not None:
        command += f" --max-documents {args.max_documents}"

    candidate_metrics = {
        "benchmark_label": "idea_foundry_candidate_generation",
        "candidates": IDEA_FOUNDRY_CANDIDATES,
        "constraint_summary": summarize_candidate_constraints(IDEA_FOUNDRY_CANDIDATES),
        "resolved_config": {
            "corpus_jsonl": str(args.corpus_jsonl),
            "candidates_output": str(args.candidates_output),
            "probe_output": str(args.probe_output),
            "max_documents": args.max_documents,
            "experiment_id": args.experiment_id,
            "seed": args.seed,
        },
    }
    candidate_record = ExperimentRecord(
        experiment_id=f"{args.experiment_id}_candidates",
        hypothesis="idea_foundry_six_distinct_local_coding_architecture_candidates",
        seed=args.seed,
        command=command,
        metrics=candidate_metrics,
        status="completed",
        notes=(
            "Six candidate mechanisms for the next research cycle. These are not "
            "quality claims until prototyped and evaluated."
        ),
    ).to_jsonable()
    candidate_record["resolved_config"] = candidate_metrics["resolved_config"]
    write_json(args.candidates_output, candidate_record)
    _append_manifest(args.candidates_output, candidate_record)

    probe_metrics = run_repository_graph_signal_probe(
        args.corpus_jsonl,
        max_documents=args.max_documents,
    )
    probe_metrics["resolved_config"] = candidate_metrics["resolved_config"]
    probe_record = ExperimentRecord(
        experiment_id=f"{args.experiment_id}_repository_graph_signal_probe",
        hypothesis="if1_repository_graph_signal_exists_in_d5",
        seed=args.seed,
        command=command,
        metrics=probe_metrics,
        status="completed",
        notes=(
            "Cheap falsifying probe for IF1 only. It checks whether regex-extracted "
            "repository import edges and roles exist; it is not a model-training result."
        ),
    ).to_jsonable()
    probe_record["resolved_config"] = probe_metrics["resolved_config"]
    write_json(args.probe_output, probe_record)
    _append_manifest(args.probe_output, probe_record)


if __name__ == "__main__":
    main()
