from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from weightlab.metrics import ExperimentRecord, write_json
from weightlab.repo_chronology import (
    run_public_patch_replay_experiment,
    run_public_repo_chronology_experiment,
    run_stale_doc_chronology_experiment,
    run_symbol_qa_chronology_experiment,
)


def _append_record(output: Path, jsonable: dict[str, object]) -> None:
    write_json(output, jsonable)

    manifest_path = output.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = [item for item in manifest if item["experiment_id"] != jsonable["experiment_id"]]
        manifest.append(jsonable)
        write_json(manifest_path, manifest)

    summary_path = output.parent / "summary.csv"
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        rows = [row for row in rows if row["experiment_id"] != jsonable["experiment_id"]]
        rows.append(
            {
                "experiment_id": jsonable["experiment_id"],
                "hypothesis": jsonable["hypothesis"],
                "seed": jsonable["seed"],
                "status": jsonable["status"],
                "notes": jsonable["notes"],
            }
        )
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["experiment_id", "hypothesis", "seed", "status", "notes"]
            )
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--max-commits", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=Path("results/E5_public_repo_chronology.json")
    )
    parser.add_argument(
        "--symbol-output",
        type=Path,
        default=Path("results/E5_public_repo_symbol_qa.json"),
    )
    parser.add_argument(
        "--stale-doc-output",
        type=Path,
        default=Path("results/E5_public_repo_stale_docs.json"),
    )
    parser.add_argument(
        "--patch-output",
        type=Path,
        default=Path("results/E5_public_patch_replay.json"),
    )
    parser.add_argument("--include-symbol-qa", action="store_true")
    parser.add_argument("--include-stale-docs", action="store_true")
    parser.add_argument("--include-patch-replay", action="store_true")
    parser.add_argument("--patch-base-commit")
    parser.add_argument("--patch-fix-commit")
    parser.add_argument(
        "--patch-test-code",
        help="Python code to execute as the regression test in the replay worktree.",
    )
    args = parser.parse_args()

    metrics = run_public_repo_chronology_experiment(
        repo_path=args.repo_path,
        max_commits=args.max_commits,
        top_k=args.top_k,
    )
    record = ExperimentRecord(
        experiment_id=args.output.stem,
        hypothesis="H5",
        seed=0,
        command=(
            "uv run python scripts/run_public_repo_chronology.py "
            f"--repo-path {args.repo_path} --max-commits {args.max_commits} --top-k {args.top_k}"
        ),
        metrics=metrics,
        notes="Explicit local-clone public repository chronology experiment.",
    )
    _append_record(args.output, record.to_jsonable())

    if args.include_symbol_qa:
        symbol_metrics = run_symbol_qa_chronology_experiment(
            repo_path=args.repo_path,
            max_commits=args.max_commits,
            top_k=args.top_k,
        )
        symbol_record = ExperimentRecord(
            experiment_id=args.symbol_output.stem,
            hypothesis="H5",
            seed=0,
            command=(
                "uv run python scripts/run_public_repo_chronology.py "
                f"--repo-path {args.repo_path} --max-commits {args.max_commits} "
                f"--top-k {args.top_k} --include-symbol-qa"
            ),
            metrics=symbol_metrics,
            notes="Explicit local-clone public repository symbol-definition QA experiment.",
        )
        _append_record(args.symbol_output, symbol_record.to_jsonable())

    if args.include_stale_docs:
        stale_doc_metrics = run_stale_doc_chronology_experiment(
            repo_path=args.repo_path,
            max_commits=args.max_commits,
            top_k=args.top_k,
        )
        stale_doc_record = ExperimentRecord(
            experiment_id=args.stale_doc_output.stem,
            hypothesis="H5",
            seed=0,
            command=(
                "uv run python scripts/run_public_repo_chronology.py "
                f"--repo-path {args.repo_path} --max-commits {args.max_commits} "
                f"--top-k {args.top_k} --include-stale-docs"
            ),
            metrics=stale_doc_metrics,
            notes="Explicit local-clone public repository stale-document detection experiment.",
        )
        _append_record(args.stale_doc_output, stale_doc_record.to_jsonable())

    if args.include_patch_replay:
        if not args.patch_base_commit or not args.patch_fix_commit or not args.patch_test_code:
            parser.error(
                "--include-patch-replay requires --patch-base-commit, "
                "--patch-fix-commit, and --patch-test-code"
            )
        patch_metrics = run_public_patch_replay_experiment(
            repo_path=args.repo_path,
            base_commit=args.patch_base_commit,
            fix_commit=args.patch_fix_commit,
            test_command=[sys.executable, "-c", args.patch_test_code],
        )
        patch_record = ExperimentRecord(
            experiment_id=args.patch_output.stem,
            hypothesis="H5",
            seed=0,
            command=(
                "uv run python scripts/run_public_repo_chronology.py "
                f"--repo-path {args.repo_path} --include-patch-replay "
                f"--patch-base-commit {args.patch_base_commit} "
                f"--patch-fix-commit {args.patch_fix_commit} "
                "--patch-test-code '<python regression code>'"
            ),
            metrics=patch_metrics,
            notes=(
                "Explicit local-clone public historical patch replay graded by executable "
                "tests, not text similarity."
            ),
        )
        _append_record(args.patch_output, patch_record.to_jsonable())


if __name__ == "__main__":
    main()
