from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from weightlab.metrics import ExperimentRecord, write_json
from weightlab.repo_chronology import run_public_patch_replay_suite_experiment

MARKUPSAFE_TEST_CODE = (
    "import sys; sys.path.insert(0, 'src'); from markupsafe import escape; "
    "Proxy = type('Proxy', (), {'__class__': property(lambda self: str), "
    "'__str__': lambda self: '<em>'}); "
    "assert str(escape(Proxy())) == '&lt;em&gt;'"
)

KILO_SAVED_HL_TEST_CODE = (
    "from pathlib import Path; text = Path('kilo.c').read_text(); "
    "start = text.index('#define FIND_RESTORE_HL'); "
    "end = text.index('} while (0)', start); "
    "macro = text[start:end]; "
    "assert 'free(saved_hl);' in macro; "
    "assert macro.index('memcpy(E.row[saved_hl_line].hl,saved_hl, "
    "E.row[saved_hl_line].rsize);') < macro.index('free(saved_hl);') < "
    "macro.index('saved_hl = NULL;')"
)


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
    parser.add_argument(
        "--markupsafe-repo",
        type=Path,
        default=Path("data/raw/public/markupsafe"),
    )
    parser.add_argument("--markupsafe-base-commit", default="54bb00b^")
    parser.add_argument("--markupsafe-fix-commit", default="54bb00b")
    parser.add_argument("--kilo-repo", type=Path, default=Path("data/raw/public/kilo"))
    parser.add_argument("--kilo-base-commit", default="8e9a9bb^")
    parser.add_argument("--kilo-fix-commit", default="8e9a9bb")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--timeout-s", type=int, default=20)
    args = parser.parse_args()

    tasks = [
        {
            "task_id": "markupsafe_proxy_exact_type",
            "repo_path": args.markupsafe_repo,
            "base_commit": args.markupsafe_base_commit,
            "fix_commit": args.markupsafe_fix_commit,
            "test_command": [sys.executable, "-c", MARKUPSAFE_TEST_CODE],
            "timeout_s": args.timeout_s,
        },
        {
            "task_id": "kilo_saved_hl_free",
            "repo_path": args.kilo_repo,
            "base_commit": args.kilo_base_commit,
            "fix_commit": args.kilo_fix_commit,
            "test_command": [sys.executable, "-c", KILO_SAVED_HL_TEST_CODE],
            "timeout_s": args.timeout_s,
        },
    ]
    metrics = run_public_patch_replay_suite_experiment(tasks)
    command = (
        "uv run python scripts/run_public_patch_replay_suite.py "
        f"--markupsafe-repo {args.markupsafe_repo} "
        f"--markupsafe-base-commit {args.markupsafe_base_commit} "
        f"--markupsafe-fix-commit {args.markupsafe_fix_commit} "
        f"--kilo-repo {args.kilo_repo} "
        f"--kilo-base-commit {args.kilo_base_commit} "
        f"--kilo-fix-commit {args.kilo_fix_commit} "
        f"--timeout-s {args.timeout_s}"
    )
    record = ExperimentRecord(
        experiment_id="E5g_public_patch_replay_suite",
        hypothesis="H5",
        seed=0,
        command=command,
        metrics=metrics,
        notes=(
            "Two-task local public historical patch replay suite covering MarkupSafe proxy "
            "exact-type behavior and Kilo saved_hl memory-leak repair. MarkupSafe uses a "
            "functional Python regression; Kilo uses an executable source-level regression."
        ),
    ).to_jsonable()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "E5g_public_patch_replay_suite.json", record)
    _append_manifest(args.output_dir, record)


if __name__ == "__main__":
    main()
