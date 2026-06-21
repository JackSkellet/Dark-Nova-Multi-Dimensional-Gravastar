from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from weightlab.repository_task_sampler import RepositoryTaskSampleConfig, sample_repository_tasks


def _write_rows(path: Path) -> None:
    rows = [
        {
            "repo": "repo-alpha",
            "path": "src/a.js",
            "split": "test",
            "language": "JavaScript",
            "content_roles": ["source"],
            "row_sha256": "row-a",
            "text_sha256": "text-a",
            "bytes": 80,
            "text": "function alpha(value) { return value + 1; }\n",
        },
        {
            "repo": "repo-alpha",
            "path": "tests/a.test.js",
            "split": "test",
            "language": "JavaScript",
            "content_roles": ["test"],
            "row_sha256": "row-a-test",
            "text_sha256": "text-a-test",
            "bytes": 90,
            "text": "test('alpha', () => alpha(1));\n",
        },
        {
            "repo": "repo-beta",
            "path": "lib/b.js",
            "split": "test",
            "language": "JavaScript",
            "content_roles": ["source"],
            "row_sha256": "row-b",
            "text_sha256": "text-b",
            "bytes": 82,
            "text": "function beta(value) { return value + 2; }\n",
        },
        {
            "repo": "repo-gamma",
            "path": "lib/c.js",
            "split": "train",
            "language": "JavaScript",
            "content_roles": ["source"],
            "row_sha256": "row-c",
            "text_sha256": "text-c",
            "bytes": 82,
            "text": "function gamma(value) { return value + 3; }\n",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_repository_task_sampler_orders_repository_file_then_task(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_rows(corpus)

    result = sample_repository_tasks(
        corpus,
        RepositoryTaskSampleConfig(
            split="test",
            seed=123,
            max_repositories=2,
            files_per_repository=1,
            task_kinds=("completion", "syntax"),
        ),
    )

    tasks = result["tasks"]
    assert result["benchmark_label"] == "repository_balanced_task_sample"
    assert result["sampling_policy"]["order"] == "repository_first_file_second_task_third"
    assert [task["repo"] for task in tasks] == [
        tasks[0]["repo"],
        tasks[0]["repo"],
        tasks[2]["repo"],
        tasks[2]["repo"],
    ]
    assert [task["task_kind"] for task in tasks[:2]] == ["completion", "syntax"]
    assert tasks[0]["source_split"] == "test"
    assert tasks[0]["seed"] == 123
    assert tasks[0]["sample_sha256"]
    assert tasks[0]["text_sha256"]
    assert tasks[0]["task_id"].startswith("repo-task-")
    assert result["repository_count"] == 2
    assert result["task_count"] == 4


def test_repository_task_sampler_cli_writes_record(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_rows(corpus)
    output = tmp_path / "tasks.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_repository_coding_tasks.py",
            "--corpus-jsonl",
            str(corpus),
            "--split",
            "test",
            "--seed",
            "123",
            "--max-repositories",
            "2",
            "--files-per-repository",
            "1",
            "--task-kind",
            "completion",
            "--task-kind",
            "syntax",
            "--output",
            str(output),
            "--experiment-id",
            "repo_tasks_test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "repo_tasks_test"
    assert record["metrics"]["task_count"] == 4
    assert record["metrics"]["tasks"][0]["task_kind"] == "completion"
    assert (tmp_path / "manifest.json").exists()
