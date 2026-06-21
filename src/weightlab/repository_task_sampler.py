from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RepositoryTaskSampleConfig:
    split: str = "test"
    seed: int = 123
    max_repositories: int = 16
    files_per_repository: int = 2
    task_kinds: tuple[str, ...] = ("completion", "infilling", "syntax")
    min_text_bytes: int = 20


def sample_repository_tasks(
    corpus_jsonl: Path,
    config: RepositoryTaskSampleConfig | None = None,
) -> dict[str, Any]:
    config = config or RepositoryTaskSampleConfig()
    rows = _load_rows(corpus_jsonl, config)
    grouped = _group_rows_by_repository(rows)
    rng = random.Random(config.seed)
    repositories = sorted(grouped)
    rng.shuffle(repositories)
    selected_repositories = repositories[: config.max_repositories]

    tasks: list[dict[str, Any]] = []
    for repository_index, repo in enumerate(selected_repositories):
        files = sorted(grouped[repo], key=lambda row: str(row["path"]))
        rng.shuffle(files)
        selected_files = files[: config.files_per_repository]
        selected_files.sort(key=lambda row: str(row["path"]))
        for file_index, row in enumerate(selected_files):
            for task_index, task_kind in enumerate(config.task_kinds):
                tasks.append(
                    _task_record(
                        row,
                        task_kind=task_kind,
                        seed=config.seed,
                        repository_index=repository_index,
                        file_index=file_index,
                        task_index=task_index,
                    )
                )

    return {
        "benchmark_label": "repository_balanced_task_sample",
        "corpus_jsonl": str(corpus_jsonl),
        "source_split": config.split,
        "seed": config.seed,
        "repository_count": len(selected_repositories),
        "file_count": sum(
            min(len(grouped[repo]), config.files_per_repository)
            for repo in selected_repositories
        ),
        "task_count": len(tasks),
        "sampling_policy": {
            "order": "repository_first_file_second_task_third",
            "max_repositories": config.max_repositories,
            "files_per_repository": config.files_per_repository,
            "task_kinds": list(config.task_kinds),
            "min_text_bytes": config.min_text_bytes,
        },
        "tasks": tasks,
    }


def _load_rows(corpus_jsonl: Path, config: RepositoryTaskSampleConfig) -> list[dict[str, Any]]:
    rows = []
    with corpus_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != config.split:
                continue
            if int(row.get("bytes", 0)) < config.min_text_bytes:
                continue
            if not row.get("repo") or not row.get("path") or not row.get("text"):
                continue
            rows.append(row)
    return rows


def _group_rows_by_repository(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["repo"])].append(row)
    return dict(grouped)


def _task_record(
    row: dict[str, Any],
    *,
    task_kind: str,
    seed: int,
    repository_index: int,
    file_index: int,
    task_index: int,
) -> dict[str, Any]:
    text = str(row["text"])
    sample_material = {
        "repo": row["repo"],
        "path": row["path"],
        "task_kind": task_kind,
        "seed": seed,
        "text_sha256": row.get("text_sha256", ""),
    }
    sample_sha256 = hashlib.sha256(
        json.dumps(sample_material, sort_keys=True).encode("utf-8")
    ).hexdigest()
    task_id = f"repo-task-{sample_sha256[:16]}"
    return {
        "task_id": task_id,
        "task_kind": task_kind,
        "repo": str(row["repo"]),
        "path": str(row["path"]),
        "source_split": str(row.get("split", "")),
        "seed": seed,
        "repository_index": repository_index,
        "file_index": file_index,
        "task_index": task_index,
        "language": str(row.get("language", "")),
        "content_roles": list(row.get("content_roles", [])),
        "row_sha256": str(row.get("row_sha256", "")),
        "text_sha256": str(row.get("text_sha256", "")),
        "text_bytes": int(row.get("bytes", len(text.encode("utf-8", errors="ignore")))),
        "sample_sha256": sample_sha256,
    }
