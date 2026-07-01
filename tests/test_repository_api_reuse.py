from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from weightlab.repository_api_reuse import (
    RepositoryApiReuseConfig,
    evaluate_repository_api_reuse,
    evaluate_repository_context_comparison,
)


def _write_rows(path: Path) -> None:
    rows = [
        {
            "repo": "repo-alpha",
            "path": "src/user.js",
            "split": "train",
            "language": "JavaScript",
            "content_roles": ["source"],
            "row_sha256": "row-alpha-source",
            "text_sha256": "text-alpha-source",
            "bytes": 190,
            "text": (
                "export function normalizeUser(rawUser) {\n"
                "  return { id: rawUser.id, name: rawUser.name.trim() };\n"
                "}\n"
            ),
        },
        {
            "repo": "repo-alpha",
            "path": "tests/user.test.js",
            "split": "validation",
            "language": "JavaScript",
            "content_roles": ["test"],
            "row_sha256": "row-alpha-test",
            "text_sha256": "text-alpha-test",
            "bytes": 160,
            "text": (
                "import { normalizeUser } from '../src/user';\n"
                "test('normalizes names', () => normalizeUser({ id: 1, name: ' Ada ' }));\n"
            ),
        },
        {
            "repo": "repo-alpha",
            "path": "src/audit.js",
            "split": "train",
            "language": "JavaScript",
            "content_roles": ["source"],
            "row_sha256": "row-alpha-audit",
            "text_sha256": "text-alpha-audit",
            "bytes": 130,
            "text": (
                "export function auditUser(rawUser) {\n"
                "  return Boolean(rawUser.id);\n"
                "}\n"
            ),
        },
        {
            "repo": "repo-beta",
            "path": "src/slug.js",
            "split": "train",
            "language": "JavaScript",
            "content_roles": ["source"],
            "row_sha256": "row-beta-source",
            "text_sha256": "text-beta-source",
            "bytes": 130,
            "text": (
                "export function slugifyTitle(title) {\n"
                "  return title.toLowerCase().replaceAll(' ', '-');\n"
                "}\n"
            ),
        },
        {
            "repo": "repo-beta",
            "path": "docs/slug.md",
            "split": "validation",
            "language": "Markdown",
            "content_roles": ["documentation"],
            "row_sha256": "row-beta-doc",
            "text_sha256": "text-beta-doc",
            "bytes": 120,
            "text": "Use slugifyTitle before persisting route names.\n",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_query_symbol_rows(path: Path) -> None:
    rows = [
        {
            "repo": "repo-gamma",
            "path": "src/target.js",
            "split": "train",
            "language": "JavaScript",
            "content_roles": ["source"],
            "row_sha256": "row-gamma-target",
            "text_sha256": "text-gamma-target",
            "bytes": 130,
            "text": (
                "export function normalizeUser(rawUser) {\n"
                "  return rawUser.name.trim();\n"
                "}\n"
            ),
        },
        {
            "repo": "repo-gamma",
            "path": "src/noisy.js",
            "split": "train",
            "language": "JavaScript",
            "content_roles": ["source"],
            "row_sha256": "row-gamma-noisy",
            "text_sha256": "text-gamma-noisy",
            "bytes": 180,
            "text": (
                "export function parseProfile(name) {\n"
                "  const normalizes = true;\n"
                "  const users = name;\n"
                "  const Ada = users;\n"
                "  return { normalizes, users, Ada };\n"
                "}\n"
            ),
        },
        {
            "repo": "repo-gamma",
            "path": "tests/user.test.js",
            "split": "validation",
            "language": "JavaScript",
            "content_roles": ["test"],
            "row_sha256": "row-gamma-test",
            "text_sha256": "text-gamma-test",
            "bytes": 130,
            "text": (
                "import { normalizeUser } from '../src/target';\n"
                "test('normalizes users', () => normalizeUser({ name: ' Ada ' }));\n"
            ),
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_repository_api_reuse_builds_tasks_and_scores_symbol_selection(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_rows(corpus)

    result = evaluate_repository_api_reuse(
        corpus,
        RepositoryApiReuseConfig(
            source_split="train",
            query_split="validation",
            seed=123,
            top_k=3,
            max_tasks=8,
            max_source_rows=8,
            max_query_rows=8,
            min_text_bytes=20,
        ),
    )

    assert result["benchmark_label"] == "repository_api_reuse_probe"
    assert result["task_count"] == 2
    assert result["symbol_count"] == 3
    assert result["source_split"] == "train"
    assert result["query_split"] == "validation"
    assert result["tasks"][0]["positive_symbols"]
    assert result["tasks"][0]["candidate_count"] >= 2
    assert result["methods"]["symbol_name_mention"]["hit_at_k"] == 1.0
    assert result["methods"]["lexical_source_overlap"]["hit_at_k"] == 1.0
    assert result["best_method"]["hit_at_k"] == 1.0
    assert result["best_method"]["name"] in {
        "symbol_name_mention",
        "lexical_source_overlap",
    }
    assert "not_code_generation" in result["limitations"]


def test_repository_api_reuse_cli_writes_machine_readable_record(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    output = tmp_path / "api_reuse.json"
    _write_rows(corpus)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_repository_api_reuse.py",
            "--corpus-jsonl",
            str(corpus),
            "--source-split",
            "train",
            "--query-split",
            "validation",
            "--seed",
            "123",
            "--top-k",
            "3",
            "--max-tasks",
            "8",
            "--max-source-rows",
            "8",
            "--max-query-rows",
            "8",
            "--output",
            str(output),
            "--experiment-id",
            "repo_api_reuse_test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "repo_api_reuse_test"
    assert record["status"] == "completed"
    assert record["metrics"]["benchmark_label"] == "repository_api_reuse_probe"
    assert record["metrics"]["task_count"] == 2
    assert "--top-k 3" in record["command"]
    assert (tmp_path / "manifest.json").exists()


def test_repository_context_comparison_cli_writes_machine_readable_record(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    output = tmp_path / "context_pairwise.json"
    _write_rows(corpus)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_repository_context_pairwise.py",
            "--corpus-jsonl",
            str(corpus),
            "--source-split",
            "train",
            "--query-split",
            "validation",
            "--seed",
            "123",
            "--top-k",
            "3",
            "--max-tasks",
            "8",
            "--max-source-rows",
            "8",
            "--max-query-rows",
            "8",
            "--output",
            str(output),
            "--experiment-id",
            "repo_context_pairwise_test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "repo_context_pairwise_test"
    assert record["status"] == "completed"
    assert record["metrics"]["benchmark_label"] == "repository_context_pairwise_probe"
    assert record["metrics"]["best_method"]["name"] == "structured_symbol_memory"
    assert "retrieval_augmented_repository_context" in record["metrics"]["pairwise_ideas"]
    assert "--top-k 3" in record["command"]
    assert (tmp_path / "manifest.json").exists()


def test_repository_context_comparison_scores_retrieval_hallucination_proxy(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_rows(corpus)

    result = evaluate_repository_context_comparison(
        corpus,
        RepositoryApiReuseConfig(
            source_split="train",
            query_split="validation",
            seed=123,
            top_k=3,
            max_tasks=8,
            max_source_rows=8,
            max_query_rows=8,
            min_text_bytes=20,
        ),
    )

    assert result["benchmark_label"] == "repository_context_pairwise_probe"
    assert result["pairwise_ideas"] == [
        "retrieval_augmented_repository_context",
        "structured_repository_memory",
    ]
    assert result["task_count"] == 2
    assert result["methods"]["structured_symbol_memory"]["hit_at_k"] == 1.0
    assert result["methods"]["structured_symbol_memory"]["hallucinated_api_rate"] == 0.0
    assert result["methods"]["symbol_aware_retrieved_snippets"]["hit_at_k"] == 1.0
    assert (
        result["methods"]["symbol_aware_retrieved_snippets"]["hallucinated_api_rate"]
        == 0.0
    )
    assert (
        result["methods"]["retrieved_snippet_identifiers"]["hallucinated_api_rate"]
        > result["methods"]["symbol_aware_retrieved_snippets"][
            "hallucinated_api_rate"
        ]
    )
    assert result["best_method"]["name"] == "structured_symbol_memory"
    assert "proxy_hallucinated_api_rate" in result["limitations"]


def test_query_symbol_aware_retrieval_uses_mentioned_symbols_before_lexical_noise(
    tmp_path,
):
    corpus = tmp_path / "corpus.jsonl"
    _write_query_symbol_rows(corpus)

    result = evaluate_repository_context_comparison(
        corpus,
        RepositoryApiReuseConfig(
            source_split="train",
            query_split="validation",
            seed=123,
            top_k=1,
            max_tasks=8,
            max_source_rows=8,
            max_query_rows=8,
            min_text_bytes=20,
        ),
    )

    assert result["task_count"] == 1
    assert result["methods"]["symbol_aware_retrieved_snippets"]["hit_at_k"] == 0.0
    assert result["methods"]["query_symbol_aware_retrieval"]["hit_at_k"] == 1.0
    assert (
        result["methods"]["query_symbol_aware_retrieval"]["hallucinated_api_rate"]
        == 0.0
    )
