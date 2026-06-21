from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from weightlab.if4_fast_repo_adaptation import run_if4_fast_repo_adaptation_probe


def _git(repo, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _commit(repo, message: str) -> None:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)


def _make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "research@example.invalid")
    _git(repo, "config", "user.name", "Research Test")

    (repo / "README.md").write_text("configuration loader package\n", encoding="utf-8")
    _commit(repo, "initial configuration loader package")

    (repo / "parser.py").write_text(
        "def parse_config(text):\n"
        "    return {'raw': text.strip()}\n",
        encoding="utf-8",
    )
    _commit(repo, "add parse_config loader")

    (repo / "writer.py").write_text(
        "def write_config(config):\n"
        "    return str(config)\n",
        encoding="utf-8",
    )
    _commit(repo, "add write_config output")

    (repo / "parser.py").write_text(
        "def parse_config(text):\n"
        "    return {'raw': text.strip(), 'valid': bool(text)}\n",
        encoding="utf-8",
    )
    _commit(repo, "extend parse_config validation")

    (repo / "docs.md").write_text(
        "parse_config loads text and write_config serializes it\n",
        encoding="utf-8",
    )
    _commit(repo, "document parser writer api")
    return repo


def test_if4_probe_compares_fast_weights_with_repository_controls(tmp_path):
    repo = _make_repo(tmp_path)

    result = run_if4_fast_repo_adaptation_probe(repo, max_commits=5, top_k=2)

    assert result["benchmark_label"] == "if4_fast_repo_adaptation_probe"
    assert result["candidate_id"] == "IF4"
    assert result["repo"]["commit_count_used"] == 5
    assert set(result["methods"]) == {
        "updated_retrieval",
        "structured_symbol_graph_memory",
        "replay_adapter_proxy",
        "fast_temporary_weights",
        "fast_weights_plus_retrieval",
        "periodic_consolidation",
    }
    assert result["steps"]
    assert all(step["future_commit_not_in_memory"] for step in result["steps"])
    assert all(step["future_changed_files"] for step in result["steps"])
    assert all("paraphrase_query" in step for step in result["steps"])
    assert result["final"]["mean_update_ms"] >= 0.0
    assert result["final"]["rollback_supported"] is True
    assert result["final"]["total_storage_bytes"] > 0
    assert result["final"]["prior_task_retention_accuracy"] >= 0.0
    assert "fast_weights_plus_retrieval_future_topk_accuracy" in result["final"]
    assert result["limitations"]


def test_if4_cli_record_is_machine_readable(tmp_path):
    repo = _make_repo(tmp_path)
    output = tmp_path / "if4.json"

    subprocess.check_call(
        [
            sys.executable,
            "scripts/run_if4_fast_repo_adaptation.py",
            "--repo-path",
            str(repo),
            "--max-commits",
            "5",
            "--top-k",
            "2",
            "--output",
            str(output),
            "--experiment-id",
            "IF4_tmp",
        ],
        cwd=Path.cwd(),
    )
    record = json.loads(output.read_text(encoding="utf-8"))

    assert record["experiment_id"] == "IF4_tmp"
    assert record["status"] == "completed"
    assert record["metrics"]["benchmark_label"] == "if4_fast_repo_adaptation_probe"
    assert json.loads(json.dumps(record))["metrics"]["repo"]["path_name"] == "repo"
