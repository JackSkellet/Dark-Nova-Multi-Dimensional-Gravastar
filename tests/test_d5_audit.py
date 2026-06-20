from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from weightlab.d5_audit import audit_d5_corpus


def _row(
    *,
    text: str,
    repo: str,
    path: str,
    split: str,
    language: str = "Python",
    dataset: str = "source/a",
    config: str = "Python-all",
    roles: list[str] | None = None,
    timestamp: str = "",
    simhash: str = "0000000000000000",
) -> str:
    payload = {
        "dataset": dataset,
        "config": config,
        "dataset_revision": "rev",
        "split": split,
        "repo_split": split,
        "temporal_split": "unknown" if not timestamp else split,
        "source_timestamp": timestamp,
        "repo": repo,
        "path": path,
        "language": language,
        "license": "mit",
        "content_roles": roles or ["source"],
        "text": text,
        "tokens": len(text.encode("utf-8")) + 1,
        "bytes": len(text.encode("utf-8")),
        "text_sha256": "sha-" + text,
        "near_duplicate_simhash64": simhash,
    }
    return json.dumps(payload)


def test_audit_d5_corpus_reports_token_mass_roles_splits_and_duplicates(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        "\n".join(
            [
                _row(
                    text="alpha",
                    repo="repo-a",
                    path="src/alpha.py",
                    split="train",
                    roles=["source", "docstring"],
                    timestamp="2024-01-01T00:00:00Z",
                    simhash="0000000000000000",
                ),
                _row(
                    text="beta",
                    repo="repo-b",
                    path="tests/test_beta.py",
                    split="validation",
                    roles=["source", "test"],
                    simhash="0000000000000001",
                ),
                _row(
                    text="alpha",
                    repo="repo-c",
                    path="src/copy.py",
                    split="test",
                    dataset="source/b",
                    roles=["source"],
                    simhash="0000000000000002",
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    audit = audit_d5_corpus(
        corpus_path,
        near_duplicate_hamming_threshold=1,
        near_duplicate_min_bytes=0,
    )

    assert audit["benchmark_label"] == "d5_corpus_audit"
    assert audit["rows"] == 3
    assert audit["token_mass"]["by_language"]["Python"]["rows"] == 3
    assert audit["token_mass"]["by_source"]["source/a::Python-all"]["rows"] == 2
    assert audit["token_mass"]["by_repository"]["repo-a"]["tokens"] == 6
    assert audit["role_overlap"]["multi_role_rows"] == 2
    assert audit["role_overlap"]["role_sets"]["source+docstring"]["rows"] == 1
    assert audit["split_integrity"]["repo_split_mismatches"] == 0
    assert audit["split_integrity"]["repositories_in_multiple_splits"] == 0
    assert audit["timestamp_coverage"]["rows_with_timestamp"] == 1
    assert audit["timestamp_coverage"]["temporal_claim_allowed"] is False
    assert audit["duplicates"]["exact_text"]["duplicate_rows"] == 1
    assert audit["duplicates"]["exact_text"]["cross_source_duplicate_groups"] == 1
    assert audit["duplicates"]["near_duplicate_simhash"]["min_bytes"] == 0
    assert audit["duplicates"]["near_duplicate_simhash"]["candidate_pairs"] >= 1


def test_d5_audit_cli_writes_record(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        _row(text="alpha", repo="repo-a", path="src/alpha.py", split="train") + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "audit.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_d5_corpus.py",
            "--corpus-jsonl",
            str(corpus_path),
            "--output",
            str(output_path),
            "--experiment-id",
            "d5_audit_test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "d5_audit_test"
    assert record["metrics"]["benchmark_label"] == "d5_corpus_audit"
    assert record["metrics"]["rows"] == 1
    assert "--corpus-jsonl" in record["command"]
