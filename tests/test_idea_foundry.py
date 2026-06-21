from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

from weightlab.idea_foundry import (
    IDEA_FOUNDRY_CANDIDATES,
    run_repository_graph_signal_probe,
    summarize_candidate_constraints,
)


def _load_run_idea_foundry_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_idea_foundry.py"
    spec = importlib.util.spec_from_file_location("run_idea_foundry", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_idea_foundry_records_six_distinct_candidates_with_required_constraints():
    summary = summarize_candidate_constraints(IDEA_FOUNDRY_CANDIDATES)

    assert summary["candidate_count"] == 6
    assert summary["without_adapters"] >= 2
    assert summary["without_moe_or_topic_routing"] >= 2
    assert summary["continual_evolution_candidates"] >= 1
    assert summary["compression_candidates"] >= 1
    assert summary["code_structure_candidates"] >= 1
    assert summary["potentially_novel_candidates"] >= 1

    for candidate in IDEA_FOUNDRY_CANDIDATES:
        assert candidate["id"].startswith("IF")
        assert candidate["mechanism"]
        assert candidate["equations"]
        assert candidate["expected_scaling"]
        assert candidate["rocm_plan"]
        assert candidate["likely_failure"]
        assert candidate["cheapest_falsifying_test"]
        assert candidate["mechanism_occurrence_evidence"]
        assert candidate["novelty_label"] in {
            "established",
            "adjacent",
            "potentially_novel",
        }
        assert candidate["closest_primary_source_prior_art"]
        for source in candidate["closest_primary_source_prior_art"]:
            assert source["title"]
            assert source["url"].startswith("https://")
            assert source["exact_difference"]


def test_repository_graph_signal_probe_detects_edges_roles_and_split_integrity(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    rows = [
        {
            "repo": "org/app",
            "path": "src/main.js",
            "split": "train",
            "repo_split": "train",
            "language": "JavaScript",
            "content_roles": ["source"],
            "tokens": 20,
            "text": "import helper from './util.js';\nexport function main(){ return helper(); }\n",
        },
        {
            "repo": "org/app",
            "path": "src/util.js",
            "split": "train",
            "repo_split": "train",
            "language": "JavaScript",
            "content_roles": ["source"],
            "tokens": 10,
            "text": "export default function helper(){ return 1; }\n",
        },
        {
            "repo": "org/app",
            "path": "test/main.test.js",
            "split": "train",
            "repo_split": "train",
            "language": "JavaScript",
            "content_roles": ["tests"],
            "tokens": 10,
            "text": "import { main } from '../src/main.js';\ntest('main', () => main());\n",
        },
        {
            "repo": "org/docs",
            "path": "README.md",
            "split": "validation",
            "repo_split": "validation",
            "language": "Markdown",
            "content_roles": ["documentation", "README"],
            "tokens": 8,
            "text": "# API\nUse main from the package.\n",
        },
    ]
    corpus_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = run_repository_graph_signal_probe(corpus_path)

    assert result["benchmark_label"] == "idea_foundry_repository_graph_signal_probe"
    assert result["candidate_id"] == "IF1"
    assert result["import_edge_count"] == 2
    assert result["resolved_local_edge_count"] >= 1
    assert result["repositories_with_edges"] == 1
    assert result["role_counts"]["tests"] == 1
    assert result["role_counts"]["documentation"] == 1
    assert result["role_counts"]["README"] == 1
    assert result["repository_aware_splits_preserved"] is True
    assert result["mechanism_signal_present"] is True


def test_idea_foundry_cli_writes_candidate_and_probe_records(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        json.dumps(
            {
                "repo": "org/app",
                "path": "src/main.js",
                "split": "train",
                "repo_split": "train",
                "language": "JavaScript",
                "content_roles": ["source"],
                "tokens": 12,
                "text": "const helper = require('./helper');\nmodule.exports = helper;\n",
            }
        )
        + "\n"
        + json.dumps(
            {
                "repo": "org/app",
                "path": "src/helper.js",
                "split": "train",
                "repo_split": "train",
                "language": "JavaScript",
                "content_roles": ["source"],
                "tokens": 6,
                "text": "module.exports = function helper() { return 1; };\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    candidates_output = tmp_path / "candidates.json"
    probe_output = tmp_path / "probe.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_idea_foundry.py",
            "--corpus-jsonl",
            str(corpus_path),
            "--candidates-output",
            str(candidates_output),
            "--probe-output",
            str(probe_output),
            "--experiment-id",
            "idea_foundry_test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    candidates = json.loads(candidates_output.read_text(encoding="utf-8"))
    probe = json.loads(probe_output.read_text(encoding="utf-8"))
    assert candidates["experiment_id"] == "idea_foundry_test_candidates"
    assert candidates["metrics"]["constraint_summary"]["candidate_count"] == 6
    assert probe["experiment_id"] == "idea_foundry_test_repository_graph_signal_probe"
    assert probe["metrics"]["candidate_id"] == "IF1"


def test_idea_foundry_builds_records_before_writing_to_preserve_clean_provenance(
    tmp_path,
    monkeypatch,
):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        json.dumps(
            {
                "repo": "org/app",
                "path": "src/main.js",
                "split": "train",
                "repo_split": "train",
                "language": "JavaScript",
                "content_roles": ["source"],
                "tokens": 12,
                "text": "const helper = require('./helper');\nmodule.exports = helper;\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("weightlab.metrics.git_commit", lambda: "cleancommit")
    monkeypatch.setattr("weightlab.metrics.git_status_short", lambda: "")

    module = _load_run_idea_foundry_module()
    candidate_record, probe_record = module.build_idea_foundry_records(
        corpus_jsonl=corpus_path,
        candidates_output=tmp_path / "candidates.json",
        probe_output=tmp_path / "probe.json",
        max_documents=None,
        experiment_id="idea_foundry_test",
        seed=123,
    )

    assert candidate_record["git_commit"] == "cleancommit"
    assert candidate_record["git_dirty"] is False
    assert probe_record["git_commit"] == "cleancommit"
    assert probe_record["git_dirty"] is False
