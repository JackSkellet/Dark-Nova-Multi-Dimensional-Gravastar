from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from weightlab.quixbugs_repair import (
    QuixBugsCandidate,
    QuixBugsCandidateConfig,
    QuixBugsRepairConfig,
    evaluate_quixbugs_candidate_repairs,
    evaluate_quixbugs_repair,
)


def _write_quixbugs_fixture(root: Path) -> None:
    (root / "python_programs").mkdir(parents=True)
    (root / "correct_python_programs").mkdir(parents=True)
    (root / "python_testcases").mkdir(parents=True)
    (root / "conftest.py").write_text(
        "import pytest\n\n"
        "def pytest_addoption(parser):\n"
        "    parser.addoption('--correct', action='store_true')\n\n"
        "def pytest_configure(config):\n"
        "    pytest.use_correct = config.getoption('--correct')\n",
        encoding="utf-8",
    )
    (root / "python_programs" / "clamp.py").write_text(
        "def clamp(value, low, high):\n"
        "    return min(value, high)\n",
        encoding="utf-8",
    )
    (root / "correct_python_programs" / "clamp.py").write_text(
        "def clamp(value, low, high):\n"
        "    return max(low, min(value, high))\n",
        encoding="utf-8",
    )
    (root / "python_testcases" / "test_clamp.py").write_text(
        "import pytest\n\n"
        "if pytest.use_correct:\n"
        "    from correct_python_programs.clamp import clamp\n"
        "else:\n"
        "    from python_programs.clamp import clamp\n\n"
        "def test_lower_bound():\n"
        "    assert clamp(-5, 0, 10) == 0\n\n"
        "def test_upper_bound():\n"
        "    assert clamp(15, 0, 10) == 10\n",
        encoding="utf-8",
    )


def test_quixbugs_repair_scores_buggy_and_oracle_correct(tmp_path):
    _write_quixbugs_fixture(tmp_path)

    result = evaluate_quixbugs_repair(
        tmp_path,
        QuixBugsRepairConfig(programs=("clamp",), timeout_seconds=10, seed=123),
    )

    assert result["benchmark_label"] == "quixbugs_python_repair_probe"
    assert result["program_count"] == 1
    assert result["final"]["buggy_pass_rate"] == 0.0
    assert result["final"]["oracle_correct_pass_rate"] == 1.0
    assert result["final"]["repair_gap"] == 1.0
    assert result["programs"][0]["program"] == "clamp"
    assert result["programs"][0]["buggy"]["passed"] is False
    assert result["programs"][0]["oracle_correct"]["passed"] is True
    assert "not_model_generated_repairs" in result["limitations"]


def test_quixbugs_repair_cli_writes_machine_readable_record(tmp_path):
    _write_quixbugs_fixture(tmp_path)
    output = tmp_path / "quixbugs.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_quixbugs_repair.py",
            "--repo-path",
            str(tmp_path),
            "--program",
            "clamp",
            "--timeout-seconds",
            "10",
            "--seed",
            "123",
            "--output",
            str(output),
            "--experiment-id",
            "quixbugs_test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "quixbugs_test"
    assert record["status"] == "completed"
    assert record["metrics"]["benchmark_label"] == "quixbugs_python_repair_probe"
    assert record["metrics"]["program_count"] == 1
    assert "--program clamp" in record["command"]
    assert (tmp_path / "manifest.json").exists()


def test_quixbugs_candidate_repairs_score_replacement_sources_without_mutating_repo(
    tmp_path,
):
    _write_quixbugs_fixture(tmp_path)
    original_buggy_source = (tmp_path / "python_programs" / "clamp.py").read_text(
        encoding="utf-8"
    )

    result = evaluate_quixbugs_candidate_repairs(
        tmp_path,
        [
            QuixBugsCandidate(
                candidate_id="identity_buggy",
                program="clamp",
                source_text=original_buggy_source,
                generator_label="identity_baseline",
            ),
            QuixBugsCandidate(
                candidate_id="manual_fixed",
                program="clamp",
                source_text=(
                    "def clamp(value, low, high):\n"
                    "    return max(low, min(value, high))\n"
                ),
                generator_label="manual_fixture_candidate",
            ),
        ],
        QuixBugsCandidateConfig(timeout_seconds=10, seed=123),
    )

    assert result["benchmark_label"] == "quixbugs_python_candidate_repair_probe"
    assert result["candidate_count"] == 2
    assert result["program_count"] == 1
    assert result["final"]["candidate_passed"] == 1
    assert result["final"]["candidate_pass_rate"] == 0.5
    assert result["final"]["programs_with_passing_candidate"] == 1
    assert result["final"]["program_repair_rate"] == 1.0
    assert result["programs"]["clamp"]["best_candidate_id"] == "manual_fixed"
    assert result["candidates"][0]["passed"] is False
    assert result["candidates"][1]["passed"] is True
    assert "not_model_generated_unless_candidate_file_is_model_output" in result["limitations"]
    assert (
        tmp_path / "python_programs" / "clamp.py"
    ).read_text(encoding="utf-8") == original_buggy_source


def test_quixbugs_candidate_repair_cli_writes_machine_readable_record(tmp_path):
    _write_quixbugs_fixture(tmp_path)
    output = tmp_path / "candidate_repairs.json"
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "candidate_id": "manual_fixed",
                "program": "clamp",
                "source_text": (
                    "def clamp(value, low, high):\n"
                    "    return max(low, min(value, high))\n"
                ),
                "generator_label": "manual_fixture_candidate",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_quixbugs_candidate_repairs.py",
            "--repo-path",
            str(tmp_path),
            "--candidates-jsonl",
            str(candidates),
            "--timeout-seconds",
            "10",
            "--seed",
            "123",
            "--output",
            str(output),
            "--experiment-id",
            "quixbugs_candidate_test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "quixbugs_candidate_test"
    assert record["status"] == "completed"
    assert (
        record["metrics"]["benchmark_label"]
        == "quixbugs_python_candidate_repair_probe"
    )
    assert record["metrics"]["final"]["program_repair_rate"] == 1.0
    assert "--candidates-jsonl" in record["command"]
    assert (tmp_path / "manifest.json").exists()


def test_quixbugs_candidate_repair_cli_can_build_buggy_and_oracle_candidates(
    tmp_path,
):
    _write_quixbugs_fixture(tmp_path)
    output = tmp_path / "candidate_repairs.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_quixbugs_candidate_repairs.py",
            "--repo-path",
            str(tmp_path),
            "--program",
            "clamp",
            "--include-buggy-identity",
            "--include-oracle-correct",
            "--timeout-seconds",
            "10",
            "--seed",
            "123",
            "--output",
            str(output),
            "--experiment-id",
            "quixbugs_builtin_candidate_test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["metrics"]["candidate_count"] == 2
    assert record["metrics"]["final"]["candidate_passed"] == 1
    assert record["metrics"]["final"]["program_repair_rate"] == 1.0
    assert "--include-oracle-correct" in record["command"]
