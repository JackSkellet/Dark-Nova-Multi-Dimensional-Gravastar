from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from weightlab.quixbugs_edit_baseline import (
    QuixBugsEditBaselineConfig,
    build_quixbugs_edit_baseline_candidates,
    evaluate_quixbugs_edit_baseline,
)


def _write_gcd_fixture(root: Path) -> None:
    (root / "python_programs").mkdir(parents=True)
    (root / "python_testcases").mkdir(parents=True)
    (root / "conftest.py").write_text(
        "def pytest_addoption(parser):\n"
        "    parser.addoption('--correct', action='store_true')\n",
        encoding="utf-8",
    )
    (root / "python_programs" / "gcd.py").write_text(
        "def gcd(a, b):\n"
        "    if b == 0:\n"
        "        return a\n"
        "    return gcd(a % b, b)\n",
        encoding="utf-8",
    )
    (root / "python_testcases" / "test_gcd.py").write_text(
        "from python_programs.gcd import gcd\n\n"
        "def test_gcd():\n"
        "    assert gcd(35, 21) == 7\n"
        "    assert gcd(10, 5) == 5\n",
        encoding="utf-8",
    )


def test_edit_baseline_builds_syntax_valid_candidates_without_oracle_source(tmp_path):
    _write_gcd_fixture(tmp_path)

    candidates = build_quixbugs_edit_baseline_candidates(
        tmp_path,
        programs=("gcd",),
        max_candidates_per_program=8,
    )

    assert candidates
    assert {candidate.generator_label for candidate in candidates} == {
        "deterministic_ast_edit_baseline"
    }
    assert all(candidate.program == "gcd" for candidate in candidates)
    assert all(":edit_" in candidate.candidate_id for candidate in candidates)
    assert not (tmp_path / "correct_python_programs").exists()
    for candidate in candidates:
        ast.parse(candidate.source_text)


def test_edit_baseline_evaluates_generated_candidates_with_existing_pytest_harness(
    tmp_path,
):
    _write_gcd_fixture(tmp_path)

    result = evaluate_quixbugs_edit_baseline(
        tmp_path,
        QuixBugsEditBaselineConfig(
            programs=("gcd",),
            timeout_seconds=10,
            seed=123,
            max_candidates_per_program=8,
        ),
    )

    assert result["benchmark_label"] == "quixbugs_python_edit_baseline_probe"
    assert result["candidate_count"] >= 1
    assert result["program_count"] == 1
    assert result["final"]["programs_with_passing_candidate"] == 1
    assert result["final"]["program_repair_rate"] == 1.0
    assert result["generator"]["label"] == "deterministic_ast_edit_baseline"
    assert "not_model_generated" in result["limitations"]
    assert "does_not_read_oracle_correct_sources" in result["limitations"]


def test_edit_baseline_cli_writes_machine_readable_record(tmp_path):
    _write_gcd_fixture(tmp_path)
    output = tmp_path / "edit_baseline.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_quixbugs_edit_baseline.py",
            "--repo-path",
            str(tmp_path),
            "--program",
            "gcd",
            "--timeout-seconds",
            "10",
            "--seed",
            "123",
            "--max-candidates-per-program",
            "8",
            "--output",
            str(output),
            "--experiment-id",
            "quixbugs_edit_baseline_test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "quixbugs_edit_baseline_test"
    assert record["status"] == "completed"
    assert record["metrics"]["benchmark_label"] == "quixbugs_python_edit_baseline_probe"
    assert record["metrics"]["final"]["program_repair_rate"] == 1.0
    assert "--max-candidates-per-program 8" in record["command"]
    assert (tmp_path / "manifest.json").exists()
