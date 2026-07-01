from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QuixBugsRepairConfig:
    programs: tuple[str, ...] = ()
    timeout_seconds: int = 20
    seed: int = 123


@dataclass(frozen=True)
class QuixBugsCandidate:
    candidate_id: str
    program: str
    source_text: str
    generator_label: str


@dataclass(frozen=True)
class QuixBugsCandidateConfig:
    timeout_seconds: int = 20
    seed: int = 123


def evaluate_quixbugs_repair(
    repo_path: Path,
    config: QuixBugsRepairConfig | None = None,
) -> dict[str, Any]:
    config = config or QuixBugsRepairConfig()
    repo_path = Path(repo_path)
    commit = _git_commit(repo_path)
    programs = tuple(config.programs) or _discover_programs(repo_path)
    if not programs:
        raise ValueError("no QuixBugs Python programs selected")
    started = time.perf_counter()
    program_results = [
        _evaluate_program(repo_path, program, timeout_seconds=config.timeout_seconds)
        for program in programs
    ]
    buggy_passed = sum(1 for row in program_results if row["buggy"]["passed"])
    oracle_passed = sum(1 for row in program_results if row["oracle_correct"]["passed"])
    program_count = len(program_results)
    buggy_pass_rate = buggy_passed / program_count
    oracle_pass_rate = oracle_passed / program_count
    return {
        "benchmark_label": "quixbugs_python_repair_probe",
        "source": {
            "repo_url": "https://github.com/jkoppel/QuixBugs",
            "repo_path": str(repo_path),
            "repo_commit": commit,
        },
        "seed": config.seed,
        "program_count": program_count,
        "programs": program_results,
        "final": {
            "buggy_passed": buggy_passed,
            "oracle_correct_passed": oracle_passed,
            "buggy_pass_rate": buggy_pass_rate,
            "oracle_correct_pass_rate": oracle_pass_rate,
            "repair_gap": oracle_pass_rate - buggy_pass_rate,
            "runtime_ms": (time.perf_counter() - started) * 1000.0,
        },
        "limitations": [
            "not_model_generated_repairs",
            "oracle_correct_is_upper_bound",
            "python_subset_only",
            "pytest_result_depends_on_local_environment",
        ],
    }


def evaluate_quixbugs_candidate_repairs(
    repo_path: Path,
    candidates: list[QuixBugsCandidate],
    config: QuixBugsCandidateConfig | None = None,
) -> dict[str, Any]:
    config = config or QuixBugsCandidateConfig()
    repo_path = Path(repo_path)
    if not candidates:
        raise ValueError("no QuixBugs repair candidates supplied")
    commit = _git_commit(repo_path)
    started = time.perf_counter()
    candidate_results = [
        _evaluate_candidate(repo_path, candidate, timeout_seconds=config.timeout_seconds)
        for candidate in candidates
    ]
    programs = _summarize_candidate_programs(candidate_results)
    candidate_passed = sum(1 for row in candidate_results if row["passed"])
    program_count = len(programs)
    programs_with_passing_candidate = sum(
        1 for row in programs.values() if row["best_candidate_id"] is not None
    )
    return {
        "benchmark_label": "quixbugs_python_candidate_repair_probe",
        "source": {
            "repo_url": "https://github.com/jkoppel/QuixBugs",
            "repo_path": str(repo_path),
            "repo_commit": commit,
        },
        "seed": config.seed,
        "candidate_count": len(candidate_results),
        "program_count": program_count,
        "programs": programs,
        "candidates": candidate_results,
        "final": {
            "candidate_passed": candidate_passed,
            "candidate_pass_rate": candidate_passed / len(candidate_results),
            "programs_with_passing_candidate": programs_with_passing_candidate,
            "program_repair_rate": programs_with_passing_candidate / program_count,
            "runtime_ms": (time.perf_counter() - started) * 1000.0,
        },
        "limitations": [
            "not_model_generated_unless_candidate_file_is_model_output",
            "replacement_source_only",
            "python_subset_only",
            "pytest_result_depends_on_local_environment",
        ],
    }


def load_quixbugs_candidates_jsonl(path: Path) -> list[QuixBugsCandidate]:
    candidates: list[QuixBugsCandidate] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            candidates.append(
                QuixBugsCandidate(
                    candidate_id=str(row["candidate_id"]),
                    program=str(row["program"]),
                    source_text=str(row["source_text"]),
                    generator_label=str(row["generator_label"]),
                )
            )
        except KeyError as exc:
            raise ValueError(
                f"candidate row {line_number} missing field: {exc.args[0]}"
            ) from exc
    return candidates


def build_quixbugs_reference_candidates(
    repo_path: Path,
    programs: tuple[str, ...],
    *,
    include_buggy_identity: bool,
    include_oracle_correct: bool,
) -> list[QuixBugsCandidate]:
    repo_path = Path(repo_path)
    if not programs:
        raise ValueError("at least one program is required for reference candidates")
    if not include_buggy_identity and not include_oracle_correct:
        raise ValueError("no reference candidate type selected")
    candidates: list[QuixBugsCandidate] = []
    for program in programs:
        if include_buggy_identity:
            candidates.append(
                QuixBugsCandidate(
                    candidate_id=f"{program}:buggy_identity",
                    program=program,
                    source_text=_read_program_source(repo_path, "python_programs", program),
                    generator_label="buggy_identity_baseline",
                )
            )
        if include_oracle_correct:
            candidates.append(
                QuixBugsCandidate(
                    candidate_id=f"{program}:oracle_correct",
                    program=program,
                    source_text=_read_program_source(
                        repo_path, "correct_python_programs", program
                    ),
                    generator_label="oracle_correct_upper_bound",
                )
            )
    return candidates


def _discover_programs(repo_path: Path) -> tuple[str, ...]:
    testcase_dir = repo_path / "python_testcases"
    return tuple(
        sorted(
            path.stem.removeprefix("test_")
            for path in testcase_dir.glob("test_*.py")
            if path.is_file()
        )
    )


def _read_program_source(repo_path: Path, directory: str, program: str) -> str:
    path = repo_path / directory / f"{program}.py"
    if not path.exists():
        raise FileNotFoundError(f"missing QuixBugs source file: {path}")
    return path.read_text(encoding="utf-8")


def _evaluate_program(
    repo_path: Path,
    program: str,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    testcase = repo_path / "python_testcases" / f"test_{program}.py"
    if not testcase.exists():
        raise FileNotFoundError(f"missing QuixBugs testcase for program: {program}")
    buggy = _run_pytest(repo_path, testcase, correct=False, timeout_seconds=timeout_seconds)
    oracle = _run_pytest(repo_path, testcase, correct=True, timeout_seconds=timeout_seconds)
    return {
        "program": program,
        "testcase": str(testcase.relative_to(repo_path)),
        "buggy": buggy,
        "oracle_correct": oracle,
    }


def _evaluate_candidate(
    repo_path: Path,
    candidate: QuixBugsCandidate,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    testcase = repo_path / "python_testcases" / f"test_{candidate.program}.py"
    source_path = repo_path / "python_programs" / f"{candidate.program}.py"
    if not testcase.exists():
        raise FileNotFoundError(
            f"missing QuixBugs testcase for program: {candidate.program}"
        )
    if not source_path.exists():
        raise FileNotFoundError(
            f"missing QuixBugs source for program: {candidate.program}"
        )

    with tempfile.TemporaryDirectory(prefix="quixbugs-candidate-") as tmp:
        working_repo = Path(tmp) / "repo"
        shutil.copytree(
            repo_path,
            working_repo,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
        )
        replacement_path = working_repo / "python_programs" / f"{candidate.program}.py"
        replacement_path.write_text(candidate.source_text, encoding="utf-8")
        result = _run_pytest(
            working_repo,
            working_repo / "python_testcases" / f"test_{candidate.program}.py",
            correct=False,
            timeout_seconds=timeout_seconds,
        )
    return {
        "candidate_id": candidate.candidate_id,
        "program": candidate.program,
        "generator_label": candidate.generator_label,
        "source_sha256": hashlib.sha256(
            candidate.source_text.encode("utf-8")
        ).hexdigest(),
        "passed": result["passed"],
        "pytest": result,
    }


def _summarize_candidate_programs(
    candidate_results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    programs: dict[str, dict[str, Any]] = {}
    for row in candidate_results:
        program = row["program"]
        summary = programs.setdefault(
            program,
            {
                "candidate_count": 0,
                "passed_count": 0,
                "best_candidate_id": None,
                "generator_labels": [],
            },
        )
        summary["candidate_count"] += 1
        if row["generator_label"] not in summary["generator_labels"]:
            summary["generator_labels"].append(row["generator_label"])
        if row["passed"]:
            summary["passed_count"] += 1
            if summary["best_candidate_id"] is None:
                summary["best_candidate_id"] = row["candidate_id"]
    return dict(sorted(programs.items()))


def _run_pytest(
    repo_path: Path,
    testcase: Path,
    *,
    correct: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    command = ["python", "-m", "pytest", "-q"]
    if correct:
        command.append("--correct")
    command.append(str(testcase.relative_to(repo_path)))
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=repo_path,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        return {
            "command": " ".join(command),
            "passed": completed.returncode == 0,
            "returncode": completed.returncode,
            "runtime_ms": (time.perf_counter() - started) * 1000.0,
            "output_tail": _tail(completed.stdout),
        }
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        return {
            "command": " ".join(command),
            "passed": False,
            "returncode": None,
            "runtime_ms": (time.perf_counter() - started) * 1000.0,
            "output_tail": _tail(output + "\nTIMEOUT"),
        }


def _git_commit(repo_path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _tail(output: str, max_chars: int = 2000) -> str:
    return output[-max_chars:]
