from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from weightlab.dense_training import DenseTrainingConfig, train_dense_decoder
from weightlab.quixbugs_model_candidates import (
    DenseQuixBugsCandidateConfig,
    evaluate_dense_quixbugs_candidate_repairs,
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


def _write_tiny_checkpoint(root: Path) -> Path:
    train_dense_decoder(
        [
            "def clamp(value, low, high):\n"
            "    return max(low, min(value, high))\n\n"
            "def other(value):\n"
            "    return value\n"
        ]
        * 4,
        DenseTrainingConfig(
            device="cpu",
            seq_len=32,
            hidden_dim=16,
            layers=1,
            heads=4,
            batch_size=2,
            steps=1,
            validation_batches=1,
            mixed_precision="fp32",
            optimizer_name="sgd",
            learning_rate=0.0,
            attention_mask_mode="finite_causal",
            block_impl="explicit_causal",
        ),
        root,
        seed=123,
    )
    return root / "dense_decoder_last_model_only.pt"


def test_dense_quixbugs_candidate_generation_scores_model_output(tmp_path):
    repo_path = tmp_path / "quixbugs"
    checkpoint_dir = tmp_path / "checkpoint"
    _write_quixbugs_fixture(repo_path)
    checkpoint = _write_tiny_checkpoint(checkpoint_dir)

    result = evaluate_dense_quixbugs_candidate_repairs(
        checkpoint,
        repo_path,
        programs=("clamp",),
        config=DenseQuixBugsCandidateConfig(
            device="cpu",
            seed=123,
            max_new_tokens=24,
            timeout_seconds=10,
        ),
    )

    assert result["benchmark_label"] == "quixbugs_python_dense_model_candidate_probe"
    assert result["program_count"] == 1
    assert result["candidate_count"] == 1
    assert result["model"]["checkpoint"] == str(checkpoint)
    assert result["generation"]["max_new_tokens"] == 24
    assert result["generated_candidates"][0]["program"] == "clamp"
    assert result["generated_candidates"][0]["source_sha256"]
    assert result["candidates"][0]["generator_label"] == "dense_checkpoint_greedy"
    assert "local_model_generated_candidate" in result["limitations"]


def test_dense_quixbugs_sampled_candidates_record_syntax_and_selection(tmp_path):
    repo_path = tmp_path / "quixbugs"
    checkpoint_dir = tmp_path / "checkpoint"
    _write_quixbugs_fixture(repo_path)
    checkpoint = _write_tiny_checkpoint(checkpoint_dir)

    result = evaluate_dense_quixbugs_candidate_repairs(
        checkpoint,
        repo_path,
        programs=("clamp",),
        config=DenseQuixBugsCandidateConfig(
            device="cpu",
            seed=123,
            max_new_tokens=24,
            timeout_seconds=10,
            samples_per_program=3,
            temperature=0.8,
            top_k=16,
            prefer_syntax_valid=True,
        ),
    )

    assert result["generation"]["method"] == "sampled_byte_generation"
    assert result["generation"]["samples_per_program"] == 3
    assert result["generation"]["temperature"] == 0.8
    assert result["generation"]["top_k"] == 16
    assert result["generation"]["prefer_syntax_valid"] is True
    assert result["generation"]["generated_candidate_count"] == 3
    assert result["candidate_count"] >= 1
    assert result["syntax"]["generated_candidate_count"] == 3
    assert "syntax_valid_candidate_count" in result["syntax"]
    assert result["generated_candidates"][0]["sample_index"] == 0
    assert "syntax_valid" in result["generated_candidates"][0]
    assert "selection_reason" in result["generated_candidates"][0]


def test_dense_quixbugs_candidate_cli_writes_machine_readable_record(tmp_path):
    repo_path = tmp_path / "quixbugs"
    checkpoint_dir = tmp_path / "checkpoint"
    output = tmp_path / "dense_quixbugs.json"
    _write_quixbugs_fixture(repo_path)
    checkpoint = _write_tiny_checkpoint(checkpoint_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_quixbugs_dense_candidates.py",
            "--checkpoint",
            str(checkpoint),
            "--repo-path",
            str(repo_path),
            "--program",
            "clamp",
            "--device",
            "cpu",
            "--seed",
            "123",
            "--max-new-tokens",
            "24",
            "--timeout-seconds",
            "10",
            "--output",
            str(output),
            "--experiment-id",
            "quixbugs_dense_candidate_test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "quixbugs_dense_candidate_test"
    assert (
        record["metrics"]["benchmark_label"]
        == "quixbugs_python_dense_model_candidate_probe"
    )
    assert record["metrics"]["candidate_count"] == 1
    assert "--max-new-tokens 24" in record["command"]
    assert (tmp_path / "manifest.json").exists()


def test_dense_quixbugs_candidate_cli_accepts_sampling_flags(tmp_path):
    repo_path = tmp_path / "quixbugs"
    checkpoint_dir = tmp_path / "checkpoint"
    output = tmp_path / "dense_quixbugs_sampled.json"
    _write_quixbugs_fixture(repo_path)
    checkpoint = _write_tiny_checkpoint(checkpoint_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_quixbugs_dense_candidates.py",
            "--checkpoint",
            str(checkpoint),
            "--repo-path",
            str(repo_path),
            "--program",
            "clamp",
            "--device",
            "cpu",
            "--seed",
            "123",
            "--max-new-tokens",
            "24",
            "--samples-per-program",
            "3",
            "--temperature",
            "0.8",
            "--top-k",
            "16",
            "--prefer-syntax-valid",
            "--timeout-seconds",
            "10",
            "--output",
            str(output),
            "--experiment-id",
            "quixbugs_dense_sampled_candidate_test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert (
        record["metrics"]["benchmark_label"]
        == "quixbugs_python_dense_model_candidate_probe"
    )
    assert record["metrics"]["generation"]["samples_per_program"] == 3
    assert record["metrics"]["generation"]["prefer_syntax_valid"] is True
    assert "--samples-per-program 3" in record["command"]
    assert "--prefer-syntax-valid" in record["command"]
