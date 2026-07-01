from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from weightlab.dense_training import DenseTrainingConfig, train_dense_decoder
from weightlab.quixbugs_edit_baseline import build_quixbugs_edit_baseline_candidates
from weightlab.quixbugs_repair import QuixBugsCandidate
from weightlab.quixbugs_syntax_mutations import (
    DenseQuixBugsSyntaxPoolRankConfig,
    _repair_aware_order_rows,
    build_quixbugs_syntax_mutation_candidates,
    evaluate_dense_ranked_quixbugs_syntax_pool,
    evaluate_dense_ranked_quixbugs_syntax_pool_topk_profile,
    evaluate_quixbugs_syntax_pool_ordering_controls,
)


def _write_possible_change_fixture(root: Path) -> None:
    (root / "python_programs").mkdir(parents=True)
    (root / "python_testcases").mkdir(parents=True)
    (root / "conftest.py").write_text(
        "def pytest_addoption(parser):\n"
        "    parser.addoption('--correct', action='store_true')\n",
        encoding="utf-8",
    )
    (root / "python_programs" / "possible_change.py").write_text(
        "def possible_change(coins, total):\n"
        "    if total == 0:\n"
        "        return 1\n"
        "    if total < 0:\n"
        "        return 0\n"
        "    first, *rest = coins\n"
        "    return possible_change(coins, total - first) + possible_change(rest, total)\n",
        encoding="utf-8",
    )
    (root / "python_testcases" / "test_possible_change.py").write_text(
        "from python_programs.possible_change import possible_change\n\n"
        "def test_possible_change():\n"
        "    assert possible_change([1, 5, 10, 25], 11) == 4\n"
        "    assert possible_change([2, 5], 0) == 1\n",
        encoding="utf-8",
    )


def _write_generic_only_fixture(root: Path) -> None:
    (root / "python_programs").mkdir(parents=True)
    (root / "python_programs" / "generic_math.py").write_text(
        "def generic_math(value):\n"
        "    return value + 1\n",
        encoding="utf-8",
    )


def _write_parenthesization_fixture(root: Path) -> None:
    (root / "python_programs").mkdir(parents=True)
    (root / "python_programs" / "is_valid_parenthesization.py").write_text(
        "def is_valid_parenthesization(parens):\n"
        "    depth = 0\n"
        "    for paren in parens:\n"
        "        if paren == '(':\n"
        "            depth += 1\n"
        "        else:\n"
        "            depth -= 1\n"
        "            if depth < 0:\n"
        "                return False\n"
        "    return True\n",
        encoding="utf-8",
    )


def _write_tiny_checkpoint(root: Path) -> Path:
    train_dense_decoder(
        [
            "def possible_change(coins, total):\n"
            "    if not coins:\n"
            "        return 0\n"
            "    return possible_change(coins, total)\n"
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


def test_syntax_mutation_pool_adds_syntax_valid_distractors_without_oracle_source(
    tmp_path,
):
    _write_possible_change_fixture(tmp_path)

    baseline = build_quixbugs_edit_baseline_candidates(
        tmp_path,
        programs=("possible_change",),
        max_candidates_per_program=16,
    )
    syntax_pool = build_quixbugs_syntax_mutation_candidates(
        tmp_path,
        programs=("possible_change",),
        max_candidates_per_program=32,
    )

    assert len(syntax_pool) > len(baseline)
    assert {candidate.generator_label for candidate in syntax_pool} == {
        "syntax_preserving_ast_mutation_pool"
    }
    assert not (tmp_path / "correct_python_programs").exists()
    for candidate in syntax_pool:
        ast.parse(candidate.source_text)


def test_syntax_mutation_pool_keeps_generic_mutations_when_edit_baseline_empty(
    tmp_path,
):
    _write_generic_only_fixture(tmp_path)

    syntax_pool = build_quixbugs_syntax_mutation_candidates(
        tmp_path,
        programs=("generic_math",),
        max_candidates_per_program=8,
    )

    assert syntax_pool
    assert {candidate.program for candidate in syntax_pool} == {"generic_math"}
    assert all(
        candidate.generator_label == "syntax_preserving_ast_mutation_pool"
        for candidate in syntax_pool
    )
    assert all("value + 1" not in candidate.source_text for candidate in syntax_pool)
    for candidate in syntax_pool:
        ast.parse(candidate.source_text)


def test_syntax_mutation_pool_can_add_final_counter_zero_check(tmp_path):
    _write_parenthesization_fixture(tmp_path)

    syntax_pool = build_quixbugs_syntax_mutation_candidates(
        tmp_path,
        programs=("is_valid_parenthesization",),
        max_candidates_per_program=64,
    )

    assert any(
        "return depth == 0" in candidate.source_text for candidate in syntax_pool
    )


def test_repair_aware_order_prioritizes_final_counter_zero_check():
    buggy_source = (
        "def is_valid_parenthesization(parens):\n"
        "    depth = 0\n"
        "    for paren in parens:\n"
        "        if paren == '(':\n"
        "            depth += 1\n"
        "        else:\n"
        "            depth -= 1\n"
        "            if depth < 0:\n"
        "                return False\n"
        "    return True\n"
    )
    repaired_source = buggy_source.replace("return True", "return depth == 0")
    rows = _repair_aware_order_rows(
        [
            QuixBugsCandidate(
                candidate_id="is_valid_parenthesization:buggy",
                program="is_valid_parenthesization",
                source_text=buggy_source,
                generator_label="test",
            ),
            QuixBugsCandidate(
                candidate_id="is_valid_parenthesization:counter_zero",
                program="is_valid_parenthesization",
                source_text=repaired_source,
                generator_label="test",
            ),
        ]
    )

    assert rows[0]["candidate_id"] == "is_valid_parenthesization:counter_zero"
    assert "final_counter_zero_check" in rows[0]["repair_aware_reasons"]


def test_dense_ranked_syntax_pool_records_broader_pool_metadata(tmp_path):
    repo_path = tmp_path / "quixbugs"
    checkpoint_dir = tmp_path / "checkpoint"
    _write_possible_change_fixture(repo_path)
    checkpoint = _write_tiny_checkpoint(checkpoint_dir)

    result = evaluate_dense_ranked_quixbugs_syntax_pool(
        checkpoint,
        repo_path,
        programs=("possible_change",),
        config=DenseQuixBugsSyntaxPoolRankConfig(
            device="cpu",
            seed=123,
            timeout_seconds=10,
            max_candidates_per_program=32,
            top_candidates_per_program=1,
        ),
    )

    assert result["benchmark_label"] == "quixbugs_python_dense_ranked_syntax_pool_probe"
    assert result["candidate_pool"]["source"] == "syntax_preserving_ast_mutation_pool"
    assert result["candidate_pool"]["generated_candidate_count"] > 3
    assert result["candidate_pool"]["selected_candidate_count"] == 1
    assert result["ranking"][0]["program"] == "possible_change"
    assert "mean_nll" in result["ranking"][0]
    assert result["candidates"][0]["generator_label"] == "dense_ranked_syntax_pool"
    assert "broader_than_deterministic_edit_baseline" in result["limitations"]


def test_dense_ranked_syntax_pool_topk_profile_reuses_one_ranked_pool(tmp_path):
    repo_path = tmp_path / "quixbugs"
    checkpoint_dir = tmp_path / "checkpoint"
    _write_possible_change_fixture(repo_path)
    checkpoint = _write_tiny_checkpoint(checkpoint_dir)

    result = evaluate_dense_ranked_quixbugs_syntax_pool_topk_profile(
        checkpoint,
        repo_path,
        programs=("possible_change",),
        top_k_values=(1, 2),
        config=DenseQuixBugsSyntaxPoolRankConfig(
            device="cpu",
            seed=123,
            timeout_seconds=10,
            max_candidates_per_program=32,
            top_candidates_per_program=1,
        ),
    )

    assert (
        result["benchmark_label"]
        == "quixbugs_python_dense_ranked_syntax_topk_probe"
    )
    assert result["candidate_pool"]["max_top_k_profiled"] == 2
    assert result["candidate_pool"]["selected_candidate_count"] == 2
    assert [row["top_candidates_per_program"] for row in result["top_k_profile"]] == [
        1,
        2,
    ]
    assert result["top_k_profile"][0]["candidate_count"] == 1
    assert result["top_k_profile"][1]["candidate_count"] == 2
    assert (
        result["top_k_profile"][1]["program_repair_rate"]
        >= result["top_k_profile"][0]["program_repair_rate"]
    )
    assert result["final"]["best_top_k"] in {1, 2}
    assert "top_k_execution_profile" in result["limitations"]


def test_syntax_pool_ordering_controls_compare_dense_with_non_model_orders(tmp_path):
    repo_path = tmp_path / "quixbugs"
    checkpoint_dir = tmp_path / "checkpoint"
    _write_possible_change_fixture(repo_path)
    checkpoint = _write_tiny_checkpoint(checkpoint_dir)

    result = evaluate_quixbugs_syntax_pool_ordering_controls(
        checkpoint,
        repo_path,
        programs=("possible_change",),
        top_k_values=(1, 2),
        config=DenseQuixBugsSyntaxPoolRankConfig(
            device="cpu",
            seed=123,
            timeout_seconds=10,
            max_candidates_per_program=32,
            top_candidates_per_program=1,
        ),
    )

    assert (
        result["benchmark_label"]
        == "quixbugs_python_syntax_pool_ordering_control_probe"
    )
    assert set(result["ordering_controls"]) == {
        "dense_likelihood",
        "deterministic_pool_order",
        "repair_aware_static_order",
        "random_seeded_order",
    }
    for control in result["ordering_controls"].values():
        assert [row["top_candidates_per_program"] for row in control["top_k_profile"]] == [
            1,
            2,
        ]
        assert control["candidate_count"] <= result["candidate_pool"][
            "selected_candidate_count"
        ]
        assert 0.0 <= control["best_program_repair_rate"] <= 1.0
    assert result["final"]["control_names"] == [
        "dense_likelihood",
        "deterministic_pool_order",
        "repair_aware_static_order",
        "random_seeded_order",
    ]
    assert "same_pool_ordering_controls" in result["limitations"]


def test_dense_ranked_syntax_pool_cli_writes_machine_readable_record(tmp_path):
    repo_path = tmp_path / "quixbugs"
    checkpoint_dir = tmp_path / "checkpoint"
    output = tmp_path / "ranked_syntax_pool.json"
    _write_possible_change_fixture(repo_path)
    checkpoint = _write_tiny_checkpoint(checkpoint_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_quixbugs_dense_ranked_syntax_pool.py",
            "--checkpoint",
            str(checkpoint),
            "--repo-path",
            str(repo_path),
            "--program",
            "possible_change",
            "--device",
            "cpu",
            "--seed",
            "123",
            "--timeout-seconds",
            "10",
            "--max-candidates-per-program",
            "32",
            "--top-candidates-per-program",
            "1",
            "--output",
            str(output),
            "--experiment-id",
            "quixbugs_dense_ranked_syntax_pool_test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "quixbugs_dense_ranked_syntax_pool_test"
    assert (
        record["metrics"]["benchmark_label"]
        == "quixbugs_python_dense_ranked_syntax_pool_probe"
    )
    assert record["metrics"]["candidate_pool"]["generated_candidate_count"] > 3
    assert "--max-candidates-per-program 32" in record["command"]
    assert (tmp_path / "manifest.json").exists()


def test_dense_ranked_syntax_topk_cli_writes_machine_readable_record(tmp_path):
    repo_path = tmp_path / "quixbugs"
    checkpoint_dir = tmp_path / "checkpoint"
    output = tmp_path / "ranked_syntax_topk.json"
    _write_possible_change_fixture(repo_path)
    checkpoint = _write_tiny_checkpoint(checkpoint_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_quixbugs_dense_ranked_syntax_topk.py",
            "--checkpoint",
            str(checkpoint),
            "--repo-path",
            str(repo_path),
            "--program",
            "possible_change",
            "--device",
            "cpu",
            "--seed",
            "123",
            "--timeout-seconds",
            "10",
            "--max-candidates-per-program",
            "32",
            "--top-k",
            "1",
            "--top-k",
            "2",
            "--output",
            str(output),
            "--experiment-id",
            "quixbugs_dense_ranked_syntax_topk_test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "quixbugs_dense_ranked_syntax_topk_test"
    assert (
        record["metrics"]["benchmark_label"]
        == "quixbugs_python_dense_ranked_syntax_topk_probe"
    )
    assert record["metrics"]["candidate_pool"]["max_top_k_profiled"] == 2
    assert "--top-k 2" in record["command"]
    assert (tmp_path / "manifest.json").exists()


def test_syntax_pool_ordering_controls_cli_writes_machine_readable_record(tmp_path):
    repo_path = tmp_path / "quixbugs"
    checkpoint_dir = tmp_path / "checkpoint"
    output = tmp_path / "syntax_controls.json"
    _write_possible_change_fixture(repo_path)
    checkpoint = _write_tiny_checkpoint(checkpoint_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_quixbugs_syntax_pool_ordering_controls.py",
            "--checkpoint",
            str(checkpoint),
            "--repo-path",
            str(repo_path),
            "--program",
            "possible_change",
            "--device",
            "cpu",
            "--seed",
            "123",
            "--timeout-seconds",
            "10",
            "--max-candidates-per-program",
            "32",
            "--top-k",
            "1",
            "--top-k",
            "2",
            "--output",
            str(output),
            "--experiment-id",
            "quixbugs_syntax_pool_ordering_controls_test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "quixbugs_syntax_pool_ordering_controls_test"
    assert (
        record["metrics"]["benchmark_label"]
        == "quixbugs_python_syntax_pool_ordering_control_probe"
    )
    assert "random_seeded_order" in record["metrics"]["ordering_controls"]
    assert "repair_aware_static_order" in record["metrics"]["ordering_controls"]
    assert "--top-k 2" in record["command"]
    assert (tmp_path / "manifest.json").exists()
