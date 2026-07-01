from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from weightlab.dense_training import DenseTrainingConfig, train_dense_decoder
from weightlab.quixbugs_model_ranked_edits import (
    DenseQuixBugsEditRankConfig,
    evaluate_dense_ranked_quixbugs_edit_candidates,
    score_candidate_mean_nll,
)


class _ConstantTokenizer:
    name = "constant_test"
    vocab_size = 257
    eos_id = 256

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text] + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(token) for token in ids if token < 256)

    def to_jsonable(self, *, include_tokenizer_json: bool = False) -> dict[str, object]:
        del include_tokenizer_json
        return {"name": self.name, "vocab_size": self.vocab_size, "eos_id": self.eos_id}


class _PreferATokenModel(torch.nn.Module):
    seq_len = 16

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, 257, device=input_ids.device)
        logits[:, :, ord("a")] = 5.0
        return logits


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


def test_candidate_mean_nll_prefers_tokens_with_higher_model_probability():
    model = _PreferATokenModel()
    tokenizer = _ConstantTokenizer()

    score_a = score_candidate_mean_nll(model, tokenizer, "prompt:", "aaa")
    score_b = score_candidate_mean_nll(model, tokenizer, "prompt:", "bbb")

    assert score_a["mean_nll"] < score_b["mean_nll"]
    assert score_a["token_count"] == 4
    assert score_b["token_count"] == 4


def test_dense_ranked_edit_candidates_scores_pool_and_evaluates_top_choice(tmp_path):
    repo_path = tmp_path / "quixbugs"
    checkpoint_dir = tmp_path / "checkpoint"
    _write_possible_change_fixture(repo_path)
    checkpoint = _write_tiny_checkpoint(checkpoint_dir)

    result = evaluate_dense_ranked_quixbugs_edit_candidates(
        checkpoint,
        repo_path,
        programs=("possible_change",),
        config=DenseQuixBugsEditRankConfig(
            device="cpu",
            seed=123,
            timeout_seconds=10,
            max_candidates_per_program=8,
            top_candidates_per_program=1,
        ),
    )

    assert result["benchmark_label"] == "quixbugs_python_dense_ranked_edit_probe"
    assert result["candidate_pool"]["generated_candidate_count"] == 3
    assert result["candidate_pool"]["selected_candidate_count"] == 1
    assert result["ranking"][0]["program"] == "possible_change"
    assert "mean_nll" in result["ranking"][0]
    assert result["candidates"][0]["generator_label"] == "dense_ranked_ast_edit"
    assert "deterministic_candidate_pool" in result["limitations"]
    assert "not_free_form_generation" in result["limitations"]


def test_dense_ranked_edit_cli_writes_machine_readable_record(tmp_path):
    repo_path = tmp_path / "quixbugs"
    checkpoint_dir = tmp_path / "checkpoint"
    output = tmp_path / "ranked_edit.json"
    _write_possible_change_fixture(repo_path)
    checkpoint = _write_tiny_checkpoint(checkpoint_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_quixbugs_dense_ranked_edits.py",
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
            "8",
            "--top-candidates-per-program",
            "1",
            "--output",
            str(output),
            "--experiment-id",
            "quixbugs_dense_ranked_edit_test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "quixbugs_dense_ranked_edit_test"
    assert (
        record["metrics"]["benchmark_label"]
        == "quixbugs_python_dense_ranked_edit_probe"
    )
    assert record["metrics"]["candidate_pool"]["selected_candidate_count"] == 1
    assert "--top-candidates-per-program 1" in record["command"]
    assert (tmp_path / "manifest.json").exists()
