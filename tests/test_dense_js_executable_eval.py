from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from weightlab.dense_js_executable_eval import (
    ExecutableJsEvalConfig,
    _line_completion_candidates,
    _node_check_javascript,
    evaluate_dense_js_executable_checkpoint,
)
from weightlab.dense_training import DenseTrainingConfig, train_dense_decoder


def test_node_check_javascript_reports_syntax_pass_and_failure():
    valid = _node_check_javascript("function demo() { return 1; }\n")
    invalid = _node_check_javascript("function demo( { return 1; }\n")

    assert valid["available"] is True
    assert valid["passed"] is True
    assert invalid["available"] is True
    assert invalid["passed"] is False
    assert invalid["stderr"]


def test_line_completion_candidates_keep_only_oracle_syntax_passing_lines():
    texts = [
        """
        const value = 1;
        const broken = ;
        return 2;
        """,
    ]

    candidates = _line_completion_candidates(
        texts,
        prefix_chars=6,
        target_chars=24,
        node_binary="node",
        timeout_s=2.0,
        max_candidates=8,
    )

    assert candidates
    assert any(candidate["line"].startswith("const value") for candidate in candidates)
    assert all(candidate["oracle_node_check"]["passed"] for candidate in candidates)
    assert not any("broken" in candidate["line"] for candidate in candidates)


def test_dense_js_executable_eval_scores_generated_completion_syntax(tmp_path):
    train_texts = [
        "const trainAlpha = 1;\nconst trainBeta = 2;\n" * 8,
        "function trainDemo() { return 3; }\n" * 8,
    ]
    heldout_texts = [
        "const heldoutAlpha = 4;\nconst heldoutBeta = 5;\n" * 8,
        "function heldoutDemo() { return 6; }\n" * 8,
    ]
    config = DenseTrainingConfig(
        device="cpu",
        seq_len=16,
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
    )
    train_dense_decoder(train_texts, config, tmp_path, seed=123)

    result = evaluate_dense_js_executable_checkpoint(
        checkpoint_path=tmp_path / "dense_decoder_last_model_only.pt",
        texts=heldout_texts,
        split_name="validation",
        config=ExecutableJsEvalConfig(
            device="cpu",
            seed=999,
            tasks=4,
            prefix_chars=8,
            target_tokens=16,
            node_binary="node",
        ),
    )

    task = result["tasks"]["line_completion_syntax"]
    assert result["benchmark_label"] == "d4_dense_js_executable_checkpoint_evaluation"
    assert result["node"]["available"] is True
    assert task["completed_tasks"] == 4
    assert 0.0 <= task["generated_node_syntax_pass_rate"] <= 1.0
    assert task["oracle_node_syntax_pass_rate"] == 1.0
    assert len(task["examples"][0]["generated_node_check"]["stdout"]) >= 0


def test_dense_js_executable_cli_writes_record(tmp_path):
    train_texts = [
        "const trainGamma = 7;\nconst trainDelta = 8;\n" * 8,
        "function trainOther() { return 9; }\n" * 8,
    ]
    heldout_texts = [
        "const heldoutGamma = 10;\nconst heldoutDelta = 11;\n" * 8,
        "function heldoutOther() { return 12; }\n" * 8,
    ]
    train_dense_decoder(
        train_texts,
        DenseTrainingConfig(
            device="cpu",
            seq_len=16,
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
        tmp_path / "train",
        seed=123,
    )
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(json.dumps({"text": text, "split": "validation"}) for text in heldout_texts)
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "js_exec.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dense_js_executable.py",
            "--checkpoint",
            str(tmp_path / "train" / "dense_decoder_last_model_only.pt"),
            "--corpus-jsonl",
            str(corpus),
            "--split",
            "validation",
            "--device",
            "cpu",
            "--seed",
            "999",
            "--tasks",
            "4",
            "--prefix-chars",
            "8",
            "--target-tokens",
            "16",
            "--output",
            str(output),
            "--experiment-id",
            "js_exec_test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "js_exec_test"
    assert record["metrics"]["benchmark_label"] == "d4_dense_js_executable_checkpoint_evaluation"
    assert record["metrics"]["resolved_config"]["tasks"] == 4
    assert "--tasks 4" in record["command"]
