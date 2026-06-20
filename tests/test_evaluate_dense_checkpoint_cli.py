from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from weightlab.dense_training import DenseTrainingConfig, train_dense_decoder


def test_evaluate_dense_checkpoint_cli_uses_requested_split(tmp_path):
    train_texts = [
        "function trainA() { return 1 }",
        "function trainB() { return 2 }",
    ]
    config = DenseTrainingConfig(
        device="cpu",
        seq_len=8,
        hidden_dim=16,
        layers=1,
        heads=4,
        batch_size=2,
        steps=1,
        validation_batches=1,
        mixed_precision="fp32",
        optimizer_name="sgd",
        learning_rate=0.0,
    )
    train_dense_decoder(train_texts, config, tmp_path / "checkpoint", seed=123)

    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        "\n".join(
            [
                json.dumps({"text": "function trainA() { return 1 }", "split": "train"}),
                json.dumps({
                    "text": "function validationA() { return 3 }",
                    "split": "validation",
                }),
                json.dumps({"text": "function testA() { return 4 }", "split": "test"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "result.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dense_checkpoint.py",
            "--checkpoint",
            str(tmp_path / "checkpoint" / "dense_decoder_last.pt"),
            "--corpus-jsonl",
            str(corpus_path),
            "--split",
            "validation",
            "--device",
            "cpu",
            "--seed",
            "999",
            "--batches",
            "2",
            "--output",
            str(output_path),
            "--experiment-id",
            "cli_checkpoint_eval",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "cli_checkpoint_eval"
    assert record["metrics"]["benchmark_label"] == "dense_checkpoint_evaluation"
    assert record["metrics"]["split"] == "validation"
    assert record["metrics"]["batches"] == 2
    assert record["metrics"]["corpus"]["document_count_used"] == 1
    assert record["metrics"]["corpus"]["split_texts_loaded_separately"] is True
    assert record["resolved_config"]["seed"] == 999
