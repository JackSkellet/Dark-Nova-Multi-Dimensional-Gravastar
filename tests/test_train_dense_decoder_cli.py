from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_train_dense_decoder_cli_accepts_attention_mask_mode(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        "\n".join(
            [
                json.dumps({"text": "def add(a, b): return a + b", "split": "train"}),
                json.dumps({
                    "text": "README: add returns the sum of two values.",
                    "split": "train",
                }),
                json.dumps({"text": "assert add(1, 2) == 3", "split": "validation"}),
                json.dumps({"text": "assert add(2, 3) == 5", "split": "validation"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "result.json"
    output_dir = tmp_path / "artifacts"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_dense_decoder.py",
            "--corpus-jsonl",
            str(corpus_path),
            "--device",
            "cpu",
            "--seq-len",
            "16",
            "--hidden-dim",
            "32",
            "--layers",
            "1",
            "--heads",
            "4",
            "--batch-size",
            "2",
            "--steps",
            "1",
            "--validation-batches",
            "1",
            "--mixed-precision",
            "fp32",
            "--optimizer-name",
            "sgd",
            "--learning-rate",
            "0.0",
            "--attention-mask-mode",
            "bool_causal",
            "--max-documents",
            "0",
            "--output-dir",
            str(output_dir),
            "--output",
            str(output_path),
            "--experiment-id",
            "cli_attention_mask_mode",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["metrics"]["model"]["config"]["attention_mask_mode"] == "bool_causal"
    assert record["metrics"]["validation"]["source"] == "provided_validation_texts"
    assert record["metrics"]["validation"]["heldout_texts_provided"] is True
    assert record["metrics"]["corpus"]["document_count_used"] == 2
    assert record["metrics"]["corpus"]["validation_document_count_used"] == 2
    assert "--optimizer-name sgd" in record["command"]
    assert "--learning-rate 0.0" in record["command"]
    assert "--max-documents 0" in record["command"]
    assert "--seed 123" in record["command"]
    assert "--experiment-id cli_attention_mask_mode" in record["command"]
    assert record["resolved_config"]["optimizer_name"] == "sgd"
    assert record["resolved_config"]["learning_rate"] == 0.0
    assert record["resolved_config"]["max_documents"] == 0
    assert record["resolved_config"]["seed"] == 123
    assert record["metrics"]["resolved_config"]["experiment_id"] == "cli_attention_mask_mode"
    resolved_config_path = output_dir / "resolved_config.json"
    assert resolved_config_path.exists()
    resolved_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    assert resolved_config["attention_mask_mode"] == "bool_causal"
