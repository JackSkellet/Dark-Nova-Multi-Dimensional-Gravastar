from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_train_fast_bpe_tokenizer_cli_writes_artifact_record_and_manifest(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        "\n".join(
            [
                json.dumps({
                    "text": "function parseConfig(value) { return value.trim(); }\n",
                    "split": "train",
                }),
                json.dumps({
                    "text": "function renderConfig(value) { return value.trim(); }\n",
                    "split": "train",
                }),
                json.dumps({
                    "text": "function validateConfig(value) { return value.trim(); }\n",
                    "split": "train",
                }),
                json.dumps({
                    "text": "function testConfig(value) { return value.trim(); }\n",
                    "split": "validation",
                }),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "result.json"
    tokenizer_path = tmp_path / "tokenizer.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_fast_bpe_tokenizer.py",
            "--corpus-jsonl",
            str(corpus_path),
            "--split",
            "train",
            "--max-train-documents",
            "2",
            "--vocab-size",
            "320",
            "--min-frequency",
            "1",
            "--seed",
            "123",
            "--tokenizer-output",
            str(tokenizer_path),
            "--output",
            str(output_path),
            "--experiment-id",
            "fast_bpe_cli_test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output_path.read_text(encoding="utf-8"))
    artifact = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert record["experiment_id"] == "fast_bpe_cli_test"
    assert record["metrics"]["tokenizer"]["checksum"] == artifact["checksum"]
    assert record["metrics"]["sampling"]["sampled_document_count"] == 2
    assert record["metrics"]["sampling"]["split"] == "train"
    assert record["metrics"]["sampling"]["deterministic_seed"] == 123
    assert record["resolved_config"]["tokenizer_output"] == str(tokenizer_path)
    assert artifact["training_config"]["vocab_size"] == 320
    assert artifact["training_config"]["min_frequency"] == 1
    assert artifact["training_config"]["sampled_document_count"] == 2
    assert manifest == [{"experiment_id": "fast_bpe_cli_test", "path": "result.json"}]
    assert "--max-train-documents 2" in record["command"]
