from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from weightlab.dense_training import ByteTokenizer, DenseDecoder


def _write_tiny_checkpoint(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteTokenizer()
    model = DenseDecoder(
        tokenizer.vocab_size,
        seq_len=32,
        hidden_dim=16,
        layers=1,
        heads=2,
        attention_mask_mode="additive_causal",
        architecture_variant="dense",
    )
    checkpoint = root / "dense_decoder_last_model_only.pt"
    torch.save(
        {
            "checkpoint_type": "model_only",
            "step": 7,
            "config": {
                "seq_len": 32,
                "hidden_dim": 16,
                "layers": 1,
                "heads": 2,
                "attention_mask_mode": "additive_causal",
                "architecture_variant": "dense",
            },
            "model": model.state_dict(),
            "tokenizer": tokenizer.to_jsonable(),
        },
        checkpoint,
    )
    return checkpoint


def test_local_dense_prediction_cli_writes_saved_outputs_for_glm_harness(
    tmp_path: Path,
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    tasks.write_text(
        json.dumps(
            {
                "task_id": "synthetic_api_reuse_001",
                "task_family": "api_reuse",
                "source_policy": "synthetic",
                "prompt": "Return a JavaScript call to normalizeUser(rawUser).",
                "expected_substrings": ["normalizeUser("],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/generate_glm_local_dense_predictions.py",
            "--tasks-jsonl",
            str(tasks),
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--seed",
            "123",
            "--max-new-tokens",
            "4",
            "--output",
            str(predictions),
            "--model-label",
            "tiny-dense-test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    rows = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["task_id"] == "synthetic_api_reuse_001"
    assert row["model"] == "tiny-dense-test"
    assert row["tool_access"] == "none"
    assert row["temperature"] == 0.0
    assert row["cost_usd"] == 0.0
    assert row["tokens_in"] > 0
    assert row["tokens_out"] <= 4
    assert row["context_length"] == row["tokens_in"]
    assert row["output_length"] == row["tokens_out"]
    assert isinstance(row["output"], str)
