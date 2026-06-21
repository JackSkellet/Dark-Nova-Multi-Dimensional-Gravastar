from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from weightlab.if3_block_codebook import (
    block_codebook_compress_state,
    run_if3_block_codebook_probe,
)


def test_block_codebook_counts_metadata_runtime_and_random_control():
    state = {
        "weight": torch.tensor(
            [
                [1.0, 1.1, -1.0, -1.1],
                [0.9, 1.2, -0.9, -1.2],
                [3.0, 3.1, -3.0, -3.1],
                [2.9, 3.2, -2.9, -3.2],
            ],
            dtype=torch.float32,
        ),
        "bias": torch.tensor([0.25, -0.25, 0.5, -0.5], dtype=torch.float32),
    }

    result = block_codebook_compress_state(
        state,
        block_size=4,
        codebook_size=2,
        residual_fraction=0.25,
        seed=123,
    )

    assert result["benchmark_label"] == "if3_block_codebook_state_compression"
    assert result["policy"]["block_size"] == 4
    assert result["policy"]["codebook_size"] == 2
    assert result["floating_parameter_count"] == 20
    assert result["learned_codebook"]["encoded_bytes"] > 0
    assert result["learned_codebook"]["metadata_bytes"] > 0
    assert result["learned_codebook"]["runtime_buffer_bytes"] == 80
    assert result["learned_codebook"]["encoded_plus_runtime_bytes"] > result[
        "learned_codebook"
    ]["encoded_bytes"]
    assert result["learned_codebook"]["residual_value_count"] > 0
    assert result["random_codebook_control"]["encoded_bytes"] == result["learned_codebook"][
        "encoded_bytes"
    ]
    assert result["learned_codebook"]["mse"] < result["random_codebook_control"]["mse"]
    assert result["learned_codebook"]["beats_random_control"] is True


def test_if3_probe_loads_model_checkpoint_and_reports_model_metadata(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model": {
                "linear.weight": torch.linspace(-1, 1, steps=32, dtype=torch.float32).reshape(4, 8),
                "linear.bias": torch.zeros(4, dtype=torch.float32),
            },
            "config": {"hidden_dim": 8, "layers": 1, "architecture_variant": "dense"},
            "step": 12,
            "checkpoint_type": "model_only",
        },
        checkpoint,
    )

    result = run_if3_block_codebook_probe(
        checkpoint,
        block_size=8,
        codebook_size=4,
        residual_fraction=0.125,
        seed=123,
    )

    assert result["candidate_id"] == "IF3"
    assert result["checkpoint"]["path"] == str(checkpoint)
    assert result["checkpoint"]["checkpoint_type"] == "model_only"
    assert result["checkpoint"]["step"] == 12
    assert result["model_config"]["hidden_dim"] == 8
    assert result["compression"]["learned_codebook"]["encoded_bytes"] > 0
    assert result["compression"]["random_codebook_control"]["mse"] >= 0


def test_if3_cli_writes_record(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model": {
                "linear.weight": torch.randn(4, 8),
                "linear.bias": torch.zeros(4),
            },
            "config": {"hidden_dim": 8, "layers": 1, "architecture_variant": "dense"},
            "step": 1,
            "checkpoint_type": "model_only",
        },
        checkpoint,
    )
    output = tmp_path / "if3.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_if3_block_codebook_probe.py",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--experiment-id",
            "if3_test",
            "--block-size",
            "8",
            "--codebook-size",
            "4",
            "--residual-fraction",
            "0.125",
            "--seed",
            "123",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "if3_test"
    assert record["metrics"]["candidate_id"] == "IF3"
    assert record["metrics"]["benchmark_label"] == "if3_block_codebook_checkpoint_probe"
