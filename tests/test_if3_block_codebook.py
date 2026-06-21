from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from weightlab.dense_training import DenseTrainingConfig, train_dense_decoder
from weightlab.if3_block_codebook import (
    block_codebook_compress_state,
    block_codebook_reconstruct_state,
    evaluate_if3_block_codebook_checkpoint,
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


def test_block_codebook_reconstructs_learned_and_random_state_dicts():
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

    result = block_codebook_reconstruct_state(
        state,
        block_size=4,
        codebook_size=2,
        residual_fraction=0.25,
        seed=123,
    )

    assert set(result["states"]) == {"learned_block_codebook", "random_block_codebook"}
    assert result["states"]["learned_block_codebook"]["weight"].shape == state["weight"].shape
    assert result["states"]["learned_block_codebook"]["bias"].shape == state["bias"].shape
    assert result["states"]["random_block_codebook"]["weight"].shape == state["weight"].shape
    assert result["compression"]["learned_codebook"]["mse"] < result["compression"][
        "random_codebook_control"
    ]["mse"]
    assert not torch.equal(result["states"]["learned_block_codebook"]["weight"], state["weight"])


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


def test_if3_validation_probe_compares_losses_on_same_heldout_batches(tmp_path):
    train_texts = [
        "function trainAlpha(value) { return value + 1; }\n" * 4,
        "function trainBeta(value) { return value + 2; }\n" * 4,
    ]
    heldout_texts = [
        "function heldoutAlpha(value) { return value + 3; }\n" * 4,
        "function heldoutBeta(value) { return value + 4; }\n" * 4,
    ]
    config = DenseTrainingConfig(
        device="cpu",
        seq_len=12,
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

    result = evaluate_if3_block_codebook_checkpoint(
        checkpoint_path=tmp_path / "dense_decoder_last_model_only.pt",
        texts=heldout_texts,
        split_name="validation",
        device="cpu",
        seed=999,
        batches=2,
        block_size=16,
        codebook_size=4,
        residual_fraction=0.0,
    )

    assert result["benchmark_label"] == "if3_block_codebook_validation_probe"
    assert result["candidate_id"] == "IF3"
    assert result["loss_evaluated"] is True
    assert result["packed_kernel_evaluated"] is False
    assert result["split"] == "validation"
    assert result["policies"]["fp32"]["loss"] > 0.0
    assert result["policies"]["learned_block_codebook"]["loss"] > 0.0
    assert result["policies"]["random_block_codebook"]["loss"] > 0.0
    assert result["policies"]["fp32"]["sample_order_sha256"] == result["policies"][
        "learned_block_codebook"
    ]["sample_order_sha256"]
    assert result["policies"]["fp32"]["sample_order_sha256"] == result["policies"][
        "random_block_codebook"
    ]["sample_order_sha256"]
    assert result["compression"]["learned_codebook"]["encoded_bytes"] > 0
    assert result["comparisons"]["learned_loss_delta_vs_fp32"] != 0.0


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


def test_if3_validation_cli_writes_loss_record(tmp_path):
    train_texts = [
        "function trainGamma(value) { return value + 5; }\n" * 4,
        "function trainDelta(value) { return value + 6; }\n" * 4,
    ]
    heldout_texts = [
        "function heldoutGamma(value) { return value + 7; }\n" * 4,
        "function heldoutDelta(value) { return value + 8; }\n" * 4,
    ]
    config = DenseTrainingConfig(
        device="cpu",
        seq_len=12,
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
    train_dense_decoder(train_texts, config, tmp_path / "train", seed=123)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(json.dumps({"text": text, "split": "validation"}) for text in heldout_texts)
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "if3_validation.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_if3_block_codebook_validation.py",
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
            "--batches",
            "2",
            "--block-size",
            "16",
            "--codebook-size",
            "4",
            "--residual-fraction",
            "0.0",
            "--output",
            str(output),
            "--experiment-id",
            "if3_validation_test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "if3_validation_test"
    assert record["metrics"]["benchmark_label"] == "if3_block_codebook_validation_probe"
    assert record["metrics"]["corpus"]["split_texts_loaded_separately"] is True
    assert record["metrics"]["resolved_config"]["batches"] == 2
    assert "--batches 2" in record["command"]
