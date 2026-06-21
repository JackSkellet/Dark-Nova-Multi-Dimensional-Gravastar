from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from weightlab.d5_tokenizer_training_analysis import summarize_d5_tokenizer_training


def _record(
    experiment_id: str,
    *,
    tokenizer_name: str,
    vocab_size: int,
    token_count: int,
    bytes_per_token: float,
    validation_loss: float,
    train_tokens: int,
    tokens_per_second: float,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "status": "completed",
        "git_commit": "abc123",
        "git_dirty": False,
        "metrics": {
            "resolved_config": {"seq_len": 128},
            "tokenizer": {"name": tokenizer_name, "vocab_size": vocab_size},
            "model": {
                "total_parameters": 10,
                "trainable_parameters": 10,
                "architecture_variant": "dense",
                "hidden_dim": 528,
                "layers": 3,
                "heads": 8,
            },
            "training": {
                "train_tokens": train_tokens,
                "elapsed_s": 2.0,
                "tokens_per_second": tokens_per_second,
                "last_completed_step": 1,
            },
            "tokenization": {
                "train": {
                    "byte_count": 1000,
                    "character_count": 1000,
                    "token_count": token_count,
                    "bytes_per_token": bytes_per_token,
                    "characters_per_token": bytes_per_token,
                    "tokens_per_byte": 1.0 / bytes_per_token,
                    "tokens_per_character": 1.0 / bytes_per_token,
                    "document_count": 5,
                },
                "validation": {
                    "byte_count": 900,
                    "character_count": 900,
                    "token_count": token_count,
                    "bytes_per_token": bytes_per_token,
                    "characters_per_token": bytes_per_token,
                    "tokens_per_byte": 1.0 / bytes_per_token,
                    "tokens_per_character": 1.0 / bytes_per_token,
                    "document_count": 4,
                },
            },
            "validation": {
                "loss": validation_loss,
                "loss_nats_per_estimated_byte": validation_loss / bytes_per_token,
                "loss_nats_per_estimated_character": validation_loss / bytes_per_token,
                "tokens": 256,
            },
            "memory": {
                "peak_allocated_bytes": 100,
                "peak_reserved_bytes": 200,
            },
            "checkpoint": {
                "training_state_bytes": 300,
                "model_only_bytes": 100,
            },
        },
    }


def _exact_eval_record(
    experiment_id: str,
    *,
    exact_nats_per_raw_byte: float,
    exact_bits_per_raw_byte: float,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "status": "completed",
        "git_commit": "abc123",
        "git_dirty": False,
        "metrics": {
            "benchmark_label": "dense_checkpoint_evaluation",
            "split": "validation",
            "seed": 424242,
            "exact_byte_accounting": {
                "evaluated_target_tokens": 12,
                "evaluated_target_bytes": 24,
                "zero_byte_target_tokens": 0,
                "total_target_nll": exact_nats_per_raw_byte * 24,
                "exact_nats_per_raw_byte": exact_nats_per_raw_byte,
                "exact_bits_per_raw_byte": exact_bits_per_raw_byte,
            },
            "token_byte_nll_records": [
                {
                    "target_index": index,
                    "token_id": index,
                    "decoded_byte_length": 2,
                    "nll": exact_nats_per_raw_byte * 2,
                }
                for index in range(12)
            ],
        },
    }


def test_summarize_d5_tokenizer_training_compares_byte_normalized_runs(tmp_path):
    records = {
        "D5_byte_dense528_seed123_5m_equal_compute.json": _record(
            "byte",
            tokenizer_name="byte_level",
            vocab_size=257,
            token_count=1000,
            bytes_per_token=1.0,
            validation_loss=1.5,
            train_tokens=1000,
            tokens_per_second=100.0,
        ),
        "D5_bpe8192_dense528_seed123_5m_equal_compute.json": _record(
            "bpe_compute",
            tokenizer_name="hf_tokenizers_bpe_bytelevel",
            vocab_size=8192,
            token_count=400,
            bytes_per_token=2.5,
            validation_loss=3.0,
            train_tokens=1000,
            tokens_per_second=60.0,
        ),
        "D5_bpe8192_dense528_seed123_equal_raw_bytes.json": _record(
            "bpe_raw",
            tokenizer_name="hf_tokenizers_bpe_bytelevel",
            vocab_size=8192,
            token_count=400,
            bytes_per_token=2.5,
            validation_loss=4.5,
            train_tokens=400,
            tokens_per_second=60.0,
        ),
    }
    for name, record in records.items():
        (tmp_path / name).write_text(json.dumps(record), encoding="utf-8")
    exact_records = {
        "D5_byte_dense528_seed123_5m_equal_compute_exact_validation_eval.json": (
            _exact_eval_record(
                "byte_exact",
                exact_nats_per_raw_byte=1.6,
                exact_bits_per_raw_byte=2.3083120654223413,
            )
        ),
        "D5_bpe8192_dense528_seed123_5m_equal_compute_exact_validation_eval.json": (
            _exact_eval_record(
                "bpe_compute_exact",
                exact_nats_per_raw_byte=1.1,
                exact_bits_per_raw_byte=1.586498294977859,
            )
        ),
        "D5_bpe8192_dense528_seed123_equal_raw_bytes_exact_validation_eval.json": (
            _exact_eval_record(
                "bpe_raw_exact",
                exact_nats_per_raw_byte=1.9,
                exact_bits_per_raw_byte=2.7411274526890304,
            )
        ),
    }
    for name, record in exact_records.items():
        (tmp_path / name).write_text(json.dumps(record), encoding="utf-8")

    summary = summarize_d5_tokenizer_training(tmp_path)

    assert summary["comparisons"]["bpe_train_token_reduction_ratio"] == 2.5
    assert summary["conclusions"]["bpe_equal_compute_improves_loss_per_estimated_byte"]
    assert not summary["conclusions"][
        "bpe_equal_raw_bytes_improves_loss_per_estimated_byte"
    ]
    assert not summary["conclusions"]["token_reduction_alone_is_sufficient"]
    assert (
        summary["runs"]["bpe_equal_compute"]["context_coverage"][
            "train_estimated_bytes_per_context"
        ]
        == 320.0
    )
    assert summary["runs"]["byte_equal_compute"]["validation_exact_byte_accounting"][
        "exact_nats_per_raw_byte"
    ] == 1.6
    assert (
        summary["runs"]["byte_equal_compute"]["validation_loss_nats_per_estimated_byte"]
        != summary["runs"]["byte_equal_compute"]["validation_exact_byte_accounting"][
            "exact_nats_per_raw_byte"
        ]
    )
    assert summary["comparisons"]["equal_compute_bpe_minus_byte_exact_nats_per_raw_byte"] == -0.5
    assert summary["comparisons"][
        "equal_raw_bpe_minus_byte_exact_nats_per_raw_byte"
    ] == pytest.approx(0.3)
    assert summary["conclusions"]["bpe_equal_compute_improves_exact_nats_per_raw_byte"]
    assert not summary["conclusions"]["bpe_equal_raw_bytes_improves_exact_nats_per_raw_byte"]


def test_summarize_d5_tokenizer_training_cli_writes_record(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for source in [
        "D5_byte_dense528_seed123_5m_equal_compute.json",
        "D5_bpe8192_dense528_seed123_5m_equal_compute.json",
        "D5_bpe8192_dense528_seed123_equal_raw_bytes.json",
    ]:
        (results_dir / source).write_text(
            json.dumps(
                _record(
                    source,
                    tokenizer_name="byte_level" if "byte_dense" in source else "bpe",
                    vocab_size=257 if "byte_dense" in source else 8192,
                    token_count=1000 if "byte_dense" in source else 500,
                    bytes_per_token=1.0 if "byte_dense" in source else 2.0,
                    validation_loss=1.0 if "byte_dense" in source else 1.5,
                    train_tokens=1000,
                    tokens_per_second=100.0,
                )
            ),
            encoding="utf-8",
        )
    output_path = results_dir / "summary.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_d5_tokenizer_training.py",
            "--results-dir",
            str(results_dir),
            "--output",
            str(output_path),
            "--experiment-id",
            "summary_test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "summary_test"
    assert record["metrics"]["benchmark_label"] == "d5_trained_tokenizer_model_comparison"
    assert (results_dir / "manifest.json").exists()
