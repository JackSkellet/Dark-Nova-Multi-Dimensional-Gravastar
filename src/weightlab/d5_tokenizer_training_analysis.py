from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_RESULT_PATHS = {
    "byte_equal_compute": "D5_byte_dense528_seed123_5m_equal_compute.json",
    "bpe_equal_compute": "D5_bpe8192_dense528_seed123_5m_equal_compute.json",
    "bpe_equal_raw_bytes": "D5_bpe8192_dense528_seed123_equal_raw_bytes.json",
}

DEFAULT_EXACT_EVAL_PATHS = {
    "byte_equal_compute": "D5_byte_dense528_seed123_5m_equal_compute_exact_validation_eval.json",
    "bpe_equal_compute": (
        "D5_bpe8192_dense528_seed123_5m_equal_compute_exact_validation_eval.json"
    ),
    "bpe_equal_raw_bytes": (
        "D5_bpe8192_dense528_seed123_equal_raw_bytes_exact_validation_eval.json"
    ),
}


def summarize_d5_tokenizer_training(
    results_dir: Path,
    *,
    result_paths: dict[str, str] | None = None,
    exact_eval_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    paths = result_paths or DEFAULT_RESULT_PATHS
    exact_paths = exact_eval_paths or DEFAULT_EXACT_EVAL_PATHS
    records = {name: _load_record(results_dir / path) for name, path in paths.items()}
    exact_eval_records = {
        name: _load_record_if_exists(results_dir / path)
        for name, path in exact_paths.items()
    }
    runs = {
        name: _run_summary(name, record, exact_eval_records.get(name))
        for name, record in records.items()
    }

    byte_run = runs["byte_equal_compute"]
    bpe_compute = runs["bpe_equal_compute"]
    bpe_raw = runs["bpe_equal_raw_bytes"]
    train_reduction = _ratio(
        byte_run["tokenization"]["train"]["token_count"],
        bpe_compute["tokenization"]["train"]["token_count"],
    )
    validation_reduction = _ratio(
        byte_run["tokenization"]["validation"]["token_count"],
        bpe_compute["tokenization"]["validation"]["token_count"],
    )
    comparisons = {
        "bpe_train_token_reduction_ratio": train_reduction,
        "bpe_validation_token_reduction_ratio": validation_reduction,
        "equal_compute_bpe_minus_byte_nats_per_estimated_byte": (
            bpe_compute["validation_loss_nats_per_estimated_byte"]
            - byte_run["validation_loss_nats_per_estimated_byte"]
        ),
        "equal_compute_bpe_to_byte_estimated_byte_throughput_ratio": _ratio(
            bpe_compute["estimated_train_bytes_per_second"],
            byte_run["estimated_train_bytes_per_second"],
        ),
        "equal_raw_bpe_minus_byte_nats_per_estimated_byte": (
            bpe_raw["validation_loss_nats_per_estimated_byte"]
            - byte_run["validation_loss_nats_per_estimated_byte"]
        ),
        "equal_raw_bpe_to_byte_train_byte_budget_ratio": _ratio(
            bpe_raw["estimated_train_bytes_consumed"],
            byte_run["estimated_train_bytes_consumed"],
        ),
    }
    if _has_exact_byte_accounting(byte_run, bpe_compute, bpe_raw):
        comparisons.update(
            {
                "equal_compute_bpe_minus_byte_exact_nats_per_raw_byte": (
                    bpe_compute["validation_exact_byte_accounting"][
                        "exact_nats_per_raw_byte"
                    ]
                    - byte_run["validation_exact_byte_accounting"][
                        "exact_nats_per_raw_byte"
                    ]
                ),
                "equal_raw_bpe_minus_byte_exact_nats_per_raw_byte": (
                    bpe_raw["validation_exact_byte_accounting"][
                        "exact_nats_per_raw_byte"
                    ]
                    - byte_run["validation_exact_byte_accounting"][
                        "exact_nats_per_raw_byte"
                    ]
                ),
            }
        )
    conclusions = {
        "bpe_equal_compute_improves_loss_per_estimated_byte": (
            comparisons["equal_compute_bpe_minus_byte_nats_per_estimated_byte"] < 0
        ),
        "bpe_equal_raw_bytes_improves_loss_per_estimated_byte": (
            comparisons["equal_raw_bpe_minus_byte_nats_per_estimated_byte"] < 0
        ),
        "exact_raw_byte_loss_measured": _has_exact_byte_accounting(
            byte_run,
            bpe_compute,
            bpe_raw,
        ),
        "bpe_equal_compute_improves_exact_nats_per_raw_byte": (
            comparisons.get("equal_compute_bpe_minus_byte_exact_nats_per_raw_byte", 0.0)
            < 0
        )
        if _has_exact_byte_accounting(byte_run, bpe_compute, bpe_raw)
        else False,
        "bpe_equal_raw_bytes_improves_exact_nats_per_raw_byte": (
            comparisons.get("equal_raw_bpe_minus_byte_exact_nats_per_raw_byte", 0.0) < 0
        )
        if _has_exact_byte_accounting(byte_run, bpe_compute, bpe_raw)
        else False,
        "token_reduction_alone_is_sufficient": False,
        "functional_quality_measured": False,
    }
    return {
        "benchmark_label": "d5_trained_tokenizer_model_comparison",
        "runs": runs,
        "comparisons": comparisons,
        "conclusions": conclusions,
        "limitations": [
            "single_seed_pilot",
            "same_128_token_context_not_equal_attention_compute_per_raw_byte",
            "estimated_byte_loss_uses_split_level_bytes_per_token",
            "exact_byte_loss_available_only_when_exact_validation_eval_records_exist",
            "no_executable_functional_quality_evaluation_yet",
            "bpe_tokenizer_trained_on_sampled_train_documents_only",
        ],
    }


def _load_record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_record_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_record(path)


def _run_summary(
    name: str,
    record: dict[str, Any],
    exact_eval_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = record["metrics"]
    training = metrics["training"]
    validation = metrics["validation"]
    train_tok = metrics["tokenization"]["train"]
    validation_tok = metrics["tokenization"]["validation"]
    seq_len = int(metrics["resolved_config"]["seq_len"])
    train_tokens = int(training["train_tokens"])
    train_bytes_per_token = float(train_tok["bytes_per_token"])
    tokens_per_second = float(training["tokens_per_second"])
    model = metrics["model"]
    model_config = model.get("config", {})
    checkpoint = metrics["checkpoint"]
    total_parameters = model.get("total_parameters")
    if total_parameters is None:
        total_parameters = model["parameter_count"]
    trainable_parameters = model.get("trainable_parameters")
    if trainable_parameters is None:
        trainable_parameters = model["trainable_parameter_count"]
    training_state_bytes = checkpoint.get("training_state_bytes")
    if training_state_bytes is None:
        training_state_bytes = checkpoint["bytes"]
    run = {
        "experiment_id": record["experiment_id"],
        "protocol": name,
        "status": record["status"],
        "git_commit": record["git_commit"],
        "git_dirty": record["git_dirty"],
        "tokenizer": metrics["tokenizer"],
        "model": {
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "architecture_variant": model.get(
                "architecture_variant",
                model_config.get("architecture_variant"),
            ),
            "hidden_dim": model.get("hidden_dim", model_config.get("hidden_dim")),
            "layers": model.get("layers", model_config.get("layers")),
            "heads": model.get("heads", model_config.get("heads")),
            "seq_len": seq_len,
        },
        "training": {
            "train_token_units": train_tokens,
            "elapsed_s": float(training["elapsed_s"]),
            "tokens_per_second": tokens_per_second,
            "last_completed_step": int(training["last_completed_step"]),
        },
        "tokenization": metrics["tokenization"],
        "estimated_train_bytes_consumed": train_tokens * train_bytes_per_token,
        "estimated_train_bytes_per_second": tokens_per_second * train_bytes_per_token,
        "validation_loss_nats_per_token": float(validation["loss"]),
        "validation_loss_nats_per_estimated_byte": float(
            validation["loss_nats_per_estimated_byte"]
        ),
        "validation_loss_nats_per_estimated_character": float(
            validation["loss_nats_per_estimated_character"]
        ),
        "validation_tokens": int(validation["tokens"]),
        "context_coverage": {
            "train_estimated_bytes_per_context": seq_len * train_bytes_per_token,
            "validation_estimated_bytes_per_context": (
                seq_len * float(validation_tok["bytes_per_token"])
            ),
            "train_estimated_characters_per_context": (
                seq_len * float(train_tok["characters_per_token"])
            ),
            "validation_estimated_characters_per_context": (
                seq_len * float(validation_tok["characters_per_token"])
            ),
        },
        "memory": {
            "peak_allocated_bytes": metrics["memory"]["peak_allocated_bytes"],
            "peak_reserved_bytes": metrics["memory"]["peak_reserved_bytes"],
        },
        "checkpoint": {
            "training_state_bytes": training_state_bytes,
            "model_only_bytes": checkpoint["model_only_bytes"],
        },
    }
    if exact_eval_record is not None:
        exact = exact_eval_record["metrics"]["exact_byte_accounting"]
        run["validation_exact_byte_accounting"] = {
            "experiment_id": exact_eval_record["experiment_id"],
            "evaluated_target_tokens": int(exact["evaluated_target_tokens"]),
            "evaluated_target_bytes": int(exact["evaluated_target_bytes"]),
            "zero_byte_target_tokens": int(exact["zero_byte_target_tokens"]),
            "total_target_nll": float(exact["total_target_nll"]),
            "exact_nats_per_raw_byte": float(exact["exact_nats_per_raw_byte"]),
            "exact_bits_per_raw_byte": float(exact["exact_bits_per_raw_byte"]),
        }
    return run


def _has_exact_byte_accounting(*runs: dict[str, Any]) -> bool:
    return all("validation_exact_byte_accounting" in run for run in runs)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
