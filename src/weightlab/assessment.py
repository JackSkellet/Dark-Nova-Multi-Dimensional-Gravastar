from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

Assessment = dict[str, Any]


NEXT_RESEARCH_OPTIONS = [
    "larger_task_sensitive_model_tensors",
    "upstream_test_suite_patch_mining",
    "retrieval_first_update_gates",
    "multilingual_file_format_security_red_team",
    "activation_residual_cache",
    "ephemeral_repository_adapters",
    "structured_external_repository_memory",
    "documentation_drift_detection",
]


def assess_record(record: dict[str, Any]) -> Assessment:
    experiment_id = str(record["experiment_id"])
    if experiment_id == "E2_compositional_storage":
        return _assess_e2(record)
    if experiment_id == "E3j_real_open_model_natural_text_precision":
        return _assess_e3j(record)
    if experiment_id == "E3k_real_open_model_internal_tensor_precision":
        return _assess_e3k(record)
    if experiment_id == "E4c_torch_batched_routed_execution":
        return _assess_e4c(record)
    if experiment_id == "E4d_rocm_transfer_scaling":
        return _assess_e4d(record)
    if experiment_id == "R1_rocm_training_validation":
        return _assess_r1(record)
    if record.get("metrics", {}).get("benchmark_label") == "licensed_repository_corpus_preparation":
        return _assess_d1(record)
    if record.get("metrics", {}).get("benchmark_label") == "hf_corpus_manifest":
        return _assess_hf_corpus_manifest(record)
    if record.get("metrics", {}).get("benchmark_label") == "hf_corpus_materialization":
        return _assess_hf_corpus_materialization(record)
    if record.get("metrics", {}).get("benchmark_label") == "dense_decoder_training_smoke":
        return _assess_t1(record)
    if record.get("metrics", {}).get("benchmark_label") == "dense_step_stability_debug":
        return _assess_t3(record)
    if experiment_id == "R2_rocm_transformer_stability":
        return _assess_r2(record)
    if experiment_id == "E5g_public_patch_replay_suite":
        return _assess_e5g(record)
    if experiment_id == "S2c_semantic_extraction_red_team":
        return _assess_s2c(record)
    if experiment_id == "E6a_structured_repository_memory":
        return _assess_e6a(record)
    if experiment_id == "E6b_public_repository_memory_qa":
        return _assess_e6b(record)
    if experiment_id == "E6c_public_repository_signature_qa":
        return _assess_e6c(record)
    if experiment_id == "E6d_public_repository_call_stub_generation":
        return _assess_e6d(record)
    if experiment_id == "E6e_public_repository_function_skeleton_generation":
        return _assess_e6e(record)
    if experiment_id == "E6f_public_repository_docstring_skeleton_generation":
        return _assess_e6f(record)
    if experiment_id == "E6g_public_repository_api_reference_generation":
        return _assess_e6g(record)
    if experiment_id == "E6h_public_repository_api_doc_coverage_qa":
        return _assess_e6h(record)
    if experiment_id == "E6i_synthetic_api_doc_drift_detection":
        return _assess_e6i(record)
    return {
        "experiment_id": experiment_id,
        "hypothesis": record.get("hypothesis"),
        "outcome": "recorded",
        "supports_pareto_improvement": False,
        "primary_reason": "not_classified_by_current_assessment_layer",
        "limitations": ["manual_review_required"],
        "evidence": {},
    }


def assess_manifest(results_dir: Path) -> Assessment:
    manifest = json.loads((results_dir / "manifest.json").read_text())
    records = [_load_manifest_record(results_dir, row) for row in manifest]
    assessments = [assess_record(record) for record in records]
    outcome_counts = Counter(str(row["outcome"]) for row in assessments)
    return {
        "record_count": len(records),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "pareto_improvement_found": any(
            bool(row["supports_pareto_improvement"]) for row in assessments
        ),
        "assessments": assessments,
        "next_research_options": NEXT_RESEARCH_OPTIONS,
    }


def _load_manifest_record(results_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    if "path" in row:
        return json.loads((results_dir / row["path"]).read_text())
    return row


def _assess_e2(record: dict[str, Any]) -> Assessment:
    methods = {row["method"]: row for row in record["metrics"]["methods"]}
    int8 = methods["int8_uniform"]
    compositional_names = [
        "low_rank_svd",
        "low_rank_sparse_residual",
        "rank1_outer_product",
        "shared_basis_coefficients",
        "tensorized_two_block",
        "product_quantized_rows",
        "kronecker_rank1",
        "tensor_train_4d",
    ]
    pareto_methods = [
        name
        for name in compositional_names
        if methods[name]["total_bytes"] < int8["total_bytes"]
        and methods[name]["reconstruction_mse"] < int8["reconstruction_mse"]
    ]
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "rejected" if not pareto_methods else "mixed",
        "supports_pareto_improvement": bool(pareto_methods),
        "primary_reason": "uniform_int8_dominates_tested_compositional_methods",
        "limitations": [
            "metadata_accounting_included",
            "cpu_reduced_matrix",
            "no_compositional_pareto_win",
        ],
        "evidence": {
            "uniform_int8_bytes": int8["total_bytes"],
            "uniform_int8_mse": int8["reconstruction_mse"],
            "pareto_compositional_methods": pareto_methods,
            "metadata_accounting": True,
        },
    }


def _assess_e3j(record: dict[str, Any]) -> Assessment:
    policies = record["metrics"]["precision_policies"]
    selected_bf16 = policies["output_error_bf16_protected"]
    selected_fp32 = policies["output_error_fp32_protected"]
    random_bf16_mean = policies["random_bf16_protected_mean"]
    random_bf16_best = policies["random_bf16_protected_best"]
    uniform_int4 = policies["uniform_int4"]
    full_fp32 = policies["full_fp32"]
    beats_random_mean = (
        selected_bf16["heldout_kl_divergence"]
        < random_bf16_mean["heldout_kl_divergence"]
    )
    beats_best_random = (
        selected_bf16["heldout_kl_divergence"]
        <= random_bf16_best["heldout_kl_divergence"]
    )
    storage_win = selected_bf16["total_bytes"] < full_fp32["total_bytes"]
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "mixed" if beats_random_mean else "rejected",
        "supports_pareto_improvement": bool(beats_best_random and storage_win),
        "primary_reason": "selected_rows_beat_random_mean_but_lose_storage",
        "limitations": [
            "not_pareto_improvement",
            "storage_loss",
            "does_not_beat_best_random",
            "tiny_open_model_smoke_test",
        ],
        "evidence": {
            "selected_bf16_kl": selected_bf16["heldout_kl_divergence"],
            "selected_fp32_kl": selected_fp32["heldout_kl_divergence"],
            "random_bf16_mean_kl": random_bf16_mean["heldout_kl_divergence"],
            "random_bf16_best_kl": random_bf16_best["heldout_kl_divergence"],
            "selected_bf16_bytes": selected_bf16["total_bytes"],
            "uniform_int4_bytes": uniform_int4["total_bytes"],
            "full_fp32_bytes": full_fp32["total_bytes"],
        },
    }


def _assess_e3k(record: dict[str, Any]) -> Assessment:
    policies = record["metrics"]["precision_policies"]
    kl_values = [
        float(policy.get("heldout_kl_divergence", 0.0)) for policy in policies.values()
    ]
    zero_signal = all(abs(value) <= 1e-12 for value in kl_values)
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "no_signal" if zero_signal else "mixed",
        "supports_pareto_improvement": False,
        "primary_reason": "all_precision_policies_have_zero_measurable_kl",
        "limitations": [
            "tiny_tensor",
            "zero_task_sensitivity",
            "selective_precision_not_supported",
        ],
        "evidence": {
            "max_abs_heldout_kl": max(abs(value) for value in kl_values),
            "tensor_shape": record["metrics"]["tensor"]["shape"],
            "protected_count": record["metrics"]["protected_count"],
        },
    }


def _assess_e4c(record: dict[str, Any]) -> Assessment:
    rows = [
        metrics["torch_exact_batched"]
        for metrics in record["metrics"].values()
        if "torch_exact_batched" in metrics
    ]
    rocm_rows = [
        row
        for row in rows
        if row.get("accelerator_backend") == "rocm"
        and float(row.get("rocm_available", 0.0)) == 1.0
        and float(row.get("uses_rocm_transfer", 0.0)) == 1.0
    ]
    speed_outlier_ratio = max(
        float(row["end_to_end_ms_p95"]) / max(float(row["end_to_end_ms_p50"]), 1e-9)
        for row in rows
    )
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "runtime_positive" if rocm_rows else "no_accelerator",
        "supports_pareto_improvement": False,
        "primary_reason": "rocm_backend_executed_torch_dispatch",
        "limitations": [
            "not_speed_claim",
            "small_dispatch_benchmark",
            "no_occupancy_measurement",
            "no_kernel_fusion",
            "p95_outlier_present" if speed_outlier_ratio > 10.0 else "limited_timing_sample",
        ],
        "evidence": {
            "accelerator_backend": rocm_rows[0]["accelerator_backend"] if rocm_rows else "cpu",
            "rocm_available": bool(rocm_rows),
            "rocm_runtime_version": rocm_rows[0].get("rocm_runtime_version", "")
            if rocm_rows
            else "",
            "uses_rocm_transfer": bool(rocm_rows),
            "uses_cuda_transfer": any(
                float(row.get("uses_cuda_transfer", 0.0)) == 1.0 for row in rows
            ),
            "max_p95_to_p50_ratio": speed_outlier_ratio,
        },
    }


def _assess_e4d(record: dict[str, Any]) -> Assessment:
    rows = [
        metrics["rocm_transfer_scaling"]
        for metrics in record["metrics"].values()
        if "rocm_transfer_scaling" in metrics
    ]
    rocm_rows = [
        row
        for row in rows
        if row.get("accelerator_backend") == "rocm"
        and float(row.get("uses_rocm_transfer", 0.0)) == 1.0
    ]
    best_h2d_bandwidth = max(
        float(row.get("host_to_device_bandwidth_gb_s_p50", 0.0)) for row in rows
    )
    best_d2h_bandwidth = max(
        float(row.get("device_to_host_bandwidth_gb_s_p50", 0.0)) for row in rows
    )
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "measurement_positive" if rocm_rows else "no_accelerator",
        "supports_pareto_improvement": False,
        "primary_reason": "rocm_transfer_scaling_recorded",
        "limitations": [
            "no_occupancy_measurement",
            "no_kernel_fusion",
            "no_power_measurement",
            "not_model_layer_benchmark",
        ],
        "evidence": {
            "accelerator_backend": rocm_rows[0]["accelerator_backend"] if rocm_rows else "cpu",
            "payload_count": len(rows),
            "rocm_runtime_version": rocm_rows[0].get("rocm_runtime_version", "")
            if rocm_rows
            else "",
            "uses_rocm_transfer": bool(rocm_rows),
            "uses_cuda_transfer": any(
                float(row.get("uses_cuda_transfer", 0.0)) == 1.0 for row in rows
            ),
            "best_host_to_device_bandwidth_gb_s_p50": best_h2d_bandwidth,
            "best_device_to_host_bandwidth_gb_s_p50": best_d2h_bandwidth,
        },
    }


def _assess_r1(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    checkpoint_resume_ok = bool(metrics["checkpoint_resume"]["resume_ok"])
    rocm_ready = bool(
        metrics.get("accelerator_backend") == "rocm"
        and metrics.get("rocm_available") is True
        and checkpoint_resume_ok
        and int(metrics.get("stable_batch_size", 0)) > 0
    )
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "training_readiness_positive" if rocm_ready else "mixed",
        "supports_pareto_improvement": False,
        "primary_reason": "rocm_training_runtime_probe_completed",
        "limitations": [
            "tiny_decoder_training_probe_only",
            "not_full_model_training",
            "not_50m_token_run",
            "not_occupancy_measurement",
            "throughput_depends_on_python_training_loop",
        ],
        "evidence": {
            "accelerator_backend": metrics.get("accelerator_backend"),
            "rocm_available": metrics.get("rocm_available"),
            "rocm_runtime_version": metrics.get("rocm_runtime_version"),
            "device_name": metrics.get("device_properties", {}).get("name", ""),
            "total_memory_bytes": metrics.get("memory", {}).get("total_bytes", 0),
            "free_memory_bytes": metrics.get("memory", {}).get("free_bytes", 0),
            "fp32_forward_backward": metrics["precision_support"][
                "fp32_forward_backward"
            ],
            "bf16_forward_backward": metrics["precision_support"][
                "bf16_forward_backward"
            ],
            "fp16_forward_backward": metrics["precision_support"][
                "fp16_forward_backward"
            ],
            "stable_batch_size": metrics.get("stable_batch_size", 0),
            "max_stable_tokens": metrics.get("max_stable_tokens", 0),
            "checkpoint_resume_ok": checkpoint_resume_ok,
        },
    }


def _assess_d1(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    meets_50m = bool(metrics.get("meets_50m_token_requirement", False))
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "corpus_prepared" if meets_50m else "corpus_prepared_insufficient_tokens",
        "supports_pareto_improvement": False,
        "primary_reason": (
            "licensed_corpus_prepared"
            if meets_50m
            else "licensed_corpus_prepared_below_required_token_floor"
        ),
        "limitations": [
            "approximate_regex_token_count",
            "heuristic_license_detection",
            "heuristic_secret_scanning",
            "no_model_tokenizer_yet",
        ]
        + ([] if meets_50m else ["below_50m_token_requirement"]),
        "evidence": {
            "repo_count": metrics.get("repo_count", 0),
            "document_count": metrics.get("document_count", 0),
            "total_tokens": metrics.get("total_tokens", 0),
            "target_min_tokens": metrics.get("target_min_tokens", 0),
            "meets_50m_token_requirement": meets_50m,
            "license_counts": metrics.get("license_counts", {}),
            "languages": metrics.get("languages", {}),
            "file_roles": metrics.get("file_roles", {}),
            "split_counts": metrics.get("split_counts", {}),
        },
    }


def _assess_hf_corpus_manifest(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    accepted_configs = int(metrics.get("accepted_config_count", 0))
    pinned_sources = all(
        bool(source.get("resolved_revision"))
        and source.get("resolved_revision") == source.get("requested_revision")
        for source in metrics.get("sources", [])
    )
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "hf_corpus_manifest_positive"
        if accepted_configs and pinned_sources
        else "hf_corpus_manifest_incomplete",
        "supports_pareto_improvement": False,
        "primary_reason": "reviewed_revision_pinned_hf_corpus_metadata_recorded",
        "limitations": [
            "metadata_manifest_only",
            "token_count_pending_streaming_materialization",
            "not_training_run",
            "row_level_quality_filters_pending",
        ],
        "evidence": {
            "source_count": metrics.get("source_count", 0),
            "accepted_config_count": accepted_configs,
            "rejected_config_count": metrics.get("rejected_config_count", 0),
            "total_rows": metrics.get("total_rows", 0),
            "total_parquet_bytes": metrics.get("total_parquet_bytes", 0),
            "pinned_sources": pinned_sources,
            "token_count_status": metrics.get("token_count", {}).get("status", ""),
        },
    }


def _assess_hf_corpus_materialization(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    train_tokens = int(metrics.get("tokens", {}).get("train", 0))
    meets_50m = bool(metrics.get("meets_50m_token_requirement", False))
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "hf_corpus_materialized"
        if meets_50m
        else "hf_corpus_materialized_insufficient_tokens",
        "supports_pareto_improvement": False,
        "primary_reason": (
            "filtered_hf_corpus_mirror_reached_required_train_tokens"
            if meets_50m
            else "filtered_hf_corpus_mirror_below_required_train_tokens"
        ),
        "limitations": [
            "exploratory_research_only_not_production_approved",
            "not_training_run",
            "near_duplicate_filter_pending",
            "mixed_license_metadata_not_blocking_exploratory_use",
        ]
        + ([] if meets_50m else ["below_50m_token_requirement"]),
        "evidence": {
            "corpus_use": metrics.get("corpus_use", ""),
            "train_tokens": train_tokens,
            "target_train_tokens": metrics.get("target_train_tokens", 0),
            "meets_50m_token_requirement": meets_50m,
            "rows_seen": metrics.get("rows_seen", 0),
            "rows_accepted": metrics.get("rows_accepted", 0),
            "rows_excluded": metrics.get("rows_excluded", 0),
            "output_sha256": metrics.get("output", {}).get("sha256", ""),
            "dataset_config_counts": metrics.get("dataset_config_counts", {}),
        },
    }


def _assess_t1(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    training = metrics.get("training", {})
    checkpoint = metrics.get("checkpoint", {})
    completed = bool(metrics.get("status") == "completed" and record.get("status") == "completed")
    parameter_count = int(metrics.get("model", {}).get("parameter_count", 0))
    train_tokens = int(training.get("train_tokens", 0))
    is_required_dense_baseline = bool(
        completed and 10_000_000 <= parameter_count <= 50_000_000 and train_tokens >= 50_000_000
    )
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "dense_50m_baseline_positive"
        if is_required_dense_baseline
        else ("training_smoke_positive" if completed else "training_failed"),
        "supports_pareto_improvement": False,
        "primary_reason": (
            "dense_decoder_10m_plus_completed_50m_token_run"
            if is_required_dense_baseline
            else (
            "dense_decoder_training_pipeline_completed"
            if completed
            else "dense_decoder_training_smoke_failed"
            )
        ),
        "limitations": [
            "no_functional_coding_evaluation_yet",
        ]
        + ([] if is_required_dense_baseline else ["training_smoke_only"])
        + ([] if train_tokens >= 50_000_000 else ["not_50m_token_run"])
        + (
            ["10m_parameter_floor_reached"]
            if parameter_count >= 10_000_000
            else ["not_10m_parameter_model"]
        )
        + ([] if completed else ["failed_training_run"]),
        "evidence": {
            "accelerator_backend": metrics.get("accelerator_backend"),
            "status": metrics.get("status"),
            "failure": metrics.get("failure", ""),
            "parameter_count": parameter_count,
            "train_tokens": train_tokens,
            "train_steps": training.get("steps", 0),
            "validation_loss": metrics.get("validation", {}).get("loss", 0.0),
            "checkpoint_resume_ok": bool(checkpoint.get("resume_ok", False)),
            "corpus_record": metrics.get("corpus", {}).get("record", {}),
        },
    }


def _assess_r2(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    failure_count = int(metrics.get("failure_count", 0))
    first_failure = metrics.get("first_failure") or {}
    passed = failure_count == 0
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": (
            "transformer_stability_positive"
            if passed
            else "transformer_stability_failed"
        ),
        "supports_pareto_improvement": False,
        "primary_reason": (
            "all_component_stability_cases_passed"
            if passed
            else "component_stability_case_failed"
        ),
        "limitations": [
            "microbench_only",
            "not_full_dense_training",
            "single_process_probe",
            "does_not_measure_kernel_occupancy",
        ]
        + ([] if passed else ["failed_component_requires_isolation"]),
        "evidence": {
            "accelerator_backend": metrics.get("accelerator_backend"),
            "rocm_available": metrics.get("rocm_available"),
            "rocm_runtime_version": metrics.get("rocm_runtime_version"),
            "case_count": metrics.get("case_count", 0),
            "failure_count": failure_count,
            "first_failure_component": first_failure.get("component", ""),
            "first_failure_dtype": first_failure.get("dtype", ""),
            "first_failure_mask_mode": first_failure.get("mask_mode", ""),
            "first_failure_error": first_failure.get("error", ""),
        },
    }


def _assess_t3(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    first_nonfinite_phase = metrics.get("first_nonfinite_phase")
    first_step = (metrics.get("step_results") or [{}])[0]
    gradients = first_step.get("gradients", {})
    first_nonfinite_tensors = gradients.get("nonfinite_tensors") or []
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": (
            "dense_step_debug_positive"
            if first_nonfinite_phase is None
            else "dense_step_debug_failed"
        ),
        "supports_pareto_improvement": False,
        "primary_reason": (
            "dense_step_debug_window_remained_finite"
            if first_nonfinite_phase is None
            else "dense_step_debug_found_nonfinite_phase"
        ),
        "limitations": [
            "debug_probe_only",
            "short_two_step_window",
            "not_training_quality_evaluation",
        ],
        "evidence": {
            "parameter_count": metrics.get("model", {}).get("parameter_count", 0),
            "layers": metrics.get("model", {}).get("config", {}).get("layers", 0),
            "hidden_dim": metrics.get("model", {}).get("config", {}).get("hidden_dim", 0),
            "first_nonfinite_phase": first_nonfinite_phase or "",
            "gradient_nonfinite_count": gradients.get("nonfinite_count", 0),
            "first_nonfinite_tensor": first_nonfinite_tensors[0].get("name", "")
            if first_nonfinite_tensors
            else "",
        },
    }


def _assess_e5g(record: dict[str, Any]) -> Assessment:
    final = record["metrics"]["final"]
    task_count = int(final["task_count"])
    generated_successes = int(final["generated_task_successes"])
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "limited_positive"
        if generated_successes == task_count
        else "mixed",
        "supports_pareto_improvement": False,
        "primary_reason": "generated_candidates_passed_narrow_public_patch_replay",
        "limitations": [
            "hand_authored_heuristics",
            "not_autonomous_repair",
            "small_task_count",
            "source_level_kilo_regression",
        ],
        "evidence": {
            "task_count": task_count,
            "generated_task_successes": generated_successes,
            "historical_patch_successes": final["historical_patch_successes"],
            "baseline_failures": final["baseline_failures"],
        },
    }


def _assess_s2c(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]["metrics"]
    critical_failures = record["metrics"]["critical_failures"]
    smoke_passed = (
        len(critical_failures) == 0
        and metrics["sensitive_alias_leaks_in_allowed_context"] == 0
        and metrics["unauthorized_retrievals"] == 0
    )
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "smoke_pass" if smoke_passed else "rejected",
        "supports_pareto_improvement": False,
        "primary_reason": "covered_security_checks_passed",
        "limitations": [
            "deterministic_heuristic",
            "not_production_proof",
            "synthetic_red_team",
        ],
        "evidence": {
            "critical_failures": len(critical_failures),
            "unauthorized_retrievals": metrics["unauthorized_retrievals"],
            "sensitive_alias_leaks": metrics[
                "sensitive_alias_leaks_in_allowed_context"
            ],
            "semantic_exfiltration_queries_blocked": metrics[
                "semantic_exfiltration_queries_blocked"
            ],
        },
    }


def _assess_e6a(record: dict[str, Any]) -> Assessment:
    methods = record["metrics"]["methods"]
    structured = methods["gated_structured_external_memory"]
    updated_retrieval = methods["updated_text_retrieval"]
    structured_wins = bool(
        structured["answer_accuracy"] > updated_retrieval["answer_accuracy"]
        and structured["poisoned_answer_count"] == 0
        and structured["restricted_denial_accuracy"] == 1.0
    )
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "exploration_positive" if structured_wins else "mixed",
        "supports_pareto_improvement": False,
        "primary_reason": (
            "structured_external_memory_beats_poisonable_text_retrieval_in_"
            "synthetic_convention_task"
        ),
        "limitations": [
            "synthetic_convention_corpus",
            "parametric_baseline_is_proxy_not_trained_lm",
            "small_question_set",
            "not_code_generation",
            "not_end_to_end_model_quality",
        ],
        "evidence": {
            "structured_answer_accuracy": structured["answer_accuracy"],
            "updated_text_retrieval_answer_accuracy": updated_retrieval[
                "answer_accuracy"
            ],
            "structured_poisoned_answer_count": structured["poisoned_answer_count"],
            "updated_text_retrieval_poisoned_answer_count": updated_retrieval[
                "poisoned_answer_count"
            ],
            "structured_restricted_denial_accuracy": structured[
                "restricted_denial_accuracy"
            ],
            "supports_model_level_claim": record["metrics"]["final"][
                "supports_model_level_claim"
            ],
        },
    }


def _assess_e6b(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    if metrics.get("status") != "completed":
        return {
            "experiment_id": record["experiment_id"],
            "hypothesis": record.get("hypothesis"),
            "outcome": "skipped",
            "supports_pareto_improvement": False,
            "primary_reason": "no_local_public_repository_symbols_available",
            "limitations": metrics.get("limitations", []),
            "evidence": {
                "repo_count_used": metrics.get("final", {}).get("repo_count_used", 0),
                "question_count": metrics.get("question_count", 0),
            },
        }

    methods = metrics["methods"]
    structured = methods["structured_symbol_memory"]
    text_retrieval = methods["text_file_retrieval"]
    structured_wins = bool(
        structured["answer_accuracy"] >= text_retrieval["answer_accuracy"]
        and structured["answer_accuracy"] == 1.0
    )
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "real_repo_measurement_positive" if structured_wins else "mixed",
        "supports_pareto_improvement": False,
        "primary_reason": (
            "structured_symbol_memory_answers_public_repo_symbol_file_qa"
        ),
        "limitations": [
            "symbol_file_location_only",
            "not_code_generation",
            "not_natural_language_reasoning",
            "structured_method_uses_extracted_symbol_index",
            "no_model_integration",
        ],
        "evidence": {
            "repo_count_used": metrics["final"]["repo_count_used"],
            "question_count": metrics["question_count"],
            "structured_answer_accuracy": structured["answer_accuracy"],
            "text_retrieval_answer_accuracy": text_retrieval["answer_accuracy"],
            "structured_storage_bytes": structured["storage_bytes"],
            "text_retrieval_storage_bytes": text_retrieval["storage_bytes"],
            "supports_model_level_claim": metrics["final"][
                "supports_model_level_claim"
            ],
        },
    }


def _assess_e6c(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    if metrics.get("status") != "completed":
        return {
            "experiment_id": record["experiment_id"],
            "hypothesis": record.get("hypothesis"),
            "outcome": "skipped",
            "supports_pareto_improvement": False,
            "primary_reason": "no_local_public_repository_signatures_available",
            "limitations": metrics.get("limitations", []),
            "evidence": {
                "repo_count_used": metrics.get("final", {}).get("repo_count_used", 0),
                "question_count": metrics.get("question_count", 0),
            },
        }

    methods = metrics["methods"]
    structured = methods["structured_signature_memory"]
    text_lookup = methods["text_signature_lookup"]
    structured_wins = bool(
        structured["answer_accuracy"] >= text_lookup["answer_accuracy"]
        and structured["answer_accuracy"] == 1.0
    )
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "real_repo_measurement_positive" if structured_wins else "mixed",
        "supports_pareto_improvement": False,
        "primary_reason": (
            "structured_signature_memory_answers_public_repo_signature_qa"
        ),
        "limitations": [
            "signature_line_only",
            "not_code_generation",
            "not_natural_language_reasoning",
            "structured_method_uses_extracted_signature_index",
            "no_model_integration",
        ],
        "evidence": {
            "repo_count_used": metrics["final"]["repo_count_used"],
            "question_count": metrics["question_count"],
            "structured_answer_accuracy": structured["answer_accuracy"],
            "text_lookup_answer_accuracy": text_lookup["answer_accuracy"],
            "structured_storage_bytes": structured["storage_bytes"],
            "text_lookup_storage_bytes": text_lookup["storage_bytes"],
            "supports_model_level_claim": metrics["final"][
                "supports_model_level_claim"
            ],
        },
    }


def _assess_e6d(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    if metrics.get("status") != "completed":
        return {
            "experiment_id": record["experiment_id"],
            "hypothesis": record.get("hypothesis"),
            "outcome": "skipped",
            "supports_pareto_improvement": False,
            "primary_reason": "no_local_public_repository_python_signatures_available",
            "limitations": metrics.get("limitations", []),
            "evidence": {
                "repo_count_used": metrics.get("final", {}).get("repo_count_used", 0),
                "question_count": metrics.get("question_count", 0),
            },
        }

    methods = metrics["methods"]
    structured = methods["structured_signature_call_stub"]
    name_only = methods["name_only_call_stub"]
    structured_wins = bool(
        structured["answer_accuracy"] > name_only["answer_accuracy"]
        and structured["answer_accuracy"] == 1.0
    )
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "codegen_proxy_positive" if structured_wins else "mixed",
        "supports_pareto_improvement": False,
        "primary_reason": (
            "structured_signature_memory_generates_public_repo_call_stubs"
        ),
        "limitations": [
            "python_signature_only",
            "canonical_call_stub_only",
            "not_executable_argument_synthesis",
            "not_semantic_code_generation",
            "no_model_integration",
        ],
        "evidence": {
            "repo_count_used": metrics["final"]["repo_count_used"],
            "question_count": metrics["question_count"],
            "structured_answer_accuracy": structured["answer_accuracy"],
            "name_only_answer_accuracy": name_only["answer_accuracy"],
            "structured_storage_bytes": structured["storage_bytes"],
            "supports_model_level_claim": metrics["final"][
                "supports_model_level_claim"
            ],
        },
    }


def _assess_e6e(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    if metrics.get("status") != "completed":
        return {
            "experiment_id": record["experiment_id"],
            "hypothesis": record.get("hypothesis"),
            "outcome": "skipped",
            "supports_pareto_improvement": False,
            "primary_reason": "no_local_public_repository_python_signatures_available",
            "limitations": metrics.get("limitations", []),
            "evidence": {
                "repo_count_used": metrics.get("final", {}).get("repo_count_used", 0),
                "question_count": metrics.get("question_count", 0),
            },
        }

    methods = metrics["methods"]
    structured = methods["structured_signature_skeleton"]
    name_only = methods["name_only_skeleton"]
    structured_wins = bool(
        structured["answer_accuracy"] > name_only["answer_accuracy"]
        and structured["answer_accuracy"] == 1.0
    )
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "codegen_proxy_positive" if structured_wins else "mixed",
        "supports_pareto_improvement": False,
        "primary_reason": (
            "structured_signature_memory_generates_public_repo_function_skeletons"
        ),
        "limitations": [
            "python_signature_only",
            "deterministic_skeleton_body_only",
            "not_executable_behavior_synthesis",
            "not_semantic_code_generation",
            "no_model_integration",
        ],
        "evidence": {
            "repo_count_used": metrics["final"]["repo_count_used"],
            "question_count": metrics["question_count"],
            "structured_answer_accuracy": structured["answer_accuracy"],
            "name_only_answer_accuracy": name_only["answer_accuracy"],
            "structured_storage_bytes": structured["storage_bytes"],
            "supports_model_level_claim": metrics["final"][
                "supports_model_level_claim"
            ],
        },
    }


def _assess_e6f(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    if metrics.get("status") != "completed":
        return {
            "experiment_id": record["experiment_id"],
            "hypothesis": record.get("hypothesis"),
            "outcome": "skipped",
            "supports_pareto_improvement": False,
            "primary_reason": (
                "no_local_public_repository_python_function_docstrings_available"
            ),
            "limitations": metrics.get("limitations", []),
            "evidence": {
                "repo_count_used": metrics.get("final", {}).get("repo_count_used", 0),
                "question_count": metrics.get("question_count", 0),
            },
        }

    methods = metrics["methods"]
    structured = methods["structured_docstring_skeleton"]
    signature_only = methods["signature_only_skeleton"]
    structured_wins = bool(
        structured["answer_accuracy"] > signature_only["answer_accuracy"]
        and structured["answer_accuracy"] == 1.0
    )
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "codegen_proxy_positive" if structured_wins else "mixed",
        "supports_pareto_improvement": False,
        "primary_reason": (
            "structured_docstring_memory_generates_public_repo_documented_skeletons"
        ),
        "limitations": [
            "python_signature_and_docstring_only",
            "first_docstring_line_only",
            "deterministic_skeleton_body_only",
            "not_executable_behavior_synthesis",
            "not_semantic_code_generation",
            "no_model_integration",
        ],
        "evidence": {
            "repo_count_used": metrics["final"]["repo_count_used"],
            "question_count": metrics["question_count"],
            "structured_answer_accuracy": structured["answer_accuracy"],
            "signature_only_answer_accuracy": signature_only["answer_accuracy"],
            "structured_storage_bytes": structured["storage_bytes"],
            "signature_only_storage_bytes": signature_only["storage_bytes"],
            "supports_model_level_claim": metrics["final"][
                "supports_model_level_claim"
            ],
        },
    }


def _assess_e6g(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    if metrics.get("status") != "completed":
        return {
            "experiment_id": record["experiment_id"],
            "hypothesis": record.get("hypothesis"),
            "outcome": "skipped",
            "supports_pareto_improvement": False,
            "primary_reason": (
                "no_local_public_repository_python_function_docstrings_available"
            ),
            "limitations": metrics.get("limitations", []),
            "evidence": {
                "repo_count_used": metrics.get("final", {}).get("repo_count_used", 0),
                "question_count": metrics.get("question_count", 0),
            },
        }

    methods = metrics["methods"]
    structured = methods["structured_docstring_api_reference"]
    signature_only = methods["signature_only_api_reference"]
    structured_wins = bool(
        structured["answer_accuracy"] > signature_only["answer_accuracy"]
        and structured["answer_accuracy"] == 1.0
    )
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "docgen_proxy_positive" if structured_wins else "mixed",
        "supports_pareto_improvement": False,
        "primary_reason": (
            "structured_docstring_memory_generates_public_repo_api_reference_entries"
        ),
        "limitations": [
            "python_signature_and_docstring_only",
            "first_docstring_line_only",
            "deterministic_api_reference_template",
            "not_quality_rated_documentation",
            "not_semantic_documentation_generation",
            "no_model_integration",
        ],
        "evidence": {
            "repo_count_used": metrics["final"]["repo_count_used"],
            "question_count": metrics["question_count"],
            "structured_answer_accuracy": structured["answer_accuracy"],
            "signature_only_answer_accuracy": signature_only["answer_accuracy"],
            "structured_storage_bytes": structured["storage_bytes"],
            "signature_only_storage_bytes": signature_only["storage_bytes"],
            "supports_model_level_claim": metrics["final"][
                "supports_model_level_claim"
            ],
        },
    }


def _assess_e6h(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    if metrics.get("status") != "completed":
        return {
            "experiment_id": record["experiment_id"],
            "hypothesis": record.get("hypothesis"),
            "outcome": "skipped",
            "supports_pareto_improvement": False,
            "primary_reason": "no_local_public_repository_api_doc_coverage_available",
            "limitations": metrics.get("limitations", []),
            "evidence": {
                "repo_count_used": metrics.get("final", {}).get("repo_count_used", 0),
                "question_count": metrics.get("question_count", 0),
            },
        }

    methods = metrics["methods"]
    structured = methods["structured_doc_directive_memory"]
    source_only = methods["source_symbol_only_memory"]
    structured_wins = bool(
        structured["answer_accuracy"] > source_only["answer_accuracy"]
        and structured["answer_accuracy"] == 1.0
    )
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "real_repo_doc_measurement_positive"
        if structured_wins
        else "mixed",
        "supports_pareto_improvement": False,
        "primary_reason": (
            "structured_doc_directive_memory_answers_public_repo_api_doc_coverage_qa"
        ),
        "limitations": [
            "sphinx_directive_coverage_only",
            "documented_python_functions_only",
            "exact_doc_file_lookup_only",
            "not_documentation_quality",
            "not_semantic_documentation_generation",
            "no_model_integration",
        ],
        "evidence": {
            "repo_count_used": metrics["final"]["repo_count_used"],
            "question_count": metrics["question_count"],
            "structured_answer_accuracy": structured["answer_accuracy"],
            "source_only_answer_accuracy": source_only["answer_accuracy"],
            "structured_storage_bytes": structured["storage_bytes"],
            "source_only_storage_bytes": source_only["storage_bytes"],
            "supports_model_level_claim": metrics["final"][
                "supports_model_level_claim"
            ],
        },
    }


def _assess_e6i(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    if metrics.get("status") != "completed":
        return {
            "experiment_id": record["experiment_id"],
            "hypothesis": record.get("hypothesis"),
            "outcome": "skipped",
            "supports_pareto_improvement": False,
            "primary_reason": "no_synthetic_api_doc_drift_fixture_available",
            "limitations": metrics.get("limitations", []),
            "evidence": {
                "repo_count_used": metrics.get("final", {}).get("repo_count_used", 0),
                "question_count": metrics.get("question_count", 0),
            },
        }

    methods = metrics["methods"]
    structured = methods["structured_source_doc_consistency_memory"]
    doc_only = methods["doc_directive_only_memory"]
    structured_wins = bool(
        metrics["final"]["stale_doc_issue_count"] >= 1
        and structured["answer_accuracy"] > doc_only["answer_accuracy"]
        and structured["answer_accuracy"] == 1.0
    )
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "doc_drift_positive_control" if structured_wins else "mixed",
        "supports_pareto_improvement": False,
        "primary_reason": (
            "structured_source_doc_consistency_memory_detects_synthetic_api_doc_drift"
        ),
        "limitations": [
            "synthetic_positive_control",
            "sphinx_directive_consistency_only",
            "python_signature_presence_only",
            "not_documentation_quality",
            "not_semantic_prose_drift_detection",
            "no_model_integration",
        ],
        "evidence": {
            "repo_count_used": metrics["final"]["repo_count_used"],
            "question_count": metrics["question_count"],
            "stale_doc_issue_count": metrics["final"]["stale_doc_issue_count"],
            "structured_answer_accuracy": structured["answer_accuracy"],
            "doc_only_answer_accuracy": doc_only["answer_accuracy"],
            "structured_storage_bytes": structured["storage_bytes"],
            "doc_only_storage_bytes": doc_only["storage_bytes"],
            "supports_model_level_claim": metrics["final"][
                "supports_model_level_claim"
            ],
        },
    }
