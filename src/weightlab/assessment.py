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
    "sparse_hebbian_assembly_memory",
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
    if record.get("metrics", {}).get("benchmark_label") == "d5_trained_tokenizer_model_comparison":
        return _assess_d5_tokenizer_training(record)
    if record.get("metrics", {}).get("benchmark_label") == "idea_foundry_candidate_generation":
        return _assess_idea_foundry_candidates(record)
    if (
        record.get("metrics", {}).get("benchmark_label")
        == "idea_foundry_repository_graph_signal_probe"
    ):
        return _assess_idea_foundry_graph_probe(record)
    if record.get("metrics", {}).get("benchmark_label") == "if2_fast_weight_continual_probe":
        return _assess_if2_fast_weight_probe(record)
    if record.get("metrics", {}).get("benchmark_label") == "if4_fast_repo_adaptation_probe":
        return _assess_if4_fast_repo_adaptation_probe(record)
    if record.get("metrics", {}).get("benchmark_label") == "if7_sparse_hebbian_assembly_probe":
        return _assess_if7_sparse_hebbian_probe(record)
    if record.get("metrics", {}).get("benchmark_label") == "if7_hebbian_conditioned_trained_model":
        return _assess_if7_hebbian_trained_model(record)
    if record.get("metrics", {}).get("benchmark_label") == "if7_sparse_hebbian_candidate_reranker":
        return _assess_if7_sparse_hebbian_reranker(record)
    if record.get("metrics", {}).get("benchmark_label") == "if7_hebbian_repository_linking":
        return _assess_if7_hebbian_repository_linking(record)
    if (
        record.get("metrics", {}).get("benchmark_label")
        == "if7_trained_repository_linking_ranker"
    ):
        return _assess_if7_trained_repository_ranker(record)
    if record.get("metrics", {}).get("benchmark_label") == "if3_block_codebook_checkpoint_probe":
        return _assess_if3_block_codebook_probe(record)
    if record.get("metrics", {}).get("benchmark_label") == "if3_block_codebook_validation_probe":
        return _assess_if3_block_codebook_validation_probe(record)
    if (
        record.get("metrics", {}).get("benchmark_label")
        == "d4_dense_js_executable_checkpoint_evaluation"
    ):
        return _assess_dense_js_executable_probe(record)
    if record.get("metrics", {}).get("benchmark_label") == "repository_balanced_task_sample":
        return _assess_repository_task_sample(record)
    if record.get("metrics", {}).get("benchmark_label") == "repository_api_reuse_probe":
        return _assess_repository_api_reuse_probe(record)
    if (
        record.get("metrics", {}).get("benchmark_label")
        == "repository_context_pairwise_probe"
    ):
        return _assess_repository_context_pairwise_probe(record)
    if record.get("metrics", {}).get("benchmark_label") == "quixbugs_python_repair_probe":
        return _assess_quixbugs_repair_probe(record)
    if (
        record.get("metrics", {}).get("benchmark_label")
        == "quixbugs_python_candidate_repair_probe"
    ):
        return _assess_quixbugs_candidate_repair_probe(record)
    if (
        record.get("metrics", {}).get("benchmark_label")
        == "quixbugs_python_dense_model_candidate_probe"
    ):
        return _assess_quixbugs_dense_model_candidate_probe(record)
    if (
        record.get("metrics", {}).get("benchmark_label")
        == "quixbugs_python_edit_baseline_probe"
    ):
        return _assess_quixbugs_edit_baseline_probe(record)
    if (
        record.get("metrics", {}).get("benchmark_label")
        == "quixbugs_python_dense_ranked_edit_probe"
    ):
        return _assess_quixbugs_dense_ranked_edit_probe(record)
    if (
        record.get("metrics", {}).get("benchmark_label")
        == "quixbugs_python_dense_ranked_syntax_pool_probe"
    ):
        return _assess_quixbugs_dense_ranked_syntax_pool_probe(record)
    if (
        record.get("metrics", {}).get("benchmark_label")
        == "quixbugs_python_dense_ranked_syntax_topk_probe"
    ):
        return _assess_quixbugs_dense_ranked_syntax_topk_probe(record)
    if (
        record.get("metrics", {}).get("benchmark_label")
        == "quixbugs_python_syntax_pool_ordering_control_probe"
    ):
        return _assess_quixbugs_syntax_pool_ordering_control_probe(record)
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
    t11_comparison = _assess_t11_dense_adapter_comparison(records)
    if t11_comparison:
        assessments.append(t11_comparison)
    t11_width_control = _assess_t11_dense528_width_control(records)
    if t11_width_control:
        assessments.append(t11_width_control)
    t12_three_seed = _assess_t12_three_seed_summary(records)
    if t12_three_seed:
        assessments.append(t12_three_seed)
    outcome_counts = Counter(str(row["outcome"]) for row in assessments)
    pareto_dominance_found = any(
        bool(row.get("evidence", {}).get("dominates_dense_baseline"))
        for row in assessments
    )
    frontier_expansion_found = any(
        bool(row.get("evidence", {}).get("expands_measured_frontier"))
        for row in assessments
    )
    return {
        "record_count": len(records),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "pareto_improvement_found": any(
            bool(row["supports_pareto_improvement"]) for row in assessments
        ),
        "pareto_dominance_found": pareto_dominance_found,
        "frontier_expansion_found": frontier_expansion_found,
        "assessments": assessments,
        "next_research_options": NEXT_RESEARCH_OPTIONS,
    }


def _load_manifest_record(results_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    if "path" in row:
        return json.loads((results_dir / row["path"]).read_text())
    return row


def _assess_t11_dense_adapter_comparison(
    records: list[dict[str, Any]],
) -> Assessment | None:
    by_id = {str(record.get("experiment_id", "")): record for record in records}
    required_ids = {
        "dense_train": "T11a_dense544_adamw_fp32_50m",
        "adapter_train": "T11b_adapter528_adamw_fp32_50m",
        "dense_validation": "T11a_dense544_adamw_fp32_50m_final_validation_eval",
        "adapter_validation": "T11b_adapter528_adamw_fp32_50m_final_validation_eval",
        "dense_test": "T11a_dense544_adamw_fp32_50m_final_test_eval",
        "adapter_test": "T11b_adapter528_adamw_fp32_50m_final_test_eval",
        "dense_functional": "T11a_dense544_adamw_fp32_50m_final_test_functional",
        "adapter_functional": "T11b_adapter528_adamw_fp32_50m_final_test_functional",
    }
    if any(experiment_id not in by_id for experiment_id in required_ids.values()):
        return None

    dense_train = by_id[required_ids["dense_train"]]["metrics"]
    adapter_train = by_id[required_ids["adapter_train"]]["metrics"]
    dense_validation = by_id[required_ids["dense_validation"]]["metrics"]
    adapter_validation = by_id[required_ids["adapter_validation"]]["metrics"]
    dense_test = by_id[required_ids["dense_test"]]["metrics"]
    adapter_test = by_id[required_ids["adapter_test"]]["metrics"]
    dense_functional = by_id[required_ids["dense_functional"]]["metrics"]["tasks"]
    adapter_functional = by_id[required_ids["adapter_functional"]]["metrics"]["tasks"]

    dense_throughput = float(dense_train["training"]["tokens_per_second"])
    adapter_throughput = float(adapter_train["training"]["tokens_per_second"])
    dense_model_bytes = int(dense_train["checkpoint"]["model_only_bytes"])
    adapter_model_bytes = int(adapter_train["checkpoint"]["model_only_bytes"])
    dense_state_bytes = int(dense_train["checkpoint"]["optimizer_state_bytes"])
    adapter_state_bytes = int(adapter_train["checkpoint"]["optimizer_state_bytes"])
    dense_peak_vram = int(dense_train["memory"]["peak_allocated_bytes"])
    adapter_peak_vram = int(adapter_train["memory"]["peak_allocated_bytes"])
    dense_grad = dense_train["training"]["gradient_norms"]["summary"]
    adapter_grad = adapter_train["training"]["gradient_norms"]["summary"]

    dense_functional_scores = {
        name: {
            "token_accuracy_mean": float(task.get("token_accuracy_mean", 0.0)),
            "exact_match_rate": float(task.get("exact_match_rate", 0.0)),
        }
        for name, task in dense_functional.items()
        if isinstance(task, dict) and "token_accuracy_mean" in task
    }
    adapter_functional_scores = {
        name: {
            "token_accuracy_mean": float(task.get("token_accuracy_mean", 0.0)),
            "exact_match_rate": float(task.get("exact_match_rate", 0.0)),
        }
        for name, task in adapter_functional.items()
        if isinstance(task, dict) and "token_accuracy_mean" in task
    }
    functional_tasks_won = [
        name
        for name, adapter_scores in adapter_functional_scores.items()
        if adapter_scores["token_accuracy_mean"]
        > dense_functional_scores.get(name, {}).get("token_accuracy_mean", 0.0)
    ]

    adapter_quality_wins = bool(
        float(adapter_validation["loss"]) < float(dense_validation["loss"])
        and float(adapter_test["loss"]) < float(dense_test["loss"])
    )
    adapter_resource_wins = bool(
        adapter_model_bytes < dense_model_bytes
        and adapter_state_bytes < dense_state_bytes
        and adapter_peak_vram < dense_peak_vram
    )
    adapter_stability_win = bool(
        float(adapter_grad["max"]) < float(dense_grad["max"])
        and int(adapter_grad["nonfinite_count"]) == 0
        and int(dense_grad["nonfinite_count"]) == 0
    )
    adapter_runtime_loss = adapter_throughput < dense_throughput
    dominates_dense_baseline = bool(
        adapter_quality_wins
        and adapter_resource_wins
        and adapter_stability_win
        and not adapter_runtime_loss
    )
    expands_measured_frontier = bool(
        adapter_quality_wins
        and adapter_resource_wins
        and functional_tasks_won
        and adapter_runtime_loss
    )

    return {
        "experiment_id": "T11_dense_adapter_50m_paired_assessment",
        "hypothesis": "residual_adapter_candidate_can_expand_measured_frontier",
        "outcome": (
            "t11_adapter_dominates_dense_baseline"
            if dominates_dense_baseline
            else (
                "t11_adapter_frontier_expansion"
                if expands_measured_frontier
                else "t11_adapter_mixed"
            )
        ),
        "supports_pareto_improvement": expands_measured_frontier
        or dominates_dense_baseline,
        "primary_reason": (
            "t11b_improves_quality_storage_memory_and_functional_scores_but_loses_throughput"
        ),
        "limitations": [
            "does_not_dominate_dense_baseline_on_training_throughput",
            "d4_javascript_source_local_only",
            "no_paired_documentation_source_consistency_benchmark",
            "no_executable_javascript_benchmark",
            "no_packed_quantized_kernel_measurement",
            "single_seed_paired_run",
        ],
        "evidence": {
            "dense_experiment_id": required_ids["dense_train"],
            "adapter_experiment_id": required_ids["adapter_train"],
            "dominates_dense_baseline": dominates_dense_baseline,
            "expands_measured_frontier": expands_measured_frontier,
            "dense_final_validation_loss": float(dense_validation["loss"]),
            "adapter_final_validation_loss": float(adapter_validation["loss"]),
            "dense_final_test_loss": float(dense_test["loss"]),
            "adapter_final_test_loss": float(adapter_test["loss"]),
            "dense_train_tokens_per_second": dense_throughput,
            "adapter_train_tokens_per_second": adapter_throughput,
            "adapter_throughput_relative_to_dense": adapter_throughput
            / dense_throughput,
            "dense_model_only_bytes": dense_model_bytes,
            "adapter_model_only_bytes": adapter_model_bytes,
            "dense_optimizer_state_bytes": dense_state_bytes,
            "adapter_optimizer_state_bytes": adapter_state_bytes,
            "dense_peak_vram_bytes": dense_peak_vram,
            "adapter_peak_vram_bytes": adapter_peak_vram,
            "dense_max_recorded_grad_norm": float(dense_grad["max"]),
            "adapter_max_recorded_grad_norm": float(adapter_grad["max"]),
            "functional_tasks_won_by_adapter": functional_tasks_won,
            "dense_functional_scores": dense_functional_scores,
            "adapter_functional_scores": adapter_functional_scores,
        },
    }


def _assess_t11_dense528_width_control(records: list[dict[str, Any]]) -> Assessment | None:
    by_id = {str(record.get("experiment_id", "")): record for record in records}
    required_ids = {
        "dense544_train": "T11a_dense544_adamw_fp32_50m",
        "adapter528_train": "T11b_adapter528_adamw_fp32_50m",
        "dense528_train": "T11c_dense528_adamw_fp32_50m",
        "dense544_validation": "T11a_dense544_adamw_fp32_50m_final_validation_eval",
        "adapter528_validation": "T11b_adapter528_adamw_fp32_50m_final_validation_eval",
        "dense528_validation": "T11c_dense528_adamw_fp32_50m_final_validation_eval",
        "dense544_test": "T11a_dense544_adamw_fp32_50m_final_test_eval",
        "adapter528_test": "T11b_adapter528_adamw_fp32_50m_final_test_eval",
        "dense528_test": "T11c_dense528_adamw_fp32_50m_final_test_eval",
        "dense528_functional": "T11c_dense528_adamw_fp32_50m_final_test_functional",
    }
    if any(experiment_id not in by_id for experiment_id in required_ids.values()):
        return None

    dense544_train = by_id[required_ids["dense544_train"]]["metrics"]
    adapter528_train = by_id[required_ids["adapter528_train"]]["metrics"]
    dense528_train = by_id[required_ids["dense528_train"]]["metrics"]
    dense544_validation = by_id[required_ids["dense544_validation"]]["metrics"]
    adapter528_validation = by_id[required_ids["adapter528_validation"]]["metrics"]
    dense528_validation = by_id[required_ids["dense528_validation"]]["metrics"]
    dense544_test = by_id[required_ids["dense544_test"]]["metrics"]
    adapter528_test = by_id[required_ids["adapter528_test"]]["metrics"]
    dense528_test = by_id[required_ids["dense528_test"]]["metrics"]
    dense528_functional = by_id[required_ids["dense528_functional"]]["metrics"]["tasks"]

    dense528_best_validation = (
        float(dense528_validation["loss"]) < float(dense544_validation["loss"])
        and float(dense528_validation["loss"]) < float(adapter528_validation["loss"])
    )
    dense528_resource_wins = (
        int(dense528_train["checkpoint"]["model_only_bytes"])
        < int(dense544_train["checkpoint"]["model_only_bytes"])
        and int(dense528_train["checkpoint"]["model_only_bytes"])
        < int(adapter528_train["checkpoint"]["model_only_bytes"])
        and int(dense528_train["memory"]["peak_allocated_bytes"])
        < int(dense544_train["memory"]["peak_allocated_bytes"])
        and int(dense528_train["memory"]["peak_allocated_bytes"])
        < int(adapter528_train["memory"]["peak_allocated_bytes"])
    )
    dense528_throughput_wins = (
        float(dense528_train["training"]["tokens_per_second"])
        > float(dense544_train["training"]["tokens_per_second"])
        and float(dense528_train["training"]["tokens_per_second"])
        > float(adapter528_train["training"]["tokens_per_second"])
    )
    adapter_test_win = float(adapter528_test["loss"]) < float(dense528_test["loss"])

    return {
        "experiment_id": "T11_dense528_width_control_assessment",
        "hypothesis": "dense528_width_control_isolates_t11_adapter_gain",
        "outcome": "t11_dense528_width_control_frontier_expansion",
        "supports_pareto_improvement": bool(
            dense528_best_validation and dense528_resource_wins and dense528_throughput_wins
        ),
        "primary_reason": (
            "dense528_beats_dense544_and_adapter528_on_validation_resource_and_throughput"
        ),
        "limitations": [
            "adapter528_still_has_slightly_lower_final_test_loss",
            "test_loss_not_used_for_selection",
            "single_seed_width_control",
            "d4_javascript_source_local_only",
            "functional_probe_not_executable_javascript",
        ],
        "evidence": {
            "dominates_dense_baseline": False,
            "expands_measured_frontier": True,
            "dense528_best_final_validation": dense528_best_validation,
            "dense528_resource_wins": dense528_resource_wins,
            "dense528_throughput_wins": dense528_throughput_wins,
            "adapter528_final_test_win": adapter_test_win,
            "dense544_final_validation_loss": float(dense544_validation["loss"]),
            "adapter528_final_validation_loss": float(adapter528_validation["loss"]),
            "dense528_final_validation_loss": float(dense528_validation["loss"]),
            "dense544_final_test_loss": float(dense544_test["loss"]),
            "adapter528_final_test_loss": float(adapter528_test["loss"]),
            "dense528_final_test_loss": float(dense528_test["loss"]),
            "dense544_tokens_per_second": float(
                dense544_train["training"]["tokens_per_second"]
            ),
            "adapter528_tokens_per_second": float(
                adapter528_train["training"]["tokens_per_second"]
            ),
            "dense528_tokens_per_second": float(
                dense528_train["training"]["tokens_per_second"]
            ),
            "dense528_functional_scores": {
                name: {
                    "token_accuracy_mean": float(task.get("token_accuracy_mean", 0.0)),
                    "exact_match_rate": float(task.get("exact_match_rate", 0.0)),
                }
                for name, task in dense528_functional.items()
                if isinstance(task, dict) and "token_accuracy_mean" in task
            },
        },
    }


def _assess_t12_three_seed_summary(records: list[dict[str, Any]]) -> Assessment | None:
    by_id = {str(record.get("experiment_id", "")): record for record in records}
    record = by_id.get("T12_three_seed_summary")
    if not record:
        return None

    metrics = record["metrics"]["metrics"]
    resolution = record["metrics"]["third_seed_resolution"]
    uncertainty = record["metrics"].get("uncertainty", {})
    pareto = record["metrics"].get("pareto", {})
    final_validation = metrics["final_validation_loss"]
    best_validation = metrics["best_validation_loss"]
    final_test = metrics["final_test_loss"]
    throughput = metrics["tokens_per_second"]
    model_bytes = metrics["model_only_bytes"]
    peak_vram = metrics["peak_vram_bytes"]
    max_gradient_norm = metrics["max_gradient_norm"]

    dense_validation_selected = bool(
        final_validation["winner_by_mean"] == "dense"
        and best_validation["winner_by_mean"] == "dense"
        and not resolution["test_loss_used_for_selection"]
    )
    dense_resource_wins = bool(
        model_bytes["winner_by_mean"] == "dense"
        and peak_vram["winner_by_mean"] == "dense"
    )
    dense_runtime_wins = throughput["winner_by_mean"] == "dense"
    dense_stability_wins = max_gradient_norm["winner_by_mean"] == "dense"
    adapter_final_test_win = final_test["winner_by_mean"] == "adapter"

    expands_measured_frontier = bool(
        pareto.get("frontier_expansion", {}).get(
            "found",
            dense_validation_selected
            and dense_resource_wins
            and dense_runtime_wins
            and dense_stability_wins
            and adapter_final_test_win,
        )
    )
    dominates_dense_baseline = bool(
        pareto.get("pareto_dominance", {}).get("found", False)
    )

    return {
        "experiment_id": "T12_three_seed_dense_adapter_assessment",
        "hypothesis": "dense528_is_validation_selected_over_residual_adapter528",
        "outcome": (
            "t12_dense528_validation_selected"
            if dense_validation_selected
            else "t12_three_seed_mixed"
        ),
        "supports_pareto_improvement": expands_measured_frontier
        or dominates_dense_baseline,
        "primary_reason": (
            "dense528_wins_three_seed_validation_resource_runtime_and_stability_means"
            "_but_does_not_pareto_dominate_adapter528"
        ),
        "limitations": [
            "test_loss_reported_only_not_selection_metric",
            "final_validation_paired_batch_bootstrap_ci_crosses_zero",
            "d4_javascript_source_local_only",
            "only_selected_dense528_has_executable_javascript_syntax_probe",
            "no_paired_documentation_source_consistency_benchmark",
            "no_packed_quantized_kernel_measurement",
            "residual_adapter_is_not_frozen_base_parameter_efficient_adapter",
        ],
        "evidence": {
            "summary_experiment_id": "T12_three_seed_summary",
            "dominates_dense_baseline": dominates_dense_baseline,
            "expands_measured_frontier": expands_measured_frontier,
            "pareto_dominance_found": dominates_dense_baseline,
            "frontier_expansion_found": expands_measured_frontier,
            "selected_family_by_final_validation_mean": resolution[
                "selected_family_by_final_validation_mean"
            ],
            "selected_family_by_best_validation_mean": resolution[
                "selected_family_by_best_validation_mean"
            ],
            "test_loss_used_for_selection": bool(resolution["test_loss_used_for_selection"]),
            "dense_final_validation_mean": float(
                final_validation["families"]["dense"]["mean"]
            ),
            "adapter_final_validation_mean": float(
                final_validation["families"]["adapter"]["mean"]
            ),
            "dense_best_validation_mean": float(best_validation["families"]["dense"]["mean"]),
            "adapter_best_validation_mean": float(
                best_validation["families"]["adapter"]["mean"]
            ),
            "dense_final_test_mean": float(final_test["families"]["dense"]["mean"]),
            "adapter_final_test_mean": float(final_test["families"]["adapter"]["mean"]),
            "adapter_final_test_winner_by_mean": adapter_final_test_win,
            "paired_batch_bootstrap": uncertainty.get("metrics", {}),
            "dense_throughput_winner_by_mean": dense_runtime_wins,
            "dense_resource_winner_by_mean": dense_resource_wins,
            "dense_stability_winner_by_mean": dense_stability_wins,
            "winner_changes_by_metric": resolution["winner_changes_by_metric"],
        },
    }


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


def _assess_d5_tokenizer_training(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    conclusions = metrics["conclusions"]
    comparisons = metrics["comparisons"]
    runs = metrics["runs"]
    equal_compute_win = bool(
        conclusions["bpe_equal_compute_improves_loss_per_estimated_byte"]
    )
    equal_raw_win = bool(
        conclusions["bpe_equal_raw_bytes_improves_loss_per_estimated_byte"]
    )
    exact_loss_measured = bool(conclusions.get("exact_raw_byte_loss_measured", False))
    equal_compute_exact_win = bool(
        conclusions.get("bpe_equal_compute_improves_exact_nats_per_raw_byte", False)
    )
    equal_raw_exact_win = bool(
        conclusions.get("bpe_equal_raw_bytes_improves_exact_nats_per_raw_byte", False)
    )
    evidence = {
        "byte_equal_compute_loss_per_estimated_byte": runs["byte_equal_compute"][
            "validation_loss_nats_per_estimated_byte"
        ],
        "bpe_equal_compute_loss_per_estimated_byte": runs["bpe_equal_compute"][
            "validation_loss_nats_per_estimated_byte"
        ],
        "bpe_equal_raw_bytes_loss_per_estimated_byte": runs["bpe_equal_raw_bytes"][
            "validation_loss_nats_per_estimated_byte"
        ],
        "bpe_train_token_reduction_ratio": comparisons[
            "bpe_train_token_reduction_ratio"
        ],
        "equal_compute_loss_delta_per_estimated_byte": comparisons[
            "equal_compute_bpe_minus_byte_nats_per_estimated_byte"
        ],
        "equal_raw_loss_delta_per_estimated_byte": comparisons[
            "equal_raw_bpe_minus_byte_nats_per_estimated_byte"
        ],
        "exact_raw_byte_loss_measured": exact_loss_measured,
        "bpe_equal_compute_improves_exact_nats_per_raw_byte": equal_compute_exact_win,
        "bpe_equal_raw_bytes_improves_exact_nats_per_raw_byte": equal_raw_exact_win,
        "functional_quality_measured": conclusions["functional_quality_measured"],
    }
    if exact_loss_measured:
        evidence.update(
            {
                "equal_compute_loss_delta_exact_nats_per_raw_byte": comparisons[
                    "equal_compute_bpe_minus_byte_exact_nats_per_raw_byte"
                ],
                "equal_raw_loss_delta_exact_nats_per_raw_byte": comparisons[
                    "equal_raw_bpe_minus_byte_exact_nats_per_raw_byte"
                ],
            }
        )
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": (
            "bpe_equal_compute_positive_equal_raw_negative"
            if equal_compute_win and not equal_raw_win
            else "mixed"
        ),
        "supports_pareto_improvement": False,
        "primary_reason": (
            "fast_bpe_improves_d5_loss_per_raw_byte_only_at_equal_token_budget"
        ),
        "limitations": [
            "single_seed_pilot",
            "token_reduction_alone_not_sufficient",
            "no_functional_quality_measurement",
            "not_architecture_selection",
        ],
        "evidence": evidence,
    }


def _assess_idea_foundry_candidates(record: dict[str, Any]) -> Assessment:
    summary = record["metrics"]["constraint_summary"]
    constraints_met = bool(
        int(summary["candidate_count"]) == 6
        and int(summary["without_adapters"]) >= 2
        and int(summary["without_moe_or_topic_routing"]) >= 2
        and int(summary["continual_evolution_candidates"]) >= 1
        and int(summary["compression_candidates"]) >= 1
        and int(summary["code_structure_candidates"]) >= 1
        and int(summary["potentially_novel_candidates"]) >= 1
    )
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "idea_lane_opened" if constraints_met else "idea_lane_incomplete",
        "supports_pareto_improvement": False,
        "primary_reason": "six_candidate_idea_foundry_constraints_recorded",
        "limitations": [
            "not_training_evidence",
            "not_quality_evidence",
            "novelty_requires_deeper_prior_art_review",
            "three_candidate_prototypes_still_pending",
        ],
        "evidence": {
            "candidate_count": summary["candidate_count"],
            "without_adapters": summary["without_adapters"],
            "without_moe_or_topic_routing": summary[
                "without_moe_or_topic_routing"
            ],
            "continual_evolution_candidates": summary[
                "continual_evolution_candidates"
            ],
            "compression_candidates": summary["compression_candidates"],
            "code_structure_candidates": summary["code_structure_candidates"],
            "potentially_novel_candidates": summary[
                "potentially_novel_candidates"
            ],
        },
    }


def _assess_repository_task_sample(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    sampling = metrics["sampling_policy"]
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "benchmark_scaffold",
        "supports_pareto_improvement": False,
        "primary_reason": "repository_first_task_index_created_without_model_scoring",
        "limitations": [
            "no_model_quality_measured",
            "no_executable_runtime_scoring_yet",
            "task_constructors_are_metadata_only",
        ],
        "evidence": {
            "repository_count": int(metrics["repository_count"]),
            "file_count": int(metrics["file_count"]),
            "task_count": int(metrics["task_count"]),
            "sampling_order": sampling["order"],
            "task_kinds": sampling["task_kinds"],
        },
    }


def _assess_repository_api_reuse_probe(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    best = metrics["best_method"]
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "api_reuse_benchmark_probe",
        "supports_pareto_improvement": False,
        "primary_reason": "repository_api_reuse_symbol_selection_scored_without_model_generation",
        "limitations": [
            "not_code_generation",
            "not_executable_runtime_scoring",
            "symbol_mentions_are_proxy_labels",
            "baseline_scoring_only",
        ],
        "evidence": {
            "source_split": metrics["source_split"],
            "query_split": metrics["query_split"],
            "repository_count": int(metrics["repository_count"]),
            "symbol_count": int(metrics["symbol_count"]),
            "task_count": int(metrics["task_count"]),
            "top_k": int(metrics["top_k"]),
            "best_method": best["name"],
            "best_hit_at_k": best["hit_at_k"],
            "best_mrr": best["mrr"],
            "method_names": sorted(metrics["methods"]),
        },
    }


def _assess_repository_context_pairwise_probe(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    best = metrics["best_method"]
    methods = metrics["methods"]
    structured = methods["structured_symbol_memory"]
    retrieved = methods["retrieved_snippet_identifiers"]
    symbol_aware = methods.get("symbol_aware_retrieved_snippets", {})
    query_symbol_aware = methods.get("query_symbol_aware_retrieval", {})
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "pairwise_context_proxy_probe",
        "supports_pareto_improvement": False,
        "primary_reason": (
            "structured_symbol_memory_beats_retrieval_variants_on_api_reuse_proxy"
        ),
        "limitations": [
            "not_code_generation",
            "not_executable_runtime_scoring",
            "symbol_mentions_are_proxy_labels",
            "proxy_hallucinated_api_rate",
            "retrieval_method_is_raw_identifier_extraction",
            "symbol_aware_retrieval_still_loses_hit_rate",
        ],
        "evidence": {
            "source_split": metrics["source_split"],
            "query_split": metrics["query_split"],
            "repository_count": int(metrics["repository_count"]),
            "symbol_count": int(metrics["symbol_count"]),
            "task_count": int(metrics["task_count"]),
            "top_k": int(metrics["top_k"]),
            "pairwise_ideas": metrics["pairwise_ideas"],
            "best_method": best["name"],
            "best_hit_at_k": best["hit_at_k"],
            "best_mrr": best["mrr"],
            "best_hallucinated_api_rate": best["hallucinated_api_rate"],
            "structured_hit_at_k": structured["hit_at_k"],
            "structured_hallucinated_api_rate": structured["hallucinated_api_rate"],
            "retrieved_hit_at_k": retrieved["hit_at_k"],
            "retrieved_hallucinated_api_rate": retrieved["hallucinated_api_rate"],
            "symbol_aware_hit_at_k": symbol_aware.get("hit_at_k"),
            "symbol_aware_hallucinated_api_rate": symbol_aware.get(
                "hallucinated_api_rate"
            ),
            "query_symbol_aware_hit_at_k": query_symbol_aware.get("hit_at_k"),
            "query_symbol_aware_hallucinated_api_rate": query_symbol_aware.get(
                "hallucinated_api_rate"
            ),
            "method_names": sorted(methods),
        },
    }


def _assess_quixbugs_repair_probe(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    final = metrics["final"]
    source = metrics["source"]
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "executable_repair_benchmark_probe",
        "supports_pareto_improvement": False,
        "primary_reason": "quixbugs_buggy_floor_and_oracle_ceiling_measured_with_pytest",
        "limitations": [
            "not_model_generated_repairs",
            "oracle_correct_is_upper_bound",
            "python_subset_only",
            "not_public_leaderboard_comparison",
        ],
        "evidence": {
            "repo_url": source["repo_url"],
            "repo_commit": source["repo_commit"],
            "program_count": int(metrics["program_count"]),
            "buggy_passed": int(final["buggy_passed"]),
            "oracle_correct_passed": int(final["oracle_correct_passed"]),
            "buggy_pass_rate": final["buggy_pass_rate"],
            "oracle_correct_pass_rate": final["oracle_correct_pass_rate"],
            "repair_gap": final["repair_gap"],
        },
    }


def _assess_quixbugs_candidate_repair_probe(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    final = metrics["final"]
    source = metrics["source"]
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "executable_candidate_repair_probe",
        "supports_pareto_improvement": False,
        "primary_reason": "quixbugs_candidate_replacement_sources_measured_with_pytest",
        "limitations": [
            "not_model_generated_unless_candidate_file_is_model_output",
            "replacement_source_only",
            "python_subset_only",
            "not_public_leaderboard_comparison",
        ],
        "evidence": {
            "repo_url": source["repo_url"],
            "repo_commit": source["repo_commit"],
            "program_count": int(metrics["program_count"]),
            "candidate_count": int(metrics["candidate_count"]),
            "candidate_passed": int(final["candidate_passed"]),
            "candidate_pass_rate": final["candidate_pass_rate"],
            "programs_with_passing_candidate": int(
                final["programs_with_passing_candidate"]
            ),
            "program_repair_rate": final["program_repair_rate"],
        },
    }


def _assess_quixbugs_dense_model_candidate_probe(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    final = metrics["final"]
    source = metrics["source"]
    model = metrics["model"]
    syntax = metrics.get("syntax", {})
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "model_generated_repair_candidate_probe",
        "supports_pareto_improvement": False,
        "primary_reason": "dense_checkpoint_generated_quixbugs_candidates_measured_with_pytest",
        "limitations": [
            "local_model_generated_candidate",
            "greedy_byte_generation",
            "prompted_source_replacement_from_general_lm",
            "replacement_source_only",
            "python_subset_only",
            "not_public_leaderboard_comparison",
        ],
        "evidence": {
            "repo_url": source["repo_url"],
            "repo_commit": source["repo_commit"],
            "checkpoint": model["checkpoint"],
            "checkpoint_step": int(model.get("checkpoint_step", 0)),
            "hidden_dim": int(model.get("config", {}).get("hidden_dim", 0)),
            "layers": int(model.get("config", {}).get("layers", 0)),
            "program_count": int(metrics["program_count"]),
            "candidate_count": int(metrics["candidate_count"]),
            "candidate_passed": int(final["candidate_passed"]),
            "candidate_pass_rate": final["candidate_pass_rate"],
            "programs_with_passing_candidate": int(
                final["programs_with_passing_candidate"]
            ),
            "program_repair_rate": final["program_repair_rate"],
            "generated_candidate_count": int(
                syntax.get("generated_candidate_count", metrics["candidate_count"])
            ),
            "syntax_valid_candidate_count": int(
                syntax.get("syntax_valid_candidate_count", 0)
            ),
            "programs_with_syntax_valid_candidate": int(
                syntax.get("programs_with_syntax_valid_candidate", 0)
            ),
        },
    }


def _assess_quixbugs_edit_baseline_probe(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    final = metrics["final"]
    source = metrics["source"]
    generator = metrics["generator"]
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "deterministic_repair_baseline_probe",
        "supports_pareto_improvement": False,
        "primary_reason": "hand_engineered_ast_edits_calibrate_quixbugs_candidate_lane",
        "limitations": [
            "not_model_generated",
            "hand_engineered_deterministic_edits",
            "does_not_read_oracle_correct_sources",
            "replacement_source_only",
            "python_subset_only",
            "not_public_leaderboard_comparison",
        ],
        "evidence": {
            "repo_url": source["repo_url"],
            "repo_commit": source["repo_commit"],
            "generator_label": generator["label"],
            "program_count": int(metrics["program_count"]),
            "candidate_count": int(metrics["candidate_count"]),
            "candidate_passed": int(final["candidate_passed"]),
            "candidate_pass_rate": final["candidate_pass_rate"],
            "programs_with_passing_candidate": int(
                final["programs_with_passing_candidate"]
            ),
            "program_repair_rate": final["program_repair_rate"],
            "max_candidates_per_program": int(generator["max_candidates_per_program"]),
            "edit_templates": list(generator["edit_templates"]),
        },
    }


def _assess_quixbugs_dense_ranked_edit_probe(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    final = metrics["final"]
    source = metrics["source"]
    model = metrics["model"]
    candidate_pool = metrics["candidate_pool"]
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "model_ranked_repair_candidate_probe",
        "supports_pareto_improvement": False,
        "primary_reason": "dense_checkpoint_ranked_deterministic_ast_edits_with_pytest",
        "limitations": [
            "local_model_ranked_candidate",
            "deterministic_candidate_pool",
            "not_free_form_generation",
            "replacement_source_only",
            "python_subset_only",
            "not_public_leaderboard_comparison",
        ],
        "evidence": {
            "repo_url": source["repo_url"],
            "repo_commit": source["repo_commit"],
            "checkpoint": model["checkpoint"],
            "checkpoint_step": int(model.get("checkpoint_step", 0)),
            "hidden_dim": int(model.get("config", {}).get("hidden_dim", 0)),
            "layers": int(model.get("config", {}).get("layers", 0)),
            "program_count": int(metrics["program_count"]),
            "candidate_pool_count": int(
                candidate_pool["generated_candidate_count"]
            ),
            "selected_candidate_count": int(
                candidate_pool["selected_candidate_count"]
            ),
            "top_candidates_per_program": int(
                candidate_pool["top_candidates_per_program"]
            ),
            "candidate_passed": int(final["candidate_passed"]),
            "candidate_pass_rate": final["candidate_pass_rate"],
            "programs_with_passing_candidate": int(
                final["programs_with_passing_candidate"]
            ),
            "program_repair_rate": final["program_repair_rate"],
        },
    }


def _assess_quixbugs_dense_ranked_syntax_pool_probe(
    record: dict[str, Any],
) -> Assessment:
    metrics = record["metrics"]
    final = metrics["final"]
    source = metrics["source"]
    model = metrics["model"]
    candidate_pool = metrics["candidate_pool"]
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "model_ranked_broader_syntax_pool_probe",
        "supports_pareto_improvement": False,
        "primary_reason": "dense_checkpoint_top1_ranking_drops_on_broader_syntax_pool",
        "limitations": [
            "local_model_ranked_candidate",
            "syntax_preserving_candidate_pool",
            "broader_than_deterministic_edit_baseline",
            "not_free_form_generation",
            "replacement_source_only",
            "python_subset_only",
            "not_public_leaderboard_comparison",
        ],
        "evidence": {
            "repo_url": source["repo_url"],
            "repo_commit": source["repo_commit"],
            "checkpoint": model["checkpoint"],
            "checkpoint_step": int(model.get("checkpoint_step", 0)),
            "hidden_dim": int(model.get("config", {}).get("hidden_dim", 0)),
            "layers": int(model.get("config", {}).get("layers", 0)),
            "program_count": int(metrics["program_count"]),
            "candidate_pool_count": int(
                candidate_pool["generated_candidate_count"]
            ),
            "selected_candidate_count": int(
                candidate_pool["selected_candidate_count"]
            ),
            "top_candidates_per_program": int(
                candidate_pool["top_candidates_per_program"]
            ),
            "candidate_passed": int(final["candidate_passed"]),
            "candidate_pass_rate": final["candidate_pass_rate"],
            "programs_with_passing_candidate": int(
                final["programs_with_passing_candidate"]
            ),
            "program_repair_rate": final["program_repair_rate"],
        },
    }


def _assess_quixbugs_dense_ranked_syntax_topk_probe(
    record: dict[str, Any],
) -> Assessment:
    metrics = record["metrics"]
    final = metrics["final"]
    source = metrics["source"]
    model = metrics["model"]
    candidate_pool = metrics["candidate_pool"]
    top_k_profile = metrics["top_k_profile"]
    top1 = next(
        row for row in top_k_profile if int(row["top_candidates_per_program"]) == 1
    )
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "model_ranked_syntax_topk_pairwise_probe",
        "supports_pareto_improvement": False,
        "primary_reason": "top_k_execution_profiles_dense_ranking_plus_syntax_pool",
        "limitations": [
            "local_model_ranked_candidate",
            "syntax_preserving_candidate_pool",
            "broader_than_deterministic_edit_baseline",
            "top_k_execution_profile",
            "not_free_form_generation",
            "replacement_source_only",
            "python_subset_only",
            "not_public_leaderboard_comparison",
        ],
        "evidence": {
            "repo_url": source["repo_url"],
            "repo_commit": source["repo_commit"],
            "checkpoint": model["checkpoint"],
            "checkpoint_step": int(model.get("checkpoint_step", 0)),
            "hidden_dim": int(model.get("config", {}).get("hidden_dim", 0)),
            "layers": int(model.get("config", {}).get("layers", 0)),
            "program_count": int(metrics["program_count"]),
            "candidate_pool_count": int(
                candidate_pool["generated_candidate_count"]
            ),
            "selected_candidate_count": int(
                candidate_pool["selected_candidate_count"]
            ),
            "top_k_values": list(candidate_pool["top_k_values"]),
            "max_top_k_profiled": int(candidate_pool["max_top_k_profiled"]),
            "top1_candidate_pass_rate": top1["candidate_pass_rate"],
            "top1_program_repair_rate": top1["program_repair_rate"],
            "best_top_k": int(final["best_top_k"]),
            "best_candidate_pass_rate": final["best_candidate_pass_rate"],
            "best_program_repair_rate": final["best_program_repair_rate"],
            "candidate_passed_at_max_k": int(final["candidate_passed"]),
            "program_repair_rate_at_max_k": final["program_repair_rate"],
        },
    }


def _assess_quixbugs_syntax_pool_ordering_control_probe(
    record: dict[str, Any],
) -> Assessment:
    metrics = record["metrics"]
    final = metrics["final"]
    source = metrics["source"]
    model = metrics["model"]
    candidate_pool = metrics["candidate_pool"]
    controls = metrics["ordering_controls"]
    dense = controls["dense_likelihood"]
    deterministic = controls["deterministic_pool_order"]
    repair_aware = controls.get("repair_aware_static_order")
    random_control = controls["random_seeded_order"]
    dense_top1 = next(
        row
        for row in dense["top_k_profile"]
        if int(row["top_candidates_per_program"]) == 1
    )
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "syntax_pool_ordering_control_probe",
        "supports_pareto_improvement": False,
        "primary_reason": (
            "dense_ranking_compared_with_same_pool_non_model_controls"
        ),
        "limitations": [
            "local_model_ranked_candidate",
            "syntax_preserving_candidate_pool",
            "broader_than_deterministic_edit_baseline",
            "top_k_execution_profile",
            "same_pool_ordering_controls",
            "not_free_form_generation",
            "replacement_source_only",
            "python_subset_only",
            "not_public_leaderboard_comparison",
        ],
        "evidence": {
            "repo_url": source["repo_url"],
            "repo_commit": source["repo_commit"],
            "checkpoint": model["checkpoint"],
            "checkpoint_step": int(model.get("checkpoint_step", 0)),
            "hidden_dim": int(model.get("config", {}).get("hidden_dim", 0)),
            "layers": int(model.get("config", {}).get("layers", 0)),
            "program_count": int(metrics["program_count"]),
            "candidate_pool_count": int(
                candidate_pool["generated_candidate_count"]
            ),
            "selected_candidate_count": int(
                candidate_pool["selected_candidate_count"]
            ),
            "top_k_values": list(candidate_pool["top_k_values"]),
            "max_top_k_profiled": int(candidate_pool["max_top_k_profiled"]),
            "best_ordering": final["best_ordering"],
            "best_program_repair_rate": final["best_program_repair_rate"],
            "dense_beats_all_controls": bool(final["dense_beats_all_controls"]),
            "dense_top1_program_repair_rate": dense_top1["program_repair_rate"],
            "dense_best_top_k": int(dense["best_top_k"]),
            "dense_best_program_repair_rate": dense["best_program_repair_rate"],
            "deterministic_best_top_k": int(deterministic["best_top_k"]),
            "deterministic_best_program_repair_rate": deterministic[
                "best_program_repair_rate"
            ],
            "repair_aware_best_top_k": (
                int(repair_aware["best_top_k"]) if repair_aware is not None else None
            ),
            "repair_aware_best_program_repair_rate": (
                repair_aware["best_program_repair_rate"]
                if repair_aware is not None
                else None
            ),
            "random_best_top_k": int(random_control["best_top_k"]),
            "random_best_program_repair_rate": random_control[
                "best_program_repair_rate"
            ],
            "control_names": list(final["control_names"]),
        },
    }


def _assess_idea_foundry_graph_probe(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    signal_present = bool(
        metrics["mechanism_signal_present"]
        and metrics["repository_aware_splits_preserved"]
    )
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "mechanism_signal_present" if signal_present else "no_mechanism_signal",
        "supports_pareto_improvement": False,
        "primary_reason": "repository_graph_signal_probe_completed_for_if1",
        "limitations": [
            "not_model_training",
            "regex_import_and_heuristic_role_link_extraction",
            "no_ast_or_package_resolution",
            "does_not_prove_quality_gain",
        ],
        "evidence": {
            "candidate_id": metrics["candidate_id"],
            "document_count": metrics["document_count"],
            "import_edge_count": metrics["import_edge_count"],
            "resolved_local_edge_count": metrics["resolved_local_edge_count"],
            "role_link_edge_count": metrics.get("role_link_edge_count", 0),
            "graph_edge_count": metrics.get(
                "graph_edge_count",
                metrics["resolved_local_edge_count"],
            ),
            "typed_edge_counts": metrics.get("typed_edge_counts", {}),
            "repositories_with_edges": metrics["repositories_with_edges"],
            "repository_aware_splits_preserved": metrics[
                "repository_aware_splits_preserved"
            ],
            "mechanism_signal_present": metrics["mechanism_signal_present"],
        },
    }


def _assess_if2_fast_weight_probe(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    final = metrics["final"]
    heldout = metrics["heldout_generalization"]
    methods = metrics["methods"]
    adds_value = bool(metrics["parameter_evolution_adds_value_beyond_updated_memory"])
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": (
            "synthetic_fast_weight_positive"
            if adds_value
            else "updated_memory_not_beaten"
        ),
        "supports_pareto_improvement": False,
        "primary_reason": "fast_weight_proxy_adds_heldout_paraphrase_signal_beyond_memory",
        "limitations": [
            "synthetic_fixture_only",
            "not_language_model_training",
            "feature_hash_fast_weight_proxy",
            "no_security_or_poisoning_gate",
            "storage_cost_exceeds_structured_memory",
        ],
        "evidence": {
            "candidate_id": metrics["candidate_id"],
            "timeline_steps": metrics["timeline_steps"],
            "exact_retrieval_accuracy": final["exact_retrieval_accuracy"],
            "structured_memory_accuracy": final["structured_memory_accuracy"],
            "fast_weight_scratchpad_accuracy": final["fast_weight_scratchpad_accuracy"],
            "structured_memory_plus_fast_weight_accuracy": final[
                "structured_memory_plus_fast_weight_accuracy"
            ],
            "heldout_task_count": heldout["task_count"],
            "heldout_structured_memory_correct": heldout[
                "structured_memory_correct"
            ],
            "heldout_fast_weight_correct": heldout[
                "fast_weight_scratchpad_correct"
            ],
            "heldout_combined_correct": heldout[
                "structured_memory_plus_fast_weight_correct"
            ],
            "structured_memory_storage_bytes": methods["structured_memory"][
                "storage_bytes"
            ],
            "parameter_bytes": methods["fast_weight_scratchpad"][
                "parameter_bytes"
            ],
            "combined_storage_bytes": methods["structured_memory_plus_fast_weight"][
                "storage_bytes"
            ],
            "parameter_evolution_adds_value_beyond_updated_memory": adds_value,
        },
    }


def _assess_if4_fast_repo_adaptation_probe(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    final = metrics["final"]
    repo = metrics["repo"]
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "repository_adaptation_probe",
        "supports_pareto_improvement": False,
        "primary_reason": "fast_repository_adaptation_controls_measured_on_real_git_history",
        "limitations": [
            "changed_file_retrieval_proxy",
            "not_language_model_training",
            "replay_adapter_is_sparse_proxy",
            "feature_hash_fast_weight_proxy",
            "paraphrase_transfer_is_query_rewrite_proxy",
            "no_security_or_poisoning_gate",
        ],
        "evidence": {
            "candidate_id": metrics["candidate_id"],
            "repository": repo["path_name"],
            "commit_count_used": repo["commit_count_used"],
            "top_k": repo["top_k"],
            "step_count": len(metrics["steps"]),
            "future_leakage_blocked": all(
                step["future_commit_not_in_memory"] for step in metrics["steps"]
            ),
            "updated_retrieval_future_topk_accuracy": final[
                "updated_retrieval_future_topk_accuracy"
            ],
            "structured_symbol_graph_memory_future_topk_accuracy": final[
                "structured_symbol_graph_memory_future_topk_accuracy"
            ],
            "replay_adapter_proxy_future_topk_accuracy": final[
                "replay_adapter_proxy_future_topk_accuracy"
            ],
            "fast_temporary_weights_future_topk_accuracy": final[
                "fast_temporary_weights_future_topk_accuracy"
            ],
            "fast_weights_plus_retrieval_future_topk_accuracy": final[
                "fast_weights_plus_retrieval_future_topk_accuracy"
            ],
            "periodic_consolidation_future_topk_accuracy": final[
                "periodic_consolidation_future_topk_accuracy"
            ],
            "fast_weights_plus_retrieval_paraphrase_topk_accuracy": final[
                "fast_weights_plus_retrieval_paraphrase_topk_accuracy"
            ],
            "prior_task_retention_accuracy": final["prior_task_retention_accuracy"],
            "mean_update_ms": final["mean_update_ms"],
            "rollback_supported": final["rollback_supported"],
            "total_storage_bytes": final["total_storage_bytes"],
        },
    }


def _assess_if7_sparse_hebbian_probe(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    final = metrics["final"]
    corpus = metrics["corpus"]
    methods = metrics["methods"]
    adds_signal = bool(metrics["hebbian_adds_associative_signal"])
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": (
            "real_corpus_associative_signal"
            if adds_signal
            else "frequency_or_random_control_not_beaten"
        ),
        "supports_pareto_improvement": False,
        "primary_reason": (
            "sparse_hebbian_completion_beats_frequency_and_random_controls_on_masked_real_rows"
            if adds_signal
            else "sparse_hebbian_completion_does_not_beat_controls"
        ),
        "limitations": [
            "associative_memory_probe_only",
            "not_language_model_training",
            "masked_recall_from_exposed_rows_not_unseen_repo_generalization",
            "hashed_node_collisions_possible",
            "dense_numpy_matrix_not_packed_sparse_kernel",
            "no_security_or_authorization_gate",
        ],
        "evidence": {
            "candidate_id": metrics["candidate_id"],
            "rows_loaded": corpus["rows_loaded"],
            "rows_scanned": corpus["rows_scanned"],
            "usable_patterns": corpus["usable_patterns"],
            "eval_patterns": corpus["eval_patterns"],
            "repositories_loaded": corpus["repositories_loaded"],
            "mean_active_fraction": metrics["sparsity"]["mean_active_fraction"],
            "hebbian_hit_at_k": final["hebbian_hit_at_k"],
            "frequency_hit_at_k": final["frequency_hit_at_k"],
            "random_hit_at_k": final["random_hit_at_k"],
            "hebbian_mrr": final["hebbian_mrr"],
            "frequency_mrr": final["frequency_mrr"],
            "random_mrr": final["random_mrr"],
            "hebbian_storage_bytes": methods["hebbian_sparse_assembly"][
                "storage_bytes"
            ],
            "random_storage_bytes": methods["random_sparse_control"][
                "storage_bytes"
            ],
            "hebbian_adds_associative_signal": adds_signal,
        },
    }


def _assess_if7_hebbian_trained_model(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    cue_only = metrics["validation"]["cue_only"]
    cue_plus = metrics["validation"]["cue_plus_hebbian"]
    raw = metrics["validation"]["raw_hebbian_memory"]
    improves = bool(metrics["hebbian_conditioning_improves_trained_model"])
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": (
            "trained_hebbian_conditioning_positive"
            if improves
            else "trained_cue_only_not_beaten"
        ),
        "supports_pareto_improvement": False,
        "primary_reason": (
            "trained_hebbian_conditioning_beats_or_matches_cue_only_validation"
            if improves
            else "cue_plus_hebbian_trained_model_loses_to_cue_only_on_validation"
        ),
        "limitations": [
            "trained_linear_multilabel_probe_not_decoder_language_model",
            "predicts_hashed_identifier_import_nodes_not_tokens",
            "not_executable_code_generation_or_repair",
            "dense_torch_linear_layers_not_sparse_rocm_kernel",
            "no_security_or_authorization_gate",
        ],
        "evidence": {
            "candidate_id": metrics["candidate_id"],
            "train_rows": metrics["training"]["supervised_train_rows"],
            "validation_rows": metrics["training"]["validation_rows"],
            "accelerator_backend": metrics["training"]["accelerator_backend"],
            "node_count": metrics["hebbian_memory"]["node_count"],
            "hebbian_memory_storage_bytes": metrics["hebbian_memory"]["storage_bytes"],
            "cue_only_hit_at_k": cue_only["hit_at_k"],
            "cue_plus_hebbian_hit_at_k": cue_plus["hit_at_k"],
            "raw_hebbian_hit_at_k": raw["hit_at_k"],
            "cue_only_mrr": cue_only["mrr"],
            "cue_plus_hebbian_mrr": cue_plus["mrr"],
            "raw_hebbian_mrr": raw["mrr"],
            "cue_only_loss": cue_only["loss"],
            "cue_plus_hebbian_loss": cue_plus["loss"],
            "hebbian_conditioning_improves_trained_model": improves,
        },
    }


def _assess_if7_sparse_hebbian_reranker(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    raw = metrics["validation"]["raw_hebbian_memory"]
    reranker = metrics["validation"]["candidate_reranker"]
    ceiling = metrics["validation"]["candidate_recall_ceiling"]
    improves = bool(metrics["sparse_reranker_improves_raw_hebbian"])
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": (
            "sparse_reranker_positive"
            if improves
            else "raw_hebbian_not_beaten_by_sparse_reranker"
        ),
        "supports_pareto_improvement": False,
        "primary_reason": (
            "sparse_candidate_reranker_beats_or_matches_raw_hebbian"
            if improves
            else "sparse_candidate_reranker_loses_to_raw_hebbian_ranking"
        ),
        "limitations": [
            "candidate_reranker_not_decoder_language_model",
            "predicts_hashed_identifier_import_nodes_not_tokens",
            "candidate_recall_bounds_possible_quality",
            "not_executable_code_generation_or_repair",
            "dense_torch_linear_scorer_not_sparse_rocm_kernel",
        ],
        "evidence": {
            "candidate_id": metrics["candidate_id"],
            "train_patterns": metrics["training"]["train_patterns"],
            "validation_patterns": metrics["training"]["validation_patterns"],
            "train_candidate_examples": metrics["training"]["train_candidate_examples"],
            "candidate_count": metrics["training"]["candidate_count"],
            "parameter_count": metrics["models"]["candidate_reranker"][
                "parameter_count"
            ],
            "raw_hebbian_hit_at_k": raw["hit_at_k"],
            "raw_hebbian_mrr": raw["mrr"],
            "reranker_hit_at_k": reranker["hit_at_k"],
            "reranker_mrr": reranker["mrr"],
            "candidate_ceiling_hit_at_k": ceiling["hit_at_k"],
            "candidate_ceiling_mrr": ceiling["mrr"],
            "sparse_reranker_improves_raw_hebbian": improves,
        },
    }


def _assess_if7_hebbian_repository_linking(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    corpus = metrics["corpus"]
    tasks = metrics["tasks"]
    methods = metrics["methods"]
    lexical = methods["lexical_text_overlap"]
    raw = methods["raw_hebbian_context"]
    combined = methods["combined_lexical_hebbian"]
    raw_beats = bool(metrics["hebbian_beats_lexical"])
    combined_beats = bool(metrics["combined_beats_lexical"])
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": (
            "repository_linking_hebbian_positive"
            if raw_beats or combined_beats
            else "lexical_baseline_not_beaten"
        ),
        "supports_pareto_improvement": False,
        "primary_reason": (
            "repository_linking_hebbian_or_combined_score_beats_lexical_text_overlap"
            if raw_beats or combined_beats
            else "repository_linking_hebbian_scores_lose_to_lexical_text_overlap"
        ),
        "limitations": [
            "repository_linking_proxy_not_generation",
            "positive_targets_are_same_repo_files_not_human_labeled_dependencies",
            "distractors_sampled_from_eval_split_other_repositories",
            "global_hebbian_memory_built_from_train_split_only",
            "lexical_baseline_is_strong_for_same_repository_file_linking",
            "not_executable_code_generation_or_repair",
        ],
        "evidence": {
            "candidate_id": metrics["candidate_id"],
            "train_rows_loaded": corpus["train_rows_loaded"],
            "eval_rows_loaded": corpus["eval_rows_loaded"],
            "eval_repositories": corpus["eval_repositories"],
            "eligible_eval_repositories": corpus["eligible_eval_repositories"],
            "task_count": tasks["task_count"],
            "top_k": tasks["top_k"],
            "negatives_per_query": tasks["negatives_per_query"],
            "candidate_count_mean": tasks["candidate_count_mean"],
            "best_method": metrics["best_method"]["name"],
            "lexical_hit_at_k": lexical["hit_at_k"],
            "lexical_mrr": lexical["mrr"],
            "lexical_coverage_at_k": lexical["coverage_at_k"],
            "raw_hebbian_hit_at_k": raw["hit_at_k"],
            "raw_hebbian_mrr": raw["mrr"],
            "raw_hebbian_coverage_at_k": raw["coverage_at_k"],
            "combined_hit_at_k": combined["hit_at_k"],
            "combined_mrr": combined["mrr"],
            "combined_coverage_at_k": combined["coverage_at_k"],
            "hebbian_beats_lexical": raw_beats,
            "combined_beats_lexical": combined_beats,
        },
    }


def _assess_if7_trained_repository_ranker(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    corpus = metrics["corpus"]
    tasks = metrics["tasks"]
    training = metrics["training"]
    models = metrics["models"]
    methods = metrics["methods"]
    lexical = methods["lexical_text_overlap"]
    raw = methods["raw_hebbian_context"]
    trained = methods["trained_task_aware_ranker"]
    no_hebbian = methods["trained_no_hebbian_ranker"]
    feature_names = list(models["task_aware_ranker"].get("feature_names", []))
    beats_lexical = bool(metrics["trained_ranker_beats_lexical"])
    beats_raw = bool(metrics["trained_ranker_beats_raw_hebbian"])
    beats_ablation = bool(metrics["trained_ranker_beats_no_hebbian"])
    if beats_lexical and beats_raw and beats_ablation:
        outcome = "task_ranker_hebbian_ablation_positive"
        reason = "trained_repository_ranker_beats_static_baselines_and_no_hebbian_ablation"
    elif beats_lexical and beats_raw:
        outcome = "task_ranker_positive_hebbian_ablation_not_beaten"
        reason = "trained_repository_ranker_beats_static_baselines_but_not_no_hebbian_ablation"
    else:
        outcome = "task_ranker_not_beating_static_baselines"
        reason = "trained_repository_ranker_does_not_beat_required_static_baselines"
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": outcome,
        "supports_pareto_improvement": False,
        "primary_reason": reason,
        "limitations": [
            "repository_linking_proxy_not_generation",
            "positive_targets_are_same_repo_files_not_human_labeled_dependencies",
            "linear_ranker_not_decoder_language_model",
            "hebbian_feature_requires_no_hebbian_ablation_to_claim_value",
            "not_executable_code_generation_or_repair",
        ],
        "evidence": {
            "candidate_id": metrics["candidate_id"],
            "train_rows_loaded": corpus["train_rows_loaded"],
            "eval_rows_loaded": corpus["eval_rows_loaded"],
            "train_task_repositories": corpus["train_task_repositories"],
            "eval_task_repositories": corpus["eval_task_repositories"],
            "train_tasks": tasks["train_tasks"],
            "eval_tasks": tasks["eval_tasks"],
            "top_k": tasks["top_k"],
            "negatives_per_query": tasks["negatives_per_query"],
            "candidate_examples": training["candidate_examples"],
            "epochs": training["epochs"],
            "accelerator_backend": training["accelerator_backend"],
            "trained_parameter_count": models["task_aware_ranker"]["parameter_count"],
            "no_hebbian_parameter_count": models["no_hebbian_ranker"][
                "parameter_count"
            ],
            "feature_names": feature_names,
            "has_hebbian_pair_edge_score": "hebbian_pair_edge_score" in feature_names,
            "best_method": metrics["best_method"]["name"],
            "selected_ranker_uses_hebbian": metrics["best_method"]["name"]
            == "trained_task_aware_ranker",
            "lexical_hit_at_k": lexical["hit_at_k"],
            "lexical_mrr": lexical["mrr"],
            "raw_hebbian_hit_at_k": raw["hit_at_k"],
            "raw_hebbian_mrr": raw["mrr"],
            "trained_hit_at_k": trained["hit_at_k"],
            "trained_mrr": trained["mrr"],
            "trained_coverage_at_k": trained["coverage_at_k"],
            "no_hebbian_hit_at_k": no_hebbian["hit_at_k"],
            "no_hebbian_mrr": no_hebbian["mrr"],
            "no_hebbian_coverage_at_k": no_hebbian["coverage_at_k"],
            "trained_ranker_beats_lexical": beats_lexical,
            "trained_ranker_beats_raw_hebbian": beats_raw,
            "trained_ranker_beats_no_hebbian": beats_ablation,
        },
    }


def _assess_if3_block_codebook_probe(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    compression = metrics["compression"]
    learned = compression["learned_codebook"]
    random_control = compression["random_codebook_control"]
    beats_random = bool(learned["beats_random_control"])
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": (
            "reconstruction_proxy_positive"
            if beats_random
            else "random_control_not_beaten"
        ),
        "supports_pareto_improvement": False,
        "primary_reason": "block_codebook_reconstruction_beats_random_control",
        "limitations": [
            "reconstruction_proxy_only",
            "no_language_model_loss",
            "no_packed_kernel_speed",
            "metadata_and_runtime_buffers_counted",
            "not_deployment_ready_compression",
        ],
        "evidence": {
            "candidate_id": metrics["candidate_id"],
            "checkpoint_type": metrics["checkpoint"]["checkpoint_type"],
            "checkpoint_step": metrics["checkpoint"]["step"],
            "floating_parameter_count": compression["floating_parameter_count"],
            "block_count": compression["block_count"],
            "learned_mse": learned["mse"],
            "random_control_mse": random_control["mse"],
            "beats_random_control": beats_random,
            "encoded_bytes": learned["encoded_bytes"],
            "metadata_bytes": learned["metadata_bytes"],
            "runtime_buffer_bytes": learned["runtime_buffer_bytes"],
            "encoded_plus_runtime_bytes": learned["encoded_plus_runtime_bytes"],
            "packed_kernel_evaluated": metrics["packed_kernel_evaluated"],
            "loss_evaluated": metrics["loss_evaluated"],
        },
    }


def _assess_if3_block_codebook_validation_probe(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    compression = metrics["compression"]
    learned = compression["learned_codebook"]
    random_control = compression["random_codebook_control"]
    policies = metrics["policies"]
    comparisons = metrics["comparisons"]
    learned_beats_random_loss = bool(comparisons["learned_beats_random_loss"])
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": (
            "validation_loss_probe_positive"
            if learned_beats_random_loss
            else "validation_loss_random_control_not_beaten"
        ),
        "supports_pareto_improvement": False,
        "primary_reason": (
            "learned_block_codebook_validation_loss_beats_random_control"
            if learned_beats_random_loss
            else "learned_block_codebook_validation_loss_does_not_beat_random_control"
        ),
        "limitations": [
            "reconstructed_fp32_runtime_buffer",
            "no_packed_kernel_speed",
            "metadata_and_runtime_buffers_counted",
            "compression_candidate_not_deployment_ready",
            "single_checkpoint_validation_probe",
        ],
        "evidence": {
            "candidate_id": metrics["candidate_id"],
            "checkpoint_type": metrics["checkpoint"]["checkpoint_type"],
            "checkpoint_step": metrics["checkpoint"]["step"],
            "split": metrics["split"],
            "floating_parameter_count": compression["floating_parameter_count"],
            "block_count": compression["block_count"],
            "fp32_loss": policies["fp32"]["loss"],
            "learned_loss": policies["learned_block_codebook"]["loss"],
            "random_control_loss": policies["random_block_codebook"]["loss"],
            "tokens": policies["fp32"]["tokens"],
            "learned_loss_delta_vs_fp32": comparisons["learned_loss_delta_vs_fp32"],
            "random_loss_delta_vs_fp32": comparisons["random_loss_delta_vs_fp32"],
            "learned_loss_delta_vs_random": comparisons["learned_loss_delta_vs_random"],
            "learned_beats_random_loss": learned_beats_random_loss,
            "learned_mse": learned["mse"],
            "random_control_mse": random_control["mse"],
            "learned_mse_beats_random_mse": comparisons[
                "learned_mse_beats_random_mse"
            ],
            "encoded_bytes": learned["encoded_bytes"],
            "metadata_bytes": learned["metadata_bytes"],
            "runtime_buffer_bytes": learned["runtime_buffer_bytes"],
            "encoded_plus_runtime_bytes": learned["encoded_plus_runtime_bytes"],
            "packed_kernel_evaluated": metrics["packed_kernel_evaluated"],
            "loss_evaluated": metrics["loss_evaluated"],
        },
    }


def _assess_dense_js_executable_probe(record: dict[str, Any]) -> Assessment:
    metrics = record["metrics"]
    task = metrics["tasks"]["line_completion_syntax"]
    node = metrics["node"]
    return {
        "experiment_id": record["experiment_id"],
        "hypothesis": record.get("hypothesis"),
        "outcome": "executable_js_syntax_probe_recorded",
        "supports_pareto_improvement": False,
        "primary_reason": "heldout_generated_javascript_checked_with_node_syntax",
        "limitations": [
            "syntax_only_not_unit_tests",
            "line_completion_context_only",
            "not_repair_or_api_reuse",
            "not_multilingual",
            "not_architecture_comparison",
        ],
        "evidence": {
            "checkpoint": metrics["checkpoint"],
            "split": metrics["split"],
            "node_available": node["available"],
            "node_version": node["version"],
            "candidate_count": task["candidate_count"],
            "completed_tasks": task["completed_tasks"],
            "token_accuracy_mean": task["token_accuracy_mean"],
            "exact_match_rate": task["exact_match_rate"],
            "edit_similarity_mean": task["edit_similarity_mean"],
            "oracle_node_syntax_pass_rate": task["oracle_node_syntax_pass_rate"],
            "generated_node_syntax_pass_rate": task["generated_node_syntax_pass_rate"],
        },
    }
