import json
from pathlib import Path

from weightlab.assessment import assess_manifest, assess_record


def _record(experiment_id: str) -> dict[str, object]:
    return json.loads((Path("results") / f"{experiment_id}.json").read_text())


def test_assessment_marks_rejected_storage_without_treating_green_run_as_success():
    assessment = assess_record(_record("E2_compositional_storage"))

    assert assessment["outcome"] == "rejected"
    assert assessment["primary_reason"] == "uniform_int8_dominates_tested_compositional_methods"
    assert assessment["limitations"]
    assert "metadata_accounting" in assessment["evidence"]


def test_assessment_marks_tiny_real_model_precision_as_mixed_or_no_signal():
    e3j = assess_record(_record("E3j_real_open_model_natural_text_precision"))
    e3k = assess_record(_record("E3k_real_open_model_internal_tensor_precision"))

    assert e3j["outcome"] == "mixed"
    assert e3j["primary_reason"] == "selected_rows_beat_random_mean_but_lose_storage"
    assert "not_pareto_improvement" in e3j["limitations"]

    assert e3k["outcome"] == "no_signal"
    assert e3k["primary_reason"] == "all_precision_policies_have_zero_measurable_kl"
    assert "tiny_tensor" in e3k["limitations"]


def test_assessment_distinguishes_smoke_security_from_production_security_proof():
    assessment = assess_record(_record("S2c_semantic_extraction_red_team"))

    assert assessment["outcome"] == "smoke_pass"
    assert assessment["primary_reason"] == "covered_security_checks_passed"
    assert "not_production_proof" in assessment["limitations"]
    assert assessment["evidence"]["critical_failures"] == 0


def test_assessment_marks_rocm_runtime_as_positive_but_not_speed_claim():
    assessment = assess_record(_record("E4c_torch_batched_routed_execution"))

    assert assessment["outcome"] == "runtime_positive"
    assert assessment["primary_reason"] == "rocm_backend_executed_torch_dispatch"
    assert assessment["supports_pareto_improvement"] is False
    assert "not_speed_claim" in assessment["limitations"]
    assert assessment["evidence"]["accelerator_backend"] == "rocm"
    assert assessment["evidence"]["uses_rocm_transfer"] is True
    assert assessment["evidence"]["uses_cuda_transfer"] is False


def test_assessment_marks_rocm_scaling_as_measurement_without_occupancy_claim():
    assessment = assess_record(_record("E4d_rocm_transfer_scaling"))

    assert assessment["outcome"] == "measurement_positive"
    assert assessment["primary_reason"] == "rocm_transfer_scaling_recorded"
    assert assessment["supports_pareto_improvement"] is False
    assert "no_occupancy_measurement" in assessment["limitations"]
    assert assessment["evidence"]["accelerator_backend"] == "rocm"
    assert assessment["evidence"]["payload_count"] >= 3
    assert assessment["evidence"]["uses_rocm_transfer"] is True


def test_assessment_marks_rocm_training_validation_as_readiness_evidence():
    assessment = assess_record(_record("R1_rocm_training_validation"))

    assert assessment["outcome"] == "training_readiness_positive"
    assert assessment["primary_reason"] == "rocm_training_runtime_probe_completed"
    assert assessment["supports_pareto_improvement"] is False
    assert "not_full_model_training" in assessment["limitations"]
    assert assessment["evidence"]["accelerator_backend"] == "rocm"
    assert assessment["evidence"]["rocm_available"] is True
    assert assessment["evidence"]["checkpoint_resume_ok"] is True
    assert assessment["evidence"]["stable_batch_size"] >= 1
    assert assessment["evidence"]["max_stable_tokens"] >= 1


def test_assessment_marks_corpus_preparation_as_insufficient_until_50m_tokens():
    assessment = assess_record(_record("D1_corpus_preparation"))

    assert assessment["outcome"] == "corpus_prepared_insufficient_tokens"
    assert assessment["primary_reason"] == "licensed_corpus_prepared_below_required_token_floor"
    assert assessment["supports_pareto_improvement"] is False
    assert "below_50m_token_requirement" in assessment["limitations"]
    assert assessment["evidence"]["repo_count"] >= 1
    assert assessment["evidence"]["total_tokens"] > 0
    assert assessment["evidence"]["meets_50m_token_requirement"] is False


def test_assessment_classifies_any_licensed_corpus_preparation_record():
    assessment = assess_record(
        {
            "experiment_id": "D2_expanded_corpus_preparation",
            "hypothesis": "real_training_data",
            "metrics": {
                "benchmark_label": "licensed_repository_corpus_preparation",
                "repo_count": 2,
                "document_count": 10,
                "total_tokens": 1234,
                "target_min_tokens": 50_000_000,
                "meets_50m_token_requirement": False,
                "license_counts": {"MIT": 2},
                "languages": {"Python": 10},
                "file_roles": {"code": 10},
                "split_counts": {"train": 10},
            },
        }
    )

    assert assessment["outcome"] == "corpus_prepared_insufficient_tokens"
    assert assessment["evidence"]["repo_count"] == 2


def test_assessment_classifies_hf_corpus_materialization():
    assessment = assess_record(
        {
            "experiment_id": "D4_hf_corpus_materialization",
            "hypothesis": "real_training_data",
            "metrics": {
                "benchmark_label": "hf_corpus_materialization",
                "corpus_use": "exploratory-research-only",
                "target_train_tokens": 50_000_000,
                "meets_50m_token_requirement": True,
                "tokens": {"train": 50_000_001},
                "rows_seen": 100,
                "rows_accepted": 80,
                "rows_excluded": 20,
                "output": {"sha256": "abc"},
                "dataset_config_counts": {"dataset::config": 80},
            },
        }
    )

    assert assessment["outcome"] == "hf_corpus_materialized"
    assert assessment["evidence"]["train_tokens"] == 50_000_001
    assert assessment["evidence"]["meets_50m_token_requirement"] is True
    assert "not_training_run" in assessment["limitations"]


def test_assessment_marks_dense_training_smoke_outcomes_without_overclaiming():
    rocm = assess_record(_record("T1_dense_decoder_training_smoke"))
    cpu = assess_record(_record("T1b_cpu_dense_decoder_training_smoke"))
    bf16_failure = assess_record(_record("T1a_rocm_dense_decoder_bf16_training_failure"))

    assert rocm["outcome"] == "training_failed"
    assert rocm["primary_reason"] == "dense_decoder_training_smoke_failed"
    assert rocm["evidence"]["accelerator_backend"] == "rocm"
    assert rocm["evidence"]["failure"]

    assert cpu["outcome"] == "training_smoke_positive"
    assert cpu["primary_reason"] == "dense_decoder_training_pipeline_completed"
    assert cpu["evidence"]["accelerator_backend"] == "cpu"
    assert cpu["evidence"]["checkpoint_resume_ok"] is True

    assert bf16_failure["outcome"] == "training_failed"
    assert bf16_failure["evidence"]["accelerator_backend"] == "rocm"


def test_assessment_marks_required_dense_50m_baseline():
    assessment = assess_record(
        {
            "experiment_id": "T6_rocm_dense_decoder_11m_hf_d4_50m_tokens",
            "hypothesis": "dense_baseline_training",
            "status": "completed",
            "metrics": {
                "benchmark_label": "dense_decoder_training_smoke",
                "status": "completed",
                "accelerator_backend": "rocm",
                "failure": "",
                "model": {"parameter_count": 11_025_505},
                "training": {"train_tokens": 50_000_128, "steps": 195_313},
                "validation": {"loss": 4.04},
                "checkpoint": {"resume_ok": True},
                "corpus": {
                    "record": {
                        "experiment_id": "D4_hf_corpus_materialization",
                        "output_sha256": "abc",
                    }
                },
            },
        }
    )

    assert assessment["outcome"] == "dense_50m_baseline_positive"
    assert assessment["primary_reason"] == "dense_decoder_10m_plus_completed_50m_token_run"
    assert "not_50m_token_run" not in assessment["limitations"]
    assert "training_smoke_only" not in assessment["limitations"]
    assert assessment["evidence"]["train_tokens"] == 50_000_128
    assert (
        assessment["evidence"]["corpus_record"]["experiment_id"]
        == "D4_hf_corpus_materialization"
    )


def test_assessment_marks_structured_external_memory_as_reduced_exploration():
    assessment = assess_record(_record("E6a_structured_repository_memory"))

    assert assessment["outcome"] == "exploration_positive"
    assert (
        assessment["primary_reason"]
        == "structured_external_memory_beats_poisonable_text_retrieval_in_synthetic_convention_task"
    )
    assert assessment["supports_pareto_improvement"] is False
    assert "parametric_baseline_is_proxy_not_trained_lm" in assessment["limitations"]
    assert assessment["evidence"]["structured_poisoned_answer_count"] == 0
    assert assessment["evidence"]["supports_model_level_claim"] is False


def test_assessment_marks_public_repository_memory_qa_as_real_repo_measurement():
    assessment = assess_record(_record("E6b_public_repository_memory_qa"))

    assert assessment["outcome"] == "real_repo_measurement_positive"
    assert (
        assessment["primary_reason"]
        == "structured_symbol_memory_answers_public_repo_symbol_file_qa"
    )
    assert assessment["supports_pareto_improvement"] is False
    assert "not_code_generation" in assessment["limitations"]
    assert assessment["evidence"]["repo_count_used"] >= 1
    assert assessment["evidence"]["question_count"] >= 1
    assert assessment["evidence"]["structured_answer_accuracy"] == 1.0
    assert assessment["evidence"]["supports_model_level_claim"] is False


def test_assessment_marks_public_repository_signature_qa_as_real_repo_measurement():
    assessment = assess_record(_record("E6c_public_repository_signature_qa"))

    assert assessment["outcome"] == "real_repo_measurement_positive"
    assert (
        assessment["primary_reason"]
        == "structured_signature_memory_answers_public_repo_signature_qa"
    )
    assert assessment["supports_pareto_improvement"] is False
    assert "signature_line_only" in assessment["limitations"]
    assert assessment["evidence"]["repo_count_used"] >= 1
    assert assessment["evidence"]["question_count"] >= 1
    assert assessment["evidence"]["structured_answer_accuracy"] == 1.0
    assert assessment["evidence"]["supports_model_level_claim"] is False


def test_assessment_marks_public_repository_call_stub_generation_as_codegen_proxy():
    assessment = assess_record(_record("E6d_public_repository_call_stub_generation"))

    assert assessment["outcome"] == "codegen_proxy_positive"
    assert (
        assessment["primary_reason"]
        == "structured_signature_memory_generates_public_repo_call_stubs"
    )
    assert assessment["supports_pareto_improvement"] is False
    assert "canonical_call_stub_only" in assessment["limitations"]
    assert "not_semantic_code_generation" in assessment["limitations"]
    assert assessment["evidence"]["repo_count_used"] >= 1
    assert assessment["evidence"]["question_count"] >= 1
    assert assessment["evidence"]["structured_answer_accuracy"] == 1.0
    assert assessment["evidence"]["supports_model_level_claim"] is False


def test_assessment_marks_public_repository_function_skeleton_generation_as_codegen_proxy():
    assessment = assess_record(
        _record("E6e_public_repository_function_skeleton_generation")
    )

    assert assessment["outcome"] == "codegen_proxy_positive"
    assert (
        assessment["primary_reason"]
        == "structured_signature_memory_generates_public_repo_function_skeletons"
    )
    assert assessment["supports_pareto_improvement"] is False
    assert "deterministic_skeleton_body_only" in assessment["limitations"]
    assert "not_semantic_code_generation" in assessment["limitations"]
    assert assessment["evidence"]["repo_count_used"] >= 1
    assert assessment["evidence"]["question_count"] >= 1
    assert assessment["evidence"]["structured_answer_accuracy"] == 1.0
    assert assessment["evidence"]["supports_model_level_claim"] is False


def test_assessment_marks_d5_tokenizer_training_as_bounded_mixed_result():
    assessment = assess_record(
        {
            "experiment_id": "D5_tokenizer_training_comparison",
            "hypothesis": "d5_fast_bpe_must_improve_loss_per_byte",
            "metrics": {
                "benchmark_label": "d5_trained_tokenizer_model_comparison",
                "conclusions": {
                    "bpe_equal_compute_improves_loss_per_estimated_byte": True,
                    "bpe_equal_raw_bytes_improves_loss_per_estimated_byte": False,
                    "functional_quality_measured": False,
                },
                "comparisons": {
                    "bpe_train_token_reduction_ratio": 3.0,
                    "equal_compute_bpe_minus_byte_nats_per_estimated_byte": -0.1,
                    "equal_raw_bpe_minus_byte_nats_per_estimated_byte": 0.2,
                },
                "runs": {
                    "byte_equal_compute": {
                        "validation_loss_nats_per_estimated_byte": 1.5,
                    },
                    "bpe_equal_compute": {
                        "validation_loss_nats_per_estimated_byte": 1.4,
                    },
                    "bpe_equal_raw_bytes": {
                        "validation_loss_nats_per_estimated_byte": 1.7,
                    },
                },
            },
        }
    )

    assert assessment["outcome"] == "bpe_equal_compute_positive_equal_raw_negative"
    assert assessment["supports_pareto_improvement"] is False
    assert "token_reduction_alone_not_sufficient" in assessment["limitations"]
    assert assessment["evidence"]["equal_raw_loss_delta_per_estimated_byte"] == 0.2


def test_assessment_marks_idea_foundry_candidates_as_design_lane_not_result():
    assessment = assess_record(
        {
            "experiment_id": "idea_foundry_candidates",
            "hypothesis": "six_candidates",
            "metrics": {
                "benchmark_label": "idea_foundry_candidate_generation",
                "constraint_summary": {
                    "candidate_count": 6,
                    "without_adapters": 5,
                    "without_moe_or_topic_routing": 5,
                    "continual_evolution_candidates": 2,
                    "compression_candidates": 1,
                    "code_structure_candidates": 3,
                    "potentially_novel_candidates": 1,
                },
            },
        }
    )

    assert assessment["outcome"] == "idea_lane_opened"
    assert assessment["supports_pareto_improvement"] is False
    assert "not_training_evidence" in assessment["limitations"]
    assert assessment["evidence"]["candidate_count"] == 6


def test_assessment_marks_if1_probe_as_mechanism_signal_not_quality_win():
    assessment = assess_record(
        {
            "experiment_id": "idea_foundry_repository_graph_signal_probe",
            "hypothesis": "if1_signal",
            "metrics": {
                "benchmark_label": "idea_foundry_repository_graph_signal_probe",
                "candidate_id": "IF1",
                "document_count": 10,
                "import_edge_count": 4,
                "resolved_local_edge_count": 2,
                "role_link_edge_count": 3,
                "graph_edge_count": 5,
                "typed_edge_counts": {
                    "doc_to_source": 1,
                    "import_local": 2,
                    "test_to_source": 2,
                },
                "repositories_with_edges": 2,
                "repository_aware_splits_preserved": True,
                "mechanism_signal_present": True,
            },
        }
    )

    assert assessment["outcome"] == "mechanism_signal_present"
    assert assessment["supports_pareto_improvement"] is False
    assert "not_model_training" in assessment["limitations"]
    assert assessment["evidence"]["resolved_local_edge_count"] == 2
    assert assessment["evidence"]["role_link_edge_count"] == 3
    assert assessment["evidence"]["typed_edge_counts"]["test_to_source"] == 2


def test_assessment_marks_if2_fast_weight_probe_as_synthetic_positive():
    assessment = assess_record(
        {
            "experiment_id": "IF2_fast_weight_continual_probe",
            "hypothesis": "if2_fast_weights",
            "metrics": {
                "benchmark_label": "if2_fast_weight_continual_probe",
                "candidate_id": "IF2",
                "timeline_steps": 4,
                "final": {
                    "exact_retrieval_accuracy": 0.2,
                    "structured_memory_accuracy": 0.6,
                    "fast_weight_scratchpad_accuracy": 0.95,
                    "structured_memory_plus_fast_weight_accuracy": 0.95,
                },
                "heldout_generalization": {
                    "task_count": 8,
                    "structured_memory_correct": 0,
                    "fast_weight_scratchpad_correct": 7,
                    "structured_memory_plus_fast_weight_correct": 7,
                },
                "methods": {
                    "structured_memory": {"storage_bytes": 596},
                    "fast_weight_scratchpad": {"parameter_bytes": 4096},
                    "structured_memory_plus_fast_weight": {"storage_bytes": 4692},
                },
                "parameter_evolution_adds_value_beyond_updated_memory": True,
            },
        }
    )

    assert assessment["outcome"] == "synthetic_fast_weight_positive"
    assert assessment["supports_pareto_improvement"] is False
    assert "synthetic_fixture_only" in assessment["limitations"]
    assert assessment["evidence"]["heldout_fast_weight_correct"] == 7
    assert assessment["evidence"]["parameter_bytes"] == 4096


def test_assessment_marks_if3_block_codebook_as_reconstruction_proxy():
    assessment = assess_record(
        {
            "experiment_id": "IF3_block_codebook_t11c_probe",
            "hypothesis": "if3_block_codebook",
            "metrics": {
                "benchmark_label": "if3_block_codebook_checkpoint_probe",
                "candidate_id": "IF3",
                "checkpoint": {
                    "path": "artifacts/T11c/dense_decoder_last_model_only.pt",
                    "checkpoint_type": "model_only",
                    "step": 195313,
                },
                "compression": {
                    "floating_parameter_count": 10_000,
                    "block_count": 100,
                    "learned_codebook": {
                        "mse": 0.01,
                        "encoded_bytes": 9000,
                        "metadata_bytes": 1000,
                        "runtime_buffer_bytes": 40_000,
                        "encoded_plus_runtime_bytes": 49_000,
                        "beats_random_control": True,
                    },
                    "random_codebook_control": {
                        "mse": 0.05,
                        "encoded_bytes": 9000,
                    },
                },
                "packed_kernel_evaluated": False,
                "loss_evaluated": False,
            },
        }
    )

    assert assessment["outcome"] == "reconstruction_proxy_positive"
    assert assessment["supports_pareto_improvement"] is False
    assert "no_language_model_loss" in assessment["limitations"]
    assert assessment["evidence"]["learned_mse"] == 0.01
    assert assessment["evidence"]["beats_random_control"] is True


def test_assessment_marks_public_repository_docstring_skeleton_generation_as_codegen_proxy():
    assessment = assess_record(
        _record("E6f_public_repository_docstring_skeleton_generation")
    )

    assert assessment["outcome"] == "codegen_proxy_positive"
    assert (
        assessment["primary_reason"]
        == "structured_docstring_memory_generates_public_repo_documented_skeletons"
    )
    assert assessment["supports_pareto_improvement"] is False
    assert "first_docstring_line_only" in assessment["limitations"]
    assert "not_semantic_code_generation" in assessment["limitations"]
    assert assessment["evidence"]["repo_count_used"] >= 1
    assert assessment["evidence"]["question_count"] >= 1
    assert assessment["evidence"]["structured_answer_accuracy"] == 1.0
    assert assessment["evidence"]["supports_model_level_claim"] is False


def test_assessment_marks_public_repository_api_reference_generation_as_docgen_proxy():
    assessment = assess_record(
        _record("E6g_public_repository_api_reference_generation")
    )

    assert assessment["outcome"] == "docgen_proxy_positive"
    assert (
        assessment["primary_reason"]
        == "structured_docstring_memory_generates_public_repo_api_reference_entries"
    )
    assert assessment["supports_pareto_improvement"] is False
    assert "deterministic_api_reference_template" in assessment["limitations"]
    assert "not_semantic_documentation_generation" in assessment["limitations"]
    assert assessment["evidence"]["repo_count_used"] >= 1
    assert assessment["evidence"]["question_count"] >= 1
    assert assessment["evidence"]["structured_answer_accuracy"] == 1.0
    assert assessment["evidence"]["supports_model_level_claim"] is False


def test_assessment_marks_public_repository_api_doc_coverage_as_real_repo_measurement():
    assessment = assess_record(
        _record("E6h_public_repository_api_doc_coverage_qa")
    )

    assert assessment["outcome"] == "real_repo_doc_measurement_positive"
    assert (
        assessment["primary_reason"]
        == "structured_doc_directive_memory_answers_public_repo_api_doc_coverage_qa"
    )
    assert assessment["supports_pareto_improvement"] is False
    assert "sphinx_directive_coverage_only" in assessment["limitations"]
    assert "not_documentation_quality" in assessment["limitations"]
    assert assessment["evidence"]["repo_count_used"] >= 1
    assert assessment["evidence"]["question_count"] >= 1
    assert assessment["evidence"]["structured_answer_accuracy"] == 1.0
    assert assessment["evidence"]["supports_model_level_claim"] is False


def test_assessment_marks_synthetic_api_doc_drift_detection_as_positive_control():
    assessment = assess_record(_record("E6i_synthetic_api_doc_drift_detection"))

    assert assessment["outcome"] == "doc_drift_positive_control"
    assert (
        assessment["primary_reason"]
        == "structured_source_doc_consistency_memory_detects_synthetic_api_doc_drift"
    )
    assert assessment["supports_pareto_improvement"] is False
    assert "synthetic_positive_control" in assessment["limitations"]
    assert "not_semantic_prose_drift_detection" in assessment["limitations"]
    assert assessment["evidence"]["repo_count_used"] == 1
    assert assessment["evidence"]["stale_doc_issue_count"] == 1
    assert assessment["evidence"]["structured_answer_accuracy"] == 1.0
    assert assessment["evidence"]["supports_model_level_claim"] is False


def test_assessment_classifies_rocm_transformer_stability_records():
    passing = {
        "experiment_id": "R2_rocm_transformer_stability",
        "hypothesis": "rocm_transformer_failure_is_component_or_kernel_path_specific",
        "metrics": {
            "failure_count": 0,
            "case_count": 2,
            "first_failure": None,
            "case_results": [
                {"component": "embedding", "status": "ok"},
                {"component": "transformer", "status": "ok"},
            ],
        },
    }
    failing = {
        "experiment_id": "R2_rocm_transformer_stability",
        "hypothesis": "rocm_transformer_failure_is_component_or_kernel_path_specific",
        "metrics": {
            "failure_count": 1,
            "case_count": 2,
            "first_failure": {
                "component": "transformer",
                "dtype": "bf16",
                "mask_mode": "bool_causal",
                "status": "failed",
                "error": "RuntimeError: gpu hang",
            },
            "case_results": [
                {"component": "embedding", "status": "ok"},
                {"component": "transformer", "status": "failed"},
            ],
        },
    }

    assert assess_record(passing)["outcome"] == "transformer_stability_positive"
    failed_assessment = assess_record(failing)
    assert failed_assessment["outcome"] == "transformer_stability_failed"
    assert failed_assessment["evidence"]["first_failure_component"] == "transformer"


def test_assessment_classifies_dense_step_debug_records():
    passing = {
        "experiment_id": "T3e_dense_step_stability_544x3_sgd_lr0",
        "hypothesis": "dense_step_two_nonfinite_origin",
        "metrics": {
            "benchmark_label": "dense_step_stability_debug",
            "first_nonfinite_phase": None,
            "model": {"parameter_count": 11_025_505, "config": {"layers": 3, "hidden_dim": 544}},
            "step_results": [
                {"gradients": {"nonfinite_count": 0, "nonfinite_tensors": []}},
                {"loss": {"finite": True}},
            ],
        },
    }
    failing = {
        "experiment_id": "T3_dense_step_stability_10m_sgd_lr0",
        "hypothesis": "dense_step_two_nonfinite_origin",
        "metrics": {
            "benchmark_label": "dense_step_stability_debug",
            "first_nonfinite_phase": "step_1_gradients",
            "model": {"parameter_count": 12_939_521, "config": {"layers": 4, "hidden_dim": 512}},
            "step_results": [
                {
                    "gradients": {
                        "nonfinite_count": 1592,
                        "nonfinite_tensors": [
                            {"name": "blocks.layers.1.norm2.weight"}
                        ],
                    }
                }
            ],
        },
    }

    assert assess_record(passing)["outcome"] == "dense_step_debug_positive"
    failed_assessment = assess_record(failing)
    assert failed_assessment["outcome"] == "dense_step_debug_failed"
    assert failed_assessment["evidence"]["first_nonfinite_phase"] == "step_1_gradients"
    assert failed_assessment["evidence"]["first_nonfinite_tensor"] == (
        "blocks.layers.1.norm2.weight"
    )


def test_assessment_manifest_includes_new_research_options_for_current_limits():
    summary = assess_manifest(Path("results"))

    assert summary["record_count"] >= 49
    assert summary["outcome_counts"]["runtime_positive"] >= 1
    assert summary["outcome_counts"]["measurement_positive"] >= 1
    assert summary["outcome_counts"]["training_readiness_positive"] >= 1
    assert summary["outcome_counts"]["corpus_prepared_insufficient_tokens"] >= 1
    assert summary["outcome_counts"]["training_smoke_positive"] >= 1
    assert summary["outcome_counts"]["training_failed"] >= 2
    assert summary["outcome_counts"]["exploration_positive"] >= 1
    assert summary["outcome_counts"]["real_repo_measurement_positive"] >= 2
    assert summary["outcome_counts"]["real_repo_doc_measurement_positive"] >= 1
    assert summary["outcome_counts"]["doc_drift_positive_control"] >= 1
    assert summary["outcome_counts"]["codegen_proxy_positive"] >= 3
    assert summary["outcome_counts"]["docgen_proxy_positive"] >= 1
    assert summary["outcome_counts"]["rejected"] >= 1
    assert summary["outcome_counts"]["no_signal"] >= 1
    assert summary["outcome_counts"]["smoke_pass"] >= 1
    assert summary["outcome_counts"]["t11_adapter_frontier_expansion"] == 1
    assert summary["outcome_counts"]["t11_dense528_width_control_frontier_expansion"] == 1
    assert summary["outcome_counts"]["t12_dense528_validation_selected"] == 1
    assert summary["pareto_improvement_found"] is True
    assert summary["pareto_dominance_found"] is False
    assert summary["frontier_expansion_found"] is True
    t11 = next(
        row
        for row in summary["assessments"]
        if row["experiment_id"] == "T11_dense_adapter_50m_paired_assessment"
    )
    assert t11["supports_pareto_improvement"] is True
    assert t11["evidence"]["dominates_dense_baseline"] is False
    assert t11["evidence"]["expands_measured_frontier"] is True
    assert (
        t11["evidence"]["adapter_final_validation_loss"]
        < t11["evidence"]["dense_final_validation_loss"]
    )
    assert (
        t11["evidence"]["adapter_final_test_loss"]
        < t11["evidence"]["dense_final_test_loss"]
    )
    assert (
        t11["evidence"]["adapter_train_tokens_per_second"]
        < t11["evidence"]["dense_train_tokens_per_second"]
    )
    t11c = next(
        row
        for row in summary["assessments"]
        if row["experiment_id"] == "T11_dense528_width_control_assessment"
    )
    assert t11c["supports_pareto_improvement"] is True
    assert t11c["evidence"]["dense528_best_final_validation"] is True
    assert t11c["evidence"]["dense528_resource_wins"] is True
    assert t11c["evidence"]["dense528_throughput_wins"] is True
    assert t11c["evidence"]["adapter528_final_test_win"] is True
    t12 = next(
        row
        for row in summary["assessments"]
        if row["experiment_id"] == "T12_three_seed_dense_adapter_assessment"
    )
    assert t12["supports_pareto_improvement"] is True
    assert t12["evidence"]["selected_family_by_final_validation_mean"] == "dense"
    assert t12["evidence"]["selected_family_by_best_validation_mean"] == "dense"
    assert t12["evidence"]["test_loss_used_for_selection"] is False
    assert t12["evidence"]["adapter_final_test_winner_by_mean"] is True
    assert t12["evidence"]["dense_throughput_winner_by_mean"] is True
    assert "activation_residual_cache" in summary["next_research_options"]
    assert "upstream_test_suite_patch_mining" in summary["next_research_options"]
    assert "structured_external_repository_memory" in summary["next_research_options"]
