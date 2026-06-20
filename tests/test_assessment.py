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
    assert "activation_residual_cache" in summary["next_research_options"]
    assert "upstream_test_suite_patch_mining" in summary["next_research_options"]
    assert "structured_external_repository_memory" in summary["next_research_options"]
