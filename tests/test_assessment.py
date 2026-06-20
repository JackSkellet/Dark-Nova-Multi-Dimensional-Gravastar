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
