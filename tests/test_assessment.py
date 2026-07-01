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


def test_assessment_marks_if4_fast_repo_adaptation_as_proxy_probe():
    assessment = assess_record(
        {
            "experiment_id": "IF4_fast_repo_adaptation_markupsafe",
            "hypothesis": "if4_fast_temporary_weights",
            "metrics": {
                "benchmark_label": "if4_fast_repo_adaptation_probe",
                "candidate_id": "IF4",
                "repo": {
                    "path_name": "markupsafe",
                    "commit_count_used": 8,
                    "top_k": 5,
                },
                "steps": [{"future_commit_not_in_memory": True}],
                "final": {
                    "updated_retrieval_future_topk_accuracy": 0.25,
                    "structured_symbol_graph_memory_future_topk_accuracy": 0.5,
                    "replay_adapter_proxy_future_topk_accuracy": 0.5,
                    "fast_temporary_weights_future_topk_accuracy": 0.25,
                    "fast_weights_plus_retrieval_future_topk_accuracy": 0.75,
                    "periodic_consolidation_future_topk_accuracy": 0.5,
                    "fast_weights_plus_retrieval_paraphrase_topk_accuracy": 0.5,
                    "prior_task_retention_accuracy": 1.0,
                    "mean_update_ms": 0.2,
                    "rollback_supported": True,
                    "total_storage_bytes": 2048,
                },
            },
        }
    )

    assert assessment["outcome"] == "repository_adaptation_probe"
    assert assessment["supports_pareto_improvement"] is False
    assert "not_language_model_training" in assessment["limitations"]
    assert assessment["evidence"]["candidate_id"] == "IF4"
    assert assessment["evidence"]["rollback_supported"] is True


def test_assessment_marks_if7_sparse_hebbian_as_real_corpus_signal():
    assessment = assess_record(
        {
            "experiment_id": "IF7_sparse_hebbian_d5_probe",
            "hypothesis": "if7_sparse_hebbian",
            "metrics": {
                "benchmark_label": "if7_sparse_hebbian_assembly_probe",
                "candidate_id": "IF7",
                "corpus": {
                    "rows_loaded": 8192,
                    "rows_scanned": 26173,
                    "usable_patterns": 8190,
                    "eval_patterns": 512,
                    "repositories_loaded": 7381,
                },
                "sparsity": {"mean_active_fraction": 0.011},
                "methods": {
                    "hebbian_sparse_assembly": {"storage_bytes": 69_399_884},
                    "random_sparse_control": {"storage_bytes": 67_141_632},
                },
                "final": {
                    "hebbian_hit_at_k": 0.95,
                    "frequency_hit_at_k": 0.88,
                    "random_hit_at_k": 0.20,
                    "hebbian_mrr": 0.48,
                    "frequency_mrr": 0.28,
                    "random_mrr": 0.03,
                },
                "hebbian_adds_associative_signal": True,
            },
        }
    )

    assert assessment["outcome"] == "real_corpus_associative_signal"
    assert assessment["supports_pareto_improvement"] is False
    assert "not_language_model_training" in assessment["limitations"]
    assert assessment["evidence"]["candidate_id"] == "IF7"
    assert assessment["evidence"]["hebbian_hit_at_k"] > assessment["evidence"][
        "frequency_hit_at_k"
    ]


def test_assessment_marks_glm_public_harness_as_external_baseline_scaffold():
    assessment = assess_record(
        {
            "experiment_id": "GLM5_2_public_eval_harness",
            "hypothesis": "glm_5_2_external_public_task_eval_harness",
            "metrics": {
                "benchmark_label": "glm_5_2_public_eval_harness",
                "baseline_category": "external_glm_5_2_baseline",
                "glm_run_status": "not_run_no_predictions",
                "privacy_gate_passed": True,
                "task_count": 3,
                "evaluated_prediction_count": 0,
                "pass_count": 0,
                "metadata_completeness": {
                    "complete_records": 0,
                    "records_with_missing_fields": 0,
                },
            },
        }
    )

    assert assessment["outcome"] == "external_baseline_harness_ready"
    assert assessment["supports_pareto_improvement"] is False
    assert assessment["primary_reason"] == "glm_public_eval_harness_has_no_model_outputs_yet"
    assert "not_same_budget_local_baseline" in assessment["limitations"]
    assert assessment["evidence"]["task_count"] == 3
    assert assessment["evidence"]["privacy_gate_passed"] is True


def test_assessment_marks_local_glm_public_smoke_as_scored_local_baseline():
    assessment = assess_record(
        {
            "experiment_id": "T11c_dense528_glm_public_smoke",
            "hypothesis": "glm_5_2_external_public_task_eval_harness",
            "metrics": {
                "benchmark_label": "glm_5_2_public_eval_harness",
                "baseline_category": "local_same_budget_baseline",
                "glm_run_status": "evaluated_saved_predictions",
                "privacy_gate_passed": True,
                "task_count": 3,
                "evaluated_prediction_count": 3,
                "pass_count": 0,
                "pass_rate": 0.0,
                "metadata_completeness": {
                    "complete_records": 3,
                    "records_with_missing_fields": 0,
                },
            },
        }
    )

    assert assessment["outcome"] == "local_baseline_predictions_scored"
    assert assessment["primary_reason"] == "local_baseline_scored_on_glm_public_tasks"
    assert assessment["supports_pareto_improvement"] is False
    assert "not_glm_5_2_result" in assessment["limitations"]
    assert assessment["evidence"]["pass_rate"] == 0.0


def test_assessment_marks_if7_trained_conditioning_when_cue_only_wins():
    assessment = assess_record(
        {
            "experiment_id": "IF7b_hebbian_trained_model_d5",
            "hypothesis": "if7_trained_conditioning",
            "metrics": {
                "benchmark_label": "if7_hebbian_conditioned_trained_model",
                "candidate_id": "IF7",
                "training": {
                    "supervised_train_rows": 8190,
                    "validation_rows": 804,
                    "accelerator_backend": "rocm",
                },
                "hebbian_memory": {
                    "node_count": 2048,
                    "storage_bytes": 19_231_715,
                },
                "validation": {
                    "cue_only": {
                        "hit_at_k": 0.9788,
                        "mrr": 0.6916,
                        "loss": 0.0673,
                    },
                    "cue_plus_hebbian": {
                        "hit_at_k": 0.9701,
                        "mrr": 0.6678,
                        "loss": 0.0708,
                    },
                    "raw_hebbian_memory": {
                        "hit_at_k": 0.9490,
                        "mrr": 0.5253,
                    },
                },
                "hebbian_conditioning_improves_trained_model": False,
            },
        }
    )

    assert assessment["outcome"] == "trained_cue_only_not_beaten"
    assert assessment["supports_pareto_improvement"] is False
    assert assessment["primary_reason"] == (
        "cue_plus_hebbian_trained_model_loses_to_cue_only_on_validation"
    )
    assert assessment["evidence"]["accelerator_backend"] == "rocm"
    assert assessment["evidence"]["cue_only_hit_at_k"] > assessment["evidence"][
        "cue_plus_hebbian_hit_at_k"
    ]


def test_assessment_marks_if7_sparse_reranker_when_raw_hebbian_wins():
    assessment = assess_record(
        {
            "experiment_id": "IF7f_sparse_hebbian_reranker_priors_d5_500k_windows",
            "hypothesis": "if7_sparse_reranker",
            "metrics": {
                "benchmark_label": "if7_sparse_hebbian_candidate_reranker",
                "candidate_id": "IF7",
                "training": {
                    "train_patterns": 494_403,
                    "validation_patterns": 24_216,
                    "train_candidate_examples": 500_000,
                    "candidate_count": 64,
                },
                "models": {
                    "candidate_reranker": {
                        "parameter_count": 9,
                    }
                },
                "validation": {
                    "raw_hebbian_memory": {
                        "hit_at_k": 0.8167,
                        "mrr": 0.1939,
                    },
                    "candidate_reranker": {
                        "hit_at_k": 0.7951,
                        "mrr": 0.1867,
                    },
                    "candidate_recall_ceiling": {
                        "hit_at_k": 0.9252,
                        "mrr": 0.9252,
                    },
                },
                "sparse_reranker_improves_raw_hebbian": False,
            },
        }
    )

    assert assessment["outcome"] == "raw_hebbian_not_beaten_by_sparse_reranker"
    assert assessment["primary_reason"] == (
        "sparse_candidate_reranker_loses_to_raw_hebbian_ranking"
    )
    assert assessment["evidence"]["candidate_ceiling_hit_at_k"] > assessment[
        "evidence"
    ]["raw_hebbian_hit_at_k"]
    assert assessment["evidence"]["raw_hebbian_hit_at_k"] > assessment["evidence"][
        "reranker_hit_at_k"
    ]


def test_assessment_marks_if7_repository_linking_when_lexical_baseline_wins():
    assessment = assess_record(
        {
            "experiment_id": "IF7g_repository_linking_d5_validation",
            "hypothesis": "if7_repository_linking",
            "metrics": {
                "benchmark_label": "if7_hebbian_repository_linking",
                "candidate_id": "IF7",
                "corpus": {
                    "train_rows_loaded": 26_000,
                    "eval_rows_loaded": 804,
                    "eval_repositories": 58,
                    "eligible_eval_repositories": 58,
                },
                "tasks": {
                    "task_count": 180,
                    "top_k": 5,
                    "negatives_per_query": 32,
                    "candidate_count_mean": 36.6,
                },
                "methods": {
                    "lexical_text_overlap": {
                        "hit_at_k": 1.0,
                        "mrr": 0.9712,
                        "coverage_at_k": 0.8652,
                    },
                    "raw_hebbian_context": {
                        "hit_at_k": 0.4388,
                        "mrr": 0.3841,
                        "coverage_at_k": 0.2191,
                    },
                    "combined_lexical_hebbian": {
                        "hit_at_k": 0.95,
                        "mrr": 0.8238,
                        "coverage_at_k": 0.7765,
                    },
                },
                "best_method": {
                    "name": "lexical_text_overlap",
                    "hit_at_k": 1.0,
                    "mrr": 0.9712,
                    "coverage_at_k": 0.8652,
                },
                "hebbian_beats_lexical": False,
                "combined_beats_lexical": False,
            },
        }
    )

    assert assessment["outcome"] == "lexical_baseline_not_beaten"
    assert assessment["primary_reason"] == (
        "repository_linking_hebbian_scores_lose_to_lexical_text_overlap"
    )
    assert assessment["supports_pareto_improvement"] is False
    assert assessment["evidence"]["task_count"] == 180
    assert assessment["evidence"]["lexical_hit_at_k"] > assessment["evidence"][
        "raw_hebbian_hit_at_k"
    ]
    assert assessment["evidence"]["lexical_mrr"] > assessment["evidence"][
        "combined_mrr"
    ]


def test_assessment_marks_if7_trained_repository_ranker_with_ablation():
    assessment = assess_record(
        {
            "experiment_id": "IF7i_trained_repository_ranker_d5_validation",
            "hypothesis": "if7_task_aware_ranker",
            "metrics": {
                "benchmark_label": "if7_trained_repository_linking_ranker",
                "candidate_id": "IF7",
                "corpus": {
                    "train_rows_loaded": 26_000,
                    "eval_rows_loaded": 804,
                    "train_task_repositories": 512,
                    "eval_task_repositories": 58,
                },
                "tasks": {
                    "train_tasks": 1_728,
                    "eval_tasks": 180,
                    "top_k": 5,
                    "negatives_per_query": 64,
                    "eval_candidate_count_mean": 68.6,
                },
                "training": {
                    "candidate_examples": 126_560,
                    "epochs": 12,
                    "accelerator_backend": "rocm",
                },
                "models": {
                    "task_aware_ranker": {
                        "feature_names": [
                            "path_token_overlap_norm",
                            "lexical_text_overlap_norm",
                            "hebbian_candidate_context_score",
                            "hebbian_pair_edge_score",
                            "same_file_extension",
                            "bias",
                        ],
                        "parameter_count": 7,
                    },
                    "no_hebbian_ranker": {
                        "parameter_count": 7,
                    },
                },
                "methods": {
                    "lexical_text_overlap": {
                        "hit_at_k": 0.8333,
                        "mrr": 0.7375,
                        "coverage_at_k": 0.6097,
                    },
                    "raw_hebbian_context": {
                        "hit_at_k": 0.3944,
                        "mrr": 0.3336,
                        "coverage_at_k": 0.1780,
                    },
                    "trained_task_aware_ranker": {
                        "hit_at_k": 0.9555,
                        "mrr": 0.8963,
                        "coverage_at_k": 0.8078,
                    },
                    "trained_no_hebbian_ranker": {
                        "hit_at_k": 0.9555,
                        "mrr": 0.8982,
                        "coverage_at_k": 0.8069,
                    },
                },
                "best_method": {
                    "name": "trained_no_hebbian_ranker",
                    "hit_at_k": 0.9555,
                    "mrr": 0.8982,
                    "coverage_at_k": 0.8069,
                },
                "trained_ranker_beats_lexical": True,
                "trained_ranker_beats_raw_hebbian": True,
                "trained_ranker_beats_no_hebbian": False,
            },
        }
    )

    assert assessment["outcome"] == "task_ranker_positive_hebbian_ablation_not_beaten"
    assert assessment["primary_reason"] == (
        "trained_repository_ranker_beats_static_baselines_but_not_no_hebbian_ablation"
    )
    assert assessment["supports_pareto_improvement"] is False
    assert assessment["evidence"]["trained_hit_at_k"] > assessment["evidence"][
        "lexical_hit_at_k"
    ]
    assert assessment["evidence"]["no_hebbian_mrr"] > assessment["evidence"][
        "trained_mrr"
    ]
    assert assessment["evidence"]["has_hebbian_pair_edge_score"] is True
    assert assessment["evidence"]["selected_ranker_uses_hebbian"] is False


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
                    "exact_raw_byte_loss_measured": True,
                    "bpe_equal_compute_improves_exact_nats_per_raw_byte": True,
                    "bpe_equal_raw_bytes_improves_exact_nats_per_raw_byte": False,
                    "functional_quality_measured": False,
                },
                "comparisons": {
                    "bpe_train_token_reduction_ratio": 3.0,
                    "equal_compute_bpe_minus_byte_nats_per_estimated_byte": -0.1,
                    "equal_raw_bpe_minus_byte_nats_per_estimated_byte": 0.2,
                    "equal_compute_bpe_minus_byte_exact_nats_per_raw_byte": -0.15,
                    "equal_raw_bpe_minus_byte_exact_nats_per_raw_byte": 0.25,
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
    assert assessment["evidence"]["exact_raw_byte_loss_measured"] is True
    assert assessment["evidence"]["equal_raw_loss_delta_exact_nats_per_raw_byte"] == 0.25


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


def test_assessment_marks_repository_task_sample_as_benchmark_scaffold():
    assessment = assess_record(
        {
            "experiment_id": "D5_repository_balanced_task_sample",
            "hypothesis": "repository_first_sampling",
            "metrics": {
                "benchmark_label": "repository_balanced_task_sample",
                "repository_count": 32,
                "file_count": 32,
                "task_count": 320,
                "sampling_policy": {
                    "order": "repository_first_file_second_task_third",
                    "task_kinds": ["completion", "syntax"],
                },
            },
        }
    )

    assert assessment["outcome"] == "benchmark_scaffold"
    assert assessment["supports_pareto_improvement"] is False
    assert "no_model_quality_measured" in assessment["limitations"]
    assert assessment["evidence"]["task_count"] == 320


def test_assessment_marks_repository_api_reuse_probe_without_overclaiming():
    assessment = assess_record(
        {
            "experiment_id": "D5_repository_api_reuse_validation",
            "hypothesis": "repository_api_reuse_symbol_selection_probe",
            "metrics": {
                "benchmark_label": "repository_api_reuse_probe",
                "source_split": "train",
                "query_split": "validation",
                "task_count": 42,
                "repository_count": 12,
                "symbol_count": 88,
                "top_k": 5,
                "methods": {
                    "symbol_name_mention": {"hit_at_k": 0.81, "mrr": 0.62},
                    "lexical_source_overlap": {"hit_at_k": 0.74, "mrr": 0.55},
                },
                "best_method": {
                    "name": "symbol_name_mention",
                    "hit_at_k": 0.81,
                    "mrr": 0.62,
                },
                "limitations": ["not_code_generation", "not_executable_runtime_scoring"],
            },
        }
    )

    assert assessment["outcome"] == "api_reuse_benchmark_probe"
    assert assessment["supports_pareto_improvement"] is False
    assert assessment["primary_reason"] == (
        "repository_api_reuse_symbol_selection_scored_without_model_generation"
    )
    assert "not_code_generation" in assessment["limitations"]
    assert assessment["evidence"]["task_count"] == 42
    assert assessment["evidence"]["best_method"] == "symbol_name_mention"


def test_assessment_marks_repository_context_pairwise_probe_without_overclaiming():
    assessment = assess_record(
        {
            "experiment_id": "D5_repository_context_pairwise_validation",
            "hypothesis": "retrieval_context_vs_structured_memory_pairwise_probe",
            "metrics": {
                "benchmark_label": "repository_context_pairwise_probe",
                "source_split": "validation",
                "query_split": "validation",
                "task_count": 23,
                "repository_count": 3,
                "symbol_count": 689,
                "top_k": 5,
                "pairwise_ideas": [
                    "retrieval_augmented_repository_context",
                    "structured_repository_memory",
                ],
                "methods": {
                    "structured_symbol_memory": {
                        "hit_at_k": 0.8695652173913043,
                        "mrr": 0.753623188405797,
                        "hallucinated_api_rate": 0.0,
                    },
                    "retrieved_snippet_identifiers": {
                        "hit_at_k": 0.0,
                        "mrr": 0.0,
                        "hallucinated_api_rate": 0.9826086956521739,
                    },
                    "symbol_aware_retrieved_snippets": {
                        "hit_at_k": 0.21739130434782608,
                        "mrr": 0.18478260869565216,
                        "hallucinated_api_rate": 0.0,
                    },
                    "query_symbol_aware_retrieval": {
                        "hit_at_k": 0.782608695652174,
                        "mrr": 0.6884057971014493,
                        "hallucinated_api_rate": 0.0,
                    },
                },
                "best_method": {
                    "name": "structured_symbol_memory",
                    "hit_at_k": 0.8695652173913043,
                    "mrr": 0.753623188405797,
                    "hallucinated_api_rate": 0.0,
                },
                "limitations": [
                    "not_code_generation",
                    "not_executable_runtime_scoring",
                    "proxy_hallucinated_api_rate",
                ],
            },
        }
    )

    assert assessment["outcome"] == "pairwise_context_proxy_probe"
    assert assessment["supports_pareto_improvement"] is False
    assert assessment["primary_reason"] == (
        "structured_symbol_memory_beats_retrieval_variants_on_api_reuse_proxy"
    )
    assert "not_code_generation" in assessment["limitations"]
    assert assessment["evidence"]["best_method"] == "structured_symbol_memory"
    assert assessment["evidence"]["retrieved_hallucinated_api_rate"] > 0.9
    assert assessment["evidence"]["symbol_aware_hallucinated_api_rate"] == 0.0
    assert assessment["evidence"]["symbol_aware_hit_at_k"] < assessment["evidence"][
        "structured_hit_at_k"
    ]
    assert assessment["evidence"]["query_symbol_aware_hit_at_k"] > assessment[
        "evidence"
    ]["symbol_aware_hit_at_k"]
    assert assessment["evidence"]["query_symbol_aware_hit_at_k"] < assessment[
        "evidence"
    ]["structured_hit_at_k"]


def test_assessment_marks_quixbugs_repair_probe_as_executable_repair_gate():
    assessment = assess_record(
        {
            "experiment_id": "QuixBugs_python_repair_smoke",
            "hypothesis": "quixbugs_python_repair_floor_and_oracle_ceiling",
            "metrics": {
                "benchmark_label": "quixbugs_python_repair_probe",
                "program_count": 3,
                "source": {
                    "repo_url": "https://github.com/jkoppel/QuixBugs",
                    "repo_commit": "abc123",
                },
                "final": {
                    "buggy_pass_rate": 0.0,
                    "oracle_correct_pass_rate": 1.0,
                    "repair_gap": 1.0,
                    "buggy_passed": 0,
                    "oracle_correct_passed": 3,
                },
                "limitations": [
                    "not_model_generated_repairs",
                    "oracle_correct_is_upper_bound",
                ],
            },
        }
    )

    assert assessment["outcome"] == "executable_repair_benchmark_probe"
    assert assessment["supports_pareto_improvement"] is False
    assert assessment["primary_reason"] == (
        "quixbugs_buggy_floor_and_oracle_ceiling_measured_with_pytest"
    )
    assert assessment["evidence"]["program_count"] == 3
    assert assessment["evidence"]["repair_gap"] == 1.0
    assert "not_model_generated_repairs" in assessment["limitations"]


def test_assessment_marks_quixbugs_candidate_repair_probe_without_overclaiming():
    assessment = assess_record(
        {
            "experiment_id": "QuixBugs_python_candidate_repair_smoke",
            "hypothesis": "quixbugs_python_candidate_repair_source_replacement_probe",
            "metrics": {
                "benchmark_label": "quixbugs_python_candidate_repair_probe",
                "source": {
                    "repo_url": "https://github.com/jkoppel/QuixBugs",
                    "repo_commit": "4257f44b0ff1181dedaedee6a447e133219fcebf",
                },
                "candidate_count": 8,
                "program_count": 4,
                "final": {
                    "candidate_passed": 4,
                    "candidate_pass_rate": 0.5,
                    "programs_with_passing_candidate": 4,
                    "program_repair_rate": 1.0,
                },
                "limitations": [
                    "not_model_generated_unless_candidate_file_is_model_output",
                    "replacement_source_only",
                    "python_subset_only",
                ],
            },
        }
    )

    assert assessment["outcome"] == "executable_candidate_repair_probe"
    assert assessment["supports_pareto_improvement"] is False
    assert assessment["primary_reason"] == (
        "quixbugs_candidate_replacement_sources_measured_with_pytest"
    )
    assert "not_model_generated_unless_candidate_file_is_model_output" in assessment[
        "limitations"
    ]
    assert assessment["evidence"]["candidate_count"] == 8
    assert assessment["evidence"]["program_repair_rate"] == 1.0


def test_assessment_marks_quixbugs_dense_model_candidate_probe_as_model_repair_eval():
    assessment = assess_record(
        {
            "experiment_id": "QuixBugs_dense528_candidate_repair_smoke",
            "hypothesis": "dense_checkpoint_generates_quixbugs_repair_candidates",
            "metrics": {
                "benchmark_label": "quixbugs_python_dense_model_candidate_probe",
                "source": {
                    "repo_url": "https://github.com/jkoppel/QuixBugs",
                    "repo_commit": "4257f44b0ff1181dedaedee6a447e133219fcebf",
                },
                "model": {
                    "checkpoint": (
                        "artifacts/T11c_dense528_adamw_fp32_50m/"
                        "dense_decoder_best_model_only.pt"
                    ),
                    "checkpoint_step": 195000,
                    "config": {"hidden_dim": 528, "layers": 3},
                },
                "candidate_count": 4,
                "program_count": 4,
                "final": {
                    "candidate_passed": 0,
                    "candidate_pass_rate": 0.0,
                    "programs_with_passing_candidate": 0,
                    "program_repair_rate": 0.0,
                },
                "syntax": {
                    "generated_candidate_count": 16,
                    "syntax_valid_candidate_count": 2,
                    "programs_with_syntax_valid_candidate": 1,
                },
                "limitations": [
                    "local_model_generated_candidate",
                    "greedy_byte_generation",
                ],
            },
        }
    )

    assert assessment["outcome"] == "model_generated_repair_candidate_probe"
    assert assessment["supports_pareto_improvement"] is False
    assert assessment["primary_reason"] == (
        "dense_checkpoint_generated_quixbugs_candidates_measured_with_pytest"
    )
    assert "local_model_generated_candidate" in assessment["limitations"]
    assert assessment["evidence"]["candidate_count"] == 4
    assert assessment["evidence"]["program_repair_rate"] == 0.0
    assert assessment["evidence"]["checkpoint_step"] == 195000
    assert assessment["evidence"]["generated_candidate_count"] == 16
    assert assessment["evidence"]["syntax_valid_candidate_count"] == 2


def test_assessment_marks_quixbugs_edit_baseline_as_hand_engineered_calibration():
    assessment = assess_record(
        {
            "experiment_id": "QuixBugs_edit_baseline_repair_smoke",
            "hypothesis": "deterministic_ast_edits_can_calibrate_quixbugs_repair_lane",
            "metrics": {
                "benchmark_label": "quixbugs_python_edit_baseline_probe",
                "source": {
                    "repo_url": "https://github.com/jkoppel/QuixBugs",
                    "repo_commit": "4257f44b0ff1181dedaedee6a447e133219fcebf",
                },
                "generator": {
                    "label": "deterministic_ast_edit_baseline",
                    "max_candidates_per_program": 8,
                    "edit_templates": [
                        "swap_recursive_call_arguments",
                        "yield_recursive_call_argument",
                    ],
                },
                "candidate_count": 6,
                "program_count": 4,
                "final": {
                    "candidate_passed": 4,
                    "candidate_pass_rate": 0.6666666666666666,
                    "programs_with_passing_candidate": 4,
                    "program_repair_rate": 1.0,
                },
                "limitations": [
                    "not_model_generated",
                    "hand_engineered_deterministic_edits",
                    "does_not_read_oracle_correct_sources",
                ],
            },
        }
    )

    assert assessment["outcome"] == "deterministic_repair_baseline_probe"
    assert assessment["supports_pareto_improvement"] is False
    assert assessment["primary_reason"] == (
        "hand_engineered_ast_edits_calibrate_quixbugs_candidate_lane"
    )
    assert "not_model_generated" in assessment["limitations"]
    assert assessment["evidence"]["candidate_count"] == 6
    assert assessment["evidence"]["program_repair_rate"] == 1.0
    assert assessment["evidence"]["generator_label"] == "deterministic_ast_edit_baseline"


def test_assessment_marks_quixbugs_dense_ranked_edits_as_model_ranking_probe():
    assessment = assess_record(
        {
            "experiment_id": "QuixBugs_T11c_dense528_ranked_edit_smoke",
            "hypothesis": "dense_checkpoint_can_rank_deterministic_quixbugs_edit_candidates",
            "metrics": {
                "benchmark_label": "quixbugs_python_dense_ranked_edit_probe",
                "source": {
                    "repo_url": "https://github.com/jkoppel/QuixBugs",
                    "repo_commit": "4257f44b0ff1181dedaedee6a447e133219fcebf",
                },
                "model": {
                    "checkpoint": (
                        "artifacts/T11c_dense528_adamw_fp32_50m/"
                        "dense_decoder_best_model_only.pt"
                    ),
                    "checkpoint_step": 195000,
                    "config": {"hidden_dim": 528, "layers": 3},
                },
                "candidate_pool": {
                    "generated_candidate_count": 6,
                    "selected_candidate_count": 4,
                    "top_candidates_per_program": 1,
                },
                "candidate_count": 4,
                "program_count": 4,
                "final": {
                    "candidate_passed": 4,
                    "candidate_pass_rate": 1.0,
                    "programs_with_passing_candidate": 4,
                    "program_repair_rate": 1.0,
                },
                "limitations": [
                    "local_model_ranked_candidate",
                    "deterministic_candidate_pool",
                    "not_free_form_generation",
                ],
            },
        }
    )

    assert assessment["outcome"] == "model_ranked_repair_candidate_probe"
    assert assessment["supports_pareto_improvement"] is False
    assert assessment["primary_reason"] == (
        "dense_checkpoint_ranked_deterministic_ast_edits_with_pytest"
    )
    assert "not_free_form_generation" in assessment["limitations"]
    assert assessment["evidence"]["candidate_pool_count"] == 6
    assert assessment["evidence"]["selected_candidate_count"] == 4
    assert assessment["evidence"]["program_repair_rate"] == 1.0


def test_assessment_marks_quixbugs_dense_ranked_syntax_pool_as_broader_pool_probe():
    assessment = assess_record(
        {
            "experiment_id": "QuixBugs_T11c_dense528_ranked_syntax_pool_smoke",
            "hypothesis": "dense_checkpoint_can_rank_broader_syntax_preserving_repair_pool",
            "metrics": {
                "benchmark_label": "quixbugs_python_dense_ranked_syntax_pool_probe",
                "source": {
                    "repo_url": "https://github.com/jkoppel/QuixBugs",
                    "repo_commit": "4257f44b0ff1181dedaedee6a447e133219fcebf",
                },
                "model": {
                    "checkpoint": (
                        "artifacts/T11c_dense528_adamw_fp32_50m/"
                        "dense_decoder_best_model_only.pt"
                    ),
                    "checkpoint_step": 195000,
                    "config": {"hidden_dim": 528, "layers": 3},
                },
                "candidate_pool": {
                    "generated_candidate_count": 63,
                    "selected_candidate_count": 4,
                    "top_candidates_per_program": 1,
                },
                "candidate_count": 4,
                "program_count": 4,
                "final": {
                    "candidate_passed": 1,
                    "candidate_pass_rate": 0.25,
                    "programs_with_passing_candidate": 1,
                    "program_repair_rate": 0.25,
                },
                "limitations": [
                    "local_model_ranked_candidate",
                    "syntax_preserving_candidate_pool",
                    "broader_than_deterministic_edit_baseline",
                ],
            },
        }
    )

    assert assessment["outcome"] == "model_ranked_broader_syntax_pool_probe"
    assert assessment["supports_pareto_improvement"] is False
    assert assessment["primary_reason"] == (
        "dense_checkpoint_top1_ranking_drops_on_broader_syntax_pool"
    )
    assert "broader_than_deterministic_edit_baseline" in assessment["limitations"]
    assert assessment["evidence"]["candidate_pool_count"] == 63
    assert assessment["evidence"]["candidate_passed"] == 1
    assert assessment["evidence"]["program_repair_rate"] == 0.25


def test_assessment_marks_quixbugs_dense_ranked_syntax_topk_as_pairwise_probe():
    assessment = assess_record(
        {
            "experiment_id": "QuixBugs_T11c_dense528_ranked_syntax_topk_smoke",
            "hypothesis": "dense_ranked_syntax_pool_topk_execution_improves_repair_selection",
            "metrics": {
                "benchmark_label": "quixbugs_python_dense_ranked_syntax_topk_probe",
                "source": {
                    "repo_url": "https://github.com/jkoppel/QuixBugs",
                    "repo_commit": "4257f44b0ff1181dedaedee6a447e133219fcebf",
                },
                "model": {
                    "checkpoint": (
                        "artifacts/T11c_dense528_adamw_fp32_50m/"
                        "dense_decoder_best_model_only.pt"
                    ),
                    "checkpoint_step": 195000,
                    "config": {"hidden_dim": 528, "layers": 3},
                },
                "candidate_pool": {
                    "generated_candidate_count": 63,
                    "selected_candidate_count": 16,
                    "top_k_values": [1, 2, 4],
                    "max_top_k_profiled": 4,
                },
                "candidate_count": 16,
                "program_count": 4,
                "top_k_profile": [
                    {
                        "top_candidates_per_program": 1,
                        "candidate_count": 4,
                        "candidate_passed": 1,
                        "candidate_pass_rate": 0.25,
                        "programs_with_passing_candidate": 1,
                        "program_repair_rate": 0.25,
                    },
                    {
                        "top_candidates_per_program": 2,
                        "candidate_count": 8,
                        "candidate_passed": 2,
                        "candidate_pass_rate": 0.25,
                        "programs_with_passing_candidate": 2,
                        "program_repair_rate": 0.5,
                    },
                    {
                        "top_candidates_per_program": 4,
                        "candidate_count": 16,
                        "candidate_passed": 2,
                        "candidate_pass_rate": 0.125,
                        "programs_with_passing_candidate": 2,
                        "program_repair_rate": 0.5,
                    },
                ],
                "final": {
                    "candidate_passed": 2,
                    "candidate_pass_rate": 0.125,
                    "programs_with_passing_candidate": 2,
                    "program_repair_rate": 0.5,
                    "best_top_k": 2,
                    "best_program_repair_rate": 0.5,
                    "best_candidate_pass_rate": 0.25,
                },
                "limitations": [
                    "local_model_ranked_candidate",
                    "syntax_preserving_candidate_pool",
                    "top_k_execution_profile",
                ],
            },
        }
    )

    assert assessment["outcome"] == "model_ranked_syntax_topk_pairwise_probe"
    assert assessment["supports_pareto_improvement"] is False
    assert assessment["primary_reason"] == (
        "top_k_execution_profiles_dense_ranking_plus_syntax_pool"
    )
    assert "top_k_execution_profile" in assessment["limitations"]
    assert assessment["evidence"]["top1_program_repair_rate"] == 0.25
    assert assessment["evidence"]["best_top_k"] == 2
    assert assessment["evidence"]["best_program_repair_rate"] == 0.5


def test_assessment_marks_quixbugs_syntax_pool_ordering_controls():
    assessment = assess_record(
        {
            "experiment_id": "QuixBugs_T11c_dense528_syntax_pool_ordering_controls_smoke",
            "hypothesis": "dense_ranking_should_beat_same_pool_non_model_ordering_controls",
            "metrics": {
                "benchmark_label": "quixbugs_python_syntax_pool_ordering_control_probe",
                "source": {
                    "repo_url": "https://github.com/jkoppel/QuixBugs",
                    "repo_commit": "4257f44b0ff1181dedaedee6a447e133219fcebf",
                },
                "model": {
                    "checkpoint": (
                        "artifacts/T11c_dense528_adamw_fp32_50m/"
                        "dense_decoder_best_model_only.pt"
                    ),
                    "checkpoint_step": 195000,
                    "config": {"hidden_dim": 528, "layers": 3},
                },
                "candidate_pool": {
                    "generated_candidate_count": 63,
                    "selected_candidate_count": 26,
                    "top_k_values": [1, 2, 4, 8],
                    "max_top_k_profiled": 8,
                },
                "program_count": 4,
                "ordering_controls": {
                    "dense_likelihood": {
                        "best_top_k": 8,
                        "best_program_repair_rate": 1.0,
                        "top_k_profile": [
                            {
                                "top_candidates_per_program": 1,
                                "program_repair_rate": 0.25,
                            },
                            {
                                "top_candidates_per_program": 8,
                                "program_repair_rate": 1.0,
                            },
                        ],
                    },
                    "deterministic_pool_order": {
                        "best_top_k": 2,
                        "best_program_repair_rate": 0.75,
                        "top_k_profile": [
                            {
                                "top_candidates_per_program": 1,
                                "program_repair_rate": 0.5,
                            },
                            {
                                "top_candidates_per_program": 2,
                                "program_repair_rate": 0.75,
                            },
                        ],
                    },
                    "repair_aware_static_order": {
                        "best_top_k": 1,
                        "best_program_repair_rate": 1.0,
                        "top_k_profile": [
                            {
                                "top_candidates_per_program": 1,
                                "program_repair_rate": 1.0,
                            },
                            {
                                "top_candidates_per_program": 2,
                                "program_repair_rate": 1.0,
                            },
                        ],
                    },
                    "random_seeded_order": {
                        "best_top_k": 8,
                        "best_program_repair_rate": 0.5,
                        "top_k_profile": [
                            {
                                "top_candidates_per_program": 1,
                                "program_repair_rate": 0.25,
                            },
                            {
                                "top_candidates_per_program": 8,
                                "program_repair_rate": 0.5,
                            },
                        ],
                    },
                },
                "final": {
                    "best_ordering": "dense_likelihood",
                    "best_program_repair_rate": 1.0,
                    "dense_beats_all_controls": True,
                        "control_names": [
                            "dense_likelihood",
                            "deterministic_pool_order",
                            "repair_aware_static_order",
                            "random_seeded_order",
                        ],
                    },
                "limitations": [
                    "same_pool_ordering_controls",
                    "syntax_preserving_candidate_pool",
                ],
            },
        }
    )

    assert assessment["outcome"] == "syntax_pool_ordering_control_probe"
    assert assessment["supports_pareto_improvement"] is False
    assert assessment["primary_reason"] == (
        "dense_ranking_compared_with_same_pool_non_model_controls"
    )
    assert assessment["evidence"]["best_ordering"] == "dense_likelihood"
    assert assessment["evidence"]["dense_top1_program_repair_rate"] == 0.25
    assert assessment["evidence"]["dense_best_program_repair_rate"] == 1.0
    assert assessment["evidence"]["deterministic_best_program_repair_rate"] == 0.75
    assert assessment["evidence"]["repair_aware_best_top_k"] == 1
    assert assessment["evidence"]["repair_aware_best_program_repair_rate"] == 1.0
    assert assessment["evidence"]["random_best_program_repair_rate"] == 0.5


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


def test_assessment_marks_if3_block_codebook_validation_probe_as_loss_probe():
    assessment = assess_record(
        {
            "experiment_id": "IF3_block_codebook_t11c_validation_probe",
            "hypothesis": "if3_block_codebook_validation_loss",
            "metrics": {
                "benchmark_label": "if3_block_codebook_validation_probe",
                "candidate_id": "IF3",
                "checkpoint": {
                    "path": "artifacts/T11c/dense_decoder_last_model_only.pt",
                    "checkpoint_type": "model_only",
                    "step": 195313,
                },
                "split": "validation",
                "compression": {
                    "floating_parameter_count": 10_000,
                    "block_count": 100,
                    "learned_codebook": {
                        "mse": 0.01,
                        "encoded_bytes": 9000,
                        "metadata_bytes": 1000,
                        "runtime_buffer_bytes": 40_000,
                        "encoded_plus_runtime_bytes": 49_000,
                    },
                    "random_codebook_control": {
                        "mse": 0.05,
                        "encoded_bytes": 9000,
                    },
                },
                "policies": {
                    "fp32": {"loss": 1.0, "tokens": 1000},
                    "learned_block_codebook": {"loss": 1.1, "tokens": 1000},
                    "random_block_codebook": {"loss": 2.0, "tokens": 1000},
                },
                "comparisons": {
                    "learned_loss_delta_vs_fp32": 0.1,
                    "random_loss_delta_vs_fp32": 1.0,
                    "learned_loss_delta_vs_random": -0.9,
                    "learned_beats_random_loss": True,
                    "learned_mse_beats_random_mse": True,
                },
                "packed_kernel_evaluated": False,
                "loss_evaluated": True,
            },
        }
    )

    assert assessment["outcome"] == "validation_loss_probe_positive"
    assert assessment["supports_pareto_improvement"] is False
    assert "no_packed_kernel_speed" in assessment["limitations"]
    assert assessment["evidence"]["learned_loss_delta_vs_fp32"] == 0.1
    assert assessment["evidence"]["learned_beats_random_loss"] is True


def test_assessment_marks_dense_js_executable_syntax_probe():
    assessment = assess_record(
        {
            "experiment_id": "T11c_dense528_adamw_fp32_50m_final_test_js_executable",
            "hypothesis": "heldout_d4_javascript_executable_syntax_checkpoint_evaluation",
            "metrics": {
                "benchmark_label": "d4_dense_js_executable_checkpoint_evaluation",
                "checkpoint": "artifacts/T11c/dense_decoder_last_model_only.pt",
                "split": "test",
                "node": {"available": True, "version": "v20.20.1"},
                "tasks": {
                    "line_completion_syntax": {
                        "candidate_count": 100,
                        "completed_tasks": 64,
                        "token_accuracy_mean": 0.05,
                        "exact_match_rate": 0.0,
                        "edit_similarity_mean": 0.2,
                        "oracle_node_syntax_pass_rate": 1.0,
                        "generated_node_syntax_pass_rate": 0.25,
                    }
                },
            },
        }
    )

    assert assessment["outcome"] == "executable_js_syntax_probe_recorded"
    assert assessment["supports_pareto_improvement"] is False
    assert "syntax_only_not_unit_tests" in assessment["limitations"]
    assert assessment["evidence"]["generated_node_syntax_pass_rate"] == 0.25
    assert assessment["evidence"]["node_available"] is True


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
