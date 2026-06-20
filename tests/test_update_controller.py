from weightlab.update_controller import (
    CandidateUpdate,
    EvaluationReport,
    ExternalUpdateController,
    sign_payload,
    simulate_update_controller_experiment,
)


def _candidate(
    *,
    version: str = "adapter-alpha-v1",
    payload: bytes = b"adapter delta",
    report: EvaluationReport | None = None,
    approved_by: str = "external-ci",
) -> CandidateUpdate:
    report = report or EvaluationReport(
        new_task_score=0.86,
        retrieval_baseline_score=0.80,
        prior_task_score=0.99,
        previous_prior_task_score=1.00,
        unauthorized_retrievals=0,
        canary_leaks=0,
        poisoning_findings=0,
    )
    return CandidateUpdate.create(
        version=version,
        payload=payload,
        training_data_ids=("commit:a1", "doc:api"),
        config_id="adapter-smoke",
        report=report,
        proposed_by="model-proposal",
        approved_by=approved_by,
        signing_key="offline-test-key",
    )


def test_update_controller_deploys_signed_candidate_and_rolls_back():
    controller = ExternalUpdateController(signing_key="offline-test-key")
    deployed = controller.evaluate_and_stage(_candidate())

    assert deployed.accepted
    assert deployed.active_version == "adapter-alpha-v1"
    assert controller.active_version == "adapter-alpha-v1"
    assert controller.rollback("base")
    assert controller.active_version == "base"
    assert any(event["event"] == "rollback" for event in controller.audit_log)


def test_update_controller_rejects_tampering_security_leak_and_regression():
    controller = ExternalUpdateController(signing_key="offline-test-key")

    tampered = _candidate(payload=b"adapter delta")
    tampered.payload = b"changed after signing"
    assert "checksum_mismatch" in controller.evaluate_and_stage(tampered).reasons

    leaking = _candidate(
        version="adapter-leak",
        report=EvaluationReport(
            new_task_score=0.90,
            retrieval_baseline_score=0.80,
            prior_task_score=1.00,
            previous_prior_task_score=1.00,
            unauthorized_retrievals=0,
            canary_leaks=1,
            poisoning_findings=0,
        ),
    )
    assert "critical_security_failure" in controller.evaluate_and_stage(leaking).reasons

    regressing = _candidate(
        version="adapter-regress",
        report=EvaluationReport(
            new_task_score=0.90,
            retrieval_baseline_score=0.80,
            prior_task_score=0.95,
            previous_prior_task_score=1.00,
            unauthorized_retrievals=0,
            canary_leaks=0,
            poisoning_findings=0,
        ),
    )
    assert "prior_regression_too_large" in controller.evaluate_and_stage(regressing).reasons


def test_simulated_update_controller_experiment_records_required_gates():
    result = simulate_update_controller_experiment()

    assert result["successful_updates"] == 1
    assert result["rejected_updates"] == 3
    assert result["rollback_success"] is True
    assert result["critical_security_failures"] == 0
    assert result["new_task_delta_vs_retrieval"] > 0
    assert result["versioned_artifacts"] >= 2


def test_signature_depends_on_checksum_and_key():
    assert sign_payload("abc", "k1") != sign_payload("abc", "k2")
    assert sign_payload("abc", "k1") != sign_payload("def", "k1")
