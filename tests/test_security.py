from weightlab.security import (
    run_adversarial_extraction_corpus_experiment,
    run_adversarial_security_experiment,
    run_security_canary_experiment,
    run_semantic_extraction_red_team_experiment,
)


def test_security_canary_experiment_blocks_cross_repo_retrieval_and_canary_leakage():
    result = run_security_canary_experiment()

    assert result["critical_failures"] == []
    assert result["metrics"]["unauthorized_retrievals"] == 0
    assert result["metrics"]["canary_leaks_in_allowed_context"] == 0
    assert result["metrics"]["blocked_secret_documents"] == 1
    assert result["metrics"]["prompt_injection_documents_quarantined"] == 1
    assert result["metrics"]["audit_event_count"] >= 4
    assert all(
        event["decision"] in {"allow", "deny", "quarantine"} for event in result["audit_log"]
    )


def test_adversarial_security_suite_blocks_bypass_extraction_and_poisoning():
    result = run_adversarial_security_experiment()

    assert result["critical_failures"] == []
    assert result["metrics"]["access_control_bypass_successes"] == 0
    assert result["metrics"]["canary_extraction_successes"] == 0
    assert result["metrics"]["prompt_injection_bypass_successes"] == 0
    assert result["metrics"]["poisoned_update_acceptances"] == 0
    assert result["metrics"]["tampered_update_acceptances"] == 0
    assert result["metrics"]["blocked_prompt_injection_documents"] >= 3
    assert result["metrics"]["rejected_poisoning_candidates"] >= 1


def test_adversarial_extraction_corpus_blocks_obfuscated_canaries_and_injections():
    result = run_adversarial_extraction_corpus_experiment()

    assert result["critical_failures"] == []
    assert result["metrics"]["query_count"] >= 10
    assert result["metrics"]["obfuscated_canary_documents"] >= 4
    assert result["metrics"]["encoded_canary_documents_quarantined"] >= 4
    assert result["metrics"]["paraphrased_prompt_injection_documents"] >= 4
    assert result["metrics"]["paraphrased_prompt_injection_documents_quarantined"] >= 4
    assert result["metrics"]["canary_leaks_in_allowed_context"] == 0
    assert result["metrics"]["unauthorized_retrievals"] == 0
    assert result["metrics"]["audit_event_count"] >= result["metrics"]["document_count"]
    allowed_context = "\n".join(result["allowed_contexts"])
    assert all(
        leaked not in allowed_context for leaked in result["synthetic_canaries"]
    )


def test_semantic_extraction_red_team_blocks_alias_secrets_and_exfil_queries():
    result = run_semantic_extraction_red_team_experiment()

    assert result["critical_failures"] == []
    assert result["metrics"]["semantic_secret_documents"] >= 3
    assert result["metrics"]["trust_label_quarantines"] >= 3
    assert result["metrics"]["semantic_exfiltration_queries"] >= 5
    assert result["metrics"]["semantic_exfiltration_queries_blocked"] >= 5
    assert result["metrics"]["sensitive_alias_leaks_in_allowed_context"] == 0
    assert result["metrics"]["unauthorized_retrievals"] == 0
    assert result["metrics"]["clean_queries_allowed"] >= 2
    allowed_context = "\n".join(result["allowed_contexts"])
    assert all(alias not in allowed_context for alias in result["sensitive_aliases"])
