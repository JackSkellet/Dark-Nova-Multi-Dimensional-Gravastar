from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


def checksum_payload(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sign_payload(checksum: str, signing_key: str) -> str:
    return hashlib.sha256(f"{signing_key}:{checksum}".encode()).hexdigest()


@dataclass(frozen=True)
class EvaluationReport:
    new_task_score: float
    retrieval_baseline_score: float
    prior_task_score: float
    previous_prior_task_score: float
    unauthorized_retrievals: int
    canary_leaks: int
    poisoning_findings: int

    @property
    def prior_regression(self) -> float:
        return self.previous_prior_task_score - self.prior_task_score

    @property
    def has_critical_security_failure(self) -> bool:
        return (
            self.unauthorized_retrievals > 0
            or self.canary_leaks > 0
            or self.poisoning_findings > 0
        )


@dataclass
class CandidateUpdate:
    version: str
    payload: bytes
    checksum: str
    signature: str
    training_data_ids: tuple[str, ...]
    config_id: str
    report: EvaluationReport
    proposed_by: str
    approved_by: str

    @classmethod
    def create(
        cls,
        *,
        version: str,
        payload: bytes,
        training_data_ids: tuple[str, ...],
        config_id: str,
        report: EvaluationReport,
        proposed_by: str,
        approved_by: str,
        signing_key: str,
    ) -> CandidateUpdate:
        checksum = checksum_payload(payload)
        return cls(
            version=version,
            payload=payload,
            checksum=checksum,
            signature=sign_payload(checksum, signing_key),
            training_data_ids=training_data_ids,
            config_id=config_id,
            report=report,
            proposed_by=proposed_by,
            approved_by=approved_by,
        )


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    active_version: str
    candidate_version: str
    reasons: tuple[str, ...]


@dataclass
class ExternalUpdateController:
    signing_key: str
    max_prior_regression: float = 0.02
    require_external_approval: bool = True
    active_version: str = "base"
    audit_log: list[dict[str, Any]] = field(default_factory=list)
    artifact_store: dict[str, str] = field(default_factory=lambda: {"base": "frozen-core"})
    _deployment_history: list[str] = field(default_factory=lambda: ["base"])

    def evaluate_and_stage(self, candidate: CandidateUpdate) -> GateDecision:
        reasons = self._gate_reasons(candidate)
        accepted = not reasons
        if accepted:
            self.artifact_store[candidate.version] = candidate.checksum
            self._deployment_history.append(candidate.version)
            self.active_version = candidate.version
            event = "deploy"
        else:
            event = "reject"

        self.audit_log.append(
            {
                "event": event,
                "candidate_version": candidate.version,
                "active_version": self.active_version,
                "reasons": list(reasons),
                "checksum": candidate.checksum,
                "training_data_ids": list(candidate.training_data_ids),
                "config_id": candidate.config_id,
                "proposed_by": candidate.proposed_by,
                "approved_by": candidate.approved_by,
            }
        )
        return GateDecision(
            accepted=accepted,
            active_version=self.active_version,
            candidate_version=candidate.version,
            reasons=tuple(reasons),
        )

    def rollback(self, target_version: str) -> bool:
        if target_version not in self.artifact_store:
            self.audit_log.append(
                {
                    "event": "rollback_failed",
                    "target_version": target_version,
                    "active_version": self.active_version,
                    "reason": "unknown_version",
                }
            )
            return False

        previous_version = self.active_version
        self.active_version = target_version
        self._deployment_history.append(target_version)
        self.audit_log.append(
            {
                "event": "rollback",
                "from_version": previous_version,
                "target_version": target_version,
                "active_version": self.active_version,
            }
        )
        return True

    def _gate_reasons(self, candidate: CandidateUpdate) -> list[str]:
        reasons: list[str] = []
        observed_checksum = checksum_payload(candidate.payload)
        if observed_checksum != candidate.checksum:
            reasons.append("checksum_mismatch")
        expected_signature = sign_payload(candidate.checksum, self.signing_key)
        if candidate.signature != expected_signature:
            reasons.append("signature_mismatch")
        if self.require_external_approval and not candidate.approved_by:
            reasons.append("missing_external_approval")
        if candidate.approved_by == candidate.proposed_by:
            reasons.append("self_approved_update")
        if candidate.report.new_task_score <= candidate.report.retrieval_baseline_score:
            reasons.append("no_new_task_improvement_over_retrieval")
        if candidate.report.prior_regression > self.max_prior_regression:
            reasons.append("prior_regression_too_large")
        if candidate.report.has_critical_security_failure:
            reasons.append("critical_security_failure")
        if not candidate.training_data_ids:
            reasons.append("missing_training_lineage")
        if not candidate.config_id:
            reasons.append("missing_config_lineage")
        return reasons


def simulate_update_controller_experiment() -> dict[str, Any]:
    controller = ExternalUpdateController(signing_key="offline-test-key")
    base_report = EvaluationReport(
        new_task_score=0.86,
        retrieval_baseline_score=0.80,
        prior_task_score=0.99,
        previous_prior_task_score=1.00,
        unauthorized_retrievals=0,
        canary_leaks=0,
        poisoning_findings=0,
    )
    accepted = CandidateUpdate.create(
        version="repo-adapter-v1",
        payload=b"repo adapter update with replay",
        training_data_ids=("commit:a1", "commit:a2", "replay:prior-api"),
        config_id="adapter-replay-smoke",
        report=base_report,
        proposed_by="model-proposal",
        approved_by="external-ci",
        signing_key=controller.signing_key,
    )
    decisions = [controller.evaluate_and_stage(accepted)]

    regression = CandidateUpdate.create(
        version="repo-adapter-no-replay",
        payload=b"repo adapter update without replay",
        training_data_ids=("commit:a3",),
        config_id="adapter-no-replay-smoke",
        report=EvaluationReport(
            new_task_score=0.90,
            retrieval_baseline_score=0.80,
            prior_task_score=0.95,
            previous_prior_task_score=1.00,
            unauthorized_retrievals=0,
            canary_leaks=0,
            poisoning_findings=0,
        ),
        proposed_by="model-proposal",
        approved_by="external-ci",
        signing_key=controller.signing_key,
    )
    decisions.append(controller.evaluate_and_stage(regression))

    leaking = CandidateUpdate.create(
        version="repo-adapter-leak",
        payload=b"repo adapter with canary memorization",
        training_data_ids=("commit:a4", "canary:synthetic"),
        config_id="adapter-security-smoke",
        report=EvaluationReport(
            new_task_score=0.91,
            retrieval_baseline_score=0.80,
            prior_task_score=1.00,
            previous_prior_task_score=1.00,
            unauthorized_retrievals=0,
            canary_leaks=1,
            poisoning_findings=0,
        ),
        proposed_by="model-proposal",
        approved_by="external-ci",
        signing_key=controller.signing_key,
    )
    decisions.append(controller.evaluate_and_stage(leaking))

    tampered = CandidateUpdate.create(
        version="repo-adapter-tampered",
        payload=b"repo adapter clean payload",
        training_data_ids=("commit:a5",),
        config_id="adapter-replay-smoke",
        report=base_report,
        proposed_by="model-proposal",
        approved_by="external-ci",
        signing_key=controller.signing_key,
    )
    tampered.payload = b"repo adapter modified after signature"
    decisions.append(controller.evaluate_and_stage(tampered))

    rollback_success = controller.rollback("base")
    successful = sum(decision.accepted for decision in decisions)
    rejected = len(decisions) - successful
    return {
        "successful_updates": successful,
        "rejected_updates": rejected,
        "rollback_success": rollback_success,
        "active_version_after_rollback": controller.active_version,
        "versioned_artifacts": len(controller.artifact_store),
        "audit_event_count": len(controller.audit_log),
        "new_task_delta_vs_retrieval": (
            base_report.new_task_score - base_report.retrieval_baseline_score
        ),
        "worst_prior_regression": max(
            decision_candidate.report.prior_regression
            for decision_candidate in [accepted, regression, leaking, tampered]
        ),
        "critical_security_failures": sum(
            1
            for event in controller.audit_log
            if event["event"] == "deploy"
            and "critical_security_failure" in event.get("reasons", [])
        ),
        "rejected_security_candidates": sum(
            1
            for event in controller.audit_log
            if event["event"] == "reject"
            and "critical_security_failure" in event.get("reasons", [])
        ),
        "decisions": [
            {
                "candidate_version": decision.candidate_version,
                "accepted": decision.accepted,
                "reasons": list(decision.reasons),
            }
            for decision in decisions
        ],
        "audit_log": controller.audit_log,
        "notes": (
            "External controller simulation for versioned adapter deployment. "
            "The controller, not the model, verifies lineage, signatures, "
            "quality/regression gates, security gates, and rollback."
        ),
    }
