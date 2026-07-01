from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_SOURCE_POLICIES = {
    "public",
    "synthetic",
    "approved",
    "explicitly_approved",
    "approved_public",
}
DISALLOWED_SOURCE_POLICIES = {
    "private",
    "confidential",
    "customer",
    "restricted",
    "secret",
}
FORBIDDEN_PROMPT_MARKERS = (
    "BEGIN PRIVATE",
    "CONFIDENTIAL",
    "AWS_SECRET_ACCESS_KEY",
    "PRIVATE KEY",
    "sk-",
)
REQUIRED_PREDICTION_METADATA = (
    "model",
    "prompt",
    "output",
    "context_length",
    "output_length",
    "thinking_effort",
    "tool_access",
    "temperature",
    "sampling",
    "latency_ms",
    "tokens_in",
    "tokens_out",
    "cost_usd",
)


@dataclass(frozen=True)
class GlmPublicEvalConfig:
    seed: int = 123
    predictions_jsonl: Path | None = None
    max_tasks: int = 256
    baseline_category: str = "external_glm_5_2_baseline"


def evaluate_glm_public_tasks(
    tasks_jsonl: Path,
    config: GlmPublicEvalConfig | None = None,
) -> dict[str, Any]:
    config = config or GlmPublicEvalConfig()
    tasks = _load_tasks(tasks_jsonl, max_tasks=config.max_tasks)
    predictions = (
        _load_predictions(config.predictions_jsonl)
        if config.predictions_jsonl is not None
        else {}
    )
    evaluated = [
        _evaluate_prediction(task, predictions[str(task["task_id"])])
        for task in tasks
        if str(task["task_id"]) in predictions
    ]
    pass_count = sum(1 for row in evaluated if row["passed"])
    missing_metadata_counts = [
        len(row["missing_metadata"])
        for row in evaluated
        if isinstance(row.get("missing_metadata"), list)
    ]
    complete_records = sum(1 for count in missing_metadata_counts if count == 0)
    return {
        "benchmark_label": "glm_5_2_public_eval_harness",
        "baseline_category": config.baseline_category,
        "published_reference_category": "published_glm_5_2_reference",
        "glm_run_status": (
            "evaluated_saved_predictions"
            if config.predictions_jsonl is not None
            else "not_run_no_predictions"
        ),
        "tasks_jsonl": str(tasks_jsonl),
        "predictions_jsonl": (
            str(config.predictions_jsonl) if config.predictions_jsonl is not None else None
        ),
        "seed": config.seed,
        "task_count": len(tasks),
        "evaluated_prediction_count": len(evaluated),
        "pass_count": pass_count,
        "pass_rate": pass_count / len(evaluated) if evaluated else math.nan,
        "privacy_gate_passed": True,
        "remote_data_policy": (
            "public_synthetic_or_explicitly_approved_only; "
            "this harness does not call a remote model"
        ),
        "metadata_completeness": {
            "required_fields": list(REQUIRED_PREDICTION_METADATA),
            "complete_records": complete_records,
            "records_with_missing_fields": len(evaluated) - complete_records,
        },
        "tasks": [_public_task(task) for task in tasks],
        "predictions": evaluated,
        "limitations": [
            "offline_harness_only",
            "saved_predictions_required_for_model_scoring",
            "contains_all_and_exact_match_judges_only",
            "not_same_budget_local_baseline",
        ],
    }


def _load_tasks(path: Path, *, max_tasks: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            task = json.loads(line)
            _validate_task(task, line_number=line_number)
            tasks.append(task)
            if len(tasks) >= max_tasks:
                break
    return tasks


def _load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            prediction = json.loads(line)
            task_id = str(prediction.get("task_id", ""))
            if not task_id:
                raise ValueError("prediction missing task_id")
            predictions[task_id] = prediction
    return predictions


def _validate_task(task: dict[str, Any], *, line_number: int) -> None:
    task_id = str(task.get("task_id", ""))
    prompt = str(task.get("prompt", ""))
    source_policy = str(task.get("source_policy", "")).strip().lower()
    if not task_id:
        raise ValueError(f"task line {line_number} missing task_id")
    if not prompt:
        raise ValueError(f"task {task_id} missing prompt")
    if source_policy in DISALLOWED_SOURCE_POLICIES or source_policy not in ALLOWED_SOURCE_POLICIES:
        raise ValueError(f"task {task_id} has disallowed source_policy {source_policy!r}")
    upper_prompt = prompt.upper()
    for marker in FORBIDDEN_PROMPT_MARKERS:
        if marker in upper_prompt:
            raise ValueError(f"task {task_id} prompt contains forbidden private marker")
    expected = task.get("expected_substrings", [])
    if expected is None:
        expected = []
    if not isinstance(expected, list):
        raise ValueError(f"task {task_id} expected_substrings must be a list")


def _evaluate_prediction(
    task: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    output = str(prediction.get("output", ""))
    expected_substrings = [str(value) for value in task.get("expected_substrings", [])]
    judge_type = str(task.get("judge_type", "contains_all"))
    if judge_type == "exact_match":
        expected = str(task.get("expected_output", ""))
        passed = output.strip() == expected.strip()
    elif judge_type == "contains_all":
        passed = all(value in output for value in expected_substrings)
    else:
        raise ValueError(f"unsupported judge_type {judge_type!r}")
    missing_metadata = [
        field for field in REQUIRED_PREDICTION_METADATA if field not in prediction
    ]
    return {
        "task_id": str(task["task_id"]),
        "task_family": str(task.get("task_family", "")),
        "source_policy": str(task.get("source_policy", "")),
        "judge_type": judge_type,
        "model": str(prediction.get("model", "")),
        "passed": bool(passed),
        "missing_metadata": missing_metadata,
        "prompt_sha256": _sha256_text(str(prediction.get("prompt", ""))),
        "output_sha256": _sha256_text(output),
        "context_length": prediction.get("context_length"),
        "output_length": prediction.get("output_length"),
        "thinking_effort": prediction.get("thinking_effort"),
        "tool_access": prediction.get("tool_access"),
        "temperature": prediction.get("temperature"),
        "sampling": prediction.get("sampling"),
        "latency_ms": prediction.get("latency_ms"),
        "tokens_in": prediction.get("tokens_in"),
        "tokens_out": prediction.get("tokens_out"),
        "cost_usd": prediction.get("cost_usd"),
        "failure_example": None if passed else output[:500],
    }


def _public_task(task: dict[str, Any]) -> dict[str, Any]:
    prompt = str(task.get("prompt", ""))
    return {
        "task_id": str(task["task_id"]),
        "task_family": str(task.get("task_family", "")),
        "source_policy": str(task.get("source_policy", "")),
        "judge_type": str(task.get("judge_type", "contains_all")),
        "prompt_sha256": _sha256_text(prompt),
        "prompt_chars": len(prompt),
        "expected_substring_count": len(task.get("expected_substrings", []) or []),
    }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
