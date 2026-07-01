from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from weightlab.glm_public_eval import (
    GlmPublicEvalConfig,
    evaluate_glm_public_tasks,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_glm_public_eval_scores_saved_public_outputs(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        tasks,
        [
            {
                "task_id": "synthetic_api_reuse_001",
                "task_family": "api_reuse",
                "source_policy": "synthetic",
                "prompt": "Use normalizeUser in a one-line call.",
                "expected_substrings": ["normalizeUser("],
                "judge_type": "contains_all",
            },
            {
                "task_id": "public_bug_repair_001",
                "task_family": "bug_repair",
                "source_policy": "public",
                "prompt": "Return the fixed line for a public toy clamp bug.",
                "expected_substrings": ["min(max(value, low), high)"],
                "judge_type": "contains_all",
            },
        ],
    )
    _write_jsonl(
        predictions,
        [
            {
                "task_id": "synthetic_api_reuse_001",
                "model": "glm-5.2",
                "prompt": "Use normalizeUser in a one-line call.",
                "output": "const user = normalizeUser(rawUser);",
                "context_length": 7,
                "output_length": 5,
                "thinking_effort": "default",
                "tool_access": "none",
                "temperature": 0.0,
                "sampling": {"top_p": 1.0},
                "latency_ms": 1234.5,
                "tokens_in": 7,
                "tokens_out": 5,
                "cost_usd": 0.01,
            },
            {
                "task_id": "public_bug_repair_001",
                "model": "glm-5.2",
                "prompt": "Return the fixed line for a public toy clamp bug.",
                "output": "return value;",
                "context_length": 9,
                "output_length": 2,
                "thinking_effort": "default",
                "tool_access": "none",
                "temperature": 0.0,
                "sampling": {"top_p": 1.0},
                "latency_ms": 1000.0,
                "tokens_in": 9,
                "tokens_out": 2,
                "cost_usd": 0.01,
            },
        ],
    )

    result = evaluate_glm_public_tasks(
        tasks,
        GlmPublicEvalConfig(
            seed=123,
            predictions_jsonl=predictions,
            baseline_category="local_same_budget_baseline",
        ),
    )

    assert result["benchmark_label"] == "glm_5_2_public_eval_harness"
    assert result["baseline_category"] == "local_same_budget_baseline"
    assert result["privacy_gate_passed"] is True
    assert result["task_count"] == 2
    assert result["evaluated_prediction_count"] == 2
    assert result["pass_count"] == 1
    assert result["pass_rate"] == 0.5
    assert result["metadata_completeness"]["complete_records"] == 2
    assert result["tasks"][0]["prompt_sha256"]
    assert "prompt" not in result["tasks"][0]
    assert result["predictions"][0]["passed"] is True
    assert result["predictions"][1]["passed"] is False


def test_glm_public_eval_rejects_private_task_policy(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(
        tasks,
        [
            {
                "task_id": "private_001",
                "task_family": "completion",
                "source_policy": "private",
                "prompt": "Confidential repository prompt",
                "expected_substrings": ["x"],
            }
        ],
    )

    with pytest.raises(ValueError, match="disallowed source_policy"):
        evaluate_glm_public_tasks(tasks, GlmPublicEvalConfig(seed=123))


def test_glm_public_eval_cli_writes_machine_readable_harness_record(
    tmp_path: Path,
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    output = tmp_path / "glm_eval.json"
    _write_jsonl(
        tasks,
        [
            {
                "task_id": "synthetic_doc_001",
                "task_family": "documentation_accuracy",
                "source_policy": "synthetic",
                "prompt": "Document function normalizeUser.",
                "expected_substrings": ["normalizeUser"],
            }
        ],
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_glm_public_tasks.py",
            "--tasks-jsonl",
            str(tasks),
            "--seed",
            "123",
            "--baseline-category",
            "local_same_budget_baseline",
            "--output",
            str(output),
            "--experiment-id",
            "glm_public_eval_test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "glm_public_eval_test"
    assert record["status"] == "completed"
    assert record["metrics"]["glm_run_status"] == "not_run_no_predictions"
    assert record["metrics"]["baseline_category"] == "local_same_budget_baseline"
    assert record["metrics"]["privacy_gate_passed"] is True
    assert record["metrics"]["task_count"] == 1
    assert "--tasks-jsonl" in record["command"]
    assert (tmp_path / "manifest.json").exists()
