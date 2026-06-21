from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from weightlab.if2_fast_weights import run_if2_fast_weight_probe


def test_if2_fast_weight_probe_compares_parameter_update_beyond_memory():
    result = run_if2_fast_weight_probe(seed=123)

    assert result["benchmark_label"] == "if2_fast_weight_continual_probe"
    assert result["candidate_id"] == "IF2"
    assert result["timeline_steps"] >= 3
    assert set(result["methods"]) == {
        "exact_retrieval",
        "structured_memory",
        "fast_weight_scratchpad",
        "structured_memory_plus_fast_weight",
    }
    assert result["methods"]["structured_memory"]["storage_bytes"] > 0
    assert result["methods"]["fast_weight_scratchpad"]["parameter_bytes"] > 0
    assert result["methods"]["fast_weight_scratchpad"]["update_count"] == result[
        "timeline_steps"
    ]
    assert result["final"]["structured_memory_accuracy"] >= result["final"][
        "exact_retrieval_accuracy"
    ]
    assert result["final"]["fast_weight_scratchpad_accuracy"] >= result["final"][
        "exact_retrieval_accuracy"
    ]
    assert result["final"]["structured_memory_plus_fast_weight_accuracy"] >= result[
        "final"
    ]["structured_memory_accuracy"]
    assert result["parameter_evolution_adds_value_beyond_updated_memory"] is True
    assert result["heldout_generalization"]["structured_memory_plus_fast_weight_correct"] > (
        result["heldout_generalization"]["structured_memory_correct"]
    )


def test_if2_fast_weight_cli_writes_record(tmp_path):
    output_path = tmp_path / "if2.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_if2_fast_weight_probe.py",
            "--output",
            str(output_path),
            "--experiment-id",
            "if2_test",
            "--seed",
            "123",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "if2_test"
    assert record["metrics"]["candidate_id"] == "IF2"
    assert record["metrics"]["benchmark_label"] == "if2_fast_weight_continual_probe"
    assert "--seed 123" in record["command"]
