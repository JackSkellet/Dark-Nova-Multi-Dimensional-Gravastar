from __future__ import annotations

import math

from weightlab.dense_functional_eval import _score_example, _summarize_examples
from weightlab.dense_training import ByteTokenizer


def test_functional_score_records_token_and_edit_similarity() -> None:
    tokenizer = ByteTokenizer()
    target = tokenizer.encode("return value;")[:-1]
    generated = tokenizer.encode("return other;")[:-1]

    result = _score_example(
        tokenizer,
        row_index=3,
        start_token=10,
        task_kind="prefix_completion",
        prefix=tokenizer.encode("function demo() {")[:-1],
        target=target,
        generated=generated,
        suffix=[],
    )

    assert result["row_index"] == 3
    assert result["target_tokens"] == len(target)
    assert 0.0 < result["token_accuracy"] < 1.0
    assert 0.0 < result["edit_similarity"] < 1.0
    assert result["exact_match"] is False
    assert len(result["target_sha256"]) == 64
    assert len(result["generated_sha256"]) == 64


def test_functional_summary_handles_empty_candidate_set() -> None:
    summary = _summarize_examples([], requested_tasks=8, candidate_count=0)

    assert summary["requested_tasks"] == 8
    assert summary["candidate_count"] == 0
    assert summary["completed_tasks"] == 0
    assert math.isnan(summary["token_accuracy_mean"])
    assert summary["examples"] == []
