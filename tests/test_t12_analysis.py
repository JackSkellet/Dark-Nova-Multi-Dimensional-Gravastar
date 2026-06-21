from __future__ import annotations

from pathlib import Path

from weightlab.t12_analysis import summarize_t12_second_seed_pair, summarize_t12_three_seed_pair


def test_t12_summary_reports_seed_spread_and_validation_selection_policy():
    summary = summarize_t12_second_seed_pair(Path("results"))

    assert summary["benchmark_label"] == "t12_second_seed_pair_summary"
    assert summary["selection_policy"]["test_loss_used_for_selection"] is False
    assert summary["runs"]["dense"]["seeds"] == [123, 456]
    assert summary["runs"]["adapter"]["seeds"] == [123, 456]
    assert summary["metrics"]["final_validation_loss"]["winner_by_mean"] == "dense"
    assert summary["metrics"]["tokens_per_second"]["winner_by_mean"] == "dense"
    assert summary["metrics"]["model_only_bytes"]["winner_by_mean"] == "dense"
    assert summary["metrics"]["final_test_loss"]["selection_allowed"] is False
    assert summary["third_seed_recommendation"]["recommended"] is True
    assert "validation_margin_is_small" in summary["third_seed_recommendation"]["reasons"]


def test_t12_three_seed_summary_reports_final_validation_winner_and_no_test_selection():
    summary = summarize_t12_three_seed_pair(Path("results"))

    assert summary["benchmark_label"] == "t12_three_seed_pair_summary"
    assert summary["selection_policy"]["test_loss_used_for_selection"] is False
    assert summary["runs"]["dense"]["seeds"] == [123, 456, 789]
    assert summary["runs"]["adapter"]["seeds"] == [123, 456, 789]
    assert summary["metrics"]["final_validation_loss"]["winner_by_mean"] == "dense"
    assert summary["metrics"]["best_validation_loss"]["winner_by_mean"] == "dense"
    assert summary["metrics"]["final_test_loss"]["selection_allowed"] is False
    assert summary["third_seed_resolution"]["third_seed_completed"] is True
    assert summary["third_seed_resolution"]["selected_family_by_final_validation_mean"] == "dense"

    assert summary["uncertainty"]["method"] == "paired_batch_bootstrap"
    assert summary["uncertainty"]["resamples"] == 2000
    final_validation_ci = summary["uncertainty"]["metrics"]["final_validation_loss"]
    assert final_validation_ci["paired_split"] == "validation"
    assert final_validation_ci["selection_allowed"] is True
    assert final_validation_ci["sample_order_verified"] is True
    assert final_validation_ci["dense_minus_adapter_mean"] < 0.0
    assert final_validation_ci["ci95"]["low"] <= final_validation_ci["ci95"]["high"]
    assert summary["uncertainty"]["metrics"]["final_test_loss"]["selection_allowed"] is False

    pareto = summary["pareto"]
    assert pareto["pareto_dominance"]["found"] is False
    assert pareto["frontier_expansion"]["found"] is True
    assert "reported_only_test_loss_tradeoff" in pareto["frontier_expansion"]["drivers"]
