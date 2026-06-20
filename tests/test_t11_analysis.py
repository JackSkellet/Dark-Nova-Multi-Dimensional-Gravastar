from __future__ import annotations

from pathlib import Path

from weightlab.t11_analysis import recompute_t11_summary


def test_t11_recompute_keeps_pareto_and_frontier_separate() -> None:
    summary = recompute_t11_summary(Path("results"))
    pairwise = summary["pairwise"]

    assert pairwise["candidate"] == "T11b"
    assert pairwise["baseline"] == "T11a"
    assert pairwise["primary_validation_loss_winner"] == "T11b"
    assert pairwise["reported_test_loss_winner"] == "T11b"
    assert pairwise["throughput_winner"] == "T11a"
    assert pairwise["frontier_expansion"] is True
    assert pairwise["pareto_dominates_baseline"] is False
    assert "final_test_loss" in pairwise["candidate_wins"]
    assert "tokens_per_second" in pairwise["candidate_losses"]
    assert summary["selection_policy"]["test_loss_used_for_selection"] is False


def test_t11_recompute_includes_dense_528_width_control() -> None:
    summary = recompute_t11_summary(Path("results"))
    comparisons = summary["pairwise_comparisons"]

    assert set(summary["runs"]) == {"T11a", "T11b", "T11c"}
    assert summary["rankings"]["final_validation_loss"][0]["run"] == "T11c"
    assert summary["rankings"]["final_test_loss"][0]["run"] == "T11b"
    assert summary["rankings"]["tokens_per_second"][0]["run"] == "T11c"
    assert summary["rankings"]["model_only_bytes"][0]["run"] == "T11c"
    assert summary["rankings"]["max_gradient_norm"][0]["run"] == "T11c"
    assert comparisons["T11c_vs_T11b"]["frontier_expansion"] is True
    assert comparisons["T11c_vs_T11b"]["pareto_dominates_baseline"] is False
    assert "final_validation_loss" in comparisons["T11c_vs_T11b"]["candidate_wins"]
    assert "final_test_loss" in comparisons["T11c_vs_T11b"]["candidate_losses"]
