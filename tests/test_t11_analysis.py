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
