from __future__ import annotations

import json
from pathlib import Path
from typing import Any

QUALITY_METRICS = [
    "prefix_token_accuracy",
    "span_token_accuracy",
    "comment_token_accuracy",
]

EFFICIENCY_METRICS = [
    "parameters",
    "model_only_bytes",
    "optimizer_state_bytes",
    "peak_vram_bytes",
    "tokens_per_second",
    "max_gradient_norm",
]


def recompute_t11_summary(results_dir: Path) -> dict[str, Any]:
    runs = {
        "T11a": _load_t11_run(results_dir, "T11a_dense544_adamw_fp32_50m"),
        "T11b": _load_t11_run(results_dir, "T11b_adapter528_adamw_fp32_50m"),
        "T11c": _load_t11_run(results_dir, "T11c_dense528_adamw_fp32_50m"),
    }
    adapter_vs_dense544 = _pairwise_comparison(
        runs["T11b"],
        runs["T11a"],
        candidate_label="T11b",
        baseline_label="T11a",
    )
    dense528_vs_dense544 = _pairwise_comparison(
        runs["T11c"],
        runs["T11a"],
        candidate_label="T11c",
        baseline_label="T11a",
    )
    dense528_vs_adapter = _pairwise_comparison(
        runs["T11c"],
        runs["T11b"],
        candidate_label="T11c",
        baseline_label="T11b",
    )
    return {
        "benchmark_label": "t11_recomputed_pareto_frontier",
        "runs": runs,
        "pairwise": {
            "candidate": "T11b",
            "baseline": "T11a",
            **adapter_vs_dense544,
        },
        "pairwise_comparisons": {
            "T11b_vs_T11a": {
                "candidate": "T11b",
                "baseline": "T11a",
                **adapter_vs_dense544,
            },
            "T11c_vs_T11a": {
                "candidate": "T11c",
                "baseline": "T11a",
                **dense528_vs_dense544,
            },
            "T11c_vs_T11b": {
                "candidate": "T11c",
                "baseline": "T11b",
                **dense528_vs_adapter,
            },
        },
        "rankings": {
            "final_validation_loss": _rank_runs(
                runs,
                "final_validation_loss",
                lower_is_better=True,
            ),
            "final_test_loss": _rank_runs(runs, "final_test_loss", lower_is_better=True),
            "tokens_per_second": _rank_runs(runs, "tokens_per_second", lower_is_better=False),
            "model_only_bytes": _rank_runs(runs, "model_only_bytes", lower_is_better=True),
            "max_gradient_norm": _rank_runs(runs, "max_gradient_norm", lower_is_better=True),
        },
        "selection_policy": {
            "checkpoint_selection": "validation_only",
            "test_loss_used_for_selection": False,
            "selected_checkpoints": {
                "T11a": "final",
                "T11b": "final",
                "T11c": "final",
            },
            "reason": (
                "Final checkpoints have lower validation loss than the best periodic "
                "checkpoints because final step 195313 is not an interval checkpoint."
            ),
        },
    }


def _load_t11_run(results_dir: Path, experiment_id: str) -> dict[str, Any]:
    train = _read(results_dir / f"{experiment_id}.json")["metrics"]
    final_validation = _read(results_dir / f"{experiment_id}_final_validation_eval.json")[
        "metrics"
    ]
    final_test = _read(results_dir / f"{experiment_id}_final_test_eval.json")["metrics"]
    best_validation = _read(results_dir / f"{experiment_id}_best_validation_eval.json")[
        "metrics"
    ]
    best_test = _read(results_dir / f"{experiment_id}_best_test_eval.json")["metrics"]
    functional = _read(results_dir / f"{experiment_id}_final_test_functional.json")[
        "metrics"
    ]
    gradient_values = [
        float(row["global_norm_before_clip"])
        for row in train["training"]["gradient_norms"]["records"]
        if row.get("finite")
    ]
    return {
        "experiment_id": experiment_id,
        "architecture": train["model"]["config"]["architecture_variant"],
        "hidden_dim": train["model"]["config"]["hidden_dim"],
        "adapter_dim": train["model"]["config"]["adapter_dim"],
        "parameters": train["model"]["parameter_count"],
        "active_parameters": train["model"]["active_parameter_count"],
        "train_tokens": train["training"]["train_tokens"],
        "tokens_per_second": train["training"]["tokens_per_second"],
        "runtime_s": train["training"]["elapsed_s"],
        "final_validation_loss": final_validation["loss"],
        "final_test_loss": final_test["loss"],
        "best_validation_loss": best_validation["loss"],
        "best_test_loss": best_test["loss"],
        "model_only_bytes": train["checkpoint"]["model_only_bytes"],
        "optimizer_state_bytes": train["checkpoint"]["optimizer_state_bytes"],
        "peak_vram_bytes": train["memory"]["peak_allocated_bytes"],
        "max_gradient_norm": max(gradient_values),
        "nonfinite_gradient_records": (
            len(train["training"]["gradient_norms"]["records"]) - len(gradient_values)
        ),
        "prefix_token_accuracy": functional["tasks"]["prefix_completion"][
            "token_accuracy_mean"
        ],
        "span_token_accuracy": functional["tasks"]["causal_span_reconstruction"][
            "token_accuracy_mean"
        ],
        "comment_token_accuracy": functional["tasks"]["comment_anchored_source_completion"][
            "token_accuracy_mean"
        ],
    }


def _pairwise_comparison(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    candidate_label: str,
    baseline_label: str,
) -> dict[str, Any]:
    wins = []
    losses = []
    ties = []
    for metric in QUALITY_METRICS:
        _classify_metric(
            metric,
            candidate[metric],
            baseline[metric],
            lower_is_better=False,
            wins=wins,
            losses=losses,
            ties=ties,
        )
    for metric in [
        "final_validation_loss",
        "final_test_loss",
        "parameters",
        "model_only_bytes",
        "optimizer_state_bytes",
        "peak_vram_bytes",
        "max_gradient_norm",
    ]:
        _classify_metric(
            metric,
            candidate[metric],
            baseline[metric],
            lower_is_better=True,
            wins=wins,
            losses=losses,
            ties=ties,
        )
    _classify_metric(
        "tokens_per_second",
        candidate["tokens_per_second"],
        baseline["tokens_per_second"],
        lower_is_better=False,
        wins=wins,
        losses=losses,
        ties=ties,
    )
    pareto_dominates = bool(wins) and not losses
    frontier_expansion = bool(wins) and bool(losses) and not pareto_dominates
    return {
        "candidate_wins": wins,
        "candidate_losses": losses,
        "ties": ties,
        "pareto_dominates_baseline": pareto_dominates,
        "frontier_expansion": frontier_expansion,
        "primary_validation_loss_winner": (
            candidate_label
            if candidate["final_validation_loss"] < baseline["final_validation_loss"]
            else baseline_label
        ),
        "reported_test_loss_winner": (
            candidate_label
            if candidate["final_test_loss"] < baseline["final_test_loss"]
            else baseline_label
        ),
        "throughput_winner": (
            candidate_label
            if candidate["tokens_per_second"] > baseline["tokens_per_second"]
            else baseline_label
        ),
    }


def _rank_runs(
    runs: dict[str, dict[str, Any]],
    metric: str,
    *,
    lower_is_better: bool,
) -> list[dict[str, Any]]:
    return [
        {"run": name, "value": run[metric]}
        for name, run in sorted(
            runs.items(),
            key=lambda item: item[1][metric],
            reverse=not lower_is_better,
        )
    ]


def _classify_metric(
    metric: str,
    candidate_value: float,
    baseline_value: float,
    *,
    lower_is_better: bool,
    wins: list[str],
    losses: list[str],
    ties: list[str],
) -> None:
    if candidate_value == baseline_value:
        ties.append(metric)
        return
    candidate_better = (
        candidate_value < baseline_value if lower_is_better else candidate_value > baseline_value
    )
    if candidate_better:
        wins.append(metric)
    else:
        losses.append(metric)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
