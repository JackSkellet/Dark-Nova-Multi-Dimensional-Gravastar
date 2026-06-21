from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

RUNS = {
    "dense": {
        123: "T11c_dense528_adamw_fp32_50m",
        456: "T12a_dense528_seed456_adamw_fp32_50m",
        789: "T12c_dense528_seed789_adamw_fp32_50m",
    },
    "adapter": {
        123: "T11b_adapter528_adamw_fp32_50m",
        456: "T12b_adapter528_seed456_adamw_fp32_50m",
        789: "T12d_adapter528_seed789_adamw_fp32_50m",
    },
}

LOWER_IS_BETTER = {
    "final_validation_loss": True,
    "best_validation_loss": True,
    "final_test_loss": True,
    "best_test_loss": True,
    "tokens_per_second": False,
    "model_only_bytes": True,
    "peak_vram_bytes": True,
    "max_gradient_norm": True,
}

SELECTION_METRICS = {"final_validation_loss", "best_validation_loss"}
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 123

EVAL_METRIC_SOURCES = {
    "final_validation_loss": ("final_validation_eval", "validation"),
    "best_validation_loss": ("best_validation_eval", "validation"),
    "final_test_loss": ("final_test_eval", "test"),
    "best_test_loss": ("best_test_eval", "test"),
}


def summarize_t12_second_seed_pair(results_dir: Path) -> dict[str, Any]:
    runs = _load_runs(results_dir, seeds=(123, 456))
    summary = _summarize_runs(
        runs,
        benchmark_label="t12_second_seed_pair_summary",
        interpretation=(
            "Dense-528 has the lower mean validation loss and better resource metrics "
            "across the two available seeds, but the final-validation mean margin is "
            "small and best-validation winner changes by seed. Test loss is reported "
            "only after validation-defined selection."
        ),
    )
    final_validation_margin = abs(
        summary["metrics"]["final_validation_loss"]["families"]["dense"]["mean"]
        - summary["metrics"]["final_validation_loss"]["families"]["adapter"]["mean"]
    )
    winner_changes = _winner_changes(summary["metrics"])
    third_seed_reasons = []
    if final_validation_margin < 0.005:
        third_seed_reasons.append("validation_margin_is_small")
    if "best_validation_loss" in winner_changes:
        third_seed_reasons.append("best_validation_winner_changes_by_seed")
    summary["third_seed_recommendation"] = {
        "recommended": bool(third_seed_reasons),
        "reasons": third_seed_reasons,
        "validation_mean_margin": final_validation_margin,
        "winner_changes_by_metric": winner_changes,
    }
    return summary


def summarize_t12_three_seed_pair(results_dir: Path) -> dict[str, Any]:
    runs = _load_runs(results_dir, seeds=(123, 456, 789))
    summary = _summarize_runs(
        runs,
        benchmark_label="t12_three_seed_pair_summary",
        interpretation=(
            "Dense-528 remains the mean validation-loss, resource, and throughput "
            "winner across three matched seeds. Residual-adapter-528 keeps a "
            "reported-only final-test-loss edge, but test loss is not used for "
            "checkpoint or architecture selection."
        ),
    )
    final_validation = summary["metrics"]["final_validation_loss"]
    best_validation = summary["metrics"]["best_validation_loss"]
    summary["third_seed_resolution"] = {
        "third_seed_completed": True,
        "selected_family_by_final_validation_mean": final_validation["winner_by_mean"],
        "selected_family_by_best_validation_mean": best_validation["winner_by_mean"],
        "test_loss_used_for_selection": False,
        "winner_changes_by_metric": _winner_changes(summary["metrics"]),
    }
    summary["uncertainty"] = _paired_batch_bootstrap_uncertainty(runs)
    summary["pareto"] = _pareto_summary(summary)
    return summary


def _load_runs(
    results_dir: Path,
    *,
    seeds: tuple[int, ...],
) -> dict[str, dict[int, dict[str, Any]]]:
    runs = {
        family: {
            seed: _load_run(results_dir, experiment_id)
            for seed, experiment_id in seed_to_experiment.items()
            if seed in seeds
        }
        for family, seed_to_experiment in RUNS.items()
    }
    for family, seed_to_run in runs.items():
        missing = set(seeds) - set(seed_to_run)
        if missing:
            raise ValueError(f"missing T12 {family} run mapping for seeds: {sorted(missing)}")
    return runs


def _summarize_runs(
    runs: dict[str, dict[int, dict[str, Any]]],
    *,
    benchmark_label: str,
    interpretation: str,
) -> dict[str, Any]:
    metric_summary = {
        metric: _metric_summary(runs, metric, lower_is_better=lower_is_better)
        for metric, lower_is_better in LOWER_IS_BETTER.items()
    }
    return {
        "benchmark_label": benchmark_label,
        "runs": {
            family: {
                "seeds": sorted(seed_to_run),
                "experiments": {
                    str(seed): row["experiment_id"] for seed, row in sorted(seed_to_run.items())
                },
            }
            for family, seed_to_run in runs.items()
        },
        "metrics": metric_summary,
        "selection_policy": {
            "checkpoint_selection": "validation_only",
            "test_loss_used_for_selection": False,
            "selection_metrics": sorted(SELECTION_METRICS),
            "reported_only_metrics": ["final_test_loss", "best_test_loss"],
        },
        "interpretation": interpretation,
    }


def _winner_changes(metric_summary: dict[str, Any]) -> list[str]:
    changed = []
    for metric, summary in metric_summary.items():
        winners = set(summary["per_seed_winners"].values())
        if len(winners) > 1:
            changed.append(metric)
    return changed


def _load_run(results_dir: Path, experiment_id: str) -> dict[str, Any]:
    train = _read(results_dir / f"{experiment_id}.json")["metrics"]
    final_validation = _read(results_dir / f"{experiment_id}_final_validation_eval.json")[
        "metrics"
    ]
    final_test = _read(results_dir / f"{experiment_id}_final_test_eval.json")["metrics"]
    best_validation = _read(results_dir / f"{experiment_id}_best_validation_eval.json")[
        "metrics"
    ]
    best_test = _read(results_dir / f"{experiment_id}_best_test_eval.json")["metrics"]
    gradient_values = [
        float(row["global_norm_before_clip"])
        for row in train["training"]["gradient_norms"]["records"]
        if row.get("finite")
    ]
    return {
        "experiment_id": experiment_id,
        "architecture": train["model"]["config"]["architecture_variant"],
        "parameters": train["model"]["parameter_count"],
        "train_tokens": train["training"]["train_tokens"],
        "tokens_per_second": train["training"]["tokens_per_second"],
        "runtime_s": train["training"]["elapsed_s"],
        "final_validation_loss": final_validation["loss"],
        "best_validation_loss": best_validation["loss"],
        "final_test_loss": final_test["loss"],
        "best_test_loss": best_test["loss"],
        "final_validation_eval": final_validation,
        "best_validation_eval": best_validation,
        "final_test_eval": final_test,
        "best_test_eval": best_test,
        "model_only_bytes": train["checkpoint"]["model_only_bytes"],
        "peak_vram_bytes": train["memory"]["peak_allocated_bytes"],
        "max_gradient_norm": max(gradient_values),
    }


def _paired_batch_bootstrap_uncertainty(
    runs: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "method": "paired_batch_bootstrap",
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "unit": "dense_minus_adapter_nats_per_token",
        "metrics": {
            metric: _paired_batch_bootstrap_metric(runs, metric, eval_key, split)
            for metric, (eval_key, split) in EVAL_METRIC_SOURCES.items()
        },
    }


def _paired_batch_bootstrap_metric(
    runs: dict[str, dict[int, dict[str, Any]]],
    metric: str,
    eval_key: str,
    split: str,
) -> dict[str, Any]:
    differences: list[float] = []
    dense_losses: list[float] = []
    adapter_losses: list[float] = []
    sample_order_by_seed: dict[str, str] = {}
    batch_pairs_by_seed: dict[str, int] = {}
    for seed in sorted(runs["dense"]):
        dense_eval = runs["dense"][seed][eval_key]
        adapter_eval = runs["adapter"][seed][eval_key]
        _verify_paired_eval(metric, seed, split, dense_eval, adapter_eval)
        dense_records = dense_eval["batch_loss_records"]
        adapter_records = adapter_eval["batch_loss_records"]
        batch_pairs_by_seed[str(seed)] = len(dense_records)
        sample_order_by_seed[str(seed)] = str(dense_eval["sample_order_sha256"])
        for dense_record, adapter_record in zip(dense_records, adapter_records, strict=True):
            dense_loss = float(dense_record["loss"])
            adapter_loss = float(adapter_record["loss"])
            dense_losses.append(dense_loss)
            adapter_losses.append(adapter_loss)
            differences.append(dense_loss - adapter_loss)
    ci_low, ci_high, better_probability = _bootstrap_mean_ci(differences)
    return {
        "paired_split": split,
        "selection_allowed": metric in SELECTION_METRICS,
        "sample_order_verified": True,
        "batch_pairs": len(differences),
        "batch_pairs_by_seed": batch_pairs_by_seed,
        "sample_order_sha256_by_seed": sample_order_by_seed,
        "dense_mean": _mean(dense_losses),
        "adapter_mean": _mean(adapter_losses),
        "dense_minus_adapter_mean": _mean(differences),
        "ci95": {"low": ci_low, "high": ci_high},
        "bootstrap_probability_dense_better": better_probability,
    }


def _verify_paired_eval(
    metric: str,
    seed: int,
    split: str,
    dense_eval: dict[str, Any],
    adapter_eval: dict[str, Any],
) -> None:
    required = ("batch_loss_records", "sample_order_sha256", "split", "seed", "batches")
    for field in required:
        if field not in dense_eval or field not in adapter_eval:
            raise ValueError(
                f"{metric} seed {seed} is missing {field}; rerun checkpoint evaluations "
                "with --include-batch-losses before generating T12 uncertainty"
            )
    if dense_eval["split"] != split or adapter_eval["split"] != split:
        raise ValueError(f"{metric} seed {seed} split mismatch")
    if dense_eval["seed"] != adapter_eval["seed"]:
        raise ValueError(f"{metric} seed {seed} evaluation RNG seed mismatch")
    if dense_eval["batches"] != adapter_eval["batches"]:
        raise ValueError(f"{metric} seed {seed} batch count mismatch")
    if dense_eval["sample_order_sha256"] != adapter_eval["sample_order_sha256"]:
        raise ValueError(f"{metric} seed {seed} sample order mismatch")
    dense_records = dense_eval["batch_loss_records"]
    adapter_records = adapter_eval["batch_loss_records"]
    if len(dense_records) != len(adapter_records):
        raise ValueError(f"{metric} seed {seed} batch-loss record count mismatch")
    for dense_record, adapter_record in zip(dense_records, adapter_records, strict=True):
        if dense_record["batch_index"] != adapter_record["batch_index"]:
            raise ValueError(f"{metric} seed {seed} batch index mismatch")
        if dense_record["sample_sha256"] != adapter_record["sample_sha256"]:
            raise ValueError(f"{metric} seed {seed} batch sample hash mismatch")


def _bootstrap_mean_ci(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(BOOTSTRAP_SEED)
    count = len(values)
    means = []
    dense_better = 0
    for _ in range(BOOTSTRAP_RESAMPLES):
        mean = sum(rng.choice(values) for _ in range(count)) / count
        means.append(mean)
        if mean < 0.0:
            dense_better += 1
    means.sort()
    low_index = int(0.025 * (BOOTSTRAP_RESAMPLES - 1))
    high_index = int(0.975 * (BOOTSTRAP_RESAMPLES - 1))
    return means[low_index], means[high_index], dense_better / BOOTSTRAP_RESAMPLES


def _pareto_summary(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["metrics"]
    final_validation = metrics["final_validation_loss"]
    best_validation = metrics["best_validation_loss"]
    final_test = metrics["final_test_loss"]
    throughput = metrics["tokens_per_second"]
    model_bytes = metrics["model_only_bytes"]
    peak_vram = metrics["peak_vram_bytes"]
    max_gradient_norm = metrics["max_gradient_norm"]

    dense_validation_selected = (
        final_validation["winner_by_mean"] == "dense"
        and best_validation["winner_by_mean"] == "dense"
    )
    dense_resource_wins = (
        model_bytes["winner_by_mean"] == "dense"
        and peak_vram["winner_by_mean"] == "dense"
    )
    dense_runtime_wins = throughput["winner_by_mean"] == "dense"
    dense_stability_wins = max_gradient_norm["winner_by_mean"] == "dense"
    adapter_reported_test_edge = final_test["winner_by_mean"] == "adapter"
    frontier_expansion = (
        dense_validation_selected
        and dense_resource_wins
        and dense_runtime_wins
        and dense_stability_wins
        and adapter_reported_test_edge
    )
    return {
        "pareto_dominance": {
            "found": False,
            "reason": (
                "dense528 wins validation/resource/runtime/stability means, while "
                "adapter528 keeps a reported-only final-test-loss edge"
            ),
        },
        "frontier_expansion": {
            "found": bool(frontier_expansion),
            "drivers": [
                "validation_selected_dense528",
                "dense528_resource_runtime_stability_advantage",
                "reported_only_test_loss_tradeoff",
            ]
            if frontier_expansion
            else [],
        },
    }


def _metric_summary(
    runs: dict[str, dict[int, dict[str, Any]]],
    metric: str,
    *,
    lower_is_better: bool,
) -> dict[str, Any]:
    families = {
        family: _spread([seed_to_run[seed][metric] for seed in sorted(seed_to_run)])
        for family, seed_to_run in runs.items()
    }
    winner_by_mean = _winner(
        {family: values["mean"] for family, values in families.items()},
        lower_is_better=lower_is_better,
    )
    per_seed_winners = {
        str(seed): _winner(
            {family: seed_to_run[seed][metric] for family, seed_to_run in runs.items()},
            lower_is_better=lower_is_better,
        )
        for seed in sorted(next(iter(runs.values())))
    }
    return {
        "lower_is_better": lower_is_better,
        "selection_allowed": metric in SELECTION_METRICS,
        "winner_by_mean": winner_by_mean,
        "per_seed_winners": per_seed_winners,
        "families": families,
    }


def _spread(values: list[float]) -> dict[str, float]:
    return {
        "values": values,
        "mean": _mean(values),
        "min": min(values),
        "max": max(values),
        "spread": max(values) - min(values),
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _winner(values: dict[str, float], *, lower_is_better: bool) -> str:
    return min(values, key=values.get) if lower_is_better else max(values, key=values.get)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
