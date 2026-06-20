from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from weightlab.metrics import set_seed


@dataclass(frozen=True)
class ApiFact:
    step: int
    symbol: str
    answer: str


class VersionedMemory:
    def __init__(self) -> None:
        self._versions: list[dict[str, str]] = [dict()]

    def update(self, facts: list[ApiFact]) -> int:
        current = dict(self._versions[-1])
        for fact in facts:
            current[fact.symbol] = fact.answer
        self._versions.append(current)
        return len(self._versions) - 1

    def rollback(self, version: int) -> None:
        self._versions = self._versions[: version + 1]

    def answer(self, symbol: str) -> str | None:
        return self._versions[-1].get(symbol)

    @property
    def version(self) -> int:
        return len(self._versions) - 1


def _accuracy(memory: VersionedMemory, tasks: list[ApiFact]) -> float:
    if not tasks:
        return 1.0
    return sum(memory.answer(task.symbol) == task.answer for task in tasks) / len(tasks)


def run_chronological_memory_experiment(seed: int = 0) -> dict[str, object]:
    rng = set_seed(seed)
    del rng
    timeline = [
        [ApiFact(0, "parse_config", "returns Config from TOML")],
        [ApiFact(1, "render_graph", "returns SVG dependency graph")],
        [ApiFact(2, "find_stale_docs", "returns doc paths whose symbols changed")],
        [ApiFact(3, "apply_patch_plan", "returns ordered edit steps")],
    ]

    frozen = VersionedMemory()
    frozen.update(timeline[0])
    retrieval = VersionedMemory()
    adapter_no_replay = VersionedMemory()
    adapter_with_replay = VersionedMemory()

    steps: list[dict[str, float]] = []
    retained_tasks: list[ApiFact] = []
    for step, facts in enumerate(timeline):
        retained_tasks.extend(facts)
        retrieval.update(facts)
        adapter_no_replay.update(facts)
        adapter_with_replay.update(retained_tasks)

        if step > 1:
            # Toy catastrophic forgetting control: the no-replay adapter loses its oldest fact.
            current = dict(adapter_no_replay._versions[-1])
            current.pop(timeline[0][0].symbol, None)
            adapter_no_replay._versions[-1] = current

        steps.append(
            {
                "step": float(step),
                "frozen_accuracy": _accuracy(frozen, retained_tasks),
                "retrieval_accuracy": _accuracy(retrieval, retained_tasks),
                "adapter_no_replay_prior_accuracy": _accuracy(
                    adapter_no_replay, retained_tasks[:-1]
                ),
                "adapter_with_replay_prior_accuracy": _accuracy(
                    adapter_with_replay, retained_tasks[:-1]
                ),
                "memory_version": float(retrieval.version),
                "storage_items": float(len(retrieval._versions[-1])),
            }
        )

    retrieval.rollback(1)
    rollback_tasks = [fact for batch in timeline[:1] for fact in batch]
    rollback = {
        "restored_version": retrieval.version,
        "accuracy_after_rollback": _accuracy(retrieval, rollback_tasks),
    }

    return {
        "steps": steps,
        "final": steps[-1],
        "rollback": rollback,
        "lineage": [
            {"version": i, "items": len(v)} for i, v in enumerate(adapter_with_replay._versions)
        ],
    }


def _one_hot(indices: list[int], width: int) -> np.ndarray:
    encoded = np.zeros((len(indices), width), dtype=np.float32)
    encoded[np.arange(len(indices)), indices] = 1.0
    return encoded


def _low_rank_fit(
    facts: list[ApiFact],
    symbol_to_idx: dict[str, int],
    answer_to_idx: dict[str, int],
    base_weights: np.ndarray,
    rank: int,
    ridge: float = 1e-3,
) -> np.ndarray:
    x = _one_hot([symbol_to_idx[fact.symbol] for fact in facts], len(symbol_to_idx))
    y = _one_hot([answer_to_idx[fact.answer] for fact in facts], len(answer_to_idx))
    residual_target = y - x @ base_weights
    gram = x.T @ x + ridge * np.eye(x.shape[1], dtype=np.float32)
    full_delta = np.linalg.solve(gram, x.T @ residual_target).astype(np.float32)
    u, singular_values, vh = np.linalg.svd(full_delta, full_matrices=False)
    kept = min(rank, singular_values.size)
    low_rank = (u[:, :kept] * singular_values[:kept]) @ vh[:kept]
    return low_rank.astype(np.float32)


def _adapter_accuracy(
    facts: list[ApiFact],
    symbol_to_idx: dict[str, int],
    answer_to_idx: dict[str, int],
    weights: np.ndarray,
) -> float:
    if not facts:
        return 1.0
    x = _one_hot([symbol_to_idx[fact.symbol] for fact in facts], len(symbol_to_idx))
    predictions = np.argmax(x @ weights, axis=1)
    targets = np.asarray([answer_to_idx[fact.answer] for fact in facts], dtype=np.int64)
    return float(np.mean(predictions == targets))


def _retrieval_accuracy(memory: dict[str, str], facts: list[ApiFact]) -> float:
    if not facts:
        return 1.0
    return sum(memory.get(fact.symbol) == fact.answer for fact in facts) / len(facts)


def _retrieval_storage_bytes(memory: dict[str, str]) -> int:
    return sum(
        len(symbol.encode()) + len(answer.encode()) + 16 for symbol, answer in memory.items()
    )


def run_trainable_adapter_vs_retrieval_experiment(seed: int = 0) -> dict[str, object]:
    rng = set_seed(seed)
    del rng
    timeline = [
        [ApiFact(0, "parse_config", "config")],
        [ApiFact(1, "load_config_defaults", "config")],
        [ApiFact(2, "render_dependency_graph", "graph")],
        [ApiFact(3, "find_stale_docs", "docs")],
        [ApiFact(4, "apply_patch_plan", "patch")],
        [ApiFact(5, "summarize_release_notes", "docs")],
    ]
    all_facts = [fact for batch in timeline for fact in batch]
    symbol_to_idx = {symbol: i for i, symbol in enumerate(sorted({f.symbol for f in all_facts}))}
    answer_to_idx = {answer: i for i, answer in enumerate(sorted({f.answer for f in all_facts}))}
    adapter_rank = len(answer_to_idx)
    base_weights = np.zeros((len(symbol_to_idx), len(answer_to_idx)), dtype=np.float32)
    for fact in timeline[0]:
        base_weights[symbol_to_idx[fact.symbol], answer_to_idx[fact.answer]] = 1.0

    retrieval: dict[str, str] = {}
    retained_tasks: list[ApiFact] = []
    steps: list[dict[str, float]] = []
    update_records: list[dict[str, float | str]] = []
    no_replay_delta = np.zeros_like(base_weights)
    replay_delta = np.zeros_like(base_weights)

    for step, facts in enumerate(timeline):
        retained_tasks.extend(facts)
        for fact in facts:
            retrieval[fact.symbol] = fact.answer

        start = time.perf_counter()
        no_replay_delta = _low_rank_fit(
            facts,
            symbol_to_idx,
            answer_to_idx,
            base_weights,
            rank=adapter_rank,
        )
        no_replay_update_ms = (time.perf_counter() - start) * 1000.0
        update_records.append(
            {
                "step": float(step),
                "adapter": "no_replay",
                "training_examples": float(len(facts)),
                "update_ms": no_replay_update_ms,
            }
        )

        start = time.perf_counter()
        replay_delta = _low_rank_fit(
            retained_tasks,
            symbol_to_idx,
            answer_to_idx,
            base_weights,
            rank=adapter_rank,
        )
        replay_update_ms = (time.perf_counter() - start) * 1000.0
        update_records.append(
            {
                "step": float(step),
                "adapter": "with_replay",
                "training_examples": float(len(retained_tasks)),
                "update_ms": replay_update_ms,
            }
        )

        prior_tasks = retained_tasks[:-len(facts)] if retained_tasks else []
        no_replay_weights = base_weights + no_replay_delta
        replay_weights = base_weights + replay_delta
        adapter_storage_bytes = (
            base_weights.size * 4
            + adapter_rank * (len(symbol_to_idx) + len(answer_to_idx)) * 4
            + 128
        )
        steps.append(
            {
                "step": float(step),
                "retrieval_accuracy": _retrieval_accuracy(retrieval, retained_tasks),
                "retrieval_new_accuracy": _retrieval_accuracy(retrieval, facts),
                "retrieval_prior_accuracy": _retrieval_accuracy(retrieval, prior_tasks),
                "adapter_no_replay_accuracy": _adapter_accuracy(
                    retained_tasks, symbol_to_idx, answer_to_idx, no_replay_weights
                ),
                "adapter_no_replay_new_accuracy": _adapter_accuracy(
                    facts, symbol_to_idx, answer_to_idx, no_replay_weights
                ),
                "adapter_no_replay_prior_accuracy": _adapter_accuracy(
                    prior_tasks, symbol_to_idx, answer_to_idx, no_replay_weights
                ),
                "adapter_with_replay_accuracy": _adapter_accuracy(
                    retained_tasks, symbol_to_idx, answer_to_idx, replay_weights
                ),
                "adapter_with_replay_new_accuracy": _adapter_accuracy(
                    facts, symbol_to_idx, answer_to_idx, replay_weights
                ),
                "adapter_with_replay_prior_accuracy": _adapter_accuracy(
                    prior_tasks, symbol_to_idx, answer_to_idx, replay_weights
                ),
                "retrieval_storage_bytes": float(_retrieval_storage_bytes(retrieval)),
                "adapter_with_replay_storage_bytes": float(adapter_storage_bytes),
                "no_replay_update_ms": no_replay_update_ms,
                "replay_update_ms": replay_update_ms,
            }
        )

    final = steps[-1]
    null_hypothesis_outcome = (
        "adapter_beats_retrieval"
        if final["adapter_with_replay_accuracy"] > final["retrieval_accuracy"]
        else "retrieval_not_beaten"
    )
    return {
        "model": {
            "architecture": "one_hot_symbol_classifier_with_low_rank_adapter_delta",
            "symbol_count": len(symbol_to_idx),
            "answer_count": len(answer_to_idx),
            "adapter_rank": adapter_rank,
            "frozen_base_facts": len(timeline[0]),
        },
        "steps": steps,
        "final": final,
        "weight_update_count": len(update_records),
        "update_records": update_records,
        "lineage": [
            {"step": step, "symbols": [fact.symbol for fact in facts]}
            for step, facts in enumerate(timeline)
        ],
        "null_hypothesis_outcome": null_hypothesis_outcome,
        "notes": (
            "Toy trainable low-rank adapter delta over chronological API facts. "
            "It compares weight updates with and without replay against continuously "
            "updated retrieval. The model is not a language model."
        ),
    }
