from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from weightlab.metrics import set_seed


@dataclass(frozen=True)
class FastWeightFact:
    step: int
    symbol: str
    answer: str
    aliases: tuple[str, ...]
    heldout_queries: tuple[str, ...]


@dataclass(frozen=True)
class QueryTask:
    step: int
    query: str
    answer: str
    kind: str


def run_if2_fast_weight_probe(seed: int = 123) -> dict[str, Any]:
    rng = set_seed(seed)
    del rng
    timeline = _timeline()
    answer_to_index = {
        answer: index
        for index, answer in enumerate(sorted({fact.answer for fact in timeline}))
    }
    index_to_answer = {index: answer for answer, index in answer_to_index.items()}
    vector_dim = 256
    fast_weights = np.zeros((vector_dim, len(answer_to_index)), dtype=np.float32)
    exact_memory: dict[str, str] = {}
    structured_memory: dict[str, str] = {}
    update_records: list[dict[str, Any]] = []

    steps: list[dict[str, Any]] = []
    for fact in timeline:
        started = time.perf_counter()
        exact_memory[fact.symbol] = fact.answer
        structured_memory[fact.symbol] = fact.answer
        for alias in fact.aliases:
            structured_memory[alias] = fact.answer
        key_vectors = [_text_features(fact.symbol, vector_dim)]
        key_vectors.extend(_text_features(alias, vector_dim) for alias in fact.aliases)
        for key_vector in key_vectors:
            fast_weights[:, answer_to_index[fact.answer]] += key_vector
        update_ms = (time.perf_counter() - started) * 1000.0
        update_records.append(
            {
                "step": fact.step,
                "symbol": fact.symbol,
                "alias_count": len(fact.aliases),
                "heldout_query_count": len(fact.heldout_queries),
                "update_ms": update_ms,
            }
        )
        tasks = _tasks_up_to(timeline, fact.step)
        steps.append(
            _score_methods(
                tasks,
                exact_memory=exact_memory,
                structured_memory=structured_memory,
                fast_weights=fast_weights,
                index_to_answer=index_to_answer,
                vector_dim=vector_dim,
                step=fact.step,
            )
        )

    final = steps[-1]
    heldout = _heldout_generalization_summary(steps[-1]["task_results"])
    parameter_evolution_adds = bool(
        final["structured_memory_plus_fast_weight_accuracy"]
        > final["structured_memory_accuracy"]
        and heldout["structured_memory_plus_fast_weight_correct"]
        > heldout["structured_memory_correct"]
    )
    exact_storage = _dict_storage_bytes(exact_memory)
    structured_storage = _dict_storage_bytes(structured_memory)
    parameter_bytes = int(fast_weights.size * fast_weights.itemsize)
    return {
        "benchmark_label": "if2_fast_weight_continual_probe",
        "candidate_id": "IF2",
        "timeline_steps": len(timeline),
        "facts": [
            {
                "step": fact.step,
                "symbol": fact.symbol,
                "answer": fact.answer,
                "aliases": list(fact.aliases),
                "heldout_queries": list(fact.heldout_queries),
            }
            for fact in timeline
        ],
        "methods": {
            "exact_retrieval": {
                "description": "Exact updated symbol-to-answer memory.",
                "storage_bytes": exact_storage,
            },
            "structured_memory": {
                "description": "Updated symbol plus explicit alias table.",
                "storage_bytes": structured_storage,
            },
            "fast_weight_scratchpad": {
                "description": (
                    "Dense associative matrix updated from verified symbols and aliases."
                ),
                "parameter_bytes": parameter_bytes,
                "update_count": len(update_records),
                "feature_dim": vector_dim,
            },
            "structured_memory_plus_fast_weight": {
                "description": (
                    "Structured memory first, falling back to fast-weight generalization."
                ),
                "storage_bytes": structured_storage + parameter_bytes,
            },
        },
        "steps": steps,
        "final": _final_summary(final),
        "heldout_generalization": heldout,
        "update_records": update_records,
        "parameter_evolution_adds_value_beyond_updated_memory": parameter_evolution_adds,
        "null_hypothesis_outcome": (
            "fast_weight_adds_value_beyond_updated_memory"
            if parameter_evolution_adds
            else "updated_memory_not_beaten"
        ),
        "limitations": [
            "synthetic_api_fact_fixture",
            "fast_weights_are_feature_hash_proxy_not_neural_lm_weights",
            "structured_memory_aliases_are_hand_provided",
            "does_not_measure_security_or_poisoning_gates",
        ],
    }


def _timeline() -> list[FastWeightFact]:
    return [
        FastWeightFact(
            step=0,
            symbol="parse_toml_config",
            answer="parse_config_api",
            aliases=("load toml config", "parse toml settings"),
            heldout_queries=("read toml configuration", "load toml settings file"),
        ),
        FastWeightFact(
            step=1,
            symbol="render_dependency_graph",
            answer="render_graph_api",
            aliases=("draw dependency graph", "render import graph"),
            heldout_queries=("show dependency diagram", "draw import graph"),
        ),
        FastWeightFact(
            step=2,
            symbol="find_stale_docs",
            answer="doc_drift_api",
            aliases=("detect stale documentation", "find outdated docs"),
            heldout_queries=("locate stale docs", "detect documentation drift"),
        ),
        FastWeightFact(
            step=3,
            symbol="apply_patch_plan",
            answer="patch_plan_api",
            aliases=("execute patch plan", "apply ordered edits"),
            heldout_queries=("run patch steps", "perform ordered edit plan"),
        ),
    ]


def _tasks_up_to(timeline: list[FastWeightFact], step: int) -> list[QueryTask]:
    tasks: list[QueryTask] = []
    for fact in timeline:
        if fact.step > step:
            continue
        tasks.append(QueryTask(fact.step, fact.symbol, fact.answer, "exact_symbol"))
        tasks.extend(
            QueryTask(fact.step, alias, fact.answer, "explicit_alias")
            for alias in fact.aliases
        )
        tasks.extend(
            QueryTask(fact.step, query, fact.answer, "heldout_paraphrase")
            for query in fact.heldout_queries
        )
    return tasks


def _score_methods(
    tasks: list[QueryTask],
    *,
    exact_memory: dict[str, str],
    structured_memory: dict[str, str],
    fast_weights: np.ndarray,
    index_to_answer: dict[int, str],
    vector_dim: int,
    step: int,
) -> dict[str, Any]:
    task_results: list[dict[str, Any]] = []
    for task in tasks:
        exact = exact_memory.get(task.query)
        structured = structured_memory.get(task.query)
        fast = _fast_weight_answer(task.query, fast_weights, index_to_answer, vector_dim)
        combined = structured if structured is not None else fast
        task_results.append(
            {
                "step": task.step,
                "kind": task.kind,
                "query": task.query,
                "answer": task.answer,
                "exact_retrieval_correct": exact == task.answer,
                "structured_memory_correct": structured == task.answer,
                "fast_weight_scratchpad_correct": fast == task.answer,
                "structured_memory_plus_fast_weight_correct": combined == task.answer,
            }
        )
    return {
        "step": step,
        "task_count": len(tasks),
        "exact_retrieval_accuracy": _accuracy(task_results, "exact_retrieval_correct"),
        "structured_memory_accuracy": _accuracy(task_results, "structured_memory_correct"),
        "fast_weight_scratchpad_accuracy": _accuracy(
            task_results,
            "fast_weight_scratchpad_correct",
        ),
        "structured_memory_plus_fast_weight_accuracy": _accuracy(
            task_results,
            "structured_memory_plus_fast_weight_correct",
        ),
        "task_results": task_results,
    }


def _fast_weight_answer(
    query: str,
    fast_weights: np.ndarray,
    index_to_answer: dict[int, str],
    vector_dim: int,
) -> str:
    scores = _text_features(query, vector_dim) @ fast_weights
    return index_to_answer[int(np.argmax(scores))]


def _text_features(text: str, width: int) -> np.ndarray:
    features = np.zeros(width, dtype=np.float32)
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    tokens = [token for token in normalized.split() if token]
    grams = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:], strict=False)]
    for gram in grams:
        digest = hashlib.sha256(gram.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % width
        features[index] += 1.0
    norm = float(np.linalg.norm(features))
    if norm:
        features /= norm
    return features


def _accuracy(results: list[dict[str, Any]], key: str) -> float:
    if not results:
        return 1.0
    return sum(bool(row[key]) for row in results) / len(results)


def _heldout_generalization_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    heldout = [row for row in results if row["kind"] == "heldout_paraphrase"]
    return {
        "task_count": len(heldout),
        "exact_retrieval_correct": sum(row["exact_retrieval_correct"] for row in heldout),
        "structured_memory_correct": sum(row["structured_memory_correct"] for row in heldout),
        "fast_weight_scratchpad_correct": sum(
            row["fast_weight_scratchpad_correct"] for row in heldout
        ),
        "structured_memory_plus_fast_weight_correct": sum(
            row["structured_memory_plus_fast_weight_correct"] for row in heldout
        ),
    }


def _final_summary(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": step["step"],
        "task_count": step["task_count"],
        "exact_retrieval_accuracy": step["exact_retrieval_accuracy"],
        "structured_memory_accuracy": step["structured_memory_accuracy"],
        "fast_weight_scratchpad_accuracy": step["fast_weight_scratchpad_accuracy"],
        "structured_memory_plus_fast_weight_accuracy": step[
            "structured_memory_plus_fast_weight_accuracy"
        ],
    }


def _dict_storage_bytes(memory: dict[str, str]) -> int:
    return sum(
        len(key.encode("utf-8")) + len(value.encode("utf-8")) + 16
        for key, value in memory.items()
    )
