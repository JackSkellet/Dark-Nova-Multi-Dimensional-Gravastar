from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weightlab.lookup import _resolve_torch_accelerator

_IDENTIFIER_RE = re.compile(r"[$A-Za-z_][$A-Za-z0-9_]*")
_IMPORT_RE = re.compile(
    r"(?:from\s+['\"]([^'\"]+)['\"]|require\(\s*['\"]([^'\"]+)['\"]\s*\)|import\s+[^;]*?\s+from\s+['\"]([^'\"]+)['\"])",
)
_JS_STOPWORDS = {
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "default",
    "delete",
    "else",
    "export",
    "false",
    "for",
    "from",
    "function",
    "if",
    "import",
    "let",
    "module",
    "new",
    "null",
    "return",
    "switch",
    "this",
    "true",
    "undefined",
    "var",
    "while",
}


@dataclass(frozen=True)
class SparseHebbianConfig:
    split: str = "train"
    seed: int = 123
    node_count: int = 4096
    max_active_nodes: int = 48
    max_train_rows: int = 8192
    max_eval_rows: int = 512
    recall_at_k: int = 32
    min_text_bytes: int = 80
    cue_node_budget: int = 16
    target_node_budget: int = 32
    learning_rate: float = 1.0


@dataclass(frozen=True)
class HebbianTrainingConfig:
    split: str = "train"
    validation_split: str = "validation"
    seed: int = 123
    node_count: int = 2048
    max_active_nodes: int = 48
    max_train_rows: int = 8192
    max_validation_rows: int = 1024
    max_train_patterns: int = 8192
    max_validation_patterns: int = 1024
    text_window_bytes: int = 0
    text_window_stride_bytes: int = 0
    recall_at_k: int = 32
    min_text_bytes: int = 80
    cue_node_budget: int = 16
    target_node_budget: int = 32
    hebbian_learning_rate: float = 1.0
    epochs: int = 3
    batch_size: int = 128
    learning_rate: float = 1e-2
    weight_decay: float = 0.0
    device: str = "rocm"


@dataclass(frozen=True)
class SparseHebbianRerankerConfig:
    split: str = "train"
    validation_split: str = "validation"
    seed: int = 123
    node_count: int = 2048
    max_active_nodes: int = 48
    max_train_rows: int = 8192
    max_validation_rows: int = 1024
    max_train_patterns: int = 8192
    max_validation_patterns: int = 1024
    text_window_bytes: int = 0
    text_window_stride_bytes: int = 0
    candidate_count: int = 64
    recall_at_k: int = 32
    min_text_bytes: int = 80
    cue_node_budget: int = 16
    target_node_budget: int = 32
    hebbian_learning_rate: float = 1.0
    max_train_candidates: int = 200_000
    epochs: int = 3
    batch_size: int = 1024
    learning_rate: float = 1e-2
    weight_decay: float = 0.0
    device: str = "rocm"


@dataclass(frozen=True)
class RepositoryLinkingRankerConfig:
    train_split: str = "train"
    eval_split: str = "validation"
    seed: int = 123
    node_count: int = 4096
    max_active_nodes: int = 48
    max_memory_rows: int = 8192
    max_train_repositories: int = 128
    max_eval_repositories: int = 64
    negatives_per_query: int = 32
    top_k: int = 5
    min_text_bytes: int = 80
    cue_node_budget: int = 16
    target_node_budget: int = 32
    hebbian_learning_rate: float = 1.0
    epochs: int = 3
    batch_size: int = 1024
    learning_rate: float = 1e-2
    weight_decay: float = 0.0
    device: str = "rocm"


@dataclass(frozen=True)
class _RowPattern:
    repo: str
    path: str
    row_sha256: str
    cue_nodes: tuple[int, ...]
    target_nodes: tuple[int, ...]
    active_nodes: tuple[int, ...]
    lexical_tokens: tuple[str, ...] = ()


class SparseHebbianMemory:
    """Fixed-size co-activation memory for sparse assembly probes."""

    def __init__(self, config: SparseHebbianConfig):
        if config.node_count <= 0:
            raise ValueError("node_count must be positive")
        if config.max_active_nodes <= 1:
            raise ValueError("max_active_nodes must be greater than 1")
        self.config = config
        self.weights = np.zeros((config.node_count, config.node_count), dtype=np.float32)
        self.firing_counts = np.zeros(config.node_count, dtype=np.int64)
        self.update_count = 0

    def observe(self, active_nodes: list[int] | tuple[int, ...]) -> None:
        active = tuple(dict.fromkeys(int(node) for node in active_nodes))
        active = active[: self.config.max_active_nodes]
        if len(active) < 2:
            return
        index = np.asarray(active, dtype=np.int64)
        self.firing_counts[index] += 1
        self.weights[np.ix_(index, index)] += self.config.learning_rate
        self.weights[index, index] = 0.0
        self.update_count += 1

    def complete(self, cue_nodes: list[int] | tuple[int, ...], top_k: int) -> list[int]:
        scores = self.completion_scores(cue_nodes)
        if top_k >= len(scores):
            ranked = np.argsort(-scores)
        else:
            candidate = np.argpartition(-scores, top_k)[:top_k]
            ranked = candidate[np.argsort(-scores[candidate])]
        return [int(node) for node in ranked[:top_k] if np.isfinite(scores[node])]

    def completion_scores(self, cue_nodes: list[int] | tuple[int, ...]) -> np.ndarray:
        cue = tuple(dict.fromkeys(int(node) for node in cue_nodes))
        if not cue:
            return np.zeros(self.config.node_count, dtype=np.float32)
        cue_array = np.asarray(cue, dtype=np.int64)
        raw_scores = self.weights[cue_array].sum(axis=0, dtype=np.float64)
        normalizer = np.sqrt(np.maximum(self.firing_counts.astype(np.float64), 1.0))
        scores = raw_scores / normalizer
        scores[cue_array] = -np.inf
        finite_scores = np.where(np.isfinite(scores), scores, 0.0).astype(np.float32)
        max_score = float(np.max(finite_scores)) if finite_scores.size else 0.0
        if max_score > 0.0:
            finite_scores /= max_score
        return finite_scores

    def connection_strength(self, left: int, right: int) -> float:
        return float(self.weights[int(left), int(right)])

    def accounted_storage_bytes(self) -> int:
        return int(self.weights.nbytes + self.firing_counts.nbytes)


def run_if7_sparse_hebbian_probe(
    corpus_jsonl: str | Path,
    config: SparseHebbianConfig | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    corpus_jsonl = Path(corpus_jsonl)
    config = config or SparseHebbianConfig()
    rows, scanned = _sample_rows(corpus_jsonl, config)
    if not rows:
        raise ValueError("IF7 sparse Hebbian probe found no usable corpus rows")

    label_counts: dict[int, Counter[str]] = {}
    patterns = [_row_pattern(row, config, label_counts) for row in rows]
    patterns = [
        pattern
        for pattern in patterns
        if pattern.cue_nodes and pattern.target_nodes and len(pattern.active_nodes) >= 2
    ]
    if not patterns:
        raise ValueError("IF7 sparse Hebbian probe found no usable row patterns")

    memory = SparseHebbianMemory(config)
    random_memory = SparseHebbianMemory(config)
    rng = random.Random(config.seed)
    for pattern in patterns:
        memory.observe(pattern.active_nodes)
        random_memory.observe(
            tuple(rng.randrange(config.node_count) for _ in range(len(pattern.active_nodes)))
        )

    eval_patterns = _sample_eval_patterns(patterns, config)
    frequency_scores = memory.firing_counts.astype(np.float64)
    task_results = [
        _score_pattern(
            pattern,
            memory=memory,
            random_memory=random_memory,
            frequency_scores=frequency_scores,
            recall_at_k=config.recall_at_k,
        )
        for pattern in eval_patterns
    ]
    final = _final_summary(task_results)
    sample_predictions = _sample_predictions(
        eval_patterns,
        memory=memory,
        label_counts=label_counts,
        recall_at_k=config.recall_at_k,
    )
    hebbian_adds = bool(
        final["hebbian_hit_at_k"] >= final["frequency_hit_at_k"]
        and final["hebbian_hit_at_k"] >= final["random_hit_at_k"]
        and final["hebbian_mrr"] >= final["frequency_mrr"]
        and final["hebbian_mrr"] > final["random_mrr"]
    )
    mean_active = float(np.mean([len(pattern.active_nodes) for pattern in patterns]))
    mean_cue = float(np.mean([len(pattern.cue_nodes) for pattern in patterns]))
    mean_target = float(np.mean([len(pattern.target_nodes) for pattern in patterns]))
    label_bytes = sum(
        len(label.encode("utf-8")) + 8
        for labels in label_counts.values()
        for label in labels
    )

    return {
        "benchmark_label": "if7_sparse_hebbian_assembly_probe",
        "candidate_id": "IF7",
        "candidate_name": "Sparse Hebbian Assembly Memory",
        "config": {
            "split": config.split,
            "seed": config.seed,
            "node_count": config.node_count,
            "max_active_nodes": config.max_active_nodes,
            "max_train_rows": config.max_train_rows,
            "max_eval_rows": config.max_eval_rows,
            "recall_at_k": config.recall_at_k,
            "min_text_bytes": config.min_text_bytes,
            "cue_node_budget": config.cue_node_budget,
            "target_node_budget": config.target_node_budget,
            "learning_rate": config.learning_rate,
        },
        "corpus": {
            "path": str(corpus_jsonl),
            "source_split": config.split,
            "rows_scanned": scanned,
            "rows_loaded": len(rows),
            "usable_patterns": len(patterns),
            "eval_patterns": len(eval_patterns),
            "repositories_loaded": len({pattern.repo for pattern in patterns}),
        },
        "sparsity": {
            "mean_cue_nodes": mean_cue,
            "mean_target_nodes": mean_target,
            "mean_active_nodes": mean_active,
            "mean_active_fraction": mean_active / config.node_count,
        },
        "methods": {
            "frequency_control": {
                "description": "Ranks nodes by global firing count, without learned pair bonds.",
                "storage_bytes": int(memory.firing_counts.nbytes),
            },
            "random_sparse_control": {
                "description": (
                    "Same update count and active-set sizes with random node assemblies."
                ),
                "storage_bytes": random_memory.accounted_storage_bytes(),
            },
            "hebbian_sparse_assembly": {
                "description": (
                    "Sparse node co-activation matrix. Nodes that fire together strengthen "
                    "pairwise bonds and later complete partial cue patterns."
                ),
                "storage_bytes": memory.accounted_storage_bytes() + label_bytes,
                "update_count": memory.update_count,
                "label_index_bytes": label_bytes,
            },
        },
        "final": final,
        "sample_predictions": sample_predictions,
        "hebbian_adds_associative_signal": hebbian_adds,
        "null_hypothesis_outcome": (
            "sparse_hebbian_adds_signal_over_frequency_and_random_controls"
            if hebbian_adds
            else "frequency_or_random_control_not_beaten"
        ),
        "runtime_ms": (time.perf_counter() - started) * 1000.0,
        "limitations": [
            "associative_memory_probe_not_language_model_training",
            (
                "evaluates masked recall from exposed real corpus rows rather than "
                "unseen-repo generalization"
            ),
            "hashed node space can collide labels",
            "dense_numpy_matrix_is_not_a_packed_sparse_or_rocm_kernel",
            "no_security_or_authorization_update_gate",
        ],
    }


def run_if7_hebbian_trained_model(
    corpus_jsonl: str | Path,
    config: HebbianTrainingConfig | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    corpus_jsonl = Path(corpus_jsonl)
    config = config or HebbianTrainingConfig()
    train_rows, train_scanned = _sample_rows(corpus_jsonl, _training_to_sparse_config(config))
    validation_rows, validation_scanned = _sample_rows(
        corpus_jsonl,
        _training_to_sparse_config(
            config,
            split=config.validation_split,
            max_rows=config.max_validation_rows,
            seed=config.seed + 17,
        ),
    )
    if not train_rows:
        raise ValueError("IF7 trained model found no usable training rows")
    if not validation_rows:
        raise ValueError("IF7 trained model found no usable validation rows")

    label_counts: dict[int, Counter[str]] = {}
    sparse_config = _training_to_sparse_config(config)
    train_patterns = _usable_patterns(
        train_rows,
        sparse_config,
        label_counts,
        max_patterns=config.max_train_patterns,
        text_window_bytes=config.text_window_bytes,
        text_window_stride_bytes=config.text_window_stride_bytes,
    )
    validation_patterns = _usable_patterns(
        validation_rows,
        sparse_config,
        label_counts,
        max_patterns=config.max_validation_patterns,
        text_window_bytes=config.text_window_bytes,
        text_window_stride_bytes=config.text_window_stride_bytes,
    )
    if not train_patterns or not validation_patterns:
        raise ValueError("IF7 trained model found no usable train/validation patterns")

    memory = SparseHebbianMemory(sparse_config)
    for pattern in train_patterns:
        memory.observe(pattern.active_nodes)

    accelerator = _resolve_torch_accelerator(config.device)
    device = accelerator.device
    torch.manual_seed(config.seed)

    cue_only = torch.nn.Linear(config.node_count, config.node_count).to(device)
    cue_plus_hebbian = torch.nn.Linear(config.node_count * 2, config.node_count).to(device)
    cue_history = _train_linear_probe(
        cue_only,
        train_patterns,
        memory,
        config,
        device,
        include_hebbian=False,
    )
    hebbian_history = _train_linear_probe(
        cue_plus_hebbian,
        train_patterns,
        memory,
        config,
        device,
        include_hebbian=True,
    )
    cue_validation = _evaluate_linear_probe(
        cue_only,
        validation_patterns,
        memory,
        device,
        include_hebbian=False,
        recall_at_k=config.recall_at_k,
    )
    hebbian_validation = _evaluate_linear_probe(
        cue_plus_hebbian,
        validation_patterns,
        memory,
        device,
        include_hebbian=True,
        recall_at_k=config.recall_at_k,
    )
    raw_hebbian = _evaluate_raw_hebbian(
        validation_patterns,
        memory=memory,
        recall_at_k=config.recall_at_k,
    )
    hebbian_improves = bool(
        hebbian_validation["hit_at_k"] >= cue_validation["hit_at_k"]
        and hebbian_validation["mrr"] >= cue_validation["mrr"]
    )
    label_bytes = sum(
        len(label.encode("utf-8")) + 8
        for labels in label_counts.values()
        for label in labels
    )

    return {
        "benchmark_label": "if7_hebbian_conditioned_trained_model",
        "candidate_id": "IF7",
        "candidate_name": "Sparse Hebbian Assembly Memory",
        "corpus": {
            "path": str(corpus_jsonl),
            "train_split": config.split,
            "validation_split": config.validation_split,
            "train_rows_scanned": train_scanned,
            "validation_rows_scanned": validation_scanned,
            "train_rows_loaded": len(train_rows),
            "validation_rows_loaded": len(validation_rows),
            "usable_train_patterns": len(train_patterns),
            "usable_validation_patterns": len(validation_patterns),
            "train_repositories": len({pattern.repo for pattern in train_patterns}),
            "validation_repositories": len({pattern.repo for pattern in validation_patterns}),
        },
        "training": {
            "supervised_train_rows": len(train_patterns),
            "supervised_train_patterns": len(train_patterns),
            "validation_rows": len(validation_patterns),
            "validation_patterns": len(validation_patterns),
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "device": str(device),
            "accelerator_backend": accelerator.backend,
            "requested_device": accelerator.requested_device,
            "rocm_available": accelerator.rocm_available,
            "rocm_runtime_version": accelerator.rocm_runtime_version,
        },
        "hebbian_memory": {
            "node_count": config.node_count,
            "max_active_nodes": config.max_active_nodes,
            "mean_train_active_nodes": float(
                np.mean([len(pattern.active_nodes) for pattern in train_patterns])
            ),
            "update_count": memory.update_count,
            "storage_bytes": memory.accounted_storage_bytes() + label_bytes,
            "label_index_bytes": label_bytes,
        },
        "models": {
            "cue_only": {
                "description": "Trained linear multilabel predictor from cue nodes only.",
                "parameter_count": _parameter_count(cue_only),
                "parameter_bytes": _parameter_bytes(cue_only),
                "loss_history": cue_history,
            },
            "cue_plus_hebbian": {
                "description": (
                    "Trained linear multilabel predictor from cue nodes plus Hebbian "
                    "completion scores."
                ),
                "parameter_count": _parameter_count(cue_plus_hebbian),
                "parameter_bytes": _parameter_bytes(cue_plus_hebbian),
                "loss_history": hebbian_history,
            },
        },
        "validation": {
            "cue_only": cue_validation,
            "cue_plus_hebbian": hebbian_validation,
            "raw_hebbian_memory": raw_hebbian,
        },
        "hebbian_conditioning_improves_trained_model": hebbian_improves,
        "null_hypothesis_outcome": (
            "hebbian_conditioning_improves_or_matches_trained_cue_only_model"
            if hebbian_improves
            else "trained_cue_only_model_not_beaten"
        ),
        "runtime_ms": (time.perf_counter() - started) * 1000.0,
        "limitations": [
            "trained_linear_multilabel_probe_not_decoder_language_model",
            "predicts_hashed_identifier_import_nodes_not_tokens",
            "not_executable_code_generation_or_repair",
            "dense_torch_linear_layers_not_sparse_rocm_kernel",
            "no_security_or_authorization_update_gate",
        ],
    }


def run_if7_hebbian_sparse_reranker(
    corpus_jsonl: str | Path,
    config: SparseHebbianRerankerConfig | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    corpus_jsonl = Path(corpus_jsonl)
    config = config or SparseHebbianRerankerConfig()
    sparse_config = _reranker_to_sparse_config(config)
    train_rows, train_scanned = _sample_rows(corpus_jsonl, sparse_config)
    validation_rows, validation_scanned = _sample_rows(
        corpus_jsonl,
        _reranker_to_sparse_config(
            config,
            split=config.validation_split,
            max_rows=config.max_validation_rows,
            seed=config.seed + 23,
        ),
    )
    if not train_rows or not validation_rows:
        raise ValueError("IF7 sparse reranker found no usable train/validation rows")

    label_counts: dict[int, Counter[str]] = {}
    train_patterns = _usable_patterns(
        train_rows,
        sparse_config,
        label_counts,
        max_patterns=config.max_train_patterns,
        text_window_bytes=config.text_window_bytes,
        text_window_stride_bytes=config.text_window_stride_bytes,
    )
    validation_patterns = _usable_patterns(
        validation_rows,
        sparse_config,
        label_counts,
        max_patterns=config.max_validation_patterns,
        text_window_bytes=config.text_window_bytes,
        text_window_stride_bytes=config.text_window_stride_bytes,
    )
    if not train_patterns or not validation_patterns:
        raise ValueError("IF7 sparse reranker found no usable train/validation patterns")

    memory = SparseHebbianMemory(sparse_config)
    for pattern in train_patterns:
        memory.observe(pattern.active_nodes)

    accelerator = _resolve_torch_accelerator(config.device)
    device = accelerator.device
    torch.manual_seed(config.seed)
    node_priors = _node_priors(train_patterns, config.node_count)
    train_examples = _candidate_examples(
        train_patterns,
        memory=memory,
        candidate_count=config.candidate_count,
        max_examples=config.max_train_candidates,
        include_targets=True,
        node_priors=node_priors,
    )
    if not train_examples:
        raise ValueError("IF7 sparse reranker found no train candidate examples")
    model = torch.nn.Linear(8, 1).to(device)
    loss_history = _train_candidate_reranker(model, train_examples, config, device)
    reranked = _evaluate_candidate_reranker(
        model,
        validation_patterns,
        memory=memory,
        candidate_count=config.candidate_count,
        recall_at_k=config.recall_at_k,
        device=device,
        node_priors=node_priors,
    )
    raw_hebbian = _evaluate_raw_hebbian(
        validation_patterns,
        memory=memory,
        recall_at_k=config.recall_at_k,
    )
    ceiling = _evaluate_candidate_ceiling(
        validation_patterns,
        memory=memory,
        candidate_count=config.candidate_count,
        recall_at_k=config.recall_at_k,
    )
    improves_raw = bool(
        reranked["hit_at_k"] >= raw_hebbian["hit_at_k"]
        and reranked["mrr"] >= raw_hebbian["mrr"]
    )
    label_bytes = sum(
        len(label.encode("utf-8")) + 8
        for labels in label_counts.values()
        for label in labels
    )
    return {
        "benchmark_label": "if7_sparse_hebbian_candidate_reranker",
        "candidate_id": "IF7",
        "candidate_name": "Sparse Hebbian Assembly Memory",
        "corpus": {
            "path": str(corpus_jsonl),
            "train_split": config.split,
            "validation_split": config.validation_split,
            "train_rows_scanned": train_scanned,
            "validation_rows_scanned": validation_scanned,
            "train_rows_loaded": len(train_rows),
            "validation_rows_loaded": len(validation_rows),
            "usable_train_patterns": len(train_patterns),
            "usable_validation_patterns": len(validation_patterns),
            "train_repositories": len({pattern.repo for pattern in train_patterns}),
            "validation_repositories": len({pattern.repo for pattern in validation_patterns}),
        },
        "training": {
            "train_patterns": len(train_patterns),
            "validation_patterns": len(validation_patterns),
            "train_candidate_examples": len(train_examples),
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "candidate_count": config.candidate_count,
            "device": str(device),
            "accelerator_backend": accelerator.backend,
            "requested_device": accelerator.requested_device,
            "rocm_available": accelerator.rocm_available,
            "rocm_runtime_version": accelerator.rocm_runtime_version,
        },
        "hebbian_memory": {
            "node_count": config.node_count,
            "max_active_nodes": config.max_active_nodes,
            "mean_train_active_nodes": float(
                np.mean([len(pattern.active_nodes) for pattern in train_patterns])
            ),
            "update_count": memory.update_count,
            "storage_bytes": memory.accounted_storage_bytes() + label_bytes,
            "label_index_bytes": label_bytes,
        },
        "models": {
            "candidate_reranker": {
                "description": (
                    "Trained sparse candidate scorer over Hebbian top-k candidates, "
                    "not a dense full-node feature concatenation."
                ),
                "parameter_count": _parameter_count(model),
                "parameter_bytes": _parameter_bytes(model),
                "loss_history": loss_history,
            }
        },
        "validation": {
            "raw_hebbian_memory": raw_hebbian,
            "candidate_reranker": reranked,
            "candidate_recall_ceiling": ceiling,
        },
        "sparse_reranker_improves_raw_hebbian": improves_raw,
        "null_hypothesis_outcome": (
            "sparse_reranker_improves_or_matches_raw_hebbian"
            if improves_raw
            else "raw_hebbian_not_beaten_by_sparse_reranker"
        ),
        "runtime_ms": (time.perf_counter() - started) * 1000.0,
        "limitations": [
            "candidate_reranker_not_decoder_language_model",
            "predicts_hashed_identifier_import_nodes_not_tokens",
            "candidate_recall_bounds_possible_quality",
            "not_executable_code_generation_or_repair",
            "dense_torch_linear_scorer_not_sparse_rocm_kernel",
        ],
    }


def run_if7_hebbian_repository_linking(
    corpus_jsonl: str | Path,
    *,
    train_split: str = "train",
    eval_split: str = "validation",
    seed: int = 123,
    node_count: int = 2048,
    max_train_rows: int = 8192,
    max_eval_repositories: int = 64,
    negatives_per_query: int = 32,
    top_k: int = 5,
    min_text_bytes: int = 80,
) -> dict[str, Any]:
    started = time.perf_counter()
    corpus_jsonl = Path(corpus_jsonl)
    sparse_config = SparseHebbianConfig(
        split=train_split,
        seed=seed,
        node_count=node_count,
        max_train_rows=max_train_rows,
        max_eval_rows=max_eval_repositories,
        min_text_bytes=min_text_bytes,
    )
    train_rows, train_scanned = _sample_rows(corpus_jsonl, sparse_config)
    eval_rows, eval_scanned = _sample_rows(
        corpus_jsonl,
        SparseHebbianConfig(
            split=eval_split,
            seed=seed + 31,
            node_count=node_count,
            max_train_rows=1_000_000,
            min_text_bytes=min_text_bytes,
        ),
    )
    if not train_rows or not eval_rows:
        raise ValueError("IF7 repository linking requires train and eval rows")

    label_counts: dict[int, Counter[str]] = {}
    train_patterns = _usable_patterns(train_rows, sparse_config, label_counts)
    memory = SparseHebbianMemory(sparse_config)
    for pattern in train_patterns:
        memory.observe(pattern.active_nodes)

    eval_patterns = _usable_patterns(
        eval_rows,
        SparseHebbianConfig(
            split=eval_split,
            seed=seed,
            node_count=node_count,
            min_text_bytes=min_text_bytes,
        ),
        label_counts,
    )
    grouped = _patterns_by_repo(eval_patterns)
    eligible_repos = sorted(repo for repo, rows in grouped.items() if len(rows) >= 2)
    rng = random.Random(seed)
    rng.shuffle(eligible_repos)
    selected_repos = eligible_repos[:max_eval_repositories]
    distractors = [pattern for pattern in eval_patterns if pattern.repo not in selected_repos]
    if not distractors:
        distractors = eval_patterns

    tasks: list[dict[str, Any]] = []
    for repo in selected_repos:
        repo_patterns = sorted(grouped[repo], key=lambda pattern: pattern.path)
        for query in repo_patterns:
            positives = [pattern for pattern in repo_patterns if pattern.path != query.path]
            if not positives:
                continue
            negatives = [
                pattern
                for pattern in rng.sample(
                    distractors,
                    k=min(negatives_per_query, len(distractors)),
                )
                if pattern.repo != query.repo
            ]
            candidates = [*positives, *negatives]
            if not candidates:
                continue
            tasks.append(
                _repository_linking_task(
                    query,
                    candidates,
                    positives={pattern.path for pattern in positives},
                    memory=memory,
                    top_k=top_k,
                )
            )

    if not tasks:
        raise ValueError("IF7 repository linking produced no tasks")

    methods = {
        method: _repository_linking_summary(tasks, method)
        for method in [
            "path_role_overlap",
            "lexical_text_overlap",
            "raw_hebbian_context",
            "combined_lexical_hebbian",
        ]
    }
    best_method = max(methods.items(), key=lambda item: (item[1]["hit_at_k"], item[1]["mrr"]))
    return {
        "benchmark_label": "if7_hebbian_repository_linking",
        "candidate_id": "IF7",
        "corpus": {
            "path": str(corpus_jsonl),
            "train_split": train_split,
            "eval_split": eval_split,
            "train_rows_scanned": train_scanned,
            "eval_rows_scanned": eval_scanned,
            "train_rows_loaded": len(train_rows),
            "eval_rows_loaded": len(eval_rows),
            "eval_repositories": len(selected_repos),
            "eligible_eval_repositories": len(eligible_repos),
        },
        "tasks": {
            "task_count": len(tasks),
            "top_k": top_k,
            "negatives_per_query": negatives_per_query,
            "candidate_count_mean": float(np.mean([task["candidate_count"] for task in tasks])),
        },
        "methods": methods,
        "best_method": {
            "name": best_method[0],
            **best_method[1],
        },
        "hebbian_beats_lexical": bool(
            methods["raw_hebbian_context"]["hit_at_k"]
            > methods["lexical_text_overlap"]["hit_at_k"]
        ),
        "combined_beats_lexical": bool(
            methods["combined_lexical_hebbian"]["hit_at_k"]
            >= methods["lexical_text_overlap"]["hit_at_k"]
            and methods["combined_lexical_hebbian"]["mrr"]
            >= methods["lexical_text_overlap"]["mrr"]
        ),
        "sample_tasks": tasks[:5],
        "runtime_ms": (time.perf_counter() - started) * 1000.0,
        "limitations": [
            "repository_linking_proxy_not_generation",
            "positive_targets_are_same_repo_files_not_human_labeled_dependencies",
            "distractors_sampled_from_same_split_other_repositories",
            "global_hebbian_memory_built_from_train_split_only",
        ],
    }


def run_if7_trained_repository_linking_ranker(
    corpus_jsonl: str | Path,
    config: RepositoryLinkingRankerConfig | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    corpus_jsonl = Path(corpus_jsonl)
    config = config or RepositoryLinkingRankerConfig()
    sparse_config = SparseHebbianConfig(
        split=config.train_split,
        seed=config.seed,
        node_count=config.node_count,
        max_active_nodes=config.max_active_nodes,
        max_train_rows=config.max_memory_rows,
        min_text_bytes=config.min_text_bytes,
        cue_node_budget=config.cue_node_budget,
        target_node_budget=config.target_node_budget,
        learning_rate=config.hebbian_learning_rate,
    )
    train_rows, train_scanned = _sample_rows(corpus_jsonl, sparse_config)
    eval_rows, eval_scanned = _sample_rows(
        corpus_jsonl,
        SparseHebbianConfig(
            split=config.eval_split,
            seed=config.seed + 31,
            node_count=config.node_count,
            max_train_rows=1_000_000,
            min_text_bytes=config.min_text_bytes,
            cue_node_budget=config.cue_node_budget,
            target_node_budget=config.target_node_budget,
        ),
    )
    if not train_rows or not eval_rows:
        raise ValueError("IF7 trained repository ranker requires train and eval rows")

    label_counts: dict[int, Counter[str]] = {}
    train_patterns = _usable_patterns(train_rows, sparse_config, label_counts)
    memory = SparseHebbianMemory(sparse_config)
    for pattern in train_patterns:
        memory.observe(pattern.active_nodes)

    eval_patterns = _usable_patterns(
        eval_rows,
        SparseHebbianConfig(
            split=config.eval_split,
            seed=config.seed,
            node_count=config.node_count,
            max_active_nodes=config.max_active_nodes,
            min_text_bytes=config.min_text_bytes,
            cue_node_budget=config.cue_node_budget,
            target_node_budget=config.target_node_budget,
        ),
        label_counts,
    )
    train_specs = _repository_linking_specs(
        train_patterns,
        max_repositories=config.max_train_repositories,
        negatives_per_query=config.negatives_per_query,
        seed=config.seed,
    )
    eval_specs = _repository_linking_specs(
        eval_patterns,
        max_repositories=config.max_eval_repositories,
        negatives_per_query=config.negatives_per_query,
        seed=config.seed + 17,
    )
    if not train_specs or not eval_specs:
        raise ValueError("IF7 trained repository ranker produced no train or eval tasks")

    accelerator = _resolve_torch_accelerator(config.device)
    device = torch.device(accelerator.device)
    train_examples = _repository_ranker_examples(
        train_specs,
        memory=memory,
        include_hebbian=True,
    )
    no_hebbian_examples = _repository_ranker_examples(
        train_specs,
        memory=memory,
        include_hebbian=False,
    )
    if not train_examples:
        raise ValueError("IF7 trained repository ranker found no candidate examples")
    model = torch.nn.Linear(len(train_examples[0][0]), 1).to(device)
    no_hebbian_model = torch.nn.Linear(len(no_hebbian_examples[0][0]), 1).to(device)
    loss_history = _train_repository_ranker(model, train_examples, config, device)
    no_hebbian_loss_history = _train_repository_ranker(
        no_hebbian_model,
        no_hebbian_examples,
        config,
        device,
    )

    tasks = [
        _repository_linking_task_with_ranker(
            spec["query"],
            spec["candidates"],
            positives=spec["positives"],
            memory=memory,
            top_k=config.top_k,
            model=model,
            no_hebbian_model=no_hebbian_model,
            device=device,
        )
        for spec in eval_specs
    ]
    methods = {
        method: _repository_linking_summary(tasks, method)
        for method in [
            "path_role_overlap",
            "lexical_text_overlap",
            "raw_hebbian_context",
            "combined_lexical_hebbian",
            "trained_task_aware_ranker",
            "trained_no_hebbian_ranker",
        ]
    }
    best_method = max(methods.items(), key=lambda item: (item[1]["hit_at_k"], item[1]["mrr"]))
    label_bytes = sum(
        len(label.encode("utf-8")) + 8
        for labels in label_counts.values()
        for label in labels
    )
    trained = methods["trained_task_aware_ranker"]
    no_hebbian = methods["trained_no_hebbian_ranker"]
    lexical = methods["lexical_text_overlap"]
    raw = methods["raw_hebbian_context"]
    train_task_repositories = len({spec["query"].repo for spec in train_specs})
    eval_task_repositories = len({spec["query"].repo for spec in eval_specs})
    return {
        "benchmark_label": "if7_trained_repository_linking_ranker",
        "candidate_id": "IF7",
        "candidate_name": "Sparse Hebbian Task-Aware Repository Ranker",
        "corpus": {
            "path": str(corpus_jsonl),
            "train_split": config.train_split,
            "eval_split": config.eval_split,
            "train_rows_scanned": train_scanned,
            "eval_rows_scanned": eval_scanned,
            "train_rows_loaded": len(train_rows),
            "eval_rows_loaded": len(eval_rows),
            "train_repositories": len({pattern.repo for pattern in train_patterns}),
            "eval_repositories": len({pattern.repo for pattern in eval_patterns}),
            "train_task_repositories": train_task_repositories,
            "eval_task_repositories": eval_task_repositories,
        },
        "tasks": {
            "train_tasks": len(train_specs),
            "eval_tasks": len(tasks),
            "top_k": config.top_k,
            "negatives_per_query": config.negatives_per_query,
            "eval_candidate_count_mean": float(
                np.mean([task["candidate_count"] for task in tasks])
            ),
        },
        "training": {
            "train_patterns": len(train_patterns),
            "eval_patterns": len(eval_patterns),
            "train_tasks": len(train_specs),
            "candidate_examples": len(train_examples),
            "no_hebbian_candidate_examples": len(no_hebbian_examples),
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "device": str(device),
            "accelerator_backend": accelerator.backend,
            "requested_device": accelerator.requested_device,
            "rocm_available": accelerator.rocm_available,
            "rocm_runtime_version": accelerator.rocm_runtime_version,
        },
        "hebbian_memory": {
            "node_count": config.node_count,
            "max_active_nodes": config.max_active_nodes,
            "update_count": memory.update_count,
            "storage_bytes": memory.accounted_storage_bytes() + label_bytes,
            "label_index_bytes": label_bytes,
        },
        "models": {
            "task_aware_ranker": {
                "description": (
                    "Linear repository-linking ranker over path, lexical, and "
                    "Hebbian context features."
                ),
                "feature_names": _repository_ranker_feature_names(),
                "parameter_count": _parameter_count(model),
                "parameter_bytes": _parameter_bytes(model),
                "loss_history": loss_history,
            },
            "no_hebbian_ranker": {
                "description": (
                    "Ablation ranker trained on the same path and lexical features "
                    "with the Hebbian context feature zeroed."
                ),
                "feature_names": _repository_ranker_feature_names(),
                "parameter_count": _parameter_count(no_hebbian_model),
                "parameter_bytes": _parameter_bytes(no_hebbian_model),
                "loss_history": no_hebbian_loss_history,
            }
        },
        "methods": methods,
        "best_method": {
            "name": best_method[0],
            **best_method[1],
        },
        "trained_ranker_beats_lexical": bool(
            trained["hit_at_k"] > lexical["hit_at_k"]
            or (
                trained["hit_at_k"] >= lexical["hit_at_k"]
                and trained["mrr"] > lexical["mrr"]
            )
        ),
        "trained_ranker_beats_raw_hebbian": bool(
            trained["hit_at_k"] > raw["hit_at_k"]
            or (
                trained["hit_at_k"] >= raw["hit_at_k"]
                and trained["mrr"] >= raw["mrr"]
            )
        ),
        "trained_ranker_beats_no_hebbian": bool(
            trained["hit_at_k"] > no_hebbian["hit_at_k"]
            or (
                trained["hit_at_k"] >= no_hebbian["hit_at_k"]
                and trained["mrr"] > no_hebbian["mrr"]
            )
        ),
        "sample_tasks": tasks[:5],
        "runtime_ms": (time.perf_counter() - started) * 1000.0,
        "limitations": [
            "repository_linking_proxy_not_generation",
            "positive_targets_are_same_repo_files_not_human_labeled_dependencies",
            "linear_ranker_not_decoder_language_model",
            "global_hebbian_memory_built_from_train_split_only",
            "no_executable_code_generation_or_repair",
        ],
    }


def _sample_rows(
    corpus_jsonl: Path,
    config: SparseHebbianConfig,
) -> tuple[list[dict[str, Any]], int]:
    rng = random.Random(config.seed)
    rows: list[dict[str, Any]] = []
    seen = 0
    with corpus_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != config.split:
                continue
            if int(row.get("bytes", 0)) < config.min_text_bytes:
                continue
            if not row.get("repo") or not row.get("path") or not row.get("text"):
                continue
            seen += 1
            if len(rows) < config.max_train_rows:
                rows.append(row)
            else:
                replacement = rng.randrange(seen)
                if replacement < config.max_train_rows:
                    rows[replacement] = row
    return rows, seen


def _training_to_sparse_config(
    config: HebbianTrainingConfig,
    *,
    split: str | None = None,
    max_rows: int | None = None,
    seed: int | None = None,
) -> SparseHebbianConfig:
    return SparseHebbianConfig(
        split=split or config.split,
        seed=config.seed if seed is None else seed,
        node_count=config.node_count,
        max_active_nodes=config.max_active_nodes,
        max_train_rows=config.max_train_rows if max_rows is None else max_rows,
        max_eval_rows=config.max_validation_rows,
        recall_at_k=config.recall_at_k,
        min_text_bytes=config.min_text_bytes,
        cue_node_budget=config.cue_node_budget,
        target_node_budget=config.target_node_budget,
        learning_rate=config.hebbian_learning_rate,
    )


def _reranker_to_sparse_config(
    config: SparseHebbianRerankerConfig,
    *,
    split: str | None = None,
    max_rows: int | None = None,
    seed: int | None = None,
) -> SparseHebbianConfig:
    return SparseHebbianConfig(
        split=split or config.split,
        seed=config.seed if seed is None else seed,
        node_count=config.node_count,
        max_active_nodes=config.max_active_nodes,
        max_train_rows=config.max_train_rows if max_rows is None else max_rows,
        max_eval_rows=config.max_validation_patterns,
        recall_at_k=config.recall_at_k,
        min_text_bytes=config.min_text_bytes,
        cue_node_budget=config.cue_node_budget,
        target_node_budget=config.target_node_budget,
        learning_rate=config.hebbian_learning_rate,
    )


def _usable_patterns(
    rows: list[dict[str, Any]],
    config: SparseHebbianConfig,
    label_counts: dict[int, Counter[str]],
    *,
    max_patterns: int | None = None,
    text_window_bytes: int = 0,
    text_window_stride_bytes: int = 0,
) -> list[_RowPattern]:
    patterns: list[_RowPattern] = []
    for row in rows:
        row_patterns = _row_patterns_from_windows(
            row,
            config,
            label_counts,
            text_window_bytes=text_window_bytes,
            text_window_stride_bytes=text_window_stride_bytes,
        )
        for pattern in row_patterns:
            patterns.append(pattern)
            if max_patterns is not None and len(patterns) >= max_patterns:
                break
        if max_patterns is not None and len(patterns) >= max_patterns:
            break
    return [
        pattern
        for pattern in patterns
        if pattern.cue_nodes and pattern.target_nodes and len(pattern.active_nodes) >= 2
    ]


def _row_patterns_from_windows(
    row: dict[str, Any],
    config: SparseHebbianConfig,
    label_counts: dict[int, Counter[str]],
    *,
    text_window_bytes: int,
    text_window_stride_bytes: int,
) -> list[_RowPattern]:
    if text_window_bytes <= 0:
        return [_row_pattern(row, config, label_counts)]
    text = str(row.get("text", ""))
    encoded = text.encode("utf-8", errors="ignore")
    if not encoded:
        return []
    stride = text_window_stride_bytes if text_window_stride_bytes > 0 else text_window_bytes
    patterns: list[_RowPattern] = []
    for start in range(0, len(encoded), stride):
        chunk_bytes = encoded[start : start + text_window_bytes]
        if len(chunk_bytes) < min(32, text_window_bytes):
            continue
        chunk_row = dict(row)
        chunk_row["text"] = chunk_bytes.decode("utf-8", errors="ignore")
        chunk_row["row_sha256"] = f"{row.get('row_sha256', '')}:chunk:{start}"
        patterns.append(_row_pattern(chunk_row, config, label_counts))
    return patterns


def _sample_eval_patterns(
    patterns: list[_RowPattern],
    config: SparseHebbianConfig,
) -> list[_RowPattern]:
    rng = random.Random(config.seed + 1)
    candidates = list(patterns)
    rng.shuffle(candidates)
    return candidates[: min(config.max_eval_rows, len(candidates))]


def _row_pattern(
    row: dict[str, Any],
    config: SparseHebbianConfig,
    label_counts: dict[int, Counter[str]],
) -> _RowPattern:
    text = str(row.get("text", ""))
    cue_labels = _cue_labels(row)
    target_labels = _target_labels(text)
    cue_nodes = _labels_to_nodes(
        cue_labels,
        config.node_count,
        config.cue_node_budget,
        label_counts,
    )
    target_nodes = _labels_to_nodes(
        target_labels,
        config.node_count,
        config.target_node_budget,
        label_counts,
    )
    active_nodes = tuple(dict.fromkeys((*cue_nodes, *target_nodes)))[
        : config.max_active_nodes
    ]
    return _RowPattern(
        repo=str(row.get("repo", "")),
        path=str(row.get("path", "")),
        row_sha256=str(row.get("row_sha256", "")),
        cue_nodes=cue_nodes,
        target_nodes=target_nodes,
        active_nodes=active_nodes,
        lexical_tokens=tuple(_row_lexical_tokens(row, text)),
    )


def _batch_tensors(
    patterns: list[_RowPattern],
    memory: SparseHebbianMemory,
    indices: list[int],
    device: torch.device,
    *,
    include_hebbian: bool,
) -> dict[str, torch.Tensor]:
    cue = np.zeros((len(indices), memory.config.node_count), dtype=np.float32)
    target = np.zeros((len(indices), memory.config.node_count), dtype=np.float32)
    hebbian = (
        np.zeros((len(indices), memory.config.node_count), dtype=np.float32)
        if include_hebbian
        else None
    )
    for batch_index, pattern_index in enumerate(indices):
        pattern = patterns[pattern_index]
        cue[batch_index, list(pattern.cue_nodes)] = 1.0
        target[batch_index, list(pattern.target_nodes)] = 1.0
        if hebbian is not None:
            hebbian[batch_index] = memory.completion_scores(pattern.cue_nodes)
    cue_tensor = torch.from_numpy(cue).to(device)
    target_tensor = torch.from_numpy(target).to(device)
    if hebbian is None:
        feature_tensor = cue_tensor
    else:
        hebbian_tensor = torch.from_numpy(hebbian).to(device)
        feature_tensor = torch.cat([cue_tensor, hebbian_tensor], dim=1)
    return {
        "features": feature_tensor,
        "target": target_tensor,
    }


def _train_linear_probe(
    model: torch.nn.Module,
    patterns: list[_RowPattern],
    memory: SparseHebbianMemory,
    config: HebbianTrainingConfig,
    device: torch.device,
    *,
    include_hebbian: bool,
) -> list[float]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    history: list[float] = []
    row_count = len(patterns)
    for _epoch in range(config.epochs):
        permutation = torch.randperm(row_count, generator=generator).tolist()
        total_loss = 0.0
        seen = 0
        model.train()
        for start in range(0, row_count, config.batch_size):
            batch_index = permutation[start : start + config.batch_size]
            batch = _batch_tensors(
                patterns,
                memory,
                batch_index,
                device,
                include_hebbian=include_hebbian,
            )
            batch_features = batch["features"]
            batch_targets = batch["target"]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_features), batch_targets)
            loss.backward()
            optimizer.step()
            batch_size = int(batch_features.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            seen += batch_size
        history.append(total_loss / max(1, seen))
    return history


def _evaluate_linear_probe(
    model: torch.nn.Module,
    patterns: list[_RowPattern],
    memory: SparseHebbianMemory,
    device: torch.device,
    *,
    include_hebbian: bool,
    recall_at_k: int,
) -> dict[str, float]:
    model.eval()
    all_hits = []
    all_coverages = []
    all_reciprocal_ranks = []
    total_loss = 0.0
    seen = 0
    with torch.no_grad():
        for start in range(0, len(patterns), 256):
            indices = list(range(start, min(start + 256, len(patterns))))
            batch = _batch_tensors(
                patterns,
                memory,
                indices,
                device,
                include_hebbian=include_hebbian,
            )
            logits = model(batch["features"])
            targets = batch["target"]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
            topk = torch.topk(
                logits,
                k=min(recall_at_k, logits.shape[1]),
                dim=1,
            ).indices.cpu()
            target_cpu = targets.cpu()
            for row_index, ranked_tensor in enumerate(topk):
                ranked = [int(value) for value in ranked_tensor.tolist()]
                target = set(
                    torch.nonzero(target_cpu[row_index], as_tuple=False).flatten().tolist()
                )
                all_hits.append(_hit_at_k(ranked, target))
                all_coverages.append(_coverage_at_k(ranked, target))
                all_reciprocal_ranks.append(_reciprocal_rank(ranked, target))
            batch_size = len(indices)
            total_loss += float(loss.detach().cpu()) * batch_size
            seen += batch_size
    return {
        "loss": total_loss / max(1, seen),
        "hit_at_k": float(np.mean(all_hits)),
        "coverage_at_k": float(np.mean(all_coverages)),
        "mrr": float(np.mean(all_reciprocal_ranks)),
        "task_count": float(len(patterns)),
    }


def _evaluate_raw_hebbian(
    patterns: list[_RowPattern],
    *,
    memory: SparseHebbianMemory,
    recall_at_k: int,
) -> dict[str, float]:
    rows = [
        {
            "hit": _hit_at_k(
                memory.complete(pattern.cue_nodes, recall_at_k),
                set(pattern.target_nodes),
            ),
            "coverage": _coverage_at_k(
                memory.complete(pattern.cue_nodes, recall_at_k),
                set(pattern.target_nodes),
            ),
            "mrr": _reciprocal_rank(
                memory.complete(pattern.cue_nodes, recall_at_k),
                set(pattern.target_nodes),
            ),
        }
        for pattern in patterns
    ]
    return {
        "hit_at_k": float(np.mean([row["hit"] for row in rows])),
        "coverage_at_k": float(np.mean([row["coverage"] for row in rows])),
        "mrr": float(np.mean([row["mrr"] for row in rows])),
        "task_count": float(len(rows)),
    }


def _candidate_examples(
    patterns: list[_RowPattern],
    *,
    memory: SparseHebbianMemory,
    candidate_count: int,
    max_examples: int,
    include_targets: bool,
    node_priors: dict[str, np.ndarray],
) -> list[tuple[list[float], float]]:
    examples: list[tuple[list[float], float]] = []
    max_frequency = max(float(np.max(memory.firing_counts)), 1.0)
    for pattern in patterns:
        candidate_nodes = memory.complete(pattern.cue_nodes, candidate_count)
        if include_targets:
            candidate_nodes = list(dict.fromkeys([*candidate_nodes, *pattern.target_nodes]))
        scores = memory.completion_scores(pattern.cue_nodes)
        target = set(pattern.target_nodes)
        cue = set(pattern.cue_nodes)
        for rank, node in enumerate(candidate_nodes, start=1):
            examples.append(
                (
                    _candidate_features(
                        node,
                        rank=rank,
                        scores=scores,
                        memory=memory,
                        max_frequency=max_frequency,
                        cue=cue,
                        node_priors=node_priors,
                    ),
                    float(node in target),
                )
            )
            if len(examples) >= max_examples:
                return examples
    return examples


def _node_priors(patterns: list[_RowPattern], node_count: int) -> dict[str, np.ndarray]:
    cue_counts = np.zeros(node_count, dtype=np.float32)
    target_counts = np.zeros(node_count, dtype=np.float32)
    for pattern in patterns:
        cue_counts[list(pattern.cue_nodes)] += 1.0
        target_counts[list(pattern.target_nodes)] += 1.0
    cue_max = max(float(np.max(cue_counts)), 1.0)
    target_max = max(float(np.max(target_counts)), 1.0)
    return {
        "cue": np.log1p(cue_counts) / np.log1p(cue_max),
        "target": np.log1p(target_counts) / np.log1p(target_max),
    }


def _candidate_features(
    node: int,
    *,
    rank: int,
    scores: np.ndarray,
    memory: SparseHebbianMemory,
    max_frequency: float,
    cue: set[int],
    node_priors: dict[str, np.ndarray],
) -> list[float]:
    target_prior = float(node_priors["target"][node])
    cue_prior = float(node_priors["cue"][node])
    return [
        float(scores[node]),
        1.0 / max(1.0, float(rank)),
        float(np.log1p(memory.firing_counts[node]) / np.log1p(max_frequency)),
        float(node in cue),
        target_prior,
        cue_prior,
        target_prior / max(1e-6, cue_prior + target_prior),
        1.0,
    ]


def _train_candidate_reranker(
    model: torch.nn.Module,
    examples: list[tuple[list[float], float]],
    config: SparseHebbianRerankerConfig,
    device: torch.device,
) -> list[float]:
    features = torch.tensor(
        [example[0] for example in examples],
        dtype=torch.float32,
        device=device,
    )
    labels = torch.tensor(
        [[example[1]] for example in examples],
        dtype=torch.float32,
        device=device,
    )
    positives = float(labels.sum().detach().cpu())
    negatives = float(labels.numel() - positives)
    pos_weight = torch.tensor(
        [max(1.0, negatives / max(1.0, positives))],
        dtype=torch.float32,
        device=device,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    history: list[float] = []
    row_count = features.shape[0]
    for _epoch in range(config.epochs):
        permutation = torch.randperm(row_count, generator=generator).to(device)
        total_loss = 0.0
        seen = 0
        model.train()
        for start in range(0, row_count, config.batch_size):
            batch_index = permutation[start : start + config.batch_size]
            batch_features = features[batch_index]
            batch_labels = labels[batch_index]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            batch_size = int(batch_features.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            seen += batch_size
        history.append(total_loss / max(1, seen))
    return history


def _evaluate_candidate_reranker(
    model: torch.nn.Module,
    patterns: list[_RowPattern],
    *,
    memory: SparseHebbianMemory,
    candidate_count: int,
    recall_at_k: int,
    device: torch.device,
    node_priors: dict[str, np.ndarray],
) -> dict[str, float]:
    model.eval()
    hits: list[float] = []
    coverages: list[float] = []
    reciprocal_ranks: list[float] = []
    candidate_hit: list[float] = []
    max_frequency = max(float(np.max(memory.firing_counts)), 1.0)
    with torch.no_grad():
        for pattern in patterns:
            target = set(pattern.target_nodes)
            cue = set(pattern.cue_nodes)
            candidate_nodes = memory.complete(pattern.cue_nodes, candidate_count)
            scores = memory.completion_scores(pattern.cue_nodes)
            if candidate_nodes:
                features = torch.tensor(
                    [
                        _candidate_features(
                            node,
                            rank=rank,
                            scores=scores,
                            memory=memory,
                            max_frequency=max_frequency,
                            cue=cue,
                            node_priors=node_priors,
                        )
                        for rank, node in enumerate(candidate_nodes, start=1)
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                logits = model(features).squeeze(-1).detach().cpu().numpy()
                order = np.argsort(-logits)
                ranked = [candidate_nodes[int(index)] for index in order[:recall_at_k]]
            else:
                ranked = []
            hits.append(_hit_at_k(ranked, target))
            coverages.append(_coverage_at_k(ranked, target))
            reciprocal_ranks.append(_reciprocal_rank(ranked, target))
            candidate_hit.append(_hit_at_k(candidate_nodes, target))
    return {
        "hit_at_k": float(np.mean(hits)),
        "coverage_at_k": float(np.mean(coverages)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "candidate_hit_rate": float(np.mean(candidate_hit)),
        "task_count": float(len(patterns)),
    }


def _evaluate_candidate_ceiling(
    patterns: list[_RowPattern],
    *,
    memory: SparseHebbianMemory,
    candidate_count: int,
    recall_at_k: int,
) -> dict[str, float]:
    hits: list[float] = []
    coverages: list[float] = []
    reciprocal_ranks: list[float] = []
    for pattern in patterns:
        target = set(pattern.target_nodes)
        candidates = memory.complete(pattern.cue_nodes, candidate_count)
        positives = [node for node in candidates if node in target]
        negatives = [node for node in candidates if node not in target]
        ranked = [*positives, *negatives][:recall_at_k]
        hits.append(_hit_at_k(ranked, target))
        coverages.append(_coverage_at_k(ranked, target))
        reciprocal_ranks.append(_reciprocal_rank(ranked, target))
    return {
        "hit_at_k": float(np.mean(hits)),
        "coverage_at_k": float(np.mean(coverages)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "task_count": float(len(patterns)),
    }


def _patterns_by_repo(patterns: list[_RowPattern]) -> dict[str, list[_RowPattern]]:
    grouped: dict[str, list[_RowPattern]] = defaultdict(list)
    for pattern in patterns:
        grouped[pattern.repo].append(pattern)
    return dict(grouped)


def _repository_linking_task(
    query: _RowPattern,
    candidates: list[_RowPattern],
    *,
    positives: set[str],
    memory: SparseHebbianMemory,
    top_k: int,
) -> dict[str, Any]:
    rankings = {
        "path_role_overlap": _rank_repository_candidates(
            query,
            candidates,
            memory=memory,
            method="path_role_overlap",
        ),
        "lexical_text_overlap": _rank_repository_candidates(
            query,
            candidates,
            memory=memory,
            method="lexical_text_overlap",
        ),
        "raw_hebbian_context": _rank_repository_candidates(
            query,
            candidates,
            memory=memory,
            method="raw_hebbian_context",
        ),
        "combined_lexical_hebbian": _rank_repository_candidates(
            query,
            candidates,
            memory=memory,
            method="combined_lexical_hebbian",
        ),
    }
    method_results = {
        method: _repository_ranking_metrics(ranked, positives, top_k=top_k)
        for method, ranked in rankings.items()
    }
    return {
        "repo": query.repo,
        "query_path": query.path,
        "positive_count": len(positives),
        "candidate_count": len(candidates),
        "method_results": method_results,
        "top_paths": {
            method: [path for path, _score in ranked[:top_k]]
            for method, ranked in rankings.items()
        },
    }


def _repository_linking_task_with_ranker(
    query: _RowPattern,
    candidates: list[_RowPattern],
    *,
    positives: set[str],
    memory: SparseHebbianMemory,
    top_k: int,
    model: torch.nn.Module,
    no_hebbian_model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    rankings = {
        "path_role_overlap": _rank_repository_candidates(
            query,
            candidates,
            memory=memory,
            method="path_role_overlap",
        ),
        "lexical_text_overlap": _rank_repository_candidates(
            query,
            candidates,
            memory=memory,
            method="lexical_text_overlap",
        ),
        "raw_hebbian_context": _rank_repository_candidates(
            query,
            candidates,
            memory=memory,
            method="raw_hebbian_context",
        ),
        "combined_lexical_hebbian": _rank_repository_candidates(
            query,
            candidates,
            memory=memory,
            method="combined_lexical_hebbian",
        ),
        "trained_task_aware_ranker": _rank_repository_candidates_with_model(
            query,
            candidates,
            memory=memory,
            model=model,
            device=device,
            include_hebbian=True,
        ),
        "trained_no_hebbian_ranker": _rank_repository_candidates_with_model(
            query,
            candidates,
            memory=memory,
            model=no_hebbian_model,
            device=device,
            include_hebbian=False,
        ),
    }
    method_results = {
        method: _repository_ranking_metrics(ranked, positives, top_k=top_k)
        for method, ranked in rankings.items()
    }
    return {
        "repo": query.repo,
        "query_path": query.path,
        "positive_count": len(positives),
        "candidate_count": len(candidates),
        "method_results": method_results,
        "top_paths": {
            method: [path for path, _score in ranked[:top_k]]
            for method, ranked in rankings.items()
        },
    }


def _repository_linking_specs(
    patterns: list[_RowPattern],
    *,
    max_repositories: int,
    negatives_per_query: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped = _patterns_by_repo(patterns)
    eligible_repos = sorted(repo for repo, rows in grouped.items() if len(rows) >= 2)
    rng = random.Random(seed)
    rng.shuffle(eligible_repos)
    selected_repos = eligible_repos[:max_repositories]
    distractors = [pattern for pattern in patterns if pattern.repo not in selected_repos]
    if not distractors:
        distractors = patterns
    specs: list[dict[str, Any]] = []
    for repo in selected_repos:
        repo_patterns = sorted(grouped[repo], key=lambda pattern: pattern.path)
        for query in repo_patterns:
            positives = [pattern for pattern in repo_patterns if pattern.path != query.path]
            negatives = [
                pattern
                for pattern in rng.sample(
                    distractors,
                    k=min(negatives_per_query, len(distractors)),
                )
                if pattern.repo != query.repo
            ]
            candidates = [*positives, *negatives]
            if not candidates:
                continue
            specs.append(
                {
                    "query": query,
                    "candidates": candidates,
                    "positives": {pattern.path for pattern in positives},
                }
            )
    return specs


def _rank_repository_candidates(
    query: _RowPattern,
    candidates: list[_RowPattern],
    *,
    memory: SparseHebbianMemory,
    method: str,
) -> list[tuple[str, float]]:
    query_path_tokens = _path_tokens(query.path)
    query_lexical = _pattern_lexical_tokens(query)
    hebbian_scores = memory.completion_scores(query.cue_nodes)
    scored: list[tuple[str, float]] = []
    for candidate in candidates:
        path_score = _counter_overlap(query_path_tokens, _path_tokens(candidate.path))
        lexical_score = _counter_overlap(query_lexical, _pattern_lexical_tokens(candidate))
        hebbian_score = _candidate_pattern_score(candidate, hebbian_scores)
        if method == "path_role_overlap":
            score = path_score
        elif method == "lexical_text_overlap":
            score = lexical_score
        elif method == "raw_hebbian_context":
            score = hebbian_score
        elif method == "combined_lexical_hebbian":
            score = _safe_norm(lexical_score) + _safe_norm(hebbian_score)
        else:
            raise ValueError(f"unknown repository linking method: {method}")
        scored.append((candidate.path, float(score)))
    return sorted(scored, key=lambda item: (-item[1], item[0]))


def _rank_repository_candidates_with_model(
    query: _RowPattern,
    candidates: list[_RowPattern],
    *,
    memory: SparseHebbianMemory,
    model: torch.nn.Module,
    device: torch.device,
    include_hebbian: bool,
) -> list[tuple[str, float]]:
    if not candidates:
        return []
    features = torch.tensor(
        [
            _repository_ranker_features(
                query,
                candidate,
                memory=memory,
                include_hebbian=include_hebbian,
            )
            for candidate in candidates
        ],
        dtype=torch.float32,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        logits = model(features).squeeze(-1).detach().cpu().numpy()
    scored = [
        (candidate.path, float(logits[index]))
        for index, candidate in enumerate(candidates)
    ]
    return sorted(scored, key=lambda item: (-item[1], item[0]))


def _repository_ranker_examples(
    specs: list[dict[str, Any]],
    *,
    memory: SparseHebbianMemory,
    include_hebbian: bool,
) -> list[tuple[list[float], float]]:
    examples: list[tuple[list[float], float]] = []
    for spec in specs:
        query = spec["query"]
        positives = spec["positives"]
        for candidate in spec["candidates"]:
            examples.append(
                (
                    _repository_ranker_features(
                        query,
                        candidate,
                        memory=memory,
                        include_hebbian=include_hebbian,
                    ),
                    float(candidate.path in positives),
                )
            )
    return examples


def _repository_ranker_features(
    query: _RowPattern,
    candidate: _RowPattern,
    *,
    memory: SparseHebbianMemory,
    include_hebbian: bool = True,
) -> list[float]:
    query_path_tokens = _path_tokens(query.path)
    candidate_path_tokens = _path_tokens(candidate.path)
    query_lexical = _pattern_lexical_tokens(query)
    candidate_lexical = _pattern_lexical_tokens(candidate)
    hebbian_scores = memory.completion_scores(query.cue_nodes)
    path_score = _counter_overlap(query_path_tokens, candidate_path_tokens)
    lexical_score = _counter_overlap(query_lexical, candidate_lexical)
    hebbian_score = (
        _candidate_pattern_score(candidate, hebbian_scores)
        if include_hebbian
        else 0.0
    )
    hebbian_edge_score = (
        _candidate_pair_edge_score(query, candidate, memory=memory)
        if include_hebbian
        else 0.0
    )
    query_ext = Path(query.path).suffix.lower()
    candidate_ext = Path(candidate.path).suffix.lower()
    same_ext = float(bool(query_ext) and query_ext == candidate_ext)
    return [
        _safe_norm(path_score),
        _safe_norm(lexical_score),
        float(hebbian_score),
        float(hebbian_edge_score),
        same_ext,
        1.0,
    ]


def _repository_ranker_feature_names() -> list[str]:
    return [
        "path_token_overlap_norm",
        "lexical_text_overlap_norm",
        "hebbian_candidate_context_score",
        "hebbian_pair_edge_score",
        "same_file_extension",
        "bias",
    ]


def _candidate_pair_edge_score(
    query: _RowPattern,
    candidate: _RowPattern,
    *,
    memory: SparseHebbianMemory,
) -> float:
    left = tuple(dict.fromkeys(query.active_nodes))
    right = tuple(dict.fromkeys(candidate.active_nodes))
    if not left or not right:
        return 0.0
    weights = memory.weights[np.ix_(np.asarray(left), np.asarray(right))]
    max_weight = max(float(np.max(memory.weights)), 1.0)
    return float(np.mean(weights) / max_weight)


def _train_repository_ranker(
    model: torch.nn.Module,
    examples: list[tuple[list[float], float]],
    config: RepositoryLinkingRankerConfig,
    device: torch.device,
) -> list[float]:
    features = torch.tensor(
        [example[0] for example in examples],
        dtype=torch.float32,
        device=device,
    )
    labels = torch.tensor(
        [[example[1]] for example in examples],
        dtype=torch.float32,
        device=device,
    )
    positives = float(labels.sum().detach().cpu())
    negatives = float(labels.numel() - positives)
    pos_weight = torch.tensor(
        [max(1.0, negatives / max(1.0, positives))],
        dtype=torch.float32,
        device=device,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    history: list[float] = []
    row_count = int(features.shape[0])
    for _epoch in range(config.epochs):
        permutation = torch.randperm(row_count, generator=generator).to(device)
        total_loss = 0.0
        seen = 0
        model.train()
        for start in range(0, row_count, config.batch_size):
            batch_index = permutation[start : start + config.batch_size]
            batch_features = features[batch_index]
            batch_labels = labels[batch_index]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            batch_size = int(batch_features.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            seen += batch_size
        history.append(total_loss / max(1, seen))
    return history


def _repository_ranking_metrics(
    ranked: list[tuple[str, float]],
    positives: set[str],
    *,
    top_k: int,
) -> dict[str, float]:
    ranked_paths = [path for path, _score in ranked]
    return {
        "hit_at_k": _hit_at_k_path(ranked_paths[:top_k], positives),
        "coverage_at_k": _coverage_at_k_path(ranked_paths[:top_k], positives),
        "mrr": _reciprocal_rank_path(ranked_paths, positives),
    }


def _repository_linking_summary(tasks: list[dict[str, Any]], method: str) -> dict[str, float]:
    rows = [task["method_results"][method] for task in tasks]
    return {
        "hit_at_k": float(np.mean([row["hit_at_k"] for row in rows])),
        "coverage_at_k": float(np.mean([row["coverage_at_k"] for row in rows])),
        "mrr": float(np.mean([row["mrr"] for row in rows])),
        "task_count": float(len(rows)),
    }


def _path_tokens(path: str) -> Counter[str]:
    return Counter(_split_words(path))


def _pattern_lexical_tokens(pattern: _RowPattern) -> Counter[str]:
    return Counter([*pattern.lexical_tokens, *_split_words(pattern.path)])


def _row_lexical_tokens(row: dict[str, Any], text: str) -> list[str]:
    tokens: list[str] = []
    tokens.extend(_split_words(str(row.get("path", ""))))
    for identifier in _IDENTIFIER_RE.findall(text):
        normalized = identifier.lower().strip("$")
        if len(normalized) < 3 or normalized in _JS_STOPWORDS:
            continue
        tokens.append(normalized)
        tokens.extend(part for part in _split_words(identifier) if len(part) >= 3)
        if len(tokens) >= 256:
            break
    return tokens[:256]


def _node_surrogate_tokens(nodes: tuple[int, ...]) -> list[str]:
    return [f"node_{node}" for node in nodes]


def _candidate_pattern_score(candidate: _RowPattern, hebbian_scores: np.ndarray) -> float:
    nodes = tuple(dict.fromkeys((*candidate.cue_nodes, *candidate.target_nodes)))
    if not nodes:
        return 0.0
    return float(np.mean(hebbian_scores[list(nodes)]))


def _counter_overlap(left: Counter[str], right: Counter[str]) -> float:
    return float(sum((left & right).values()))


def _safe_norm(value: float) -> float:
    return float(value / (1.0 + abs(value)))


def _hit_at_k_path(ranked: list[str], positives: set[str]) -> float:
    return float(any(path in positives for path in ranked))


def _coverage_at_k_path(ranked: list[str], positives: set[str]) -> float:
    if not positives:
        return 0.0
    return len(set(ranked) & positives) / len(positives)


def _reciprocal_rank_path(ranked: list[str], positives: set[str]) -> float:
    for index, path in enumerate(ranked, start=1):
        if path in positives:
            return 1.0 / index
    return 0.0


def _parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _parameter_bytes(model: torch.nn.Module) -> int:
    return int(
        sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    )


def _cue_labels(row: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    repo = str(row.get("repo", ""))
    path = str(row.get("path", ""))
    language = str(row.get("language", ""))
    if repo:
        labels.append(f"repo:{repo.lower()}")
        labels.extend(f"repo_part:{part}" for part in _split_words(repo))
    if language:
        labels.append(f"lang:{language.lower()}")
    for role in row.get("content_roles", []) or []:
        labels.append(f"role:{str(role).lower()}")
    if path:
        path_obj = Path(path)
        labels.append(f"ext:{path_obj.suffix.lower()}")
        labels.extend(f"path:{part}" for part in _split_words(path_obj.as_posix()))
        labels.extend(f"stem:{part}" for part in _split_words(path_obj.stem))
        if path_obj.parent.as_posix() not in ("", "."):
            labels.extend(f"dir:{part}" for part in _split_words(path_obj.parent.as_posix()))
    return labels


def _target_labels(text: str) -> list[str]:
    counts: Counter[str] = Counter()
    for match in _IMPORT_RE.finditer(text):
        module = next(group for group in match.groups() if group)
        for part in _split_words(module):
            counts[f"import:{part}"] += 3
    for identifier in _IDENTIFIER_RE.findall(text):
        normalized = identifier.lower().strip("$")
        if len(normalized) < 3 or normalized in _JS_STOPWORDS:
            continue
        counts[f"ident:{normalized}"] += 1
        for part in _split_words(identifier):
            if len(part) >= 3 and part not in _JS_STOPWORDS:
                counts[f"ident_part:{part}"] += 1
    return [
        label
        for label, _count in sorted(
            counts.items(),
            key=lambda item: (-item[1], _stable_int(item[0])),
        )
    ]


def _split_words(text: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    parts = re.split(r"[^A-Za-z0-9_$]+", spaced)
    return [part.lower().strip("_$") for part in parts if part.strip("_$")]


def _labels_to_nodes(
    labels: list[str],
    node_count: int,
    budget: int,
    label_counts: dict[int, Counter[str]],
) -> tuple[int, ...]:
    nodes: list[int] = []
    seen: set[int] = set()
    for label in labels:
        node = _stable_int(label) % node_count
        label_counts.setdefault(node, Counter())[label] += 1
        if node in seen:
            continue
        seen.add(node)
        nodes.append(node)
        if len(nodes) >= budget:
            break
    return tuple(nodes)


def _score_pattern(
    pattern: _RowPattern,
    *,
    memory: SparseHebbianMemory,
    random_memory: SparseHebbianMemory,
    frequency_scores: np.ndarray,
    recall_at_k: int,
) -> dict[str, Any]:
    target = set(pattern.target_nodes)
    hebbian = memory.complete(pattern.cue_nodes, recall_at_k)
    random_ranked = random_memory.complete(pattern.cue_nodes, recall_at_k)
    frequency_ranked = _rank_frequency(frequency_scores, pattern.cue_nodes, recall_at_k)
    return {
        "repo": pattern.repo,
        "path": pattern.path,
        "row_sha256": pattern.row_sha256,
        "cue_count": len(pattern.cue_nodes),
        "target_count": len(pattern.target_nodes),
        "hebbian_hit_at_k": _hit_at_k(hebbian, target),
        "random_hit_at_k": _hit_at_k(random_ranked, target),
        "frequency_hit_at_k": _hit_at_k(frequency_ranked, target),
        "hebbian_coverage_at_k": _coverage_at_k(hebbian, target),
        "random_coverage_at_k": _coverage_at_k(random_ranked, target),
        "frequency_coverage_at_k": _coverage_at_k(frequency_ranked, target),
        "hebbian_reciprocal_rank": _reciprocal_rank(hebbian, target),
        "random_reciprocal_rank": _reciprocal_rank(random_ranked, target),
        "frequency_reciprocal_rank": _reciprocal_rank(frequency_ranked, target),
    }


def _rank_frequency(
    scores: np.ndarray,
    cue_nodes: tuple[int, ...],
    top_k: int,
) -> list[int]:
    adjusted = scores.copy()
    adjusted[np.asarray(cue_nodes, dtype=np.int64)] = -np.inf
    candidate = np.argpartition(-adjusted, min(top_k, len(adjusted) - 1))[:top_k]
    ranked = candidate[np.argsort(-adjusted[candidate])]
    return [int(node) for node in ranked[:top_k] if np.isfinite(adjusted[node])]


def _hit_at_k(ranked: list[int], target: set[int]) -> float:
    return float(any(node in target for node in ranked))


def _coverage_at_k(ranked: list[int], target: set[int]) -> float:
    if not target:
        return 0.0
    return len(set(ranked) & target) / len(target)


def _reciprocal_rank(ranked: list[int], target: set[int]) -> float:
    for index, node in enumerate(ranked, start=1):
        if node in target:
            return 1.0 / index
    return 0.0


def _final_summary(task_results: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "task_count": float(len(task_results)),
        "hebbian_hit_at_k": _mean(task_results, "hebbian_hit_at_k"),
        "random_hit_at_k": _mean(task_results, "random_hit_at_k"),
        "frequency_hit_at_k": _mean(task_results, "frequency_hit_at_k"),
        "hebbian_coverage_at_k": _mean(task_results, "hebbian_coverage_at_k"),
        "random_coverage_at_k": _mean(task_results, "random_coverage_at_k"),
        "frequency_coverage_at_k": _mean(task_results, "frequency_coverage_at_k"),
        "hebbian_mrr": _mean(task_results, "hebbian_reciprocal_rank"),
        "random_mrr": _mean(task_results, "random_reciprocal_rank"),
        "frequency_mrr": _mean(task_results, "frequency_reciprocal_rank"),
    }


def _sample_predictions(
    patterns: list[_RowPattern],
    *,
    memory: SparseHebbianMemory,
    label_counts: dict[int, Counter[str]],
    recall_at_k: int,
) -> list[dict[str, Any]]:
    samples = []
    for pattern in patterns[:5]:
        ranked = memory.complete(pattern.cue_nodes, recall_at_k)
        target = set(pattern.target_nodes)
        samples.append(
            {
                "repo": pattern.repo,
                "path": pattern.path,
                "cue_node_count": len(pattern.cue_nodes),
                "target_node_count": len(pattern.target_nodes),
                "hit": any(node in target for node in ranked),
                "top_predicted_labels": [
                    _node_label(node, label_counts) for node in ranked[:8]
                ],
                "target_labels_sample": [
                    _node_label(node, label_counts) for node in pattern.target_nodes[:8]
                ],
            }
        )
    return samples


def _node_label(node: int, label_counts: dict[int, Counter[str]]) -> str:
    labels = label_counts.get(node)
    if not labels:
        return f"node:{node}"
    return labels.most_common(1)[0][0]


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(row[key]) for row in rows) / len(rows))


def _stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
