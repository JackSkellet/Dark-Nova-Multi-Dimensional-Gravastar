from __future__ import annotations

import hashlib
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from weightlab.repo_chronology import (
    _changed_files,
    _commits,
    _corpus_at,
    _git,
    _rank_files,
    _structured_boosts,
    _subject,
    _tokens,
    _topk_hit,
)

FEATURE_DIM = 256


def run_if4_fast_repo_adaptation_probe(
    repo_path: str | Path,
    max_commits: int = 12,
    top_k: int = 5,
    consolidation_interval: int = 3,
) -> dict[str, Any]:
    repo_path = Path(repo_path)
    start = time.perf_counter()
    commits = _commits(repo_path, max_commits=max_commits)
    if len(commits) < 3:
        raise ValueError("IF4 fast repository adaptation requires at least three commits")

    steps: list[dict[str, Any]] = []
    update_ms: list[float] = []

    for index in range(1, len(commits)):
        exposed_commits = commits[:index]
        exposed_commit = exposed_commits[-1]
        future_commit = commits[index]
        future_targets = _changed_files(repo_path, future_commit)
        if not future_targets:
            continue

        corpus = _corpus_at(repo_path, exposed_commit)
        if not corpus:
            continue

        future_query = _subject(repo_path, future_commit)
        paraphrase_query = _paraphrase_query(future_query, future_targets)

        update_started = time.perf_counter()
        replay_boosts = _replay_adapter_boosts(repo_path, exposed_commits, future_query)
        consolidated_boosts = _periodic_consolidation_boosts(
            repo_path,
            exposed_commits,
            future_query,
            consolidation_interval=consolidation_interval,
        )
        fast_state = _build_fast_weight_state(repo_path, exposed_commits, corpus)
        update_ms.append((time.perf_counter() - update_started) * 1000.0)

        updated_ranked = _rank_files(future_query, corpus)
        structured_ranked = _rank_files(
            future_query,
            corpus,
            _structured_symbol_graph_boosts(repo_path, exposed_commits, future_query),
        )
        replay_ranked = _rank_files(future_query, corpus, replay_boosts)
        fast_ranked = _rank_fast_weight_files(future_query, fast_state)
        combined_ranked = _rank_combined_files(future_query, corpus, fast_state)
        consolidated_ranked = _rank_files(future_query, corpus, consolidated_boosts)

        combined_paraphrase = _rank_combined_files(paraphrase_query, corpus, fast_state)

        steps.append(
            {
                "step": index,
                "exposed_commit": exposed_commit[:12],
                "future_commit": future_commit[:12],
                "future_commit_not_in_memory": future_commit not in exposed_commits,
                "future_changed_files": future_targets,
                "query": future_query,
                "paraphrase_query": paraphrase_query,
                "candidate_files": len(corpus),
                "updated_retrieval_future_topk_hit": _topk_hit(
                    updated_ranked,
                    future_targets,
                    top_k,
                ),
                "structured_symbol_graph_memory_future_topk_hit": _topk_hit(
                    structured_ranked,
                    future_targets,
                    top_k,
                ),
                "replay_adapter_proxy_future_topk_hit": _topk_hit(
                    replay_ranked,
                    future_targets,
                    top_k,
                ),
                "fast_temporary_weights_future_topk_hit": _topk_hit(
                    fast_ranked,
                    future_targets,
                    top_k,
                ),
                "fast_weights_plus_retrieval_future_topk_hit": _topk_hit(
                    combined_ranked,
                    future_targets,
                    top_k,
                ),
                "periodic_consolidation_future_topk_hit": _topk_hit(
                    consolidated_ranked,
                    future_targets,
                    top_k,
                ),
                "fast_weights_plus_retrieval_paraphrase_topk_hit": _topk_hit(
                    combined_paraphrase,
                    future_targets,
                    top_k,
                ),
                "prior_task_retention_accuracy": _prior_retention_accuracy(
                    repo_path,
                    exposed_commits,
                    corpus,
                    fast_state,
                    top_k,
                ),
            }
        )

    if not steps:
        raise ValueError("IF4 fast repository adaptation found no changed-file steps")

    final_exposed_commits = commits[: int(steps[-1]["step"])]
    last_corpus = _corpus_at(repo_path, final_exposed_commits[-1])
    final_fast_state = _build_fast_weight_state(repo_path, final_exposed_commits, last_corpus)
    applied_commits: list[str] = final_fast_state["applied_commits"]
    rollback_state = _build_fast_weight_state(repo_path, applied_commits[:-1], last_corpus)
    rollback_supported = (
        bool(applied_commits)
        and final_fast_state["update_count"] == rollback_state["update_count"] + 1
        and final_fast_state["checksum"] != rollback_state["checksum"]
    )

    methods = _method_metadata(final_fast_state)
    final = _final_summary(
        steps,
        update_ms=update_ms,
        methods=methods,
        rollback_supported=rollback_supported,
        runtime_ms=(time.perf_counter() - start) * 1000.0,
    )

    return {
        "benchmark_label": "if4_fast_repo_adaptation_probe",
        "candidate_id": "IF4",
        "repo": {
            "path_name": repo_path.name,
            "head": _git(repo_path, "rev-parse", "--short=12", "HEAD"),
            "commit_count_used": len(commits),
            "top_k": top_k,
            "consolidation_interval": consolidation_interval,
        },
        "methods": methods,
        "steps": steps,
        "final": final,
        "null_hypothesis_outcome": (
            "fast_repository_adaptation_measured_no_promotion"
        ),
        "notes": (
            "Uses only local Git state at or before exposed_commit to rank files changed "
            "in the next commit. Fast weights and replay adapters are measured proxies, "
            "not trained language-model adapters."
        ),
        "limitations": [
            "changed_file_retrieval_proxy_not_patch_generation",
            "replay_adapter_is_sparse_commit_subject_proxy",
            "fast_weights_are_feature_hash_matrix_not_neural_lm_weights",
            "paraphrase_transfer_is_deterministic_query_rewrite_proxy",
            "no_security_or_poisoning_update_gate",
            "no_rocm_kernel_or_language_model_training",
        ],
    }


def _structured_symbol_graph_boosts(
    repo_path: Path,
    exposed_commits: list[str],
    query: str,
) -> dict[str, float]:
    boosts = _structured_boosts(repo_path, exposed_commits, query)
    query_tokens = _tokens(query)
    for commit in exposed_commits:
        for file_path in _changed_files(repo_path, commit):
            path_tokens = _tokens(file_path.replace("/", " ").replace(".", " "))
            overlap = sum((query_tokens & path_tokens).values())
            if overlap:
                boosts[file_path] = boosts.get(file_path, 0.0) + overlap * 5.0
    return boosts


def _replay_adapter_boosts(
    repo_path: Path,
    exposed_commits: list[str],
    query: str,
) -> dict[str, float]:
    query_tokens = _tokens(query)
    boosts: dict[str, float] = {}
    for commit in exposed_commits:
        subject_overlap = sum((_tokens(_subject(repo_path, commit)) & query_tokens).values())
        if subject_overlap == 0:
            continue
        recency = 1.0 + exposed_commits.index(commit) / max(1, len(exposed_commits))
        for file_path in _changed_files(repo_path, commit):
            boosts[file_path] = boosts.get(file_path, 0.0) + subject_overlap * recency * 4.0
    return boosts


def _periodic_consolidation_boosts(
    repo_path: Path,
    exposed_commits: list[str],
    query: str,
    *,
    consolidation_interval: int,
) -> dict[str, float]:
    retained = [
        commit
        for index, commit in enumerate(exposed_commits, start=1)
        if index % consolidation_interval == 0 or index == len(exposed_commits)
    ]
    return _replay_adapter_boosts(repo_path, retained, query)


def _build_fast_weight_state(
    repo_path: Path,
    exposed_commits: list[str],
    corpus: dict[str, Counter[str]],
) -> dict[str, Any]:
    files = sorted(corpus)
    file_to_index = {file_path: index for index, file_path in enumerate(files)}
    weights = np.zeros((FEATURE_DIM, len(files)), dtype=np.float32)
    update_count = 0
    applied_commits: list[str] = []
    for commit in exposed_commits:
        target_files = [
            file_path
            for file_path in _changed_files(repo_path, commit)
            if file_path in file_to_index
        ]
        if not target_files:
            continue
        key = _text_features(_subject(repo_path, commit), FEATURE_DIM)
        for file_path in target_files:
            weights[:, file_to_index[file_path]] += key
        update_count += 1
        applied_commits.append(commit)
    return {
        "files": files,
        "file_to_index": file_to_index,
        "weights": weights,
        "parameter_bytes": int(weights.size * weights.itemsize),
        "update_count": update_count,
        "applied_commits": applied_commits,
        "checksum": hashlib.sha256(weights.tobytes()).hexdigest(),
    }


def _rank_fast_weight_files(query: str, state: dict[str, Any]) -> list[str]:
    files: list[str] = state["files"]
    weights: np.ndarray = state["weights"]
    if weights.size == 0:
        return []
    scores = _text_features(query, FEATURE_DIM) @ weights
    return [
        files[index]
        for index in np.argsort(-scores, kind="stable")
    ]


def _rank_combined_files(
    query: str,
    corpus: dict[str, Counter[str]],
    state: dict[str, Any],
) -> list[str]:
    files: list[str] = state["files"]
    weights: np.ndarray = state["weights"]
    retrieval_scores = _retrieval_score_map(query, corpus)
    fast_scores = _text_features(query, FEATURE_DIM) @ weights if weights.size else np.array([])
    scored: list[tuple[float, str]] = []
    for index, file_path in enumerate(files):
        retrieval = retrieval_scores.get(file_path, 0.0)
        fast = float(fast_scores[index]) if fast_scores.size else 0.0
        scored.append((retrieval + fast * 8.0, file_path))
    return [path for _, path in sorted(scored, key=lambda item: (-item[0], item[1]))]


def _retrieval_score_map(query: str, corpus: dict[str, Counter[str]]) -> dict[str, float]:
    query_tokens = _tokens(query)
    return {
        path: float(sum(query_tokens[token] * tokens.get(token, 0) for token in query_tokens))
        for path, tokens in corpus.items()
    }


def _prior_retention_accuracy(
    repo_path: Path,
    exposed_commits: list[str],
    corpus: dict[str, Counter[str]],
    fast_state: dict[str, Any],
    top_k: int,
) -> float:
    hits = []
    for commit in exposed_commits:
        targets = _changed_files(repo_path, commit)
        ranked = _rank_combined_files(_subject(repo_path, commit), corpus, fast_state)
        hits.append(_topk_hit(ranked, targets, top_k))
    return sum(hits) / len(hits) if hits else 1.0


def _paraphrase_query(query: str, targets: list[str]) -> str:
    stems = [
        part
        for path in targets
        for part in path.replace("/", " ").replace(".", " ").split()
        if part
    ]
    query_tokens = query.replace("_", " ").replace("-", " ").split()
    return " ".join(["change", "repository", *stems[:3], *reversed(query_tokens)])


def _text_features(text: str, width: int) -> np.ndarray:
    features = np.zeros(width, dtype=np.float32)
    tokens = [token for token in _tokens(text)]
    grams = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:], strict=False)]
    for gram in grams:
        digest = hashlib.sha256(gram.encode("utf-8")).digest()
        features[int.from_bytes(digest[:4], "little") % width] += 1.0
    norm = float(np.linalg.norm(features))
    if norm:
        features /= norm
    return features


def _method_metadata(fast_state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    fast_bytes = int(fast_state["parameter_bytes"])
    sparse_proxy_bytes = int(fast_state["update_count"] * 96)
    return {
        "updated_retrieval": {
            "description": (
                "Rank exposed repository files by lexical overlap with the next commit subject."
            ),
            "storage_bytes": 0,
        },
        "structured_symbol_graph_memory": {
            "description": "Retrieval plus explicit path/symbol-style boosts from exposed changes.",
            "storage_bytes": sparse_proxy_bytes,
        },
        "replay_adapter_proxy": {
            "description": "Sparse replay table keyed by exposed commit-subject tokens.",
            "storage_bytes": sparse_proxy_bytes,
        },
        "fast_temporary_weights": {
            "description": (
                "Feature-hashed associative matrix from commit subjects to changed files."
            ),
            "parameter_bytes": fast_bytes,
            "update_count": int(fast_state["update_count"]),
        },
        "fast_weights_plus_retrieval": {
            "description": "Updated retrieval score combined with temporary fast-weight score.",
            "storage_bytes": fast_bytes,
        },
        "periodic_consolidation": {
            "description": "Replay proxy periodically folded into a retained sparse table.",
            "storage_bytes": max(96, sparse_proxy_bytes // 2),
        },
    }


def _final_summary(
    steps: list[dict[str, Any]],
    *,
    update_ms: list[float],
    methods: dict[str, dict[str, Any]],
    rollback_supported: bool,
    runtime_ms: float,
) -> dict[str, Any]:
    method_keys = [
        "updated_retrieval",
        "structured_symbol_graph_memory",
        "replay_adapter_proxy",
        "fast_temporary_weights",
        "fast_weights_plus_retrieval",
        "periodic_consolidation",
    ]
    summary = {
        f"{method}_future_topk_accuracy": _mean(
            step[f"{method}_future_topk_hit"] for step in steps
        )
        for method in method_keys
    }
    summary.update(
        {
            "fast_weights_plus_retrieval_paraphrase_topk_accuracy": _mean(
                step["fast_weights_plus_retrieval_paraphrase_topk_hit"] for step in steps
            ),
            "prior_task_retention_accuracy": steps[-1]["prior_task_retention_accuracy"],
            "mean_update_ms": _mean(update_ms),
            "max_update_ms": max(update_ms) if update_ms else 0.0,
            "rollback_supported": rollback_supported,
            "total_storage_bytes": _total_method_storage(methods),
            "runtime_ms": runtime_ms,
        }
    )
    return summary


def _mean(values) -> float:
    materialized = list(values)
    return float(sum(materialized) / len(materialized)) if materialized else 0.0


def _total_method_storage(methods: dict[str, dict[str, Any]]) -> int:
    total = 0
    for method in methods.values():
        total += int(method.get("storage_bytes", 0))
        total += int(method.get("parameter_bytes", 0))
    return total
