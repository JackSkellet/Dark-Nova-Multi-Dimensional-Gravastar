from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_IDENTIFIER_RE = re.compile(r"[$A-Za-z_][$A-Za-z0-9_]*")
_FUNCTION_PATTERNS = (
    re.compile(r"\bexport\s+function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\("),
    re.compile(r"\bfunction\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\("),
    re.compile(r"\bexport\s+class\s+([A-Za-z_$][A-Za-z0-9_$]*)\b"),
    re.compile(r"\bclass\s+([A-Za-z_$][A-Za-z0-9_$]*)\b"),
    re.compile(
        r"\bexport\s+const\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][A-Za-z0-9_$]*\s*=>)"
    ),
    re.compile(
        r"\bconst\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][A-Za-z0-9_$]*\s*=>)"
    ),
)
_STOPWORDS = {
    "and",
    "class",
    "const",
    "export",
    "false",
    "for",
    "from",
    "function",
    "import",
    "let",
    "may",
    "or",
    "return",
    "should",
    "that",
    "test",
    "to",
    "true",
    "var",
    "with",
}
_SOURCE_LANGUAGES = {"javascript", "typescript", "jsx", "tsx"}


@dataclass(frozen=True)
class RepositoryApiReuseConfig:
    source_split: str = "train"
    query_split: str = "validation"
    seed: int = 123
    top_k: int = 5
    max_tasks: int = 256
    max_source_rows: int = 50_000
    max_query_rows: int = 10_000
    min_text_bytes: int = 80


def evaluate_repository_api_reuse(
    corpus_jsonl: Path,
    config: RepositoryApiReuseConfig | None = None,
) -> dict[str, Any]:
    config = config or RepositoryApiReuseConfig()
    source_rows, source_scanned = _load_rows(
        corpus_jsonl,
        split=config.source_split,
        max_rows=config.max_source_rows,
        min_text_bytes=config.min_text_bytes,
    )
    query_rows, query_scanned = _load_rows(
        corpus_jsonl,
        split=config.query_split,
        max_rows=config.max_query_rows,
        min_text_bytes=config.min_text_bytes,
    )
    symbols_by_repo = _extract_symbols_by_repo(source_rows)
    tasks = _build_tasks(query_rows, symbols_by_repo, config)
    method_results = {
        "symbol_name_mention": _score_method(tasks, "name_mention", config.top_k),
        "lexical_source_overlap": _score_method(tasks, "lexical_overlap", config.top_k),
    }
    best_method_name, best_method = max(
        method_results.items(),
        key=lambda item: (item[1]["hit_at_k"], item[1]["mrr"], item[0]),
    )
    repository_names = {task["repo"] for task in tasks}
    return {
        "benchmark_label": "repository_api_reuse_probe",
        "corpus_jsonl": str(corpus_jsonl),
        "source_split": config.source_split,
        "query_split": config.query_split,
        "seed": config.seed,
        "top_k": config.top_k,
        "source_rows_scanned": source_scanned,
        "source_rows_loaded": len(source_rows),
        "query_rows_scanned": query_scanned,
        "query_rows_loaded": len(query_rows),
        "repository_count": len(repository_names),
        "symbol_count": sum(len(symbols) for symbols in symbols_by_repo.values()),
        "task_count": len(tasks),
        "methods": method_results,
        "best_method": {"name": best_method_name, **best_method},
        "tasks": [_public_task(task) for task in tasks],
        "limitations": [
            "not_code_generation",
            "not_executable_runtime_scoring",
            "symbol_mentions_are_proxy_labels",
            "javascript_style_symbol_extraction_only",
        ],
    }


def evaluate_repository_context_comparison(
    corpus_jsonl: Path,
    config: RepositoryApiReuseConfig | None = None,
) -> dict[str, Any]:
    config = config or RepositoryApiReuseConfig()
    source_rows, source_scanned = _load_rows(
        corpus_jsonl,
        split=config.source_split,
        max_rows=config.max_source_rows,
        min_text_bytes=config.min_text_bytes,
    )
    query_rows, query_scanned = _load_rows(
        corpus_jsonl,
        split=config.query_split,
        max_rows=config.max_query_rows,
        min_text_bytes=config.min_text_bytes,
    )
    symbols_by_repo = _extract_symbols_by_repo(source_rows)
    tasks = _build_tasks(query_rows, symbols_by_repo, config)
    method_results = {
        "structured_symbol_memory": _score_prediction_method(
            tasks,
            "structured_symbol_memory",
            config.top_k,
        ),
        "retrieved_snippet_identifiers": _score_prediction_method(
            tasks,
            "retrieved_snippet_identifiers",
            config.top_k,
        ),
        "symbol_aware_retrieved_snippets": _score_prediction_method(
            tasks,
            "symbol_aware_retrieved_snippets",
            config.top_k,
        ),
        "query_symbol_aware_retrieval": _score_prediction_method(
            tasks,
            "query_symbol_aware_retrieval",
            config.top_k,
        ),
    }
    best_method_name, best_method = max(
        method_results.items(),
        key=lambda item: (
            item[1]["hit_at_k"],
            item[1]["mrr"],
            -item[1]["hallucinated_api_rate"],
            -_context_method_complexity(item[0]),
        ),
    )
    return {
        "benchmark_label": "repository_context_pairwise_probe",
        "pairwise_ideas": [
            "retrieval_augmented_repository_context",
            "structured_repository_memory",
        ],
        "corpus_jsonl": str(corpus_jsonl),
        "source_split": config.source_split,
        "query_split": config.query_split,
        "seed": config.seed,
        "top_k": config.top_k,
        "source_rows_scanned": source_scanned,
        "source_rows_loaded": len(source_rows),
        "query_rows_scanned": query_scanned,
        "query_rows_loaded": len(query_rows),
        "repository_count": len({task["repo"] for task in tasks}),
        "symbol_count": sum(len(symbols) for symbols in symbols_by_repo.values()),
        "task_count": len(tasks),
        "methods": method_results,
        "best_method": {"name": best_method_name, **best_method},
        "tasks": [_public_task(task) for task in tasks],
        "limitations": [
            "not_code_generation",
            "not_executable_runtime_scoring",
            "symbol_mentions_are_proxy_labels",
            "javascript_style_symbol_extraction_only",
            "proxy_hallucinated_api_rate",
        ],
    }


def _load_rows(
    corpus_jsonl: Path,
    *,
    split: str,
    max_rows: int,
    min_text_bytes: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    scanned = 0
    with corpus_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != split:
                continue
            scanned += 1
            text = str(row.get("text", ""))
            text_bytes = int(row.get("bytes", len(text.encode("utf-8", errors="ignore"))))
            if text_bytes < min_text_bytes:
                continue
            if not row.get("repo") or not row.get("path") or not text:
                continue
            rows.append(row)
            if len(rows) >= max_rows:
                break
    return rows, scanned


def _extract_symbols_by_repo(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    symbols_by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        text = str(row.get("text", ""))
        language = str(row.get("language", "")).strip().lower()
        if language not in _SOURCE_LANGUAGES:
            continue
        repo = str(row["repo"])
        path = str(row["path"])
        for symbol in _extract_symbols(text):
            key = (repo, path, symbol)
            if key in seen:
                continue
            seen.add(key)
            symbols_by_repo[repo].append(
                {
                    "symbol": symbol,
                    "path": path,
                    "language": str(row.get("language", "")),
                    "content_roles": list(row.get("content_roles", [])),
                    "row_sha256": str(row.get("row_sha256", "")),
                    "text_sha256": str(row.get("text_sha256", "")),
                    "text": text,
                }
            )
    return {repo: symbols for repo, symbols in symbols_by_repo.items() if symbols}


def _extract_symbols(text: str) -> list[str]:
    code = _strip_javascript_comments(text)
    symbols: list[str] = []
    seen: set[str] = set()
    for pattern in _FUNCTION_PATTERNS:
        for match in pattern.finditer(code):
            symbol = match.group(1)
            if len(symbol) < 3 or symbol.lower() in _STOPWORDS:
                continue
            if symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)
    return symbols


def _strip_javascript_comments(text: str) -> str:
    without_blocks = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return re.sub(r"//.*", " ", without_blocks)


def _build_tasks(
    query_rows: list[dict[str, Any]],
    symbols_by_repo: dict[str, list[dict[str, Any]]],
    config: RepositoryApiReuseConfig,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for row in query_rows:
        repo = str(row["repo"])
        candidates = symbols_by_repo.get(repo, [])
        if not candidates:
            continue
        text = str(row["text"])
        positives = [
            candidate
            for candidate in candidates
            if _contains_identifier(text, str(candidate["symbol"]))
            and str(candidate["path"]) != str(row["path"])
        ]
        if not positives:
            continue
        sample_material = {
            "repo": repo,
            "path": row["path"],
            "query_split": config.query_split,
            "source_split": config.source_split,
            "seed": config.seed,
            "positive_symbols": sorted(str(item["symbol"]) for item in positives),
            "text_sha256": row.get("text_sha256", ""),
        }
        sample_sha256 = hashlib.sha256(
            json.dumps(sample_material, sort_keys=True).encode("utf-8")
        ).hexdigest()
        tasks.append(
            {
                "task_id": f"repo-api-reuse-{sample_sha256[:16]}",
                "repo": repo,
                "path": str(row["path"]),
                "query_language": str(row.get("language", "")),
                "query_roles": list(row.get("content_roles", [])),
                "query_text": text,
                "query_row_sha256": str(row.get("row_sha256", "")),
                "query_text_sha256": str(row.get("text_sha256", "")),
                "sample_sha256": sample_sha256,
                "positive_symbols": sorted(str(item["symbol"]) for item in positives),
                "candidates": sorted(
                    candidates,
                    key=lambda item: (str(item["symbol"]), str(item["path"])),
                ),
            }
        )
    rng = random.Random(config.seed)
    rng.shuffle(tasks)
    tasks = tasks[: config.max_tasks]
    tasks.sort(key=lambda task: task["task_id"])
    return tasks


def _contains_identifier(text: str, symbol: str) -> bool:
    return symbol in _identifier_tokens(text)


def _identifier_tokens(text: str) -> set[str]:
    return {
        token
        for token in _IDENTIFIER_RE.findall(text)
        if token and token.lower() not in _STOPWORDS
    }


def _score_method(tasks: list[dict[str, Any]], method: str, top_k: int) -> dict[str, Any]:
    ranks: list[int | None] = []
    coverages: list[float] = []
    for task in tasks:
        ranked = _rank_candidates(task, method)
        positives = set(task["positive_symbols"])
        first_rank = None
        for index, candidate in enumerate(ranked[:top_k], start=1):
            if str(candidate["symbol"]) in positives:
                first_rank = index
                break
        ranks.append(first_rank)
        covered = {
            str(candidate["symbol"])
            for candidate in ranked[:top_k]
            if str(candidate["symbol"]) in positives
        }
        coverages.append(len(covered) / max(len(positives), 1))
    task_count = len(tasks)
    hits = sum(rank is not None for rank in ranks)
    reciprocal_ranks = [0.0 if rank is None else 1.0 / rank for rank in ranks]
    return {
        "task_count": task_count,
        "hit_at_k": hits / task_count if task_count else 0.0,
        "mrr": sum(reciprocal_ranks) / task_count if task_count else 0.0,
        "coverage_at_k": sum(coverages) / task_count if task_count else 0.0,
    }


def _rank_candidates(task: dict[str, Any], method: str) -> list[dict[str, Any]]:
    query_tokens = _identifier_tokens(str(task["query_text"]))
    scored = []
    for candidate in task["candidates"]:
        symbol = str(candidate["symbol"])
        source_tokens = _identifier_tokens(str(candidate["text"]))
        if method == "name_mention":
            score = 1.0 if symbol in query_tokens else 0.0
        elif method == "lexical_overlap":
            denominator = math.sqrt(max(len(query_tokens), 1) * max(len(source_tokens), 1))
            score = len(query_tokens & source_tokens) / denominator
        elif method == "query_symbol_mention_then_lexical":
            denominator = math.sqrt(max(len(query_tokens), 1) * max(len(source_tokens), 1))
            lexical = len(query_tokens & source_tokens) / denominator
            score = (1.0 if symbol in query_tokens else 0.0) + lexical
        else:
            raise ValueError(f"unknown repository API reuse method: {method}")
        scored.append((score, symbol, str(candidate["path"]), candidate))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [item[3] for item in scored]


def _score_prediction_method(
    tasks: list[dict[str, Any]],
    method: str,
    top_k: int,
) -> dict[str, Any]:
    ranks: list[int | None] = []
    coverages: list[float] = []
    hallucinated = 0
    prediction_total = 0
    for task in tasks:
        predictions = _predict_symbols(task, method, top_k)
        positives = set(task["positive_symbols"])
        known_symbols = {str(candidate["symbol"]) for candidate in task["candidates"]}
        first_rank = None
        for index, symbol in enumerate(predictions, start=1):
            if symbol in positives:
                first_rank = index
                break
        ranks.append(first_rank)
        coverages.append(len(set(predictions) & positives) / max(len(positives), 1))
        prediction_total += len(predictions)
        hallucinated += sum(symbol not in known_symbols for symbol in predictions)
    task_count = len(tasks)
    reciprocal_ranks = [0.0 if rank is None else 1.0 / rank for rank in ranks]
    return {
        "task_count": task_count,
        "hit_at_k": sum(rank is not None for rank in ranks) / task_count
        if task_count
        else 0.0,
        "mrr": sum(reciprocal_ranks) / task_count if task_count else 0.0,
        "coverage_at_k": sum(coverages) / task_count if task_count else 0.0,
        "hallucinated_api_rate": hallucinated / prediction_total
        if prediction_total
        else 0.0,
        "prediction_count": prediction_total,
    }


def _predict_symbols(task: dict[str, Any], method: str, top_k: int) -> list[str]:
    if method == "structured_symbol_memory":
        ranked = _rank_candidates(task, "name_mention")
        return _unique_symbols(str(candidate["symbol"]) for candidate in ranked[:top_k])
    if method == "retrieved_snippet_identifiers":
        ranked = _rank_candidates(task, "lexical_overlap")
        identifiers: list[str] = []
        for candidate in ranked:
            identifiers.extend(sorted(_identifier_tokens(str(candidate["text"]))))
            if len(_unique_symbols(identifiers)) >= top_k:
                break
        return _unique_symbols(identifiers)[:top_k]
    if method == "symbol_aware_retrieved_snippets":
        ranked = _rank_candidates(task, "lexical_overlap")
        return _unique_symbols(str(candidate["symbol"]) for candidate in ranked[:top_k])
    if method == "query_symbol_aware_retrieval":
        ranked = _rank_candidates(task, "query_symbol_mention_then_lexical")
        return _unique_symbols(str(candidate["symbol"]) for candidate in ranked[:top_k])
    raise ValueError(f"unknown repository context prediction method: {method}")


def _context_method_complexity(method: str) -> int:
    return {
        "structured_symbol_memory": 0,
        "query_symbol_aware_retrieval": 1,
        "symbol_aware_retrieved_snippets": 1,
        "retrieved_snippet_identifiers": 2,
    }.get(method, 99)


def _unique_symbols(symbols: Any) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        unique.append(symbol)
    return unique


def _public_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "repo": task["repo"],
        "path": task["path"],
        "query_language": task["query_language"],
        "query_roles": task["query_roles"],
        "query_row_sha256": task["query_row_sha256"],
        "query_text_sha256": task["query_text_sha256"],
        "sample_sha256": task["sample_sha256"],
        "positive_symbols": task["positive_symbols"],
        "candidate_count": len(task["candidates"]),
    }
