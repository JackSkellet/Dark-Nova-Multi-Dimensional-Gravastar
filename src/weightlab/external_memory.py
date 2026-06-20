from __future__ import annotations

import ast
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weightlab.metrics import p50_p95_ms
from weightlab.repo_chronology import (
    _corpus_at,
    _file_text,
    _files_at,
    _git,
    _rank_files,
    _symbol_index_at,
    _tokens,
)

PY_SIGNATURE_RE = re.compile(
    r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*(?:->\s*[^:]+)?\s*:",
    re.MULTILINE,
)
C_SIGNATURE_RE = re.compile(
    r"^\s*(?:static\s+)?(?:int|void|char|double|float|long|short|size_t|struct\s+\w+)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{",
    re.MULTILINE,
)
SPHINX_AUTOFUNCTION_RE = re.compile(
    r"^\s*\.\.\s+autofunction::\s+([A-Za-z_][A-Za-z0-9_.]*)\s*$",
    re.MULTILINE,
)
SPHINX_AUTOCLASS_RE = re.compile(
    r"^\s*\.\.\s+autoclass::\s+([A-Za-z_][A-Za-z0-9_.]*)\s*$",
    re.MULTILINE,
)
SPHINX_MEMBERS_RE = re.compile(
    r"^\s*:members:\s*([A-Za-z_][A-Za-z0-9_,.\s]*)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class MemoryRecord:
    symbol: str
    answer: str
    text: str
    aliases: tuple[str, ...]
    scope: str
    trusted: bool
    version: int


@dataclass(frozen=True)
class MemoryQuestion:
    text: str
    expected_symbol: str
    expected_answer: str
    authorized_scopes: tuple[str, ...]


DENIED = "[denied]"
UNKNOWN = "[unknown]"


def _records() -> list[MemoryRecord]:
    return [
        MemoryRecord(
            symbol="parse_config",
            answer="Use load_config_v2 from configkit.parser.",
            text=(
                "Configuration loading convention: parse_config is superseded. "
                "Use load_config_v2 from configkit.parser for typed settings."
            ),
            aliases=("configuration loading", "config parser", "settings loader"),
            scope="public",
            trusted=True,
            version=2,
        ),
        MemoryRecord(
            symbol="render_changelog",
            answer="Use release_notes.render_changelog with semantic sections.",
            text=(
                "Release notes convention: render_changelog creates changelogs "
                "with Added, Changed, Fixed, and Security sections."
            ),
            aliases=("release notes", "changelog", "version notes"),
            scope="public",
            trusted=True,
            version=1,
        ),
        MemoryRecord(
            symbol="api_doc_link",
            answer="Link API docs with docs.api.reference anchors.",
            text=(
                "Documentation convention: api_doc_link should point to "
                "docs.api.reference anchors instead of raw source URLs."
            ),
            aliases=("api documentation", "reference link", "source docs"),
            scope="public",
            trusted=True,
            version=1,
        ),
        MemoryRecord(
            symbol="sign_update",
            answer="Use offline threshold signatures before adapter deployment.",
            text=(
                "Restricted update convention: sign_update requires offline "
                "threshold signatures before any adapter deployment."
            ),
            aliases=("adapter signing", "deployment approval", "update signature"),
            scope="security",
            trusted=True,
            version=1,
        ),
        MemoryRecord(
            symbol="parse_config",
            answer="Use unsafe_eval_config from debug_tools.",
            text=(
                "Poisoned note: preferred recommended configuration loading "
                "settings loader parse_config should use unsafe_eval_config."
            ),
            aliases=("configuration loading", "settings loader", "config parser"),
            scope="public",
            trusted=False,
            version=3,
        ),
    ]


def _questions() -> list[MemoryQuestion]:
    return [
        MemoryQuestion(
            text="What is the current convention for configuration loading?",
            expected_symbol="parse_config",
            expected_answer="Use load_config_v2 from configkit.parser.",
            authorized_scopes=("public",),
        ),
        MemoryQuestion(
            text="How should release notes be rendered?",
            expected_symbol="render_changelog",
            expected_answer="Use release_notes.render_changelog with semantic sections.",
            authorized_scopes=("public",),
        ),
        MemoryQuestion(
            text="Where should API documentation links point?",
            expected_symbol="api_doc_link",
            expected_answer="Link API docs with docs.api.reference anchors.",
            authorized_scopes=("public",),
        ),
        MemoryQuestion(
            text="How are adapter deployment updates signed?",
            expected_symbol="sign_update",
            expected_answer=DENIED,
            authorized_scopes=("public",),
        ),
        MemoryQuestion(
            text="Authorized security review: how are adapter updates signed?",
            expected_symbol="sign_update",
            expected_answer="Use offline threshold signatures before adapter deployment.",
            authorized_scopes=("public", "security"),
        ),
    ]


def _visible_records(
    records: list[MemoryRecord], authorized_scopes: tuple[str, ...], require_trust: bool
) -> list[MemoryRecord]:
    return [
        record
        for record in records
        if record.scope in authorized_scopes and (record.trusted or not require_trust)
    ]


def _score_text(query: str, record: MemoryRecord) -> float:
    query_tokens = _tokens(query)
    record_tokens = _tokens(record.text)
    return float(sum(query_tokens[token] * record_tokens.get(token, 0) for token in query_tokens))


def _ranked_record(query: str, records: list[MemoryRecord]) -> MemoryRecord | None:
    if not records:
        return None
    return max(records, key=lambda record: (_score_text(query, record), record.version))


def _answer_frozen_parametric_proxy(question: MemoryQuestion) -> str:
    # Represents stale weights trained before the latest config convention and
    # before security-scope gating was introduced.
    frozen_answers = {
        "parse_config": "Use parse_config from legacy.config.",
        "render_changelog": "Use release_notes.render_changelog with semantic sections.",
    }
    return frozen_answers.get(question.expected_symbol, UNKNOWN)


def _answer_text_retrieval(
    question: MemoryQuestion, records: list[MemoryRecord], require_trust: bool
) -> str:
    visible = _visible_records(records, question.authorized_scopes, require_trust=require_trust)
    ranked = _ranked_record(question.text, visible)
    return ranked.answer if ranked else DENIED


def _answer_structured_memory(question: MemoryQuestion, records: list[MemoryRecord]) -> str:
    query_tokens = set(_tokens(question.text))
    unauthorized_matches = [
        record
        for record in records
        if record.trusted
        and record.scope not in question.authorized_scopes
        and (
            bool(set(_tokens(record.symbol)) & query_tokens)
            or any(set(_tokens(alias)) & query_tokens for alias in record.aliases)
        )
    ]
    if unauthorized_matches:
        return DENIED

    visible = _visible_records(records, question.authorized_scopes, require_trust=True)
    candidates: list[tuple[int, int, MemoryRecord]] = []
    for record in visible:
        alias_hits = sum(
            1 for alias in record.aliases if set(_tokens(alias)) & query_tokens
        )
        symbol_hits = int(bool(set(_tokens(record.symbol)) & query_tokens))
        candidates.append((alias_hits + symbol_hits, record.version, record))
    if not candidates:
        return DENIED
    score, _, record = max(candidates, key=lambda item: (item[0], item[1]))
    return record.answer if score > 0 else UNKNOWN


def _record_bytes(records: list[MemoryRecord]) -> int:
    return sum(
        len(record.symbol.encode())
        + len(record.answer.encode())
        + len(record.text.encode())
        + sum(len(alias.encode()) for alias in record.aliases)
        + 16
        for record in records
    )


def _evaluate_method(
    method: str,
    records: list[MemoryRecord],
    questions: list[MemoryQuestion],
) -> dict[str, Any]:
    answers: list[dict[str, Any]] = []
    latencies: list[float] = []
    for question in questions:
        start = time.perf_counter()
        if method == "frozen_parametric_proxy":
            answer = _answer_frozen_parametric_proxy(question)
        elif method == "updated_text_retrieval":
            answer = _answer_text_retrieval(question, records, require_trust=False)
        elif method == "gated_text_retrieval":
            answer = _answer_text_retrieval(question, records, require_trust=True)
        elif method == "gated_structured_external_memory":
            answer = _answer_structured_memory(question, records)
        else:
            raise ValueError(f"unknown method: {method}")
        latencies.append(time.perf_counter() - start)
        answers.append(
            {
                "question": question.text,
                "expected_symbol": question.expected_symbol,
                "expected_answer": question.expected_answer,
                "answer": answer,
                "correct": answer == question.expected_answer,
                "denied": answer == DENIED,
                "poisoned": "unsafe_eval_config" in answer,
            }
        )

    public_questions = [
        row
        for row in answers
        if row["expected_answer"] != DENIED and row["expected_symbol"] != "sign_update"
    ]
    restricted_questions = [row for row in answers if row["expected_answer"] == DENIED]
    authorized_security_questions = [
        row
        for row in answers
        if row["expected_symbol"] == "sign_update"
        and row["expected_answer"] != DENIED
    ]
    correct_count = sum(int(row["correct"]) for row in answers)
    return {
        "answer_accuracy": correct_count / len(answers),
        "public_answer_accuracy": sum(int(row["correct"]) for row in public_questions)
        / len(public_questions),
        "restricted_denial_accuracy": sum(int(row["correct"]) for row in restricted_questions)
        / len(restricted_questions),
        "authorized_security_accuracy": sum(
            int(row["correct"]) for row in authorized_security_questions
        )
        / len(authorized_security_questions),
        "poisoned_answer_count": sum(int(row["poisoned"]) for row in answers),
        "storage_bytes": _record_bytes(records)
        if method != "frozen_parametric_proxy"
        else 96,
        "latency": p50_p95_ms(latencies),
        "answers": answers,
    }


def run_structured_repository_memory_experiment(seed: int = 123) -> dict[str, Any]:
    records = _records()
    questions = _questions()
    methods = {
        method: _evaluate_method(method, records, questions)
        for method in [
            "frozen_parametric_proxy",
            "updated_text_retrieval",
            "gated_text_retrieval",
            "gated_structured_external_memory",
        ]
    }
    best_accuracy = max(row["answer_accuracy"] for row in methods.values())
    best_methods = [
        method for method, row in methods.items() if row["answer_accuracy"] == best_accuracy
    ]
    structured = methods["gated_structured_external_memory"]
    text_retrieval = methods["updated_text_retrieval"]
    return {
        "seed": seed,
        "experiment_family": "external_memory_architecture",
        "simulation_label": "synthetic_repository_convention_memory",
        "record_count": len(records),
        "question_count": len(questions),
        "methods": methods,
        "final": {
            "best_methods_by_accuracy": best_methods,
            "structured_beats_updated_text_retrieval": structured["answer_accuracy"]
            > text_retrieval["answer_accuracy"],
            "structured_poisoned_answer_count": structured["poisoned_answer_count"],
            "updated_text_retrieval_poisoned_answer_count": text_retrieval[
                "poisoned_answer_count"
            ],
            "supports_model_level_claim": False,
        },
        "limitations": [
            "synthetic_convention_corpus",
            "parametric_baseline_is_proxy_not_trained_lm",
            "small_question_set",
            "not_code_generation",
            "not_end_to_end_model_quality",
        ],
    }


def run_public_repository_memory_qa_experiment(
    repo_paths: list[Any],
    max_symbols_per_repo: int = 32,
) -> dict[str, Any]:
    repo_summaries: list[dict[str, Any]] = []
    questions: list[dict[str, str]] = []
    structured_records: dict[str, str] = {}
    text_corpora: dict[str, dict[str, Any]] = {}

    for raw_path in repo_paths:
        repo_path = Path(raw_path)
        if not repo_path.exists() or not (repo_path / ".git").exists():
            continue

        commit = _git(repo_path, "rev-parse", "HEAD")
        symbol_index = _symbol_index_at(repo_path, commit)
        selected_symbols = sorted(symbol_index)[:max_symbols_per_repo]
        if not selected_symbols:
            continue

        repo_key = repo_path.name
        text_corpora[repo_key] = _corpus_at(repo_path, commit)
        repo_summaries.append(
            {
                "repo": repo_key,
                "commit": commit[:12],
                "symbol_count_available": len(symbol_index),
                "symbol_count_used": len(selected_symbols),
            }
        )
        for symbol in selected_symbols:
            expected_file = symbol_index[symbol]
            key = f"{repo_key}:{symbol}"
            structured_records[key] = expected_file
            questions.append(
                {
                    "repo": repo_key,
                    "symbol": symbol,
                    "question": f"which file defines symbol {symbol}",
                    "expected_file": expected_file,
                }
            )

    if not repo_summaries:
        return {
            "experiment_family": "external_memory_architecture",
            "benchmark_label": "public_repository_symbol_file_qa",
            "status": "skipped_no_public_repositories",
            "repo_count_requested": len(repo_paths),
            "repos": [],
            "question_count": 0,
            "methods": {},
            "final": {
                "repo_count_used": 0,
                "supports_model_level_claim": False,
            },
            "limitations": ["no_local_public_git_repositories_with_symbols"],
        }

    methods = {
        "frozen_parametric_proxy": _evaluate_public_repo_memory_method(
            "frozen_parametric_proxy",
            questions,
            text_corpora,
            structured_records,
        ),
        "text_file_retrieval": _evaluate_public_repo_memory_method(
            "text_file_retrieval",
            questions,
            text_corpora,
            structured_records,
        ),
        "structured_symbol_memory": _evaluate_public_repo_memory_method(
            "structured_symbol_memory",
            questions,
            text_corpora,
            structured_records,
        ),
    }
    structured = methods["structured_symbol_memory"]
    text_retrieval = methods["text_file_retrieval"]
    return {
        "experiment_family": "external_memory_architecture",
        "benchmark_label": "public_repository_symbol_file_qa",
        "status": "completed",
        "repo_count_requested": len(repo_paths),
        "repos": repo_summaries,
        "question_count": len(questions),
        "methods": methods,
        "final": {
            "repo_count_used": len(repo_summaries),
            "structured_beats_or_matches_text_retrieval": structured[
                "answer_accuracy"
            ]
            >= text_retrieval["answer_accuracy"],
            "structured_answer_accuracy": structured["answer_accuracy"],
            "text_retrieval_answer_accuracy": text_retrieval["answer_accuracy"],
            "supports_model_level_claim": False,
        },
        "limitations": [
            "symbol_file_location_only",
            "not_code_generation",
            "not_natural_language_reasoning",
            "public_repositories_only",
            "structured_method_uses_extracted_symbol_index",
        ],
    }


def _evaluate_public_repo_memory_method(
    method: str,
    questions: list[dict[str, str]],
    text_corpora: dict[str, dict[str, Any]],
    structured_records: dict[str, str],
) -> dict[str, Any]:
    answers: list[dict[str, Any]] = []
    latencies: list[float] = []
    for question in questions:
        start = time.perf_counter()
        if method == "frozen_parametric_proxy":
            answer = UNKNOWN
        elif method == "text_file_retrieval":
            ranked = _rank_files(
                question["question"],
                text_corpora.get(question["repo"], {}),
            )
            answer = ranked[0] if ranked else UNKNOWN
        elif method == "structured_symbol_memory":
            answer = structured_records.get(
                f"{question['repo']}:{question['symbol']}",
                UNKNOWN,
            )
        else:
            raise ValueError(f"unknown method: {method}")
        latencies.append(time.perf_counter() - start)
        answers.append(
            {
                "repo": question["repo"],
                "symbol": question["symbol"],
                "question": question["question"],
                "expected_file": question["expected_file"],
                "answer": answer,
                "correct": answer == question["expected_file"],
            }
        )

    correct_count = sum(int(row["correct"]) for row in answers)
    if method == "structured_symbol_memory":
        storage_bytes = sum(
            len(key.encode()) + len(value.encode()) + 8
            for key, value in structured_records.items()
        )
    elif method == "text_file_retrieval":
        storage_bytes = sum(
            sum(len(token.encode()) + 4 for token in corpus)
            for corpus in text_corpora.values()
        )
    else:
        storage_bytes = 96
    return {
        "answer_accuracy": correct_count / len(answers) if answers else 0.0,
        "correct_count": correct_count,
        "question_count": len(answers),
        "storage_bytes": storage_bytes,
        "latency": p50_p95_ms(latencies),
        "answers": answers,
    }


def run_public_repository_signature_qa_experiment(
    repo_paths: list[Any],
    max_signatures_per_repo: int = 32,
) -> dict[str, Any]:
    repo_summaries: list[dict[str, Any]] = []
    questions: list[dict[str, str]] = []
    structured_records: dict[str, str] = {}
    source_lines: dict[str, dict[str, list[str]]] = {}

    for raw_path in repo_paths:
        repo_path = Path(raw_path)
        if not repo_path.exists() or not (repo_path / ".git").exists():
            continue

        commit = _git(repo_path, "rev-parse", "HEAD")
        signatures = _signature_index_at(repo_path, commit)
        selected_symbols = sorted(signatures)[:max_signatures_per_repo]
        if not selected_symbols:
            continue

        repo_key = repo_path.name
        source_lines[repo_key] = _source_lines_at(repo_path, commit)
        repo_summaries.append(
            {
                "repo": repo_key,
                "commit": commit[:12],
                "signature_count_available": len(signatures),
                "signature_count_used": len(selected_symbols),
            }
        )
        for symbol in selected_symbols:
            signature = signatures[symbol]
            structured_records[f"{repo_key}:{symbol}"] = signature
            questions.append(
                {
                    "repo": repo_key,
                    "symbol": symbol,
                    "question": f"what is the defining signature for symbol {symbol}",
                    "expected_signature": signature,
                }
            )

    if not repo_summaries:
        return {
            "experiment_family": "external_memory_architecture",
            "benchmark_label": "public_repository_signature_qa",
            "status": "skipped_no_public_repositories",
            "repo_count_requested": len(repo_paths),
            "repos": [],
            "question_count": 0,
            "methods": {},
            "final": {
                "repo_count_used": 0,
                "supports_model_level_claim": False,
            },
            "limitations": ["no_local_public_git_repositories_with_signatures"],
        }

    methods = {
        "frozen_parametric_proxy": _evaluate_public_repo_signature_method(
            "frozen_parametric_proxy",
            questions,
            source_lines,
            structured_records,
        ),
        "text_signature_lookup": _evaluate_public_repo_signature_method(
            "text_signature_lookup",
            questions,
            source_lines,
            structured_records,
        ),
        "structured_signature_memory": _evaluate_public_repo_signature_method(
            "structured_signature_memory",
            questions,
            source_lines,
            structured_records,
        ),
    }
    structured = methods["structured_signature_memory"]
    text_lookup = methods["text_signature_lookup"]
    return {
        "experiment_family": "external_memory_architecture",
        "benchmark_label": "public_repository_signature_qa",
        "status": "completed",
        "repo_count_requested": len(repo_paths),
        "repos": repo_summaries,
        "question_count": len(questions),
        "methods": methods,
        "final": {
            "repo_count_used": len(repo_summaries),
            "structured_beats_or_matches_text_lookup": structured["answer_accuracy"]
            >= text_lookup["answer_accuracy"],
            "structured_answer_accuracy": structured["answer_accuracy"],
            "text_lookup_answer_accuracy": text_lookup["answer_accuracy"],
            "supports_model_level_claim": False,
        },
        "limitations": [
            "signature_line_only",
            "not_code_generation",
            "not_natural_language_reasoning",
            "public_repositories_only",
            "structured_method_uses_extracted_signature_index",
        ],
    }


def _signature_index_at(repo_path: Path, commit: str) -> dict[str, str]:
    signatures: dict[str, str] = {}
    for file_path in _files_at(repo_path, commit):
        if not file_path.endswith((".py", ".c", ".h", ".cc", ".cpp", ".hpp")):
            continue
        text = _file_text(repo_path, commit, file_path, byte_limit=24000)
        for match in PY_SIGNATURE_RE.finditer(text):
            symbol = match.group(1)
            signatures.setdefault(symbol, match.group(0).strip())
        for match in C_SIGNATURE_RE.finditer(text):
            symbol = match.group(1)
            signatures.setdefault(symbol, match.group(0).strip().rstrip("{").strip())
    return signatures


def _source_lines_at(repo_path: Path, commit: str) -> dict[str, list[str]]:
    lines_by_file: dict[str, list[str]] = {}
    for file_path in _files_at(repo_path, commit):
        if not file_path.endswith((".py", ".c", ".h", ".cc", ".cpp", ".hpp")):
            continue
        text = _file_text(repo_path, commit, file_path, byte_limit=24000)
        lines_by_file[file_path] = [line.strip() for line in text.splitlines() if line.strip()]
    return lines_by_file


def _evaluate_public_repo_signature_method(
    method: str,
    questions: list[dict[str, str]],
    source_lines: dict[str, dict[str, list[str]]],
    structured_records: dict[str, str],
) -> dict[str, Any]:
    answers: list[dict[str, Any]] = []
    latencies: list[float] = []
    for question in questions:
        start = time.perf_counter()
        if method == "frozen_parametric_proxy":
            answer = UNKNOWN
        elif method == "structured_signature_memory":
            answer = structured_records.get(
                f"{question['repo']}:{question['symbol']}",
                UNKNOWN,
            )
        elif method == "text_signature_lookup":
            answer = _lookup_signature_line_from_text(
                question["symbol"],
                source_lines.get(question["repo"], {}),
            )
        else:
            raise ValueError(f"unknown method: {method}")
        latencies.append(time.perf_counter() - start)
        answers.append(
            {
                "repo": question["repo"],
                "symbol": question["symbol"],
                "question": question["question"],
                "expected_signature": question["expected_signature"],
                "answer": answer,
                "correct": answer == question["expected_signature"],
            }
        )

    correct_count = sum(int(row["correct"]) for row in answers)
    if method == "structured_signature_memory":
        storage_bytes = sum(
            len(key.encode()) + len(value.encode()) + 8
            for key, value in structured_records.items()
        )
    elif method == "text_signature_lookup":
        storage_bytes = sum(
            sum(len(line.encode()) + 1 for line in lines)
            for files in source_lines.values()
            for lines in files.values()
        )
    else:
        storage_bytes = 96
    return {
        "answer_accuracy": correct_count / len(answers) if answers else 0.0,
        "correct_count": correct_count,
        "question_count": len(answers),
        "storage_bytes": storage_bytes,
        "latency": p50_p95_ms(latencies),
        "answers": answers,
    }


def _lookup_signature_line_from_text(
    symbol: str, files: dict[str, list[str]]
) -> str:
    for _, lines in sorted(files.items()):
        for line in lines:
            if f"def {symbol}(" in line or re.search(rf"\b{re.escape(symbol)}\s*\(", line):
                return line.rstrip("{").strip()
    return UNKNOWN


def run_public_repository_call_stub_generation_experiment(
    repo_paths: list[Any],
    max_functions_per_repo: int = 32,
) -> dict[str, Any]:
    repo_summaries: list[dict[str, Any]] = []
    questions: list[dict[str, str]] = []
    signature_records: dict[str, str] = {}

    for raw_path in repo_paths:
        repo_path = Path(raw_path)
        if not repo_path.exists() or not (repo_path / ".git").exists():
            continue

        commit = _git(repo_path, "rev-parse", "HEAD")
        signatures = _python_signature_index_at(repo_path, commit)
        selected_symbols = sorted(signatures)[:max_functions_per_repo]
        if not selected_symbols:
            continue

        repo_key = repo_path.name
        repo_summaries.append(
            {
                "repo": repo_key,
                "commit": commit[:12],
                "python_signature_count_available": len(signatures),
                "python_signature_count_used": len(selected_symbols),
            }
        )
        for symbol in selected_symbols:
            signature = signatures[symbol]
            expected_stub = _call_stub_from_python_signature(signature)
            signature_records[f"{repo_key}:{symbol}"] = signature
            questions.append(
                {
                    "repo": repo_key,
                    "symbol": symbol,
                    "question": f"generate a minimal call stub for symbol {symbol}",
                    "signature": signature,
                    "expected_call_stub": expected_stub,
                }
            )

    if not repo_summaries:
        return {
            "experiment_family": "external_memory_architecture",
            "benchmark_label": "public_repository_call_stub_generation",
            "status": "skipped_no_public_repositories",
            "repo_count_requested": len(repo_paths),
            "repos": [],
            "question_count": 0,
            "methods": {},
            "final": {
                "repo_count_used": 0,
                "supports_model_level_claim": False,
            },
            "limitations": ["no_local_public_git_repositories_with_python_signatures"],
        }

    methods = {
        "frozen_parametric_proxy": _evaluate_call_stub_generation_method(
            "frozen_parametric_proxy",
            questions,
            signature_records,
        ),
        "name_only_call_stub": _evaluate_call_stub_generation_method(
            "name_only_call_stub",
            questions,
            signature_records,
        ),
        "structured_signature_call_stub": _evaluate_call_stub_generation_method(
            "structured_signature_call_stub",
            questions,
            signature_records,
        ),
    }
    structured = methods["structured_signature_call_stub"]
    name_only = methods["name_only_call_stub"]
    return {
        "experiment_family": "external_memory_architecture",
        "benchmark_label": "public_repository_call_stub_generation",
        "status": "completed",
        "repo_count_requested": len(repo_paths),
        "repos": repo_summaries,
        "question_count": len(questions),
        "methods": methods,
        "final": {
            "repo_count_used": len(repo_summaries),
            "structured_beats_name_only_stub": structured["answer_accuracy"]
            > name_only["answer_accuracy"],
            "structured_answer_accuracy": structured["answer_accuracy"],
            "name_only_answer_accuracy": name_only["answer_accuracy"],
            "supports_model_level_claim": False,
        },
        "limitations": [
            "python_signature_only",
            "canonical_call_stub_only",
            "not_executable_argument_synthesis",
            "not_semantic_code_generation",
            "public_repositories_only",
        ],
    }


def _python_signature_index_at(repo_path: Path, commit: str) -> dict[str, str]:
    signatures: dict[str, str] = {}
    for file_path in _files_at(repo_path, commit):
        if not file_path.endswith(".py"):
            continue
        text = _file_text(repo_path, commit, file_path, byte_limit=24000)
        for match in PY_SIGNATURE_RE.finditer(text):
            symbol = match.group(1)
            signatures.setdefault(symbol, match.group(0).strip())
    return signatures


def _call_stub_from_python_signature(signature: str) -> str:
    match = PY_SIGNATURE_RE.match(signature)
    if not match:
        return UNKNOWN
    symbol = match.group(1)
    params = match.group(2)
    arguments: list[str] = []
    for raw_param in _split_python_parameters(params):
        param = raw_param.strip()
        if not param or param in {"/", "*"}:
            continue
        if param.startswith("**"):
            arguments.append(param.split(":", 1)[0].strip())
            continue
        if param.startswith("*"):
            name = param.split(":", 1)[0].strip()
            arguments.append(name)
            continue
        name = param.split(":", 1)[0].split("=", 1)[0].strip()
        if name in {"self", "cls"}:
            continue
        if "=" in param:
            arguments.append(f"{name}=...")
        else:
            arguments.append(name)
    return f"{symbol}({', '.join(arguments)})"


def _split_python_parameters(params: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    for char in params:
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current))
    return parts


def _evaluate_call_stub_generation_method(
    method: str,
    questions: list[dict[str, str]],
    signature_records: dict[str, str],
) -> dict[str, Any]:
    answers: list[dict[str, Any]] = []
    latencies: list[float] = []
    for question in questions:
        start = time.perf_counter()
        if method == "frozen_parametric_proxy":
            answer = UNKNOWN
        elif method == "name_only_call_stub":
            answer = f"{question['symbol']}()"
        elif method == "structured_signature_call_stub":
            signature = signature_records.get(
                f"{question['repo']}:{question['symbol']}",
                "",
            )
            answer = _call_stub_from_python_signature(signature)
        else:
            raise ValueError(f"unknown method: {method}")
        latencies.append(time.perf_counter() - start)
        answers.append(
            {
                "repo": question["repo"],
                "symbol": question["symbol"],
                "question": question["question"],
                "expected_call_stub": question["expected_call_stub"],
                "answer": answer,
                "correct": answer == question["expected_call_stub"],
            }
        )

    correct_count = sum(int(row["correct"]) for row in answers)
    if method == "structured_signature_call_stub":
        storage_bytes = sum(
            len(key.encode()) + len(value.encode()) + 8
            for key, value in signature_records.items()
        )
    else:
        storage_bytes = 96
    return {
        "answer_accuracy": correct_count / len(answers) if answers else 0.0,
        "correct_count": correct_count,
        "question_count": len(answers),
        "storage_bytes": storage_bytes,
        "latency": p50_p95_ms(latencies),
        "answers": answers,
    }


def run_public_repository_function_skeleton_generation_experiment(
    repo_paths: list[Any],
    max_functions_per_repo: int = 32,
) -> dict[str, Any]:
    repo_summaries: list[dict[str, Any]] = []
    questions: list[dict[str, str]] = []
    signature_records: dict[str, str] = {}

    for raw_path in repo_paths:
        repo_path = Path(raw_path)
        if not repo_path.exists() or not (repo_path / ".git").exists():
            continue

        commit = _git(repo_path, "rev-parse", "HEAD")
        signatures = _python_signature_index_at(repo_path, commit)
        selected_symbols = sorted(signatures)[:max_functions_per_repo]
        if not selected_symbols:
            continue

        repo_key = repo_path.name
        repo_summaries.append(
            {
                "repo": repo_key,
                "commit": commit[:12],
                "python_signature_count_available": len(signatures),
                "python_signature_count_used": len(selected_symbols),
            }
        )
        for symbol in selected_symbols:
            signature = signatures[symbol]
            expected_skeleton = _function_skeleton_from_python_signature(signature)
            signature_records[f"{repo_key}:{symbol}"] = signature
            questions.append(
                {
                    "repo": repo_key,
                    "symbol": symbol,
                    "question": f"generate a minimal function skeleton for symbol {symbol}",
                    "signature": signature,
                    "expected_skeleton": expected_skeleton,
                }
            )

    if not repo_summaries:
        return {
            "experiment_family": "external_memory_architecture",
            "benchmark_label": "public_repository_function_skeleton_generation",
            "status": "skipped_no_public_repositories",
            "repo_count_requested": len(repo_paths),
            "repos": [],
            "question_count": 0,
            "methods": {},
            "final": {
                "repo_count_used": 0,
                "supports_model_level_claim": False,
            },
            "limitations": ["no_local_public_git_repositories_with_python_signatures"],
        }

    methods = {
        "frozen_parametric_proxy": _evaluate_function_skeleton_generation_method(
            "frozen_parametric_proxy",
            questions,
            signature_records,
        ),
        "name_only_skeleton": _evaluate_function_skeleton_generation_method(
            "name_only_skeleton",
            questions,
            signature_records,
        ),
        "structured_signature_skeleton": _evaluate_function_skeleton_generation_method(
            "structured_signature_skeleton",
            questions,
            signature_records,
        ),
    }
    structured = methods["structured_signature_skeleton"]
    name_only = methods["name_only_skeleton"]
    return {
        "experiment_family": "external_memory_architecture",
        "benchmark_label": "public_repository_function_skeleton_generation",
        "status": "completed",
        "repo_count_requested": len(repo_paths),
        "repos": repo_summaries,
        "question_count": len(questions),
        "methods": methods,
        "final": {
            "repo_count_used": len(repo_summaries),
            "structured_beats_name_only_skeleton": structured["answer_accuracy"]
            > name_only["answer_accuracy"],
            "structured_answer_accuracy": structured["answer_accuracy"],
            "name_only_answer_accuracy": name_only["answer_accuracy"],
            "supports_model_level_claim": False,
        },
        "limitations": [
            "python_signature_only",
            "deterministic_skeleton_body_only",
            "not_executable_behavior_synthesis",
            "not_semantic_code_generation",
            "public_repositories_only",
        ],
    }


def _function_skeleton_from_python_signature(signature: str) -> str:
    match = PY_SIGNATURE_RE.match(signature)
    if not match:
        return UNKNOWN
    return f"{signature}\n    raise NotImplementedError(\"generated skeleton\")"


def _evaluate_function_skeleton_generation_method(
    method: str,
    questions: list[dict[str, str]],
    signature_records: dict[str, str],
) -> dict[str, Any]:
    answers: list[dict[str, Any]] = []
    latencies: list[float] = []
    for question in questions:
        start = time.perf_counter()
        if method == "frozen_parametric_proxy":
            answer = UNKNOWN
        elif method == "name_only_skeleton":
            answer = (
                f"def {question['symbol']}():\n"
                "    raise NotImplementedError(\"generated skeleton\")"
            )
        elif method == "structured_signature_skeleton":
            signature = signature_records.get(
                f"{question['repo']}:{question['symbol']}",
                "",
            )
            answer = _function_skeleton_from_python_signature(signature)
        else:
            raise ValueError(f"unknown method: {method}")
        latencies.append(time.perf_counter() - start)
        answers.append(
            {
                "repo": question["repo"],
                "symbol": question["symbol"],
                "question": question["question"],
                "expected_skeleton": question["expected_skeleton"],
                "answer": answer,
                "correct": answer == question["expected_skeleton"],
            }
        )

    correct_count = sum(int(row["correct"]) for row in answers)
    if method == "structured_signature_skeleton":
        storage_bytes = sum(
            len(key.encode()) + len(value.encode()) + 8
            for key, value in signature_records.items()
        )
    else:
        storage_bytes = 96
    return {
        "answer_accuracy": correct_count / len(answers) if answers else 0.0,
        "correct_count": correct_count,
        "question_count": len(answers),
        "storage_bytes": storage_bytes,
        "latency": p50_p95_ms(latencies),
        "answers": answers,
    }


def run_public_repository_docstring_skeleton_generation_experiment(
    repo_paths: list[Any],
    max_functions_per_repo: int = 32,
) -> dict[str, Any]:
    repo_summaries: list[dict[str, Any]] = []
    questions: list[dict[str, str]] = []
    docstring_records: dict[str, dict[str, str]] = {}

    for raw_path in repo_paths:
        repo_path = Path(raw_path)
        if not repo_path.exists() or not (repo_path / ".git").exists():
            continue

        commit = _git(repo_path, "rev-parse", "HEAD")
        documented_functions = _python_documented_function_index_at(repo_path, commit)
        selected_symbols = sorted(documented_functions)[:max_functions_per_repo]
        if not selected_symbols:
            continue

        repo_key = repo_path.name
        repo_summaries.append(
            {
                "repo": repo_key,
                "commit": commit[:12],
                "documented_function_count_available": len(documented_functions),
                "documented_function_count_used": len(selected_symbols),
            }
        )
        for symbol in selected_symbols:
            record = documented_functions[symbol]
            docstring_records[f"{repo_key}:{symbol}"] = record
            expected_skeleton = _docstring_skeleton_from_record(record)
            questions.append(
                {
                    "repo": repo_key,
                    "symbol": symbol,
                    "question": (
                        "generate a minimal documented function skeleton "
                        f"for symbol {symbol}"
                    ),
                    "signature": record["signature"],
                    "docstring_summary": record["docstring_summary"],
                    "expected_skeleton": expected_skeleton,
                }
            )

    if not repo_summaries:
        return {
            "experiment_family": "external_memory_architecture",
            "benchmark_label": "public_repository_docstring_skeleton_generation",
            "status": "skipped_no_public_repositories",
            "repo_count_requested": len(repo_paths),
            "repos": [],
            "question_count": 0,
            "methods": {},
            "final": {
                "repo_count_used": 0,
                "supports_model_level_claim": False,
            },
            "limitations": [
                "no_local_public_git_repositories_with_python_function_docstrings"
            ],
        }

    methods = {
        "frozen_parametric_proxy": _evaluate_docstring_skeleton_generation_method(
            "frozen_parametric_proxy",
            questions,
            docstring_records,
        ),
        "signature_only_skeleton": _evaluate_docstring_skeleton_generation_method(
            "signature_only_skeleton",
            questions,
            docstring_records,
        ),
        "structured_docstring_skeleton": _evaluate_docstring_skeleton_generation_method(
            "structured_docstring_skeleton",
            questions,
            docstring_records,
        ),
    }
    structured = methods["structured_docstring_skeleton"]
    signature_only = methods["signature_only_skeleton"]
    return {
        "experiment_family": "external_memory_architecture",
        "benchmark_label": "public_repository_docstring_skeleton_generation",
        "status": "completed",
        "repo_count_requested": len(repo_paths),
        "repos": repo_summaries,
        "question_count": len(questions),
        "methods": methods,
        "final": {
            "repo_count_used": len(repo_summaries),
            "structured_beats_signature_only_skeleton": structured["answer_accuracy"]
            > signature_only["answer_accuracy"],
            "structured_answer_accuracy": structured["answer_accuracy"],
            "signature_only_answer_accuracy": signature_only["answer_accuracy"],
            "supports_model_level_claim": False,
        },
        "limitations": [
            "python_signature_and_docstring_only",
            "first_docstring_line_only",
            "deterministic_skeleton_body_only",
            "not_executable_behavior_synthesis",
            "not_semantic_code_generation",
            "public_repositories_only",
        ],
    }


def _python_documented_function_index_at(
    repo_path: Path, commit: str
) -> dict[str, dict[str, str]]:
    documented: dict[str, dict[str, str]] = {}
    for file_path in _files_at(repo_path, commit):
        if not file_path.endswith(".py"):
            continue
        text = _file_text(repo_path, commit, file_path, byte_limit=24000)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            docstring = ast.get_docstring(node)
            if not docstring:
                continue
            signature = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
            if not PY_SIGNATURE_RE.match(signature):
                continue
            summary = docstring.strip().splitlines()[0].strip()
            if not summary:
                continue
            documented.setdefault(
                node.name,
                {
                    "signature": signature,
                    "docstring_summary": summary,
                    "file": file_path,
                },
            )
    return documented


def _docstring_skeleton_from_record(record: dict[str, str]) -> str:
    signature = record.get("signature", UNKNOWN)
    summary = record.get("docstring_summary", "")
    if signature == UNKNOWN or not summary:
        return UNKNOWN
    escaped_summary = summary.replace('"""', r"\"\"\"")
    return (
        f"{signature}\n"
        f"    \"\"\"{escaped_summary}\"\"\"\n"
        "    raise NotImplementedError(\"generated skeleton\")"
    )


def _evaluate_docstring_skeleton_generation_method(
    method: str,
    questions: list[dict[str, str]],
    docstring_records: dict[str, dict[str, str]],
) -> dict[str, Any]:
    answers: list[dict[str, Any]] = []
    latencies: list[float] = []
    for question in questions:
        start = time.perf_counter()
        if method == "frozen_parametric_proxy":
            answer = UNKNOWN
        elif method == "signature_only_skeleton":
            answer = _function_skeleton_from_python_signature(question["signature"])
        elif method == "structured_docstring_skeleton":
            record = docstring_records.get(
                f"{question['repo']}:{question['symbol']}",
                {},
            )
            answer = _docstring_skeleton_from_record(record)
        else:
            raise ValueError(f"unknown method: {method}")
        latencies.append(time.perf_counter() - start)
        answers.append(
            {
                "repo": question["repo"],
                "symbol": question["symbol"],
                "question": question["question"],
                "expected_skeleton": question["expected_skeleton"],
                "answer": answer,
                "correct": answer == question["expected_skeleton"],
            }
        )

    correct_count = sum(int(row["correct"]) for row in answers)
    if method == "structured_docstring_skeleton":
        storage_bytes = sum(
            len(key.encode())
            + len(record["signature"].encode())
            + len(record["docstring_summary"].encode())
            + len(record["file"].encode())
            + 8
            for key, record in docstring_records.items()
        )
    elif method == "signature_only_skeleton":
        storage_bytes = sum(
            len(key.encode()) + len(record["signature"].encode()) + 8
            for key, record in docstring_records.items()
        )
    else:
        storage_bytes = 96
    return {
        "answer_accuracy": correct_count / len(answers) if answers else 0.0,
        "correct_count": correct_count,
        "question_count": len(answers),
        "storage_bytes": storage_bytes,
        "latency": p50_p95_ms(latencies),
        "answers": answers,
    }


def run_public_repository_api_reference_generation_experiment(
    repo_paths: list[Any],
    max_functions_per_repo: int = 32,
) -> dict[str, Any]:
    repo_summaries: list[dict[str, Any]] = []
    questions: list[dict[str, str]] = []
    docstring_records: dict[str, dict[str, str]] = {}

    for raw_path in repo_paths:
        repo_path = Path(raw_path)
        if not repo_path.exists() or not (repo_path / ".git").exists():
            continue

        commit = _git(repo_path, "rev-parse", "HEAD")
        documented_functions = _python_documented_function_index_at(repo_path, commit)
        selected_symbols = sorted(documented_functions)[:max_functions_per_repo]
        if not selected_symbols:
            continue

        repo_key = repo_path.name
        repo_summaries.append(
            {
                "repo": repo_key,
                "commit": commit[:12],
                "documented_function_count_available": len(documented_functions),
                "documented_function_count_used": len(selected_symbols),
            }
        )
        for symbol in selected_symbols:
            record = documented_functions[symbol]
            docstring_records[f"{repo_key}:{symbol}"] = record
            expected_reference = _api_reference_from_record(symbol, record)
            questions.append(
                {
                    "repo": repo_key,
                    "symbol": symbol,
                    "question": f"generate an API reference entry for symbol {symbol}",
                    "signature": record["signature"],
                    "docstring_summary": record["docstring_summary"],
                    "expected_reference": expected_reference,
                }
            )

    if not repo_summaries:
        return {
            "experiment_family": "external_memory_architecture",
            "benchmark_label": "public_repository_api_reference_generation",
            "status": "skipped_no_public_repositories",
            "repo_count_requested": len(repo_paths),
            "repos": [],
            "question_count": 0,
            "methods": {},
            "final": {
                "repo_count_used": 0,
                "supports_model_level_claim": False,
            },
            "limitations": [
                "no_local_public_git_repositories_with_python_function_docstrings"
            ],
        }

    methods = {
        "frozen_parametric_proxy": _evaluate_api_reference_generation_method(
            "frozen_parametric_proxy",
            questions,
            docstring_records,
        ),
        "signature_only_api_reference": _evaluate_api_reference_generation_method(
            "signature_only_api_reference",
            questions,
            docstring_records,
        ),
        "structured_docstring_api_reference": _evaluate_api_reference_generation_method(
            "structured_docstring_api_reference",
            questions,
            docstring_records,
        ),
    }
    structured = methods["structured_docstring_api_reference"]
    signature_only = methods["signature_only_api_reference"]
    return {
        "experiment_family": "external_memory_architecture",
        "benchmark_label": "public_repository_api_reference_generation",
        "status": "completed",
        "repo_count_requested": len(repo_paths),
        "repos": repo_summaries,
        "question_count": len(questions),
        "methods": methods,
        "final": {
            "repo_count_used": len(repo_summaries),
            "structured_beats_signature_only_reference": structured[
                "answer_accuracy"
            ]
            > signature_only["answer_accuracy"],
            "structured_answer_accuracy": structured["answer_accuracy"],
            "signature_only_answer_accuracy": signature_only["answer_accuracy"],
            "supports_model_level_claim": False,
        },
        "limitations": [
            "python_signature_and_docstring_only",
            "first_docstring_line_only",
            "deterministic_api_reference_template",
            "not_quality_rated_documentation",
            "not_semantic_documentation_generation",
            "public_repositories_only",
        ],
    }


def _api_reference_from_record(symbol: str, record: dict[str, str]) -> str:
    signature = record.get("signature", UNKNOWN)
    summary = record.get("docstring_summary", "")
    file_path = record.get("file", "")
    if signature == UNKNOWN or not summary or not file_path:
        return UNKNOWN
    return (
        f"### {symbol}\n\n"
        f"- Defined in: `{file_path}`\n"
        f"- Signature: `{signature}`\n"
        f"- Summary: {summary}"
    )


def _evaluate_api_reference_generation_method(
    method: str,
    questions: list[dict[str, str]],
    docstring_records: dict[str, dict[str, str]],
) -> dict[str, Any]:
    answers: list[dict[str, Any]] = []
    latencies: list[float] = []
    for question in questions:
        start = time.perf_counter()
        if method == "frozen_parametric_proxy":
            answer = UNKNOWN
        elif method == "signature_only_api_reference":
            answer = (
                f"### {question['symbol']}\n\n"
                f"- Signature: `{question['signature']}`"
            )
        elif method == "structured_docstring_api_reference":
            record = docstring_records.get(
                f"{question['repo']}:{question['symbol']}",
                {},
            )
            answer = _api_reference_from_record(question["symbol"], record)
        else:
            raise ValueError(f"unknown method: {method}")
        latencies.append(time.perf_counter() - start)
        answers.append(
            {
                "repo": question["repo"],
                "symbol": question["symbol"],
                "question": question["question"],
                "expected_reference": question["expected_reference"],
                "answer": answer,
                "correct": answer == question["expected_reference"],
            }
        )

    correct_count = sum(int(row["correct"]) for row in answers)
    if method == "structured_docstring_api_reference":
        storage_bytes = sum(
            len(key.encode())
            + len(record["signature"].encode())
            + len(record["docstring_summary"].encode())
            + len(record["file"].encode())
            + 8
            for key, record in docstring_records.items()
        )
    elif method == "signature_only_api_reference":
        storage_bytes = sum(
            len(key.encode()) + len(record["signature"].encode()) + 8
            for key, record in docstring_records.items()
        )
    else:
        storage_bytes = 96
    return {
        "answer_accuracy": correct_count / len(answers) if answers else 0.0,
        "correct_count": correct_count,
        "question_count": len(answers),
        "storage_bytes": storage_bytes,
        "latency": p50_p95_ms(latencies),
        "answers": answers,
    }


def run_public_repository_api_doc_coverage_qa_experiment(
    repo_paths: list[Any],
    max_symbols_per_repo: int = 32,
) -> dict[str, Any]:
    repo_summaries: list[dict[str, Any]] = []
    questions: list[dict[str, str]] = []
    source_records: dict[str, dict[str, str]] = {}
    doc_records: dict[str, str] = {}

    for raw_path in repo_paths:
        repo_path = Path(raw_path)
        if not repo_path.exists() or not (repo_path / ".git").exists():
            continue

        commit = _git(repo_path, "rev-parse", "HEAD")
        documented_functions = _python_documented_function_index_at(repo_path, commit)
        doc_index = _sphinx_api_doc_index_at(repo_path, commit)
        selected_symbols = [
            symbol
            for symbol in sorted(documented_functions)
            if symbol in doc_index
        ][:max_symbols_per_repo]
        if not selected_symbols:
            continue

        repo_key = repo_path.name
        repo_summaries.append(
            {
                "repo": repo_key,
                "commit": commit[:12],
                "documented_function_count_available": len(documented_functions),
                "doc_directive_symbol_count_available": len(doc_index),
                "covered_symbol_count_used": len(selected_symbols),
            }
        )
        for symbol in selected_symbols:
            source_records[f"{repo_key}:{symbol}"] = documented_functions[symbol]
            doc_records[f"{repo_key}:{symbol}"] = doc_index[symbol]
            questions.append(
                {
                    "repo": repo_key,
                    "symbol": symbol,
                    "question": f"which documentation file covers API symbol {symbol}",
                    "expected_doc_file": doc_index[symbol],
                }
            )

    if not repo_summaries:
        return {
            "experiment_family": "external_memory_architecture",
            "benchmark_label": "public_repository_api_doc_coverage_qa",
            "status": "skipped_no_public_repositories",
            "repo_count_requested": len(repo_paths),
            "repos": [],
            "question_count": 0,
            "methods": {},
            "final": {
                "repo_count_used": 0,
                "supports_model_level_claim": False,
            },
            "limitations": [
                "no_local_public_git_repositories_with_api_doc_directive_coverage"
            ],
        }

    methods = {
        "frozen_parametric_proxy": _evaluate_api_doc_coverage_method(
            "frozen_parametric_proxy",
            questions,
            source_records,
            doc_records,
        ),
        "source_symbol_only_memory": _evaluate_api_doc_coverage_method(
            "source_symbol_only_memory",
            questions,
            source_records,
            doc_records,
        ),
        "structured_doc_directive_memory": _evaluate_api_doc_coverage_method(
            "structured_doc_directive_memory",
            questions,
            source_records,
            doc_records,
        ),
    }
    structured = methods["structured_doc_directive_memory"]
    source_only = methods["source_symbol_only_memory"]
    return {
        "experiment_family": "external_memory_architecture",
        "benchmark_label": "public_repository_api_doc_coverage_qa",
        "status": "completed",
        "repo_count_requested": len(repo_paths),
        "repos": repo_summaries,
        "question_count": len(questions),
        "methods": methods,
        "final": {
            "repo_count_used": len(repo_summaries),
            "structured_beats_source_only": structured["answer_accuracy"]
            > source_only["answer_accuracy"],
            "structured_answer_accuracy": structured["answer_accuracy"],
            "source_only_answer_accuracy": source_only["answer_accuracy"],
            "supports_model_level_claim": False,
        },
        "limitations": [
            "sphinx_directive_coverage_only",
            "documented_python_functions_only",
            "exact_doc_file_lookup_only",
            "not_documentation_quality",
            "not_semantic_documentation_generation",
            "public_repositories_only",
        ],
    }


def _sphinx_api_doc_index_at(repo_path: Path, commit: str) -> dict[str, str]:
    doc_index: dict[str, str] = {}
    for file_path in _files_at(repo_path, commit):
        if not file_path.endswith((".rst", ".md")):
            continue
        text = _file_text(repo_path, commit, file_path, byte_limit=32000)
        for match in SPHINX_AUTOFUNCTION_RE.finditer(text):
            symbol = match.group(1).rsplit(".", 1)[-1]
            doc_index.setdefault(symbol, file_path)
        for match in SPHINX_AUTOCLASS_RE.finditer(text):
            class_symbol = match.group(1).rsplit(".", 1)[-1]
            doc_index.setdefault(class_symbol, file_path)
            tail = text[match.end(): match.end() + 400]
            members = SPHINX_MEMBERS_RE.search(tail)
            if members:
                for raw_member in members.group(1).split(","):
                    member = raw_member.strip().split(".")[-1]
                    if member:
                        doc_index.setdefault(member, file_path)
    return doc_index


def _evaluate_api_doc_coverage_method(
    method: str,
    questions: list[dict[str, str]],
    source_records: dict[str, dict[str, str]],
    doc_records: dict[str, str],
) -> dict[str, Any]:
    answers: list[dict[str, Any]] = []
    latencies: list[float] = []
    for question in questions:
        start = time.perf_counter()
        if method == "frozen_parametric_proxy":
            answer = UNKNOWN
        elif method == "source_symbol_only_memory":
            key = f"{question['repo']}:{question['symbol']}"
            answer = source_records[key]["file"] if key in source_records else UNKNOWN
        elif method == "structured_doc_directive_memory":
            answer = doc_records.get(
                f"{question['repo']}:{question['symbol']}",
                UNKNOWN,
            )
        else:
            raise ValueError(f"unknown method: {method}")
        latencies.append(time.perf_counter() - start)
        answers.append(
            {
                "repo": question["repo"],
                "symbol": question["symbol"],
                "question": question["question"],
                "expected_doc_file": question["expected_doc_file"],
                "answer": answer,
                "correct": answer == question["expected_doc_file"],
            }
        )

    correct_count = sum(int(row["correct"]) for row in answers)
    if method == "structured_doc_directive_memory":
        storage_bytes = sum(
            len(key.encode()) + len(value.encode()) + 8
            for key, value in doc_records.items()
        )
    elif method == "source_symbol_only_memory":
        storage_bytes = sum(
            len(key.encode())
            + len(record["signature"].encode())
            + len(record["file"].encode())
            + 8
            for key, record in source_records.items()
        )
    else:
        storage_bytes = 96
    return {
        "answer_accuracy": correct_count / len(answers) if answers else 0.0,
        "correct_count": correct_count,
        "question_count": len(answers),
        "storage_bytes": storage_bytes,
        "latency": p50_p95_ms(latencies),
        "answers": answers,
    }


def run_synthetic_api_doc_drift_detection_experiment(
    repo_path: Any | None = None,
) -> dict[str, Any]:
    if repo_path is None:
        with tempfile.TemporaryDirectory(prefix="weightlab_api_doc_drift_") as tmp:
            repo = Path(tmp) / "synthetic_api_doc_drift"
            _create_synthetic_api_doc_drift_repo(repo)
            return run_synthetic_api_doc_drift_detection_experiment(repo)

    repo = Path(repo_path)
    if not repo.exists() or not (repo / ".git").exists():
        return {
            "experiment_family": "external_memory_architecture",
            "benchmark_label": "synthetic_api_doc_drift_detection",
            "status": "skipped_no_repository",
            "repos": [],
            "question_count": 0,
            "methods": {},
            "final": {
                "repo_count_used": 0,
                "stale_doc_issue_count": 0,
                "supports_model_level_claim": False,
            },
            "limitations": ["no_local_synthetic_git_repository"],
        }

    commit = _git(repo, "rev-parse", "HEAD")
    signatures = _python_signature_index_at(repo, commit)
    doc_index = _sphinx_api_doc_index_at(repo, commit)
    questions: list[dict[str, str]] = []
    for symbol, doc_file in sorted(doc_index.items()):
        expected_status = (
            f"consistent:{doc_file}"
            if symbol in signatures
            else f"stale_missing_source:{doc_file}"
        )
        questions.append(
            {
                "repo": repo.name,
                "symbol": symbol,
                "question": (
                    f"is documentation for API symbol {symbol} consistent with "
                    "current source?"
                ),
                "doc_file": doc_file,
                "expected_status": expected_status,
            }
        )

    if not questions:
        return {
            "experiment_family": "external_memory_architecture",
            "benchmark_label": "synthetic_api_doc_drift_detection",
            "status": "skipped_no_api_doc_directives",
            "repos": [
                {
                    "repo": repo.name,
                    "commit": commit[:12],
                    "python_signature_count_available": len(signatures),
                    "doc_directive_symbol_count_available": 0,
                }
            ],
            "question_count": 0,
            "methods": {},
            "final": {
                "repo_count_used": 1,
                "stale_doc_issue_count": 0,
                "supports_model_level_claim": False,
            },
            "limitations": ["no_sphinx_api_doc_directives"],
        }

    methods = {
        "frozen_parametric_proxy": _evaluate_api_doc_drift_method(
            "frozen_parametric_proxy",
            questions,
            signatures,
            doc_index,
        ),
        "doc_directive_only_memory": _evaluate_api_doc_drift_method(
            "doc_directive_only_memory",
            questions,
            signatures,
            doc_index,
        ),
        "structured_source_doc_consistency_memory": _evaluate_api_doc_drift_method(
            "structured_source_doc_consistency_memory",
            questions,
            signatures,
            doc_index,
        ),
    }
    structured = methods["structured_source_doc_consistency_memory"]
    doc_only = methods["doc_directive_only_memory"]
    stale_doc_issue_count = sum(
        int(question["expected_status"].startswith("stale_missing_source:"))
        for question in questions
    )
    return {
        "experiment_family": "external_memory_architecture",
        "benchmark_label": "synthetic_api_doc_drift_detection",
        "status": "completed",
        "repos": [
            {
                "repo": repo.name,
                "commit": commit[:12],
                "python_signature_count_available": len(signatures),
                "doc_directive_symbol_count_available": len(doc_index),
                "stale_doc_issue_count": stale_doc_issue_count,
            }
        ],
        "question_count": len(questions),
        "methods": methods,
        "final": {
            "repo_count_used": 1,
            "stale_doc_issue_count": stale_doc_issue_count,
            "structured_beats_doc_only": structured["answer_accuracy"]
            > doc_only["answer_accuracy"],
            "structured_answer_accuracy": structured["answer_accuracy"],
            "doc_only_answer_accuracy": doc_only["answer_accuracy"],
            "supports_model_level_claim": False,
        },
        "limitations": [
            "synthetic_positive_control",
            "sphinx_directive_consistency_only",
            "python_signature_presence_only",
            "not_documentation_quality",
            "not_semantic_prose_drift_detection",
            "no_model_integration",
        ],
    }


def _evaluate_api_doc_drift_method(
    method: str,
    questions: list[dict[str, str]],
    signatures: dict[str, str],
    doc_index: dict[str, str],
) -> dict[str, Any]:
    answers: list[dict[str, Any]] = []
    latencies: list[float] = []
    for question in questions:
        start = time.perf_counter()
        symbol = question["symbol"]
        doc_file = doc_index.get(symbol, UNKNOWN)
        if method == "frozen_parametric_proxy":
            answer = UNKNOWN
        elif method == "doc_directive_only_memory":
            answer = f"covered:{doc_file}" if doc_file != UNKNOWN else UNKNOWN
        elif method == "structured_source_doc_consistency_memory":
            if doc_file == UNKNOWN:
                answer = UNKNOWN
            elif symbol in signatures:
                answer = f"consistent:{doc_file}"
            else:
                answer = f"stale_missing_source:{doc_file}"
        else:
            raise ValueError(f"unknown method: {method}")
        latencies.append(time.perf_counter() - start)
        answers.append(
            {
                "repo": question["repo"],
                "symbol": symbol,
                "question": question["question"],
                "expected_status": question["expected_status"],
                "answer": answer,
                "correct": answer == question["expected_status"],
            }
        )

    correct_count = sum(int(row["correct"]) for row in answers)
    if method == "structured_source_doc_consistency_memory":
        storage_bytes = sum(
            len(symbol.encode())
            + len(signature.encode())
            + 8
            for symbol, signature in signatures.items()
        ) + sum(
            len(symbol.encode()) + len(doc_file.encode()) + 8
            for symbol, doc_file in doc_index.items()
        )
    elif method == "doc_directive_only_memory":
        storage_bytes = sum(
            len(symbol.encode()) + len(doc_file.encode()) + 8
            for symbol, doc_file in doc_index.items()
        )
    else:
        storage_bytes = 96
    return {
        "answer_accuracy": correct_count / len(answers) if answers else 0.0,
        "correct_count": correct_count,
        "question_count": len(answers),
        "storage_bytes": storage_bytes,
        "latency": p50_p95_ms(latencies),
        "answers": answers,
    }


def _create_synthetic_api_doc_drift_repo(repo: Path) -> None:
    repo.mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "research@example.invalid")
    _git(repo, "config", "user.name", "Research Test")
    (repo / "config.py").write_text(
        "def parse_config(text: str) -> dict:\n"
        "    \"\"\"Parse legacy configuration text.\"\"\"\n"
        "    return {}\n",
        encoding="utf-8",
    )
    docs = repo / "docs"
    docs.mkdir()
    (docs / "api.rst").write_text(
        ".. module:: config\n\n"
        ".. autofunction:: parse_config\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add documented config api")
    (repo / "config.py").write_text(
        "def load_config(text: str) -> dict:\n"
        "    \"\"\"Load current configuration text.\"\"\"\n"
        "    return {}\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "rename source api without updating docs")
