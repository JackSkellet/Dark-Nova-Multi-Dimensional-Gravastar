from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any

TEXT_SUFFIXES = {
    ".py",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".md",
    ".rst",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
}

SECRET_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|secret|token)\s*=\s*['\"][^'\"]{16,}['\"]"),
]


def prepare_repository_corpus(
    repo_paths: list[Path],
    min_tokens: int = 50_000_000,
    max_file_bytes: int = 256_000,
) -> dict[str, Any]:
    documents: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    rejected_repositories: list[dict[str, str]] = []
    seen_hashes: set[str] = set()
    license_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()

    accepted_repos: list[Path] = []
    repo_license: dict[str, str] = {}
    for repo in repo_paths:
        repo = Path(repo)
        license_name = _detect_license(repo)
        if license_name not in {"MIT", "BSD", "Apache-2.0", "ISC"}:
            rejected_repositories.append(
                {
                    "repo": repo.name,
                    "reason": "unsupported_license",
                    "license": license_name or "unknown",
                }
            )
            continue
        accepted_repos.append(repo)
        repo_license[repo.name] = license_name
        license_counts[license_name] += 1

    split_by_repo = _repo_splits([repo.name for repo in accepted_repos])

    for repo in accepted_repos:
        for path in sorted(repo.rglob("*"), key=_file_priority):
            if not path.is_file() or _skip_path(path):
                continue
            relative = path.relative_to(repo).as_posix()
            if path.suffix.lower() not in TEXT_SUFFIXES and path.name.lower() not in {
                "license",
                "readme",
            }:
                continue
            if path.stat().st_size > max_file_bytes:
                excluded.append(_excluded(repo, relative, "too_large"))
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                excluded.append(_excluded(repo, relative, "non_utf8"))
                continue
            if _looks_generated(text, relative):
                excluded.append(_excluded(repo, relative, "generated_file"))
                continue
            if _contains_secret(text):
                excluded.append(_excluded(repo, relative, "secret_pattern"))
                continue
            normalized = _normalize_text(text)
            content_hash = hashlib.sha256(normalized.encode()).hexdigest()
            if content_hash in seen_hashes:
                excluded.append(_excluded(repo, relative, "duplicate_content"))
                continue
            seen_hashes.add(content_hash)
            token_count = _approx_tokens(text)
            if token_count == 0:
                excluded.append(_excluded(repo, relative, "empty"))
                continue
            language = _language_for(path)
            role = _role_for(relative)
            split = split_by_repo.get(repo.name, "train")
            language_counts[language] += 1
            role_counts[role] += 1
            split_counts[split] += 1
            documents.append(
                {
                    "repo": repo.name,
                    "relative_path": relative,
                    "license": repo_license[repo.name],
                    "language": language,
                    "role": role,
                    "split": split,
                    "tokens": token_count,
                    "bytes": len(text.encode()),
                    "sha256": content_hash,
                }
            )

    total_tokens = sum(int(doc["tokens"]) for doc in documents)
    return {
        "benchmark_label": "licensed_repository_corpus_preparation",
        "status": "completed" if total_tokens >= min_tokens else "insufficient_tokens",
        "repo_count": len(accepted_repos),
        "rejected_repositories": rejected_repositories,
        "documents": documents,
        "excluded_documents": excluded,
        "document_count": len(documents),
        "total_tokens": total_tokens,
        "target_min_tokens": min_tokens,
        "meets_50m_token_requirement": total_tokens >= 50_000_000,
        "license_counts": dict(sorted(license_counts.items())),
        "languages": dict(sorted(language_counts.items())),
        "file_roles": dict(sorted(role_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "limitations": [
            "approximate_regex_token_count",
            "heuristic_license_detection",
            "heuristic_secret_scanning",
            "no_model_tokenizer_yet",
        ],
    }


def _detect_license(repo: Path) -> str:
    for name in [
        "LICENSE",
        "LICENSE.txt",
        "LICENSE.rst",
        "LICENSE-MIT",
        "LICENSE-APACHE",
        "COPYING",
    ]:
        path = repo / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if "mit license" in text:
            return "MIT"
        if "permission is hereby granted, free of charge" in text and (
            'the "software"' in text or "the software" in text or name == "LICENSE-MIT"
        ):
            return "MIT"
        if "apache license" in text and "version 2.0" in text:
            return "Apache-2.0"
        if "isc license" in text:
            return "ISC"
        if "redistribution and use in source and binary forms" in text:
            return "BSD"
        if "bsd" in text and "redistribution" in text:
            return "BSD"
    return "unknown"


def _skip_path(path: Path) -> bool:
    parts = set(path.parts)
    return bool(
        parts
        & {
            ".git",
            ".github",
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
            "node_modules",
            "dist",
            "build",
            ".mypy_cache",
        }
    )


def _looks_generated(text: str, relative: str) -> bool:
    lowered = text[:2000].lower()
    name = relative.lower()
    return bool(
        "generated by" in lowered
        or "do not edit" in lowered
        or name.endswith(".min.js")
        or "/vendor/" in f"/{name}"
    )


def _contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def _normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _approx_tokens(text: str) -> int:
    return len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]", text))


def _language_for(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".py": "Python",
        ".rs": "Rust",
        ".c": "C",
        ".h": "C/C++ Header",
        ".cpp": "C++",
        ".hpp": "C++ Header",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".jsx": "JavaScript",
        ".md": "Markdown",
        ".rst": "reStructuredText",
        ".toml": "TOML",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".json": "JSON",
    }.get(suffix, "Text")


def _role_for(relative: str) -> str:
    lowered = relative.lower()
    if "test" in lowered:
        return "test"
    if lowered.startswith("readme") or "/readme" in lowered:
        return "readme"
    if lowered.startswith("docs/") or "/docs/" in lowered or lowered.endswith(".rst"):
        return "docs"
    if lowered.endswith((".md", ".txt")):
        return "docs"
    return "code"


def _repo_splits(repo_names: list[str]) -> dict[str, str]:
    names = sorted(repo_names)
    if len(names) == 1:
        return {names[0]: "train"}
    if len(names) == 2:
        return {names[0]: "train", names[1]: "validation"}
    splits = {}
    for index, name in enumerate(names):
        if index % 10 == 0:
            splits[name] = "test"
        elif index % 5 == 0:
            splits[name] = "validation"
        else:
            splits[name] = "train"
    return splits


def _excluded(repo: Path, relative: str, reason: str) -> dict[str, str]:
    return {"repo": repo.name, "relative_path": relative, "reason": reason}


def _file_priority(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    relative = path.as_posix().lower()
    if name.startswith("readme"):
        priority = 0
    elif relative.endswith((".py", ".rs", ".c", ".h", ".cpp", ".hpp")):
        priority = 1
    elif "docs/" in relative:
        priority = 2
    else:
        priority = 3
    return priority, relative
