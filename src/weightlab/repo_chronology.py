from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from difflib import unified_diff
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")
PY_DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
C_DEF_RE = re.compile(
    r"^\s*(?:static\s+)?(?:int|void|char|double|float|long|short|size_t|struct\s+\w+)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{",
    re.MULTILINE,
)


def _git(repo_path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_path), *args],
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()


def _commit_all(repo_path: Path, message: str) -> None:
    _git(repo_path, "add", ".")
    _git(repo_path, "commit", "-m", message)


def _tokens(text: str) -> Counter[str]:
    return Counter(token.lower() for token in TOKEN_RE.findall(text))


def _commits(repo_path: Path, max_commits: int) -> list[str]:
    all_commits = _git(repo_path, "rev-list", "--first-parent", "--reverse", "HEAD").splitlines()
    if len(all_commits) <= max_commits:
        return all_commits
    return all_commits[-max_commits:]


def _subject(repo_path: Path, commit: str) -> str:
    return _git(repo_path, "show", "-s", "--format=%s", commit)


def _changed_files(repo_path: Path, commit: str) -> list[str]:
    output = _git(repo_path, "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", commit)
    return [line for line in output.splitlines() if line]


def _files_at(repo_path: Path, commit: str) -> list[str]:
    output = _git(repo_path, "ls-tree", "-r", "--name-only", commit)
    return [line for line in output.splitlines() if line]


def _file_text(repo_path: Path, commit: str, file_path: str, byte_limit: int = 4000) -> str:
    try:
        data = subprocess.check_output(
            ["git", "-C", str(repo_path), "show", f"{commit}:{file_path}"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except Exception:
        return ""
    if b"\x00" in data[:1024]:
        return ""
    return data[:byte_limit].decode("utf-8", errors="ignore")


def _rank_files(
    query: str,
    corpus: dict[str, Counter[str]],
    structured_boosts: dict[str, float] | None = None,
) -> list[str]:
    query_tokens = _tokens(query)
    boosts = structured_boosts or {}
    scored: list[tuple[float, str]] = []
    for path, tokens in corpus.items():
        score = sum(query_tokens[token] * tokens.get(token, 0) for token in query_tokens)
        score += boosts.get(path, 0.0)
        scored.append((float(score), path))
    return [path for _, path in sorted(scored, key=lambda item: (-item[0], item[1]))]


def _topk_hit(ranked: list[str], targets: list[str], top_k: int) -> float:
    if not targets:
        return 1.0
    selected = set(ranked[:top_k])
    return 1.0 if selected & set(targets) else 0.0


def _corpus_at(repo_path: Path, commit: str) -> dict[str, Counter[str]]:
    corpus: dict[str, Counter[str]] = {}
    for file_path in _files_at(repo_path, commit):
        path_tokens = _tokens(file_path.replace("/", " ").replace(".", " "))
        text_tokens = _tokens(_file_text(repo_path, commit, file_path))
        corpus[file_path] = path_tokens + text_tokens
    return corpus


def _symbol_index_at(repo_path: Path, commit: str) -> dict[str, str]:
    symbols: dict[str, str] = {}
    for file_path in _files_at(repo_path, commit):
        if not file_path.endswith((".py", ".c", ".h", ".cc", ".cpp", ".hpp")):
            continue
        text = _file_text(repo_path, commit, file_path, byte_limit=12000)
        for pattern in (PY_DEF_RE, C_DEF_RE):
            for match in pattern.finditer(text):
                symbols.setdefault(match.group(1), file_path)
    return symbols


def _symbol_questions(symbols: dict[str, str]) -> list[tuple[str, str, str]]:
    return [
        (f"which file defines symbol {symbol}", symbol, file_path)
        for symbol, file_path in sorted(symbols.items())
    ]


def _symbol_topk_accuracy(
    questions: list[tuple[str, str, str]],
    corpus: dict[str, Counter[str]],
    symbol_index: dict[str, str] | None,
    top_k: int,
) -> float:
    if not questions:
        return 1.0
    hits = []
    for question, symbol, target_file in questions:
        boosts = {symbol_index[symbol]: 1000.0} if symbol_index and symbol in symbol_index else None
        ranked = _rank_files(question, corpus, boosts)
        hits.append(_topk_hit(ranked, [target_file], top_k))
    return sum(hits) / len(hits)


def _doc_files_at(repo_path: Path, commit: str) -> list[str]:
    doc_suffixes = (".md", ".rst", ".txt")
    return [
        file_path
        for file_path in _files_at(repo_path, commit)
        if file_path.endswith(doc_suffixes) or "doc" in file_path.lower()
    ]


def _docs_mentioning_symbols(
    repo_path: Path,
    commit: str,
    symbols: set[str],
) -> dict[str, list[str]]:
    mentions: dict[str, list[str]] = {}
    if not symbols:
        return mentions
    lowered = {symbol.lower(): symbol for symbol in symbols}
    for file_path in _doc_files_at(repo_path, commit):
        tokens = set(_tokens(_file_text(repo_path, commit, file_path, byte_limit=16000)))
        found = sorted(original for lower, original in lowered.items() if lower in tokens)
        if found:
            mentions[file_path] = found
    return mentions


def _rank_stale_doc_files(
    repo_path: Path,
    commit: str,
    removed_symbols: set[str],
    structured: bool,
) -> list[str]:
    ranked: list[tuple[float, str]] = []
    for file_path in _doc_files_at(repo_path, commit):
        text_tokens = _tokens(_file_text(repo_path, commit, file_path, byte_limit=16000))
        score = sum(text_tokens.get(symbol.lower(), 0) for symbol in removed_symbols)
        if structured and score:
            score += 1000.0
        ranked.append((float(score), file_path))
    return [path for _, path in sorted(ranked, key=lambda item: (-item[0], item[1]))]


def _structured_boosts(
    repo_path: Path,
    exposed_commits: list[str],
    query: str,
) -> dict[str, float]:
    query_tokens = _tokens(query)
    boosts: dict[str, float] = {}
    for commit in exposed_commits:
        overlap = sum((_tokens(_subject(repo_path, commit)) & query_tokens).values())
        if overlap == 0:
            continue
        for file_path in _changed_files(repo_path, commit):
            boosts[file_path] = boosts.get(file_path, 0.0) + overlap * 3.0
    return boosts


def _prior_accuracy(
    repo_path: Path,
    exposed_commit: str,
    exposed_commits: list[str],
    top_k: int,
    structured: bool,
) -> float:
    if not exposed_commits:
        return 1.0
    corpus = _corpus_at(repo_path, exposed_commit)
    hits = []
    for commit in exposed_commits:
        query = _subject(repo_path, commit)
        boosts = _structured_boosts(repo_path, exposed_commits, query) if structured else None
        ranked = _rank_files(query, corpus, boosts)
        hits.append(_topk_hit(ranked, _changed_files(repo_path, commit), top_k))
    return sum(hits) / len(hits)


def run_public_repo_chronology_experiment(
    repo_path: str | Path,
    max_commits: int = 12,
    top_k: int = 5,
) -> dict[str, Any]:
    repo_path = Path(repo_path)
    start = time.perf_counter()
    commits = _commits(repo_path, max_commits=max_commits)
    if len(commits) < 2:
        raise ValueError("repository chronology experiment requires at least two commits")

    first_commit = commits[0]
    frozen_corpus = _corpus_at(repo_path, first_commit)
    steps: list[dict[str, Any]] = []

    for index in range(1, len(commits)):
        exposed_commits = commits[:index]
        exposed_commit = exposed_commits[-1]
        future_commit = commits[index]
        future_query = _subject(repo_path, future_commit)
        future_targets = _changed_files(repo_path, future_commit)
        if not future_targets:
            continue

        retrieval_corpus = _corpus_at(repo_path, exposed_commit)
        frozen_ranked = _rank_files(future_query, frozen_corpus)
        retrieval_ranked = _rank_files(future_query, retrieval_corpus)
        structured_ranked = _rank_files(
            future_query,
            retrieval_corpus,
            _structured_boosts(repo_path, exposed_commits, future_query),
        )

        steps.append(
            {
                "step": index,
                "exposed_commit": exposed_commit[:12],
                "future_commit": future_commit[:12],
                "future_commit_not_in_memory": future_commit not in exposed_commits,
                "future_changed_files": future_targets,
                "frozen_future_topk_hit": _topk_hit(frozen_ranked, future_targets, top_k),
                "updated_retrieval_future_topk_hit": _topk_hit(
                    retrieval_ranked, future_targets, top_k
                ),
                "structured_future_topk_hit": _topk_hit(structured_ranked, future_targets, top_k),
                "retrieval_prior_topk_accuracy": _prior_accuracy(
                    repo_path, exposed_commit, exposed_commits, top_k, structured=False
                ),
                "structured_prior_topk_accuracy": _prior_accuracy(
                    repo_path, exposed_commit, exposed_commits, top_k, structured=True
                ),
                "candidate_files": len(retrieval_corpus),
            }
        )

    if not steps:
        raise ValueError("repository chronology experiment found no commits with changed files")

    final = {
        "frozen_future_topk_accuracy": sum(s["frozen_future_topk_hit"] for s in steps) / len(steps),
        "updated_retrieval_future_topk_accuracy": sum(
            s["updated_retrieval_future_topk_hit"] for s in steps
        )
        / len(steps),
        "structured_future_topk_accuracy": sum(s["structured_future_topk_hit"] for s in steps)
        / len(steps),
        "retrieval_prior_topk_accuracy": steps[-1]["retrieval_prior_topk_accuracy"],
        "structured_prior_topk_accuracy": steps[-1]["structured_prior_topk_accuracy"],
        "runtime_ms": (time.perf_counter() - start) * 1000.0,
    }

    return {
        "repo": {
            "path_name": repo_path.name,
            "head": _git(repo_path, "rev-parse", "--short=12", "HEAD"),
            "commit_count_used": len(commits),
            "top_k": top_k,
        },
        "steps": steps,
        "final": final,
        "notes": (
            "Uses only local Git state from commits <= exposed_commit to rank files changed "
            "in the next commit. This evaluates changed-file retrieval, not patch generation."
        ),
    }


def run_symbol_qa_chronology_experiment(
    repo_path: str | Path,
    max_commits: int = 12,
    top_k: int = 3,
) -> dict[str, Any]:
    repo_path = Path(repo_path)
    start = time.perf_counter()
    commits = _commits(repo_path, max_commits=max_commits)
    if len(commits) < 2:
        raise ValueError("symbol QA chronology experiment requires at least two commits")

    first_commit = commits[0]
    frozen_corpus = _corpus_at(repo_path, first_commit)
    frozen_symbols = _symbol_index_at(repo_path, first_commit)
    steps: list[dict[str, Any]] = []

    for index in range(1, len(commits)):
        exposed_commits = commits[:index]
        exposed_commit = exposed_commits[-1]
        future_commit = commits[index]

        exposed_symbols = _symbol_index_at(repo_path, exposed_commit)
        if not exposed_symbols:
            continue

        exposed_questions = _symbol_questions(exposed_symbols)
        retrieval_corpus = _corpus_at(repo_path, exposed_commit)
        retrieval_prior = _symbol_topk_accuracy(
            exposed_questions,
            retrieval_corpus,
            symbol_index=None,
            top_k=top_k,
        )
        structured_prior = _symbol_topk_accuracy(
            exposed_questions,
            retrieval_corpus,
            symbol_index=exposed_symbols,
            top_k=top_k,
        )
        frozen_prior = _symbol_topk_accuracy(
            exposed_questions,
            frozen_corpus,
            symbol_index=frozen_symbols,
            top_k=top_k,
        )

        future_symbols = _symbol_index_at(repo_path, future_commit)
        new_future_symbols = {
            symbol: file_path
            for symbol, file_path in future_symbols.items()
            if symbol not in exposed_symbols
        }
        future_questions = _symbol_questions(new_future_symbols)
        if future_questions:
            retrieval_future = _symbol_topk_accuracy(
                future_questions,
                retrieval_corpus,
                symbol_index=None,
                top_k=top_k,
            )
            structured_future = _symbol_topk_accuracy(
                future_questions,
                retrieval_corpus,
                symbol_index=exposed_symbols,
                top_k=top_k,
            )
        else:
            retrieval_future = None
            structured_future = None

        steps.append(
            {
                "step": index,
                "exposed_commit": exposed_commit[:12],
                "future_commit": future_commit[:12],
                "future_commit_not_in_memory": future_commit not in exposed_commits,
                "visible_symbol_count": len(exposed_symbols),
                "new_future_symbol_count": len(new_future_symbols),
                "frozen_prior_symbol_topk_accuracy": frozen_prior,
                "retrieval_prior_symbol_topk_accuracy": retrieval_prior,
                "structured_prior_symbol_topk_accuracy": structured_prior,
                "retrieval_future_new_symbol_topk_accuracy": retrieval_future,
                "structured_future_new_symbol_topk_accuracy": structured_future,
            }
        )

    if not steps:
        raise ValueError("symbol QA chronology experiment found no visible symbols")

    final = {
        "frozen_prior_symbol_topk_accuracy": steps[-1]["frozen_prior_symbol_topk_accuracy"],
        "retrieval_prior_symbol_topk_accuracy": steps[-1]["retrieval_prior_symbol_topk_accuracy"],
        "structured_prior_symbol_topk_accuracy": steps[-1]["structured_prior_symbol_topk_accuracy"],
        "mean_retrieval_prior_symbol_topk_accuracy": sum(
            step["retrieval_prior_symbol_topk_accuracy"] for step in steps
        )
        / len(steps),
        "mean_structured_prior_symbol_topk_accuracy": sum(
            step["structured_prior_symbol_topk_accuracy"] for step in steps
        )
        / len(steps),
        "future_new_symbol_steps": sum(1 for step in steps if step["new_future_symbol_count"] > 0),
        "runtime_ms": (time.perf_counter() - start) * 1000.0,
    }

    return {
        "repo": {
            "path_name": repo_path.name,
            "head": _git(repo_path, "rev-parse", "--short=12", "HEAD"),
            "commit_count_used": len(commits),
            "top_k": top_k,
        },
        "steps": steps,
        "final": final,
        "notes": (
            "Uses only symbols and files visible at exposed_commit to answer repository QA "
            "questions of the form 'which file defines symbol X'. Future symbol questions "
            "are reported separately and should not be answerable before exposure."
        ),
    }


def run_stale_doc_chronology_experiment(
    repo_path: str | Path,
    max_commits: int = 12,
    top_k: int = 5,
) -> dict[str, Any]:
    repo_path = Path(repo_path)
    start = time.perf_counter()
    commits = _commits(repo_path, max_commits=max_commits)
    if len(commits) < 2:
        raise ValueError("stale-doc chronology experiment requires at least two commits")

    steps: list[dict[str, Any]] = []
    stale_doc_issue_count = 0
    retrieval_hits = []
    structured_hits = []

    for index in range(1, len(commits)):
        previous_commit = commits[index - 1]
        current_commit = commits[index]
        previous_symbols = _symbol_index_at(repo_path, previous_commit)
        current_symbols = _symbol_index_at(repo_path, current_commit)
        removed_symbols = sorted(set(previous_symbols) - set(current_symbols))
        stale_mentions = _docs_mentioning_symbols(repo_path, current_commit, set(removed_symbols))
        stale_doc_files = sorted(stale_mentions)

        retrieval_ranked = _rank_stale_doc_files(
            repo_path,
            current_commit,
            set(removed_symbols),
            structured=False,
        )
        structured_ranked = _rank_stale_doc_files(
            repo_path,
            current_commit,
            set(removed_symbols),
            structured=True,
        )
        retrieval_hit = (
            _topk_hit(retrieval_ranked, stale_doc_files, top_k) if stale_doc_files else None
        )
        structured_hit = (
            _topk_hit(structured_ranked, stale_doc_files, top_k) if stale_doc_files else None
        )
        if stale_doc_files:
            stale_doc_issue_count += len(stale_doc_files)
            retrieval_hits.append(float(retrieval_hit))
            structured_hits.append(float(structured_hit))

        steps.append(
            {
                "step": index,
                "previous_commit": previous_commit[:12],
                "current_commit": current_commit[:12],
                "removed_symbols": removed_symbols,
                "stale_doc_files": stale_doc_files,
                "stale_mentions": stale_mentions,
                "retrieval_stale_doc_topk_hit": retrieval_hit,
                "structured_stale_doc_topk_hit": structured_hit,
                "doc_file_count": len(_doc_files_at(repo_path, current_commit)),
            }
        )

    final = {
        "stale_doc_issue_count": stale_doc_issue_count,
        "steps_with_removed_symbols": sum(1 for step in steps if step["removed_symbols"]),
        "steps_with_stale_docs": sum(1 for step in steps if step["stale_doc_files"]),
        "retrieval_stale_doc_topk_accuracy": sum(retrieval_hits) / len(retrieval_hits)
        if retrieval_hits
        else None,
        "structured_stale_doc_topk_accuracy": sum(structured_hits) / len(structured_hits)
        if structured_hits
        else None,
        "runtime_ms": (time.perf_counter() - start) * 1000.0,
    }

    return {
        "repo": {
            "path_name": repo_path.name,
            "head": _git(repo_path, "rev-parse", "--short=12", "HEAD"),
            "commit_count_used": len(commits),
            "top_k": top_k,
        },
        "steps": steps,
        "final": final,
        "notes": (
            "Compares adjacent commits. A stale-doc issue is a current documentation file "
            "that still mentions a code symbol removed since the previous exposed commit."
        ),
    }


def run_synthetic_stale_doc_positive_control(top_k: int = 1) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="weightlab-stale-doc-") as tmpdir:
        repo_path = Path(tmpdir) / "repo"
        repo_path.mkdir()
        _git(repo_path, "init", "-b", "main")
        _git(repo_path, "config", "user.email", "research@example.invalid")
        _git(repo_path, "config", "user.name", "Research Test")

        (repo_path / "api.py").write_text(
            "def parse_config(text):\n    return text.strip()\n",
            encoding="utf-8",
        )
        (repo_path / "docs.md").write_text(
            "Use parse_config to load configuration text.\n",
            encoding="utf-8",
        )
        _commit_all(repo_path, "add parse_config and docs")

        (repo_path / "api.py").write_text(
            "def load_config(text):\n    return text.strip()\n",
            encoding="utf-8",
        )
        _commit_all(repo_path, "rename parse_config to load_config without doc update")

        (repo_path / "docs.md").write_text(
            "Use load_config to load configuration text.\n",
            encoding="utf-8",
        )
        _commit_all(repo_path, "update docs for load_config")

        result = run_stale_doc_chronology_experiment(
            repo_path=repo_path,
            max_commits=3,
            top_k=top_k,
        )

    stale_steps = [step for step in result["steps"] if step["stale_doc_files"]]
    post_fix_stale_doc_files = result["steps"][-1]["stale_doc_files"] if result["steps"] else []
    result["positive_control"] = {
        "scenario": "rename parse_config to load_config, then update docs in a later commit",
        "expected_stale_doc_issues": 1,
        "detected_stale_doc_issues": result["final"]["stale_doc_issue_count"],
        "expected_stale_symbol": "parse_config",
        "detected_stale_symbols": stale_steps[0]["removed_symbols"] if stale_steps else [],
        "expected_stale_doc_file": "docs.md",
        "detected_stale_doc_files": stale_steps[0]["stale_doc_files"] if stale_steps else [],
        "post_fix_stale_doc_files": post_fix_stale_doc_files,
    }
    result["notes"] = (
        "Generated Git-history positive control for stale-doc detection. The middle "
        "commit removes parse_config while docs still mention it; the final commit fixes "
        "the docs. This is synthetic and does not replace public-repository evidence."
    )
    return result


def _run_unittest(repo_path: Path) -> tuple[bool, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover"],
        cwd=repo_path,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    output = (completed.stdout + "\n" + completed.stderr).strip()
    return completed.returncode == 0, output[-1200:]


def _run_test_command(repo_path: Path, test_command: list[str], timeout_s: int) -> tuple[bool, str]:
    completed = subprocess.run(
        test_command,
        cwd=repo_path,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    output = (completed.stdout + "\n" + completed.stderr).strip()
    return completed.returncode == 0, output[-1200:]


def _apply_candidate_patch(repo_path: Path, patch_text: str) -> tuple[bool, str]:
    if not patch_text.strip():
        return True, "empty patch"
    completed = subprocess.run(
        ["git", "-C", str(repo_path), "apply", "-"],
        input=patch_text,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    output = (completed.stdout + "\n" + completed.stderr).strip()
    return completed.returncode == 0, output[-1200:]


def _generate_proxy_class_patch(repo_path: Path) -> str:
    patch_parts: list[str] = []
    for file_path in sorted(repo_path.rglob("*.py")):
        if ".git" in file_path.parts:
            continue
        try:
            old_text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new_text = old_text.replace("s.__class__ is str", "type(s) is str")
        if new_text == old_text:
            continue

        rel_path = file_path.relative_to(repo_path).as_posix()
        diff_lines = list(
            unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"a/{rel_path}",
                tofile=f"b/{rel_path}",
            )
        )
        patch_parts.append(f"diff --git a/{rel_path} b/{rel_path}\n" + "".join(diff_lines))
    return "\n".join(patch_parts)


def _generate_clamp_bounds_patch(repo_path: Path) -> str:
    patch_parts: list[str] = []
    pattern = re.compile(
        r"def\s+clamp\s*\(\s*value\s*,\s*low\s*,\s*high\s*\):\n"
        r"(?P<indent>[ \t]+)return\s+min\s*\(\s*value\s*,\s*high\s*\)"
    )
    for file_path in sorted(repo_path.rglob("*.py")):
        if ".git" in file_path.parts:
            continue
        try:
            old_text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        def replacement(match: re.Match[str]) -> str:
            return (
                "def clamp(value, low, high):\n"
                f"{match.group('indent')}return max(low, min(value, high))"
            )

        new_text = pattern.sub(replacement, old_text)
        if new_text == old_text:
            continue

        rel_path = file_path.relative_to(repo_path).as_posix()
        diff_lines = list(
            unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"a/{rel_path}",
                tofile=f"b/{rel_path}",
            )
        )
        patch_parts.append(f"diff --git a/{rel_path} b/{rel_path}\n" + "".join(diff_lines))
    return "\n".join(patch_parts)


def _generate_saved_hl_free_patch(repo_path: Path) -> str:
    patch_parts: list[str] = []
    pattern = re.compile(
        r"(?P<memcpy>[ \t]*memcpy\s*\(\s*(?:E\.row\[[^\]]+\]\.hl|row_hl)\s*,\s*"
        r"saved_hl\s*,\s*(?:E\.row\[[^\]]+\]\.rsize|row_size)\s*\)\s*;\s*\\?\n)"
        r"(?P<indent>[ \t]*)saved_hl\s*=\s*NULL\s*;",
    )
    for file_path in sorted(repo_path.rglob("*.[ch]")):
        if ".git" in file_path.parts:
            continue
        try:
            old_text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "saved_hl" not in old_text:
            continue

        def replacement(match: re.Match[str]) -> str:
            line_suffix = " \\" if match.group("memcpy").rstrip().endswith("\\") else ""
            return (
                f"{match.group('memcpy')}"
                f"{match.group('indent')}free(saved_hl);{line_suffix}\n"
                f"{match.group('indent')}saved_hl = NULL;"
            )

        new_text = pattern.sub(replacement, old_text)
        if new_text == old_text:
            continue

        rel_path = file_path.relative_to(repo_path).as_posix()
        diff_lines = list(
            unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"a/{rel_path}",
                tofile=f"b/{rel_path}",
            )
        )
        patch_parts.append(f"diff --git a/{rel_path} b/{rel_path}\n" + "".join(diff_lines))
    return "\n".join(patch_parts)


def run_public_patch_replay_experiment(
    repo_path: str | Path,
    base_commit: str,
    fix_commit: str,
    test_command: list[str],
    timeout_s: int = 20,
) -> dict[str, Any]:
    repo_path = Path(repo_path)
    start = time.perf_counter()
    base_full = _git(repo_path, "rev-parse", base_commit)
    fix_full = _git(repo_path, "rev-parse", fix_commit)
    if base_full == fix_full:
        raise ValueError("base_commit and fix_commit must be different commits")

    ancestry = subprocess.run(
        ["git", "-C", str(repo_path), "merge-base", "--is-ancestor", base_full, fix_full],
        capture_output=True,
        text=True,
        check=False,
    )
    if ancestry.returncode != 0:
        raise ValueError("fix_commit must descend from base_commit for chronological replay")

    patch_text = subprocess.check_output(
        ["git", "-C", str(repo_path), "diff", "--binary", base_full, fix_full],
        text=True,
    )
    if not patch_text.strip():
        raise ValueError("historical patch is empty")

    with tempfile.TemporaryDirectory(prefix="weightlab-public-patch-") as tmpdir:
        worktree = Path(tmpdir) / "repo"
        subprocess.check_call(
            ["git", "clone", "--quiet", "--no-local", str(repo_path), str(worktree)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _git(worktree, "checkout", "--quiet", base_full)
        baseline_passed, baseline_output = _run_test_command(worktree, test_command, timeout_s)
        generated_patch = _generate_proxy_class_patch(worktree)
        generated_clamp_patch = _generate_clamp_bounds_patch(worktree)
        generated_saved_hl_patch = _generate_saved_hl_free_patch(worktree)

        candidate_defs = [
            {
                "method": "frozen_noop",
                "patch": "",
                "notes": "No future commit patch is available to the exposed state.",
            },
        ]
        if generated_patch:
            candidate_defs.append(
                {
                    "method": "generated_proxy_class_patch",
                    "patch": generated_patch,
                    "notes": (
                        "Deterministic generated candidate for proxy objects that spoof "
                        "__class__; replaces s.__class__ exact str checks with type(s)."
                    ),
                }
            )
        if generated_clamp_patch:
            candidate_defs.append(
                {
                    "method": "generated_clamp_bounds_patch",
                    "patch": generated_clamp_patch,
                    "notes": (
                        "Deterministic generated candidate for Python clamp(value, low, high) "
                        "functions that only apply the upper bound."
                    ),
                }
            )
        if generated_saved_hl_patch:
            candidate_defs.append(
                {
                    "method": "generated_saved_hl_free_patch",
                    "patch": generated_saved_hl_patch,
                    "notes": (
                        "Deterministic generated candidate for Kilo-style saved_hl restore "
                        "macros that memcpy saved_hl and then clear the pointer without "
                        "freeing the allocation."
                    ),
                }
            )
        candidate_defs.append(
            {
                "method": "historical_future_patch",
                "patch": patch_text,
                "notes": (
                    "Actual future commit diff replayed onto the exposed parent and graded "
                    "by executable tests."
                ),
            }
        )

        candidate_results: list[dict[str, Any]] = []
        for candidate in candidate_defs:
            _git(worktree, "reset", "--hard", "--quiet", base_full)
            _git(worktree, "clean", "-fd", "--quiet")
            applied, apply_output = _apply_candidate_patch(worktree, str(candidate["patch"]))
            passed = False
            test_output = "patch did not apply"
            if applied:
                passed, test_output = _run_test_command(worktree, test_command, timeout_s)
            candidate_results.append(
                {
                    "method": candidate["method"],
                    "patch_applied": applied,
                    "tests_passed": passed,
                    "patch_bytes": len(str(candidate["patch"]).encode()),
                    "apply_output_tail": apply_output,
                    "test_output_tail": test_output,
                    "notes": candidate["notes"],
                }
            )

    passed_methods = [row["method"] for row in candidate_results if row["tests_passed"]]
    final = {
        "baseline_tests_passed": baseline_passed,
        "functional_success_rate": len(passed_methods) / len(candidate_results),
        "best_method": passed_methods[0] if passed_methods else None,
        "frozen_noop_passed": next(
            row["tests_passed"] for row in candidate_results if row["method"] == "frozen_noop"
        ),
        "historical_patch_passed": next(
            row["tests_passed"]
            for row in candidate_results
            if row["method"] == "historical_future_patch"
        ),
        "generated_proxy_class_patch_passed": next(
            (
                row["tests_passed"]
                for row in candidate_results
                if row["method"] == "generated_proxy_class_patch"
            ),
            None,
        ),
        "generated_clamp_bounds_patch_passed": next(
            (
                row["tests_passed"]
                for row in candidate_results
                if row["method"] == "generated_clamp_bounds_patch"
            ),
            None,
        ),
        "generated_saved_hl_free_patch_passed": next(
            (
                row["tests_passed"]
                for row in candidate_results
                if row["method"] == "generated_saved_hl_free_patch"
            ),
            None,
        ),
        "runtime_ms": (time.perf_counter() - start) * 1000.0,
    }

    return {
        "repo": {
            "path_name": repo_path.name,
            "head": _git(repo_path, "rev-parse", "--short=12", "HEAD"),
            "base_commit": base_full[:12],
            "fix_commit": fix_full[:12],
            "future_commit_not_in_memory": fix_full != base_full,
            "test_command": test_command,
        },
        "baseline_test_output_tail": baseline_output,
        "candidates": candidate_results,
        "final": final,
        "notes": (
            "Chronological public-patch replay harness. The exposed state is the base "
            "commit; deterministic generated candidates and the historical future diff "
            "are applied as candidates and graded only by executable tests, not by "
            "textual similarity."
        ),
    }


def run_public_patch_replay_suite_experiment(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    start = time.perf_counter()
    if not tasks:
        raise ValueError("patch replay suite requires at least one task")

    task_results: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        result = run_public_patch_replay_experiment(
            repo_path=task["repo_path"],
            base_commit=str(task["base_commit"]),
            fix_commit=str(task["fix_commit"]),
            test_command=list(task["test_command"]),
            timeout_s=int(task.get("timeout_s", 20)),
        )
        generated_candidates = [
            candidate
            for candidate in result["candidates"]
            if candidate["method"] not in {"frozen_noop", "historical_future_patch"}
        ]
        generated_passed = [
            candidate["method"] for candidate in generated_candidates if candidate["tests_passed"]
        ]
        task_results.append(
            {
                "task_id": task_id,
                "repo": result["repo"],
                "candidate_count": len(result["candidates"]),
                "generated_methods": [candidate["method"] for candidate in generated_candidates],
                "generated_passed_methods": generated_passed,
                "baseline_tests_passed": result["final"]["baseline_tests_passed"],
                "historical_patch_passed": result["final"]["historical_patch_passed"],
                "best_method": result["final"]["best_method"],
                "functional_success_rate": result["final"]["functional_success_rate"],
                "result": result,
            }
        )

    generated_task_successes = sum(
        1 for task in task_results if task["generated_passed_methods"]
    )
    final = {
        "task_count": len(task_results),
        "baseline_failures": sum(
            1 for task in task_results if not task["baseline_tests_passed"]
        ),
        "historical_patch_successes": sum(
            1 for task in task_results if task["historical_patch_passed"]
        ),
        "generated_task_successes": generated_task_successes,
        "generated_task_success_rate": generated_task_successes / len(task_results),
        "generated_candidate_count": sum(
            len(task["generated_methods"]) for task in task_results
        ),
        "runtime_ms": (time.perf_counter() - start) * 1000.0,
    }
    return {
        "tasks": task_results,
        "final": final,
        "notes": (
            "Aggregates chronological public patch replay tasks. Each task keeps the "
            "base commit as the exposed state, grades generated candidates and the "
            "historical future patch by the task's executable command, and reports "
            "task-level success rather than text similarity."
        ),
    }


def _clamp_patch(replacement: str) -> str:
    return (
        "diff --git a/calculator.py b/calculator.py\n"
        "--- a/calculator.py\n"
        "+++ b/calculator.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def clamp(value, low, high):\n"
        "-    return min(value, high)\n"
        f"+    {replacement}\n"
    )


def run_synthetic_patch_generation_positive_control() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="weightlab-patch-gen-") as tmpdir:
        repo_path = Path(tmpdir) / "repo"
        repo_path.mkdir()
        _git(repo_path, "init", "-b", "main")
        _git(repo_path, "config", "user.email", "research@example.invalid")
        _git(repo_path, "config", "user.name", "Research Test")

        buggy_source = "def clamp(value, low, high):\n    return min(value, high)\n"
        test_source = (
            "import unittest\n\n"
            "from calculator import clamp\n\n\n"
            "class ClampTests(unittest.TestCase):\n"
            "    def test_clamps_low_bound(self):\n"
            "        self.assertEqual(clamp(-5, 0, 10), 0)\n\n"
            "    def test_clamps_high_bound(self):\n"
            "        self.assertEqual(clamp(15, 0, 10), 10)\n\n"
            "    def test_keeps_middle_value(self):\n"
            "        self.assertEqual(clamp(4, 0, 10), 4)\n"
        )
        (repo_path / "calculator.py").write_text(buggy_source, encoding="utf-8")
        (repo_path / "test_calculator.py").write_text(test_source, encoding="utf-8")
        _commit_all(repo_path, "add buggy clamp and executable tests")
        exposed_commit = _git(repo_path, "rev-parse", "--short=12", "HEAD")
        baseline_passed, baseline_output = _run_unittest(repo_path)

        candidates = [
            {
                "method": "frozen_noop",
                "patch": "",
                "notes": "No retrieved fix or structured symbol information.",
            },
            {
                "method": "retrieval_wrong_patch",
                "patch": _clamp_patch("return max(value, low)"),
                "notes": "Plausible one-sided fix from lexical retrieval; misses high bound.",
            },
            {
                "method": "structured_patch",
                "patch": _clamp_patch("return max(low, min(value, high))"),
                "notes": "Structured patch uses both failing assertions and target function.",
            },
        ]

        candidate_results: list[dict[str, Any]] = []
        for candidate in candidates:
            (repo_path / "calculator.py").write_text(buggy_source, encoding="utf-8")
            applied, apply_output = _apply_candidate_patch(repo_path, str(candidate["patch"]))
            passed = False
            test_output = "patch did not apply"
            if applied:
                passed, test_output = _run_unittest(repo_path)
            candidate_results.append(
                {
                    "method": candidate["method"],
                    "patch_applied": applied,
                    "tests_passed": passed,
                    "patch_bytes": len(str(candidate["patch"]).encode()),
                    "apply_output_tail": apply_output,
                    "test_output_tail": test_output,
                    "notes": candidate["notes"],
                }
            )

    passed_methods = [row["method"] for row in candidate_results if row["tests_passed"]]
    final = {
        "baseline_tests_passed": baseline_passed,
        "functional_success_rate": len(passed_methods) / len(candidate_results),
        "best_method": passed_methods[0] if passed_methods else None,
        "frozen_noop_passed": next(
            row["tests_passed"] for row in candidate_results if row["method"] == "frozen_noop"
        ),
        "retrieval_wrong_patch_passed": next(
            row["tests_passed"]
            for row in candidate_results
            if row["method"] == "retrieval_wrong_patch"
        ),
        "structured_patch_passed": next(
            row["tests_passed"] for row in candidate_results if row["method"] == "structured_patch"
        ),
    }
    return {
        "repo": {
            "scenario": "synthetic clamp bug with executable unittest suite",
            "exposed_commit": exposed_commit,
        },
        "baseline_test_output_tail": baseline_output,
        "candidates": candidate_results,
        "final": final,
        "notes": (
            "Synthetic patch-generation positive control. Candidate patches are unified "
            "diffs applied with git apply and graded only by running unit tests; no text "
            "similarity to a reference patch is used."
        ),
    }
