from __future__ import annotations

import subprocess
import sys

from weightlab.repo_chronology import (
    run_public_patch_replay_experiment,
    run_public_patch_replay_suite_experiment,
    run_public_repo_chronology_experiment,
    run_stale_doc_chronology_experiment,
    run_symbol_qa_chronology_experiment,
    run_synthetic_patch_generation_positive_control,
    run_synthetic_stale_doc_positive_control,
)


def _git(repo, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _commit(repo, message: str) -> None:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)


def test_public_repo_chronology_uses_only_past_commits(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "research@example.invalid")
    _git(repo, "config", "user.name", "Research Test")

    (repo / "README.md").write_text("config loader project\n", encoding="utf-8")
    _commit(repo, "initial config loader project")

    (repo / "parser.py").write_text("def parse_config(text):\n    return text\n", encoding="utf-8")
    _commit(repo, "add parser config loader")

    (repo / "docs.md").write_text("parse_config reads configuration text\n", encoding="utf-8")
    _commit(repo, "document parser config behavior")

    result = run_public_repo_chronology_experiment(repo_path=repo, max_commits=3, top_k=3)

    assert result["repo"]["commit_count_used"] == 3
    assert result["steps"]
    assert all(step["future_commit_not_in_memory"] for step in result["steps"])
    assert all(step["future_changed_files"] for step in result["steps"])
    assert (
        result["final"]["structured_prior_topk_accuracy"]
        >= result["final"]["retrieval_prior_topk_accuracy"]
    )
    assert (
        result["final"]["updated_retrieval_future_topk_accuracy"]
        >= result["final"]["frozen_future_topk_accuracy"]
    )


def test_symbol_qa_chronology_answers_visible_symbol_definition_questions(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "research@example.invalid")
    _git(repo, "config", "user.name", "Research Test")

    (repo / "README.md").write_text("parser project\n", encoding="utf-8")
    _commit(repo, "initial parser project")

    (repo / "parser.py").write_text(
        "def parse_config(text):\n    return text.strip()\n",
        encoding="utf-8",
    )
    _commit(repo, "add parse_config")

    (repo / "writer.py").write_text(
        "def write_config(config):\n    return str(config)\n",
        encoding="utf-8",
    )
    _commit(repo, "add write_config")

    result = run_symbol_qa_chronology_experiment(repo_path=repo, max_commits=3, top_k=1)

    assert result["repo"]["commit_count_used"] == 3
    assert result["steps"]
    assert all(step["future_commit_not_in_memory"] for step in result["steps"])
    assert result["final"]["structured_prior_symbol_topk_accuracy"] == 1.0
    assert (
        result["final"]["structured_prior_symbol_topk_accuracy"]
        >= result["final"]["retrieval_prior_symbol_topk_accuracy"]
    )


def test_stale_doc_chronology_detects_docs_mentioning_removed_symbols(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "research@example.invalid")
    _git(repo, "config", "user.name", "Research Test")

    (repo / "api.py").write_text(
        "def parse_config(text):\n    return text.strip()\n",
        encoding="utf-8",
    )
    (repo / "docs.md").write_text(
        "Use parse_config to load configuration text.\n",
        encoding="utf-8",
    )
    _commit(repo, "add parse_config and docs")

    (repo / "api.py").write_text(
        "def load_config(text):\n    return text.strip()\n",
        encoding="utf-8",
    )
    _commit(repo, "rename parse_config to load_config without doc update")

    result = run_stale_doc_chronology_experiment(repo_path=repo, max_commits=2, top_k=1)

    assert result["repo"]["commit_count_used"] == 2
    assert result["final"]["stale_doc_issue_count"] == 1
    assert result["steps"][0]["removed_symbols"] == ["parse_config"]
    assert result["steps"][0]["stale_doc_files"] == ["docs.md"]
    assert result["final"]["structured_stale_doc_topk_accuracy"] == 1.0


def test_synthetic_stale_doc_positive_control_scores_positive_and_fix():
    result = run_synthetic_stale_doc_positive_control(top_k=1)

    assert result["positive_control"]["expected_stale_doc_issues"] == 1
    assert result["positive_control"]["detected_stale_doc_issues"] == 1
    assert result["positive_control"]["post_fix_stale_doc_files"] == []
    assert result["final"]["steps_with_stale_docs"] == 1
    assert result["final"]["structured_stale_doc_topk_accuracy"] == 1.0
    assert result["final"]["retrieval_stale_doc_topk_accuracy"] == 1.0


def test_synthetic_patch_generation_positive_control_uses_functional_tests():
    result = run_synthetic_patch_generation_positive_control()

    assert result["final"]["functional_success_rate"] == 1 / 3
    assert result["final"]["best_method"] == "structured_patch"
    assert result["final"]["frozen_noop_passed"] is False
    assert result["final"]["retrieval_wrong_patch_passed"] is False
    assert result["final"]["structured_patch_passed"] is True
    assert all("patch_text_similarity" not in candidate for candidate in result["candidates"])


def test_public_patch_replay_uses_future_commit_patch_and_executable_tests(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "research@example.invalid")
    _git(repo, "config", "user.name", "Research Test")

    (repo / "calculator.py").write_text(
        "def clamp(value, low, high):\n"
        "    return min(value, high)\n",
        encoding="utf-8",
    )
    (repo / "test_calculator.py").write_text(
        "import unittest\n\n"
        "from calculator import clamp\n\n\n"
        "class ClampTests(unittest.TestCase):\n"
        "    def test_clamps_low_bound(self):\n"
        "        self.assertEqual(clamp(-5, 0, 10), 0)\n\n"
        "    def test_clamps_high_bound(self):\n"
        "        self.assertEqual(clamp(15, 0, 10), 10)\n\n"
        "    def test_keeps_middle_value(self):\n"
        "        self.assertEqual(clamp(4, 0, 10), 4)\n",
        encoding="utf-8",
    )
    _commit(repo, "public bug: add clamp with executable regression tests")
    base_commit = _git(repo, "rev-parse", "HEAD")

    (repo / "calculator.py").write_text(
        "def clamp(value, low, high):\n"
        "    return max(low, min(value, high))\n",
        encoding="utf-8",
    )
    _commit(repo, "public fix: clamp lower bound")
    fix_commit = _git(repo, "rev-parse", "HEAD")

    result = run_public_patch_replay_experiment(
        repo_path=repo,
        base_commit=base_commit,
        fix_commit=fix_commit,
        test_command=[sys.executable, "-m", "unittest", "discover"],
    )

    assert result["repo"]["base_commit"] == base_commit[:12]
    assert result["repo"]["fix_commit"] == fix_commit[:12]
    assert result["repo"]["future_commit_not_in_memory"] is True
    assert result["final"]["baseline_tests_passed"] is False
    assert result["final"]["frozen_noop_passed"] is False
    assert result["final"]["generated_clamp_bounds_patch_passed"] is True
    assert result["final"]["historical_patch_passed"] is True
    assert result["final"]["functional_success_rate"] == 2 / 3
    assert result["final"]["best_method"] == "generated_clamp_bounds_patch"
    assert all("patch_text_similarity" not in candidate for candidate in result["candidates"])


def test_public_patch_replay_includes_generated_clamp_candidate(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "research@example.invalid")
    _git(repo, "config", "user.name", "Research Test")

    (repo / "calculator.py").write_text(
        "def clamp(value, low, high):\n"
        "    return min(value, high)\n",
        encoding="utf-8",
    )
    (repo / "test_calculator.py").write_text(
        "from calculator import clamp\n"
        "assert clamp(-5, 0, 10) == 0\n"
        "assert clamp(15, 0, 10) == 10\n"
        "assert clamp(4, 0, 10) == 4\n",
        encoding="utf-8",
    )
    _commit(repo, "public bug: clamp lower bound missing")
    base_commit = _git(repo, "rev-parse", "HEAD")

    (repo / "calculator.py").write_text(
        "def clamp(value, low, high):\n"
        "    return max(low, min(value, high))\n",
        encoding="utf-8",
    )
    _commit(repo, "public fix: clamp lower bound")
    fix_commit = _git(repo, "rev-parse", "HEAD")

    result = run_public_patch_replay_experiment(
        repo_path=repo,
        base_commit=base_commit,
        fix_commit=fix_commit,
        test_command=[sys.executable, "test_calculator.py"],
    )

    candidates = {candidate["method"]: candidate for candidate in result["candidates"]}
    assert result["final"]["baseline_tests_passed"] is False
    assert result["final"]["generated_clamp_bounds_patch_passed"] is True
    assert result["final"]["historical_patch_passed"] is True
    assert result["final"]["functional_success_rate"] == 2 / 3
    assert result["final"]["best_method"] == "generated_clamp_bounds_patch"
    assert candidates["generated_clamp_bounds_patch"]["patch_applied"] is True
    assert candidates["generated_clamp_bounds_patch"]["patch_bytes"] > 0


def test_public_patch_replay_includes_generated_proxy_class_candidate(tmp_path):
    repo = tmp_path / "repo"
    package_dir = repo / "src" / "markupsafe"
    package_dir.mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "research@example.invalid")
    _git(repo, "config", "user.name", "Research Test")

    buggy_source = (
        "def _escape_inner(s):\n"
        "    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')\n\n\n"
        "class Markup(str):\n"
        "    pass\n\n\n"
        "def escape(s):\n"
        "    if s.__class__ is str:\n"
        "        return Markup(_escape_inner(s))\n"
        "    if hasattr(s, '__html__'):\n"
        "        return Markup(s.__html__())\n"
        "    return Markup(_escape_inner(str(s)))\n"
    )
    fixed_source = buggy_source.replace("if s.__class__ is str:", "if type(s) is str:")
    (package_dir / "__init__.py").write_text(buggy_source, encoding="utf-8")
    _commit(repo, "public bug: proxy reports proxied __class__")
    base_commit = _git(repo, "rev-parse", "HEAD")

    (package_dir / "__init__.py").write_text(fixed_source, encoding="utf-8")
    _commit(repo, "public fix: use exact type check")
    fix_commit = _git(repo, "rev-parse", "HEAD")

    test_code = (
        "import sys; sys.path.insert(0, 'src'); from markupsafe import escape; "
        "Proxy = type('Proxy', (), {'__class__': property(lambda self: str), "
        "'__str__': lambda self: '<em>'}); "
        "assert str(escape(Proxy())) == '&lt;em&gt;'"
    )
    result = run_public_patch_replay_experiment(
        repo_path=repo,
        base_commit=base_commit,
        fix_commit=fix_commit,
        test_command=[sys.executable, "-c", test_code],
    )

    candidates = {candidate["method"]: candidate for candidate in result["candidates"]}
    assert result["final"]["baseline_tests_passed"] is False
    assert result["final"]["frozen_noop_passed"] is False
    assert result["final"]["generated_proxy_class_patch_passed"] is True
    assert result["final"]["historical_patch_passed"] is True
    assert result["final"]["functional_success_rate"] == 2 / 3
    assert result["final"]["best_method"] == "generated_proxy_class_patch"
    assert candidates["generated_proxy_class_patch"]["patch_applied"] is True
    assert candidates["generated_proxy_class_patch"]["patch_bytes"] > 0


def test_public_patch_replay_includes_generated_saved_hl_free_candidate(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "research@example.invalid")
    _git(repo, "config", "user.name", "Research Test")

    buggy_source = (
        "#include <stdlib.h>\n"
        "#include <string.h>\n\n"
        "unsigned char *saved_hl;\n"
        "unsigned char *row_hl;\n"
        "int row_size;\n\n"
        "#define FIND_RESTORE_HL do { \\\n"
        "    if (saved_hl) { \\\n"
        "        memcpy(row_hl, saved_hl, row_size); \\\n"
        "        saved_hl = NULL; \\\n"
        "    } \\\n"
        "} while (0)\n"
    )
    fixed_source = buggy_source.replace(
        "        memcpy(row_hl, saved_hl, row_size); \\\n"
        "        saved_hl = NULL; \\\n",
        "        memcpy(row_hl, saved_hl, row_size); \\\n"
        "        free(saved_hl); \\\n"
        "        saved_hl = NULL; \\\n",
    )
    (repo / "kilo.c").write_text(buggy_source, encoding="utf-8")
    _commit(repo, "public bug: saved highlight restore leaks")
    base_commit = _git(repo, "rev-parse", "HEAD")

    (repo / "kilo.c").write_text(fixed_source, encoding="utf-8")
    _commit(repo, "public fix: free saved highlight buffer")
    fix_commit = _git(repo, "rev-parse", "HEAD")

    test_code = (
        "from pathlib import Path; text = Path('kilo.c').read_text(); "
        "assert 'free(saved_hl);' in text; "
        "assert text.index('memcpy(row_hl, saved_hl, row_size);') < "
        "text.index('free(saved_hl);') < text.index('saved_hl = NULL;')"
    )
    result = run_public_patch_replay_experiment(
        repo_path=repo,
        base_commit=base_commit,
        fix_commit=fix_commit,
        test_command=[sys.executable, "-c", test_code],
    )

    candidates = {candidate["method"]: candidate for candidate in result["candidates"]}
    assert result["final"]["baseline_tests_passed"] is False
    assert result["final"]["generated_saved_hl_free_patch_passed"] is True
    assert result["final"]["historical_patch_passed"] is True
    assert result["final"]["functional_success_rate"] == 2 / 3
    assert result["final"]["best_method"] == "generated_saved_hl_free_patch"
    assert candidates["generated_saved_hl_free_patch"]["patch_applied"] is True
    assert candidates["generated_saved_hl_free_patch"]["patch_bytes"] > 0


def test_public_patch_replay_suite_aggregates_multiple_tasks(tmp_path):
    proxy_repo = tmp_path / "proxy"
    proxy_package = proxy_repo / "src" / "markupsafe"
    proxy_package.mkdir(parents=True)
    _git(proxy_repo, "init", "-b", "main")
    _git(proxy_repo, "config", "user.email", "research@example.invalid")
    _git(proxy_repo, "config", "user.name", "Research Test")
    proxy_bug = "def escape(s):\n    return 'raw' if s.__class__ is str else str(s)\n"
    proxy_fix = proxy_bug.replace("s.__class__ is str", "type(s) is str")
    (proxy_package / "__init__.py").write_text(proxy_bug, encoding="utf-8")
    _commit(proxy_repo, "public bug: proxy exact class")
    proxy_base = _git(proxy_repo, "rev-parse", "HEAD")
    (proxy_package / "__init__.py").write_text(proxy_fix, encoding="utf-8")
    _commit(proxy_repo, "public fix: proxy exact type")
    proxy_fix_commit = _git(proxy_repo, "rev-parse", "HEAD")

    leak_repo = tmp_path / "leak"
    leak_repo.mkdir()
    _git(leak_repo, "init", "-b", "main")
    _git(leak_repo, "config", "user.email", "research@example.invalid")
    _git(leak_repo, "config", "user.name", "Research Test")
    leak_bug = (
        "#include <string.h>\n"
        "#define FIND_RESTORE_HL do { \\\n"
        "    if (saved_hl) { \\\n"
        "        memcpy(row_hl, saved_hl, row_size); \\\n"
        "        saved_hl = NULL; \\\n"
        "    } \\\n"
        "} while (0)\n"
    )
    leak_fix = leak_bug.replace(
        "        memcpy(row_hl, saved_hl, row_size); \\\n"
        "        saved_hl = NULL; \\\n",
        "        memcpy(row_hl, saved_hl, row_size); \\\n"
        "        free(saved_hl); \\\n"
        "        saved_hl = NULL; \\\n",
    )
    (leak_repo / "kilo.c").write_text(leak_bug, encoding="utf-8")
    _commit(leak_repo, "public bug: saved highlight leak")
    leak_base = _git(leak_repo, "rev-parse", "HEAD")
    (leak_repo / "kilo.c").write_text(leak_fix, encoding="utf-8")
    _commit(leak_repo, "public fix: saved highlight free")
    leak_fix_commit = _git(leak_repo, "rev-parse", "HEAD")

    result = run_public_patch_replay_suite_experiment(
        tasks=[
            {
                "task_id": "proxy_exact_type",
                "repo_path": proxy_repo,
                "base_commit": proxy_base,
                "fix_commit": proxy_fix_commit,
                "test_command": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; "
                    "assert 'type(s) is str' in Path('src/markupsafe/__init__.py').read_text()",
                ],
            },
            {
                "task_id": "saved_hl_free",
                "repo_path": leak_repo,
                "base_commit": leak_base,
                "fix_commit": leak_fix_commit,
                "test_command": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; assert 'free(saved_hl);' in "
                    "Path('kilo.c').read_text()",
                ],
            },
        ]
    )

    assert result["final"]["task_count"] == 2
    assert result["final"]["baseline_failures"] == 2
    assert result["final"]["historical_patch_successes"] == 2
    assert result["final"]["generated_task_successes"] == 2
    assert result["final"]["generated_task_success_rate"] == 1.0
    assert {task["task_id"] for task in result["tasks"]} == {
        "proxy_exact_type",
        "saved_hl_free",
    }
