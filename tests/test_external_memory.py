import subprocess

from weightlab.external_memory import (
    run_public_repository_api_doc_coverage_qa_experiment,
    run_public_repository_api_reference_generation_experiment,
    run_public_repository_call_stub_generation_experiment,
    run_public_repository_docstring_skeleton_generation_experiment,
    run_public_repository_function_skeleton_generation_experiment,
    run_public_repository_memory_qa_experiment,
    run_public_repository_signature_qa_experiment,
    run_structured_repository_memory_experiment,
    run_synthetic_api_doc_drift_detection_experiment,
)


def _git(repo, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _commit(repo, message: str) -> None:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)


def test_structured_repository_memory_beats_poisonable_text_retrieval():
    result = run_structured_repository_memory_experiment(seed=123)
    methods = result["methods"]
    structured = methods["gated_structured_external_memory"]
    updated_retrieval = methods["updated_text_retrieval"]

    assert result["experiment_family"] == "external_memory_architecture"
    assert result["simulation_label"] == "synthetic_repository_convention_memory"
    assert structured["answer_accuracy"] > updated_retrieval["answer_accuracy"]
    assert structured["poisoned_answer_count"] == 0
    assert updated_retrieval["poisoned_answer_count"] >= 1
    assert structured["restricted_denial_accuracy"] == 1.0
    assert structured["authorized_security_accuracy"] == 1.0


def test_structured_repository_memory_is_not_model_level_claim():
    result = run_structured_repository_memory_experiment(seed=123)

    assert result["final"]["supports_model_level_claim"] is False
    assert "parametric_baseline_is_proxy_not_trained_lm" in result["limitations"]
    assert "not_code_generation" in result["limitations"]


def test_public_repository_memory_qa_answers_symbol_file_questions(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "research@example.invalid")
    _git(repo, "config", "user.name", "Research Test")

    (repo / "parser.py").write_text(
        "def parse_config(text):\n"
        "    return text.strip()\n\n"
        "def load_config(path):\n"
        "    return parse_config(path)\n",
        encoding="utf-8",
    )
    (repo / "writer.py").write_text(
        "def render_changelog(entries):\n"
        "    return '\\n'.join(entries)\n",
        encoding="utf-8",
    )
    _commit(repo, "add public repository symbols")

    result = run_public_repository_memory_qa_experiment(
        repo_paths=[repo],
        max_symbols_per_repo=8,
    )

    structured = result["methods"]["structured_symbol_memory"]
    text_retrieval = result["methods"]["text_file_retrieval"]
    assert result["experiment_family"] == "external_memory_architecture"
    assert result["benchmark_label"] == "public_repository_symbol_file_qa"
    assert result["final"]["repo_count_used"] == 1
    assert result["question_count"] >= 3
    assert structured["answer_accuracy"] == 1.0
    assert structured["answer_accuracy"] >= text_retrieval["answer_accuracy"]
    assert structured["storage_bytes"] > 0
    assert result["final"]["supports_model_level_claim"] is False


def test_public_repository_memory_qa_skips_missing_repositories(tmp_path):
    result = run_public_repository_memory_qa_experiment(
        repo_paths=[tmp_path / "missing"],
        max_symbols_per_repo=8,
    )

    assert result["final"]["repo_count_used"] == 0
    assert result["question_count"] == 0
    assert result["status"] == "skipped_no_public_repositories"
    assert result["final"]["supports_model_level_claim"] is False


def test_public_repository_signature_qa_answers_function_signatures(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "research@example.invalid")
    _git(repo, "config", "user.name", "Research Test")

    (repo / "parser.py").write_text(
        "def parse_config(text, strict=False):\n"
        "    return text.strip()\n\n"
        "def load_config(path, *, encoding='utf-8'):\n"
        "    return path\n",
        encoding="utf-8",
    )
    _commit(repo, "add public repository signatures")

    result = run_public_repository_signature_qa_experiment(
        repo_paths=[repo],
        max_signatures_per_repo=8,
    )

    structured = result["methods"]["structured_signature_memory"]
    text_lookup = result["methods"]["text_signature_lookup"]
    assert result["benchmark_label"] == "public_repository_signature_qa"
    assert result["final"]["repo_count_used"] == 1
    assert result["question_count"] == 2
    assert structured["answer_accuracy"] == 1.0
    assert structured["answer_accuracy"] >= text_lookup["answer_accuracy"]
    assert structured["storage_bytes"] > 0
    assert result["final"]["supports_model_level_claim"] is False


def test_public_repository_signature_qa_skips_missing_repositories(tmp_path):
    result = run_public_repository_signature_qa_experiment(
        repo_paths=[tmp_path / "missing"],
        max_signatures_per_repo=8,
    )

    assert result["final"]["repo_count_used"] == 0
    assert result["question_count"] == 0
    assert result["status"] == "skipped_no_public_repositories"
    assert result["final"]["supports_model_level_claim"] is False


def test_public_repository_call_stub_generation_uses_signature_parameters(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "research@example.invalid")
    _git(repo, "config", "user.name", "Research Test")

    (repo / "parser.py").write_text(
        "def parse_config(text, strict=False):\n"
        "    return text.strip()\n\n"
        "def load_config(path, *, encoding='utf-8'):\n"
        "    return path\n",
        encoding="utf-8",
    )
    _commit(repo, "add public repository call targets")

    result = run_public_repository_call_stub_generation_experiment(
        repo_paths=[repo],
        max_functions_per_repo=8,
    )

    structured = result["methods"]["structured_signature_call_stub"]
    name_only = result["methods"]["name_only_call_stub"]
    answers = {
        row["symbol"]: row["answer"]
        for row in structured["answers"]
    }
    assert result["benchmark_label"] == "public_repository_call_stub_generation"
    assert result["final"]["repo_count_used"] == 1
    assert result["question_count"] == 2
    assert answers["parse_config"] == "parse_config(text, strict=...)"
    assert answers["load_config"] == "load_config(path, encoding=...)"
    assert structured["answer_accuracy"] == 1.0
    assert structured["answer_accuracy"] > name_only["answer_accuracy"]
    assert result["final"]["supports_model_level_claim"] is False


def test_public_repository_call_stub_generation_skips_missing_repositories(tmp_path):
    result = run_public_repository_call_stub_generation_experiment(
        repo_paths=[tmp_path / "missing"],
        max_functions_per_repo=8,
    )

    assert result["final"]["repo_count_used"] == 0
    assert result["question_count"] == 0
    assert result["status"] == "skipped_no_public_repositories"
    assert result["final"]["supports_model_level_claim"] is False


def test_public_repository_function_skeleton_generation_preserves_signatures(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "research@example.invalid")
    _git(repo, "config", "user.name", "Research Test")

    (repo / "parser.py").write_text(
        "def parse_config(text, strict=False):\n"
        "    return text.strip()\n\n"
        "def load_config(path, *, encoding='utf-8'):\n"
        "    return path\n",
        encoding="utf-8",
    )
    _commit(repo, "add public repository skeleton targets")

    result = run_public_repository_function_skeleton_generation_experiment(
        repo_paths=[repo],
        max_functions_per_repo=8,
    )

    structured = result["methods"]["structured_signature_skeleton"]
    name_only = result["methods"]["name_only_skeleton"]
    answers = {row["symbol"]: row["answer"] for row in structured["answers"]}
    assert result["benchmark_label"] == "public_repository_function_skeleton_generation"
    assert result["final"]["repo_count_used"] == 1
    assert result["question_count"] == 2
    assert answers["parse_config"] == (
        "def parse_config(text, strict=False):\n"
        "    raise NotImplementedError(\"generated skeleton\")"
    )
    assert answers["load_config"] == (
        "def load_config(path, *, encoding='utf-8'):\n"
        "    raise NotImplementedError(\"generated skeleton\")"
    )
    assert structured["answer_accuracy"] == 1.0
    assert structured["answer_accuracy"] > name_only["answer_accuracy"]
    assert result["final"]["supports_model_level_claim"] is False


def test_public_repository_function_skeleton_generation_skips_missing_repositories(tmp_path):
    result = run_public_repository_function_skeleton_generation_experiment(
        repo_paths=[tmp_path / "missing"],
        max_functions_per_repo=8,
    )

    assert result["final"]["repo_count_used"] == 0
    assert result["question_count"] == 0
    assert result["status"] == "skipped_no_public_repositories"
    assert result["final"]["supports_model_level_claim"] is False


def test_public_repository_docstring_skeleton_generation_preserves_signature_and_doc(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "research@example.invalid")
    _git(repo, "config", "user.name", "Research Test")

    (repo / "html.py").write_text(
        "def escape(value: str, quote=True) -> str:\n"
        "    \"\"\"Escape HTML-sensitive characters.\"\"\"\n"
        "    return value\n\n"
        "def soft_str(value: object) -> str:\n"
        "    \"\"\"Preserve Markup instances while converting other values.\"\"\"\n"
        "    return str(value)\n",
        encoding="utf-8",
    )
    _commit(repo, "add public repository documented functions")

    result = run_public_repository_docstring_skeleton_generation_experiment(
        repo_paths=[repo],
        max_functions_per_repo=8,
    )

    structured = result["methods"]["structured_docstring_skeleton"]
    signature_only = result["methods"]["signature_only_skeleton"]
    answers = {row["symbol"]: row["answer"] for row in structured["answers"]}
    assert result["benchmark_label"] == "public_repository_docstring_skeleton_generation"
    assert result["final"]["repo_count_used"] == 1
    assert result["question_count"] == 2
    assert answers["escape"] == (
        "def escape(value: str, quote=True) -> str:\n"
        "    \"\"\"Escape HTML-sensitive characters.\"\"\"\n"
        "    raise NotImplementedError(\"generated skeleton\")"
    )
    assert answers["soft_str"] == (
        "def soft_str(value: object) -> str:\n"
        "    \"\"\"Preserve Markup instances while converting other values.\"\"\"\n"
        "    raise NotImplementedError(\"generated skeleton\")"
    )
    assert structured["answer_accuracy"] == 1.0
    assert structured["answer_accuracy"] > signature_only["answer_accuracy"]
    assert result["final"]["supports_model_level_claim"] is False


def test_public_repository_docstring_skeleton_generation_skips_missing_repositories(tmp_path):
    result = run_public_repository_docstring_skeleton_generation_experiment(
        repo_paths=[tmp_path / "missing"],
        max_functions_per_repo=8,
    )

    assert result["final"]["repo_count_used"] == 0
    assert result["question_count"] == 0
    assert result["status"] == "skipped_no_public_repositories"
    assert result["final"]["supports_model_level_claim"] is False


def test_public_repository_api_reference_generation_uses_signature_doc_and_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "research@example.invalid")
    _git(repo, "config", "user.name", "Research Test")

    (repo / "html.py").write_text(
        "def escape(value: str, quote=True) -> str:\n"
        "    \"\"\"Escape HTML-sensitive characters.\"\"\"\n"
        "    return value\n\n"
        "def soft_str(value: object) -> str:\n"
        "    \"\"\"Preserve Markup instances while converting other values.\"\"\"\n"
        "    return str(value)\n",
        encoding="utf-8",
    )
    _commit(repo, "add public repository documented api targets")

    result = run_public_repository_api_reference_generation_experiment(
        repo_paths=[repo],
        max_functions_per_repo=8,
    )

    structured = result["methods"]["structured_docstring_api_reference"]
    signature_only = result["methods"]["signature_only_api_reference"]
    answers = {row["symbol"]: row["answer"] for row in structured["answers"]}
    assert result["benchmark_label"] == "public_repository_api_reference_generation"
    assert result["final"]["repo_count_used"] == 1
    assert result["question_count"] == 2
    assert answers["escape"] == (
        "### escape\n\n"
        "- Defined in: `html.py`\n"
        "- Signature: `def escape(value: str, quote=True) -> str:`\n"
        "- Summary: Escape HTML-sensitive characters."
    )
    assert answers["soft_str"] == (
        "### soft_str\n\n"
        "- Defined in: `html.py`\n"
        "- Signature: `def soft_str(value: object) -> str:`\n"
        "- Summary: Preserve Markup instances while converting other values."
    )
    assert structured["answer_accuracy"] == 1.0
    assert structured["answer_accuracy"] > signature_only["answer_accuracy"]
    assert result["final"]["supports_model_level_claim"] is False


def test_public_repository_api_reference_generation_skips_missing_repositories(tmp_path):
    result = run_public_repository_api_reference_generation_experiment(
        repo_paths=[tmp_path / "missing"],
        max_functions_per_repo=8,
    )

    assert result["final"]["repo_count_used"] == 0
    assert result["question_count"] == 0
    assert result["status"] == "skipped_no_public_repositories"
    assert result["final"]["supports_model_level_claim"] is False


def test_public_repository_api_doc_coverage_qa_maps_symbols_to_docs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "research@example.invalid")
    _git(repo, "config", "user.name", "Research Test")

    (repo / "pkg.py").write_text(
        "def escape(value: str) -> str:\n"
        "    \"\"\"Escape HTML-sensitive characters.\"\"\"\n"
        "    return value\n\n"
        "def soft_str(value: object) -> str:\n"
        "    \"\"\"Preserve Markup instances while converting other values.\"\"\"\n"
        "    return str(value)\n\n"
        "def undocumented(value):\n"
        "    return value\n",
        encoding="utf-8",
    )
    docs = repo / "docs"
    docs.mkdir()
    (docs / "api.rst").write_text(
        ".. module:: pkg\n\n"
        ".. autofunction:: escape\n\n"
        ".. autofunction:: soft_str\n",
        encoding="utf-8",
    )
    _commit(repo, "add public api docs")

    result = run_public_repository_api_doc_coverage_qa_experiment(
        repo_paths=[repo],
        max_symbols_per_repo=8,
    )

    structured = result["methods"]["structured_doc_directive_memory"]
    source_only = result["methods"]["source_symbol_only_memory"]
    answers = {row["symbol"]: row["answer"] for row in structured["answers"]}
    assert result["benchmark_label"] == "public_repository_api_doc_coverage_qa"
    assert result["final"]["repo_count_used"] == 1
    assert result["question_count"] == 2
    assert answers["escape"] == "docs/api.rst"
    assert answers["soft_str"] == "docs/api.rst"
    assert structured["answer_accuracy"] == 1.0
    assert structured["answer_accuracy"] > source_only["answer_accuracy"]
    assert result["final"]["supports_model_level_claim"] is False


def test_public_repository_api_doc_coverage_qa_skips_missing_repositories(tmp_path):
    result = run_public_repository_api_doc_coverage_qa_experiment(
        repo_paths=[tmp_path / "missing"],
        max_symbols_per_repo=8,
    )

    assert result["final"]["repo_count_used"] == 0
    assert result["question_count"] == 0
    assert result["status"] == "skipped_no_public_repositories"
    assert result["final"]["supports_model_level_claim"] is False


def test_synthetic_api_doc_drift_detection_flags_stale_directive(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
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
    _commit(repo, "add documented config api")

    (repo / "config.py").write_text(
        "def load_config(text: str) -> dict:\n"
        "    \"\"\"Load current configuration text.\"\"\"\n"
        "    return {}\n",
        encoding="utf-8",
    )
    _commit(repo, "rename source api without updating docs")

    result = run_synthetic_api_doc_drift_detection_experiment(repo_path=repo)

    structured = result["methods"]["structured_source_doc_consistency_memory"]
    doc_only = result["methods"]["doc_directive_only_memory"]
    answers = {row["symbol"]: row["answer"] for row in structured["answers"]}
    assert result["benchmark_label"] == "synthetic_api_doc_drift_detection"
    assert result["status"] == "completed"
    assert result["final"]["stale_doc_issue_count"] == 1
    assert answers["parse_config"] == "stale_missing_source:docs/api.rst"
    assert structured["answer_accuracy"] == 1.0
    assert structured["answer_accuracy"] > doc_only["answer_accuracy"]
    assert result["final"]["supports_model_level_claim"] is False
