from __future__ import annotations

import ast
import copy
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weightlab.quixbugs_edit_baseline import build_quixbugs_edit_baseline_candidates
from weightlab.quixbugs_model_candidates import (
    _load_dense_checkpoint,
    _prompt_for_program,
    _sha256,
)
from weightlab.quixbugs_model_ranked_edits import (
    score_candidate_mean_nll,
)
from weightlab.quixbugs_repair import (
    QuixBugsCandidate,
    QuixBugsCandidateConfig,
    evaluate_quixbugs_candidate_repairs,
)

POOL_GENERATOR_LABEL = "syntax_preserving_ast_mutation_pool"
RANKED_GENERATOR_LABEL = "dense_ranked_syntax_pool"


@dataclass(frozen=True)
class DenseQuixBugsSyntaxPoolRankConfig:
    device: str = "rocm"
    seed: int = 123
    timeout_seconds: int = 20
    max_candidates_per_program: int = 32
    top_candidates_per_program: int = 1


def build_quixbugs_syntax_mutation_candidates(
    repo_path: Path,
    *,
    programs: tuple[str, ...],
    max_candidates_per_program: int,
) -> list[QuixBugsCandidate]:
    if not programs:
        raise ValueError("at least one program is required")
    if max_candidates_per_program < 1:
        raise ValueError("max_candidates_per_program must be positive")
    repo_path = Path(repo_path)
    candidates: list[QuixBugsCandidate] = []
    for program in programs:
        source_path = repo_path / "python_programs" / f"{program}.py"
        source = source_path.read_text(encoding="utf-8")
        sources = _syntax_mutation_sources(repo_path, program, source)
        for index, candidate_source in enumerate(
            sources[:max_candidates_per_program],
            start=1,
        ):
            candidates.append(
                QuixBugsCandidate(
                    candidate_id=f"{program}:syntax_pool_{index:03d}",
                    program=program,
                    source_text=candidate_source,
                    generator_label=POOL_GENERATOR_LABEL,
                )
            )
    if not candidates:
        raise ValueError("syntax mutation pool generated no candidates")
    return candidates


def evaluate_dense_ranked_quixbugs_syntax_pool(
    checkpoint_path: Path,
    repo_path: Path,
    *,
    programs: tuple[str, ...],
    config: DenseQuixBugsSyntaxPoolRankConfig | None = None,
) -> dict[str, Any]:
    config = config or DenseQuixBugsSyntaxPoolRankConfig()
    if not programs:
        raise ValueError("at least one QuixBugs program is required")
    if config.top_candidates_per_program < 1:
        raise ValueError("top_candidates_per_program must be positive")
    started = time.perf_counter()
    repo_path = Path(repo_path)
    model, tokenizer, model_metadata = _load_dense_checkpoint(
        checkpoint_path,
        config.device,
    )
    candidate_pool = build_quixbugs_syntax_mutation_candidates(
        repo_path,
        programs=programs,
        max_candidates_per_program=config.max_candidates_per_program,
    )
    ranked = _rank_syntax_pool_candidates(model, tokenizer, repo_path, candidate_pool)
    selected_rows = _select_top_ranked(ranked, config.top_candidates_per_program)
    selected_candidates = [
        QuixBugsCandidate(
            candidate_id=row["candidate_id"],
            program=row["program"],
            source_text=row["source_text"],
            generator_label=RANKED_GENERATOR_LABEL,
        )
        for row in selected_rows
    ]
    metrics = evaluate_quixbugs_candidate_repairs(
        repo_path,
        selected_candidates,
        QuixBugsCandidateConfig(
            timeout_seconds=config.timeout_seconds,
            seed=config.seed,
        ),
    )
    metrics["benchmark_label"] = "quixbugs_python_dense_ranked_syntax_pool_probe"
    metrics["model"] = {
        "checkpoint": str(checkpoint_path),
        **model_metadata,
    }
    metrics["candidate_pool"] = {
        "source": POOL_GENERATOR_LABEL,
        "generated_candidate_count": len(candidate_pool),
        "selected_candidate_count": len(selected_candidates),
        "max_candidates_per_program": config.max_candidates_per_program,
        "top_candidates_per_program": config.top_candidates_per_program,
    }
    metrics["ranking"] = [
        {key: value for key, value in row.items() if key not in {"source_text"}}
        for row in ranked
    ]
    metrics["limitations"] = [
        "local_model_ranked_candidate",
        "syntax_preserving_candidate_pool",
        "broader_than_deterministic_edit_baseline",
        "not_free_form_generation",
        "replacement_source_only",
        "python_subset_only",
        "pytest_result_depends_on_local_environment",
        "ranking_uses_teacher_forced_candidate_likelihood",
    ]
    metrics["final"]["end_to_end_runtime_ms"] = (time.perf_counter() - started) * 1000.0
    return metrics


def evaluate_dense_ranked_quixbugs_syntax_pool_topk_profile(
    checkpoint_path: Path,
    repo_path: Path,
    *,
    programs: tuple[str, ...],
    top_k_values: tuple[int, ...],
    config: DenseQuixBugsSyntaxPoolRankConfig | None = None,
) -> dict[str, Any]:
    config = config or DenseQuixBugsSyntaxPoolRankConfig()
    if not programs:
        raise ValueError("at least one QuixBugs program is required")
    normalized_top_k = _normalize_top_k_values(top_k_values)
    max_top_k = normalized_top_k[-1]
    if max_top_k > config.max_candidates_per_program:
        raise ValueError("top_k_values cannot exceed max_candidates_per_program")
    started = time.perf_counter()
    repo_path = Path(repo_path)
    model, tokenizer, model_metadata = _load_dense_checkpoint(
        checkpoint_path,
        config.device,
    )
    candidate_pool = build_quixbugs_syntax_mutation_candidates(
        repo_path,
        programs=programs,
        max_candidates_per_program=config.max_candidates_per_program,
    )
    ranked = _with_rank_within_program(
        _rank_syntax_pool_candidates(model, tokenizer, repo_path, candidate_pool)
    )
    selected_rows = _select_top_ranked(ranked, max_top_k)
    selected_candidates = [
        QuixBugsCandidate(
            candidate_id=row["candidate_id"],
            program=row["program"],
            source_text=row["source_text"],
            generator_label=RANKED_GENERATOR_LABEL,
        )
        for row in selected_rows
    ]
    evaluated = evaluate_quixbugs_candidate_repairs(
        repo_path,
        selected_candidates,
        QuixBugsCandidateConfig(
            timeout_seconds=config.timeout_seconds,
            seed=config.seed,
        ),
    )
    top_k_profile = [
        _summarize_top_k_candidate_results(
            evaluated["candidates"],
            ranked,
            top_k,
            programs,
        )
        for top_k in normalized_top_k
    ]
    best_profile = max(
        top_k_profile,
        key=lambda row: (
            row["program_repair_rate"],
            row["candidate_pass_rate"],
            -row["top_candidates_per_program"],
        ),
    )
    evaluated["benchmark_label"] = "quixbugs_python_dense_ranked_syntax_topk_probe"
    evaluated["model"] = {
        "checkpoint": str(checkpoint_path),
        **model_metadata,
    }
    evaluated["candidate_pool"] = {
        "source": POOL_GENERATOR_LABEL,
        "generated_candidate_count": len(candidate_pool),
        "selected_candidate_count": len(selected_candidates),
        "max_candidates_per_program": config.max_candidates_per_program,
        "top_k_values": list(normalized_top_k),
        "max_top_k_profiled": max_top_k,
    }
    evaluated["ranking"] = [
        {key: value for key, value in row.items() if key not in {"source_text"}}
        for row in ranked
    ]
    evaluated["top_k_profile"] = top_k_profile
    evaluated["limitations"] = [
        "local_model_ranked_candidate",
        "syntax_preserving_candidate_pool",
        "broader_than_deterministic_edit_baseline",
        "top_k_execution_profile",
        "not_free_form_generation",
        "replacement_source_only",
        "python_subset_only",
        "pytest_result_depends_on_local_environment",
        "ranking_uses_teacher_forced_candidate_likelihood",
    ]
    evaluated["final"]["best_top_k"] = best_profile["top_candidates_per_program"]
    evaluated["final"]["best_program_repair_rate"] = best_profile[
        "program_repair_rate"
    ]
    evaluated["final"]["best_candidate_pass_rate"] = best_profile[
        "candidate_pass_rate"
    ]
    evaluated["final"]["end_to_end_runtime_ms"] = (
        time.perf_counter() - started
    ) * 1000.0
    return evaluated


def evaluate_quixbugs_syntax_pool_ordering_controls(
    checkpoint_path: Path,
    repo_path: Path,
    *,
    programs: tuple[str, ...],
    top_k_values: tuple[int, ...],
    config: DenseQuixBugsSyntaxPoolRankConfig | None = None,
) -> dict[str, Any]:
    config = config or DenseQuixBugsSyntaxPoolRankConfig()
    if not programs:
        raise ValueError("at least one QuixBugs program is required")
    normalized_top_k = _normalize_top_k_values(top_k_values)
    max_top_k = normalized_top_k[-1]
    if max_top_k > config.max_candidates_per_program:
        raise ValueError("top_k_values cannot exceed max_candidates_per_program")
    started = time.perf_counter()
    repo_path = Path(repo_path)
    model, tokenizer, model_metadata = _load_dense_checkpoint(
        checkpoint_path,
        config.device,
    )
    candidate_pool = build_quixbugs_syntax_mutation_candidates(
        repo_path,
        programs=programs,
        max_candidates_per_program=config.max_candidates_per_program,
    )
    dense_order = _with_rank_within_program(
        _rank_syntax_pool_candidates(model, tokenizer, repo_path, candidate_pool)
    )
    deterministic_order = _with_rank_within_program(
        _rows_from_candidates(candidate_pool, ordering_name="deterministic_pool_order")
    )
    repair_aware_order = _with_rank_within_program(
        _repair_aware_order_rows(candidate_pool)
    )
    random_order = _with_rank_within_program(
        _random_order_rows(candidate_pool, seed=config.seed)
    )
    orderings = {
        "dense_likelihood": dense_order,
        "deterministic_pool_order": deterministic_order,
        "repair_aware_static_order": repair_aware_order,
        "random_seeded_order": random_order,
    }
    selected_rows = _union_top_rows(orderings, max_top_k)
    selected_candidates = [
        QuixBugsCandidate(
            candidate_id=row["candidate_id"],
            program=row["program"],
            source_text=row["source_text"],
            generator_label=RANKED_GENERATOR_LABEL,
        )
        for row in selected_rows
    ]
    evaluated = evaluate_quixbugs_candidate_repairs(
        repo_path,
        selected_candidates,
        QuixBugsCandidateConfig(
            timeout_seconds=config.timeout_seconds,
            seed=config.seed,
        ),
    )
    ordering_controls = {
        name: _summarize_ordering_control(
            name,
            ordered_rows,
            evaluated["candidates"],
            normalized_top_k,
            programs,
        )
        for name, ordered_rows in orderings.items()
    }
    best_ordering_name = max(
        ordering_controls,
        key=lambda name: (
            ordering_controls[name]["best_program_repair_rate"],
            ordering_controls[name]["best_candidate_pass_rate"],
            -ordering_controls[name]["best_top_k"],
        ),
    )
    dense_best = ordering_controls["dense_likelihood"]["best_program_repair_rate"]
    non_model_best = [
        control["best_program_repair_rate"]
        for name, control in ordering_controls.items()
        if name != "dense_likelihood"
    ]
    evaluated["benchmark_label"] = (
        "quixbugs_python_syntax_pool_ordering_control_probe"
    )
    evaluated["model"] = {
        "checkpoint": str(checkpoint_path),
        **model_metadata,
    }
    evaluated["candidate_pool"] = {
        "source": POOL_GENERATOR_LABEL,
        "generated_candidate_count": len(candidate_pool),
        "selected_candidate_count": len(selected_candidates),
        "max_candidates_per_program": config.max_candidates_per_program,
        "top_k_values": list(normalized_top_k),
        "max_top_k_profiled": max_top_k,
    }
    evaluated["ranking"] = [
        {key: value for key, value in row.items() if key not in {"source_text"}}
        for row in dense_order
    ]
    evaluated["ordering_controls"] = ordering_controls
    evaluated["limitations"] = [
        "local_model_ranked_candidate",
        "syntax_preserving_candidate_pool",
        "broader_than_deterministic_edit_baseline",
        "top_k_execution_profile",
        "same_pool_ordering_controls",
        "not_free_form_generation",
        "replacement_source_only",
        "python_subset_only",
        "pytest_result_depends_on_local_environment",
        "ranking_uses_teacher_forced_candidate_likelihood",
    ]
    evaluated["final"]["best_ordering"] = best_ordering_name
    evaluated["final"]["best_program_repair_rate"] = ordering_controls[
        best_ordering_name
    ]["best_program_repair_rate"]
    evaluated["final"]["best_candidate_pass_rate"] = ordering_controls[
        best_ordering_name
    ]["best_candidate_pass_rate"]
    evaluated["final"]["dense_beats_all_controls"] = all(
        dense_best > score for score in non_model_best
    )
    evaluated["final"]["control_names"] = list(ordering_controls)
    evaluated["final"]["end_to_end_runtime_ms"] = (
        time.perf_counter() - started
    ) * 1000.0
    return evaluated


def _normalize_top_k_values(top_k_values: tuple[int, ...]) -> tuple[int, ...]:
    if not top_k_values:
        raise ValueError("at least one top-k value is required")
    normalized = tuple(sorted(set(top_k_values)))
    if any(value < 1 for value in normalized):
        raise ValueError("top-k values must be positive")
    return normalized


def _with_rank_within_program(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters: dict[str, int] = {}
    ranked_with_position = []
    for row in ranked:
        program = str(row["program"])
        counters[program] = counters.get(program, 0) + 1
        ranked_with_position.append({**row, "rank_within_program": counters[program]})
    return ranked_with_position


def _summarize_top_k_candidate_results(
    candidate_results: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    top_k: int,
    programs: tuple[str, ...],
) -> dict[str, Any]:
    eligible_ids = {
        row["candidate_id"]
        for row in ranked
        if int(row["rank_within_program"]) <= top_k
    }
    rows = [row for row in candidate_results if row["candidate_id"] in eligible_ids]
    if not rows:
        raise ValueError("no candidate results available for top-k summary")
    by_program: dict[str, dict[str, Any]] = {
        program: {
            "candidate_count": 0,
            "passed_count": 0,
            "best_candidate_id": None,
        }
        for program in programs
    }
    for row in rows:
        summary = by_program[str(row["program"])]
        summary["candidate_count"] += 1
        if row["passed"]:
            summary["passed_count"] += 1
            if summary["best_candidate_id"] is None:
                summary["best_candidate_id"] = row["candidate_id"]
    candidate_passed = sum(1 for row in rows if row["passed"])
    programs_with_passing_candidate = sum(
        1 for summary in by_program.values() if summary["best_candidate_id"] is not None
    )
    program_count = len(programs)
    return {
        "top_candidates_per_program": top_k,
        "candidate_count": len(rows),
        "candidate_passed": candidate_passed,
        "candidate_pass_rate": candidate_passed / len(rows),
        "program_count": program_count,
        "programs_with_passing_candidate": programs_with_passing_candidate,
        "program_repair_rate": programs_with_passing_candidate / program_count,
        "programs": dict(sorted(by_program.items())),
    }


def _rows_from_candidates(
    candidates: list[QuixBugsCandidate],
    *,
    ordering_name: str,
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": candidate.candidate_id,
            "program": candidate.program,
            "source_text": candidate.source_text,
            "source_sha256": _sha256(candidate.source_text),
            "generator_label": RANKED_GENERATOR_LABEL,
            "ordering": ordering_name,
        }
        for candidate in sorted(
            candidates,
            key=lambda candidate: (candidate.program, candidate.candidate_id),
        )
    ]


def _random_order_rows(
    candidates: list[QuixBugsCandidate],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for program in sorted({candidate.program for candidate in candidates}):
        program_candidates = [
            candidate for candidate in candidates if candidate.program == program
        ]
        rng = random.Random(f"{seed}:{program}")
        shuffled = list(program_candidates)
        rng.shuffle(shuffled)
        rows.extend(
            {
                "candidate_id": candidate.candidate_id,
                "program": candidate.program,
                "source_text": candidate.source_text,
                "source_sha256": _sha256(candidate.source_text),
                "generator_label": RANKED_GENERATOR_LABEL,
                "ordering": "random_seeded_order",
            }
            for candidate in shuffled
        )
    return rows


def _repair_aware_order_rows(candidates: list[QuixBugsCandidate]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        score, reasons = _repair_aware_candidate_score(candidate.source_text)
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "program": candidate.program,
                "source_text": candidate.source_text,
                "source_sha256": _sha256(candidate.source_text),
                "generator_label": RANKED_GENERATOR_LABEL,
                "ordering": "repair_aware_static_order",
                "repair_aware_score": score,
                "repair_aware_reasons": reasons,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["program"],
            -float(row["repair_aware_score"]),
            row["candidate_id"],
        ),
    )


def _repair_aware_candidate_score(source_text: str) -> tuple[float, list[str]]:
    tree = ast.parse(source_text)
    reasons: list[str] = []
    score = 0.0
    if _has_empty_guard_before_star_unpack(tree):
        score += 10.0
        reasons.append("empty_guard_before_star_unpack")
    if _has_value_type_isinstance_call(tree):
        score += 8.0
        reasons.append("value_type_isinstance_call")
    if _has_recursive_modulo_call(tree):
        score += 8.0
        reasons.append("recursive_modulo_call")
    if _has_squared_residual_check(tree):
        score += 8.0
        reasons.append("squared_residual_check")
    if _has_final_counter_zero_check(tree):
        score += 8.0
        reasons.append("final_counter_zero_check")
    return score, reasons


def _has_empty_guard_before_star_unpack(tree: ast.Module) -> bool:
    for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
        for previous, current in zip(function.body, function.body[1:], strict=False):
            if not _is_star_unpack_assign(current):
                continue
            assert isinstance(current, ast.Assign)
            if _is_not_name_guard(previous, current.value):
                return True
    return False


def _is_not_name_guard(statement: ast.stmt, guarded_value: ast.AST) -> bool:
    return (
        isinstance(statement, ast.If)
        and isinstance(guarded_value, ast.Name)
        and isinstance(statement.test, ast.UnaryOp)
        and isinstance(statement.test.op, ast.Not)
        and isinstance(statement.test.operand, ast.Name)
        and statement.test.operand.id == guarded_value.id
    )


def _has_value_type_isinstance_call(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
            and len(node.args) >= 2
            and not _looks_like_type_expr(node.args[0])
            and _looks_like_type_expr(node.args[1])
        ):
            return True
    return False


def _looks_like_type_expr(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in {
            "bool",
            "bytes",
            "dict",
            "float",
            "frozenset",
            "int",
            "list",
            "set",
            "str",
            "tuple",
        }
    if isinstance(node, ast.Tuple):
        return all(_looks_like_type_expr(element) for element in node.elts)
    return False


def _has_recursive_modulo_call(tree: ast.Module) -> bool:
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in function_names
            and any(_contains_operator(arg, ast.Mod) for arg in node.args)
        ):
            return True
    return False


def _has_squared_residual_check(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and any(
            _is_square_expression(descendant) for descendant in ast.walk(node)
        ):
            return True
    return False


def _contains_operator(node: ast.AST, op_type: type[ast.operator]) -> bool:
    return any(
        isinstance(descendant, ast.BinOp) and isinstance(descendant.op, op_type)
        for descendant in ast.walk(node)
    )


def _is_square_expression(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Mult)
        and ast.dump(node.left, include_attributes=False)
        == ast.dump(node.right, include_attributes=False)
    )


def _has_final_counter_zero_check(tree: ast.Module) -> bool:
    for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
        if not function.body:
            continue
        final_statement = function.body[-1]
        if (
            isinstance(final_statement, ast.Return)
            and isinstance(final_statement.value, ast.Compare)
            and len(final_statement.value.ops) == 1
            and isinstance(final_statement.value.ops[0], ast.Eq)
            and len(final_statement.value.comparators) == 1
            and isinstance(final_statement.value.left, ast.Name)
            and isinstance(final_statement.value.comparators[0], ast.Constant)
            and final_statement.value.comparators[0].value == 0
            and final_statement.value.left.id in _updated_numeric_names(function)
        ):
            return True
    return False


def _union_top_rows(
    orderings: dict[str, list[dict[str, Any]]],
    top_k: int,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for ordered_rows in orderings.values():
        for row in ordered_rows:
            if int(row["rank_within_program"]) <= top_k:
                by_id.setdefault(str(row["candidate_id"]), row)
    return sorted(by_id.values(), key=lambda row: (row["program"], row["candidate_id"]))


def _summarize_ordering_control(
    name: str,
    ordered_rows: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
    top_k_values: tuple[int, ...],
    programs: tuple[str, ...],
) -> dict[str, Any]:
    top_k_profile = [
        _summarize_top_k_candidate_results(
            candidate_results,
            ordered_rows,
            top_k,
            programs,
        )
        for top_k in top_k_values
    ]
    best_profile = max(
        top_k_profile,
        key=lambda row: (
            row["program_repair_rate"],
            row["candidate_pass_rate"],
            -row["top_candidates_per_program"],
        ),
    )
    selected_ids = {
        row["candidate_id"]
        for row in ordered_rows
        if int(row["rank_within_program"]) <= top_k_values[-1]
    }
    return {
        "name": name,
        "candidate_count": len(selected_ids),
        "best_top_k": best_profile["top_candidates_per_program"],
        "best_program_repair_rate": best_profile["program_repair_rate"],
        "best_candidate_pass_rate": best_profile["candidate_pass_rate"],
        "top_k_profile": top_k_profile,
    }


def _syntax_mutation_sources(repo_path: Path, program: str, source: str) -> list[str]:
    seen = {source}
    sources: list[str] = []
    try:
        baseline_candidates = build_quixbugs_edit_baseline_candidates(
            repo_path,
            programs=(program,),
            max_candidates_per_program=64,
        )
    except ValueError as exc:
        if str(exc) != "deterministic edit baseline generated no candidates":
            raise
        baseline_candidates = []
    for candidate in baseline_candidates:
        _append_unique_source(sources, seen, candidate.source_text)
    tree = ast.parse(source)
    for mutated in _generic_mutations(tree):
        _append_unique_source(sources, seen, _to_source(mutated))
    return sources


def _append_unique_source(sources: list[str], seen: set[str], source: str) -> None:
    try:
        ast.parse(source)
    except SyntaxError:
        return
    if source not in seen:
        seen.add(source)
        sources.append(source)


def _generic_mutations(tree: ast.Module) -> list[ast.Module]:
    mutations: list[ast.Module] = []
    mutations.extend(_BinaryOperatorMutation.mutations(tree))
    mutations.extend(_CompareOperatorMutation.mutations(tree))
    mutations.extend(_ReturnConstantFlipMutation.mutations(tree))
    mutations.extend(_CallArgumentSwapMutation.mutations(tree))
    mutations.extend(_YieldCallUnwrapMutation.mutations(tree))
    mutations.extend(_StarUnpackGuardMutation.mutations(tree))
    mutations.extend(_AbsResidualSquareMutation.mutations(tree))
    mutations.extend(_FinalCounterZeroReturnMutation.mutations(tree))
    return mutations


def _to_source(tree: ast.AST) -> str:
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


class _BinaryOperatorMutation(ast.NodeTransformer):
    OPS = (ast.Add, ast.Sub, ast.Mult, ast.Mod, ast.FloorDiv)

    def __init__(self, target_index: int, op_type: type[ast.operator]) -> None:
        self.target_index = target_index
        self.op_type = op_type
        self.index = 0
        self.changed = False

    @classmethod
    def mutations(cls, tree: ast.Module) -> list[ast.Module]:
        nodes = [node for node in ast.walk(tree) if isinstance(node, ast.BinOp)]
        mutations = []
        for index, node in enumerate(nodes):
            for op_type in cls.OPS:
                if isinstance(node.op, op_type):
                    continue
                transformer = cls(index, op_type)
                mutated = transformer.visit(copy.deepcopy(tree))
                if transformer.changed:
                    mutations.append(mutated)
        return mutations

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if self.index == self.target_index:
            node.op = self.op_type()
            self.changed = True
        self.index += 1
        return node


class _CompareOperatorMutation(ast.NodeTransformer):
    OPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)

    def __init__(self, target_index: int, op_type: type[ast.cmpop]) -> None:
        self.target_index = target_index
        self.op_type = op_type
        self.index = 0
        self.changed = False

    @classmethod
    def mutations(cls, tree: ast.Module) -> list[ast.Module]:
        nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Compare) and len(node.ops) == 1
        ]
        mutations = []
        for index, node in enumerate(nodes):
            for op_type in cls.OPS:
                if isinstance(node.ops[0], op_type):
                    continue
                transformer = cls(index, op_type)
                mutated = transformer.visit(copy.deepcopy(tree))
                if transformer.changed:
                    mutations.append(mutated)
        return mutations

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if len(node.ops) == 1:
            if self.index == self.target_index:
                node.ops = [self.op_type()]
                self.changed = True
            self.index += 1
        return node


class _ReturnConstantFlipMutation(ast.NodeTransformer):
    def __init__(self, target_index: int) -> None:
        self.target_index = target_index
        self.index = 0
        self.changed = False

    @classmethod
    def mutations(cls, tree: ast.Module) -> list[ast.Module]:
        nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Constant)
            and node.value.value in {0, 1}
        ]
        mutations = []
        for index, _node in enumerate(nodes):
            transformer = cls(index)
            mutated = transformer.visit(copy.deepcopy(tree))
            if transformer.changed:
                mutations.append(mutated)
        return mutations

    def visit_Return(self, node: ast.Return) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Constant) and node.value.value in {0, 1}:
            if self.index == self.target_index:
                node.value = ast.Constant(value=1 - int(node.value.value))
                self.changed = True
            self.index += 1
        return node


class _CallArgumentSwapMutation(ast.NodeTransformer):
    def __init__(self, target_index: int) -> None:
        self.target_index = target_index
        self.index = 0
        self.changed = False

    @classmethod
    def mutations(cls, tree: ast.Module) -> list[ast.Module]:
        nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and len(node.args) == 2
        ]
        mutations = []
        for index, _node in enumerate(nodes):
            transformer = cls(index)
            mutated = transformer.visit(copy.deepcopy(tree))
            if transformer.changed:
                mutations.append(mutated)
        return mutations

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if len(node.args) == 2:
            if self.index == self.target_index:
                node.args = [node.args[1], node.args[0]]
                self.changed = True
            self.index += 1
        return node


class _YieldCallUnwrapMutation(ast.NodeTransformer):
    def __init__(self, target_index: int) -> None:
        self.target_index = target_index
        self.index = 0
        self.changed = False

    @classmethod
    def mutations(cls, tree: ast.Module) -> list[ast.Module]:
        nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Yield)
            and isinstance(node.value, ast.Call)
            and len(node.value.args) == 1
        ]
        mutations = []
        for index, _node in enumerate(nodes):
            transformer = cls(index)
            mutated = transformer.visit(copy.deepcopy(tree))
            if transformer.changed:
                mutations.append(mutated)
        return mutations

    def visit_Yield(self, node: ast.Yield) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Call) and len(node.value.args) == 1:
            if self.index == self.target_index:
                node.value = node.value.args[0]
                self.changed = True
            self.index += 1
        return node


class _StarUnpackGuardMutation(ast.NodeTransformer):
    def __init__(self, target_index: int, return_value: int) -> None:
        self.target_index = target_index
        self.return_value = return_value
        self.index = 0
        self.changed = False

    @classmethod
    def mutations(cls, tree: ast.Module) -> list[ast.Module]:
        nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], (ast.Tuple, ast.List))
            and any(isinstance(element, ast.Starred) for element in node.targets[0].elts)
            and isinstance(node.value, ast.Name)
        ]
        mutations = []
        for index, _node in enumerate(nodes):
            for return_value in (0, 1):
                transformer = cls(index, return_value)
                mutated = transformer.visit(copy.deepcopy(tree))
                if transformer.changed:
                    mutations.append(mutated)
        return mutations

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        new_body: list[ast.stmt] = []
        for statement in node.body:
            if _is_star_unpack_assign(statement):
                if self.index == self.target_index:
                    new_body.append(_empty_guard(statement, self.return_value))
                    self.changed = True
                self.index += 1
            new_body.append(statement)
        node.body = new_body
        return self.generic_visit(node)


def _is_star_unpack_assign(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], (ast.Tuple, ast.List))
        and any(isinstance(element, ast.Starred) for element in statement.targets[0].elts)
        and isinstance(statement.value, ast.Name)
    )


def _empty_guard(statement: ast.Assign, return_value: int) -> ast.If:
    assert isinstance(statement.value, ast.Name)
    return ast.If(
        test=ast.UnaryOp(
            op=ast.Not(),
            operand=ast.Name(id=statement.value.id, ctx=ast.Load()),
        ),
        body=[ast.Return(value=ast.Constant(value=return_value))],
        orelse=[],
    )


class _AbsResidualSquareMutation(ast.NodeTransformer):
    def __init__(self, target_index: int) -> None:
        self.target_index = target_index
        self.index = 0
        self.changed = False

    @classmethod
    def mutations(cls, tree: ast.Module) -> list[ast.Module]:
        nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Compare) and _abs_sub_operands(node.left) is not None
        ]
        mutations = []
        for index, _node in enumerate(nodes):
            transformer = cls(index)
            mutated = transformer.visit(copy.deepcopy(tree))
            if transformer.changed:
                mutations.append(mutated)
        return mutations

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        operands = _abs_sub_operands(node.left)
        if operands is not None:
            if self.index == self.target_index:
                left, right = operands
                node.left = ast.Call(
                    func=ast.Name(id="abs", ctx=ast.Load()),
                    args=[
                        ast.BinOp(
                            left=left,
                            op=ast.Sub(),
                            right=ast.BinOp(
                                left=copy.deepcopy(right),
                                op=ast.Mult(),
                                right=copy.deepcopy(right),
                            ),
                        )
                    ],
                    keywords=[],
                )
                self.changed = True
            self.index += 1
        return node


def _abs_sub_operands(node: ast.AST) -> tuple[ast.expr, ast.expr] | None:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "abs"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.BinOp)
        and isinstance(node.args[0].op, ast.Sub)
    ):
        return copy.deepcopy(node.args[0].left), copy.deepcopy(node.args[0].right)
    return None


class _FinalCounterZeroReturnMutation(ast.NodeTransformer):
    def __init__(self, target_index: int, counter_name: str) -> None:
        self.target_index = target_index
        self.counter_name = counter_name
        self.index = 0
        self.changed = False

    @classmethod
    def mutations(cls, tree: ast.Module) -> list[ast.Module]:
        sites: list[tuple[int, str]] = []
        for function in (
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ):
            counter_names = _updated_numeric_names(function)
            if (
                counter_names
                and function.body
                and _is_return_true(function.body[-1])
            ):
                for counter_name in sorted(counter_names):
                    sites.append((len(sites), counter_name))
        mutations = []
        for index, counter_name in sites:
            transformer = cls(index, counter_name)
            mutated = transformer.visit(copy.deepcopy(tree))
            if transformer.changed:
                mutations.append(mutated)
        return mutations

    def visit_Return(self, node: ast.Return) -> ast.AST:
        self.generic_visit(node)
        if _is_return_true(node):
            if self.index == self.target_index:
                node.value = ast.Compare(
                    left=ast.Name(id=self.counter_name, ctx=ast.Load()),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value=0)],
                )
                self.changed = True
            self.index += 1
        return node


def _updated_numeric_names(function: ast.FunctionDef) -> set[str]:
    initialized = {
        statement.targets[0].id
        for statement in ast.walk(function)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, int)
    }
    updated = {
        statement.target.id
        for statement in ast.walk(function)
        if isinstance(statement, ast.AugAssign)
        and isinstance(statement.target, ast.Name)
        and isinstance(statement.op, (ast.Add, ast.Sub))
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, int)
    }
    return initialized & updated


def _is_return_true(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Return)
        and isinstance(statement.value, ast.Constant)
        and statement.value.value is True
    )


def _rank_syntax_pool_candidates(
    model: Any,
    tokenizer: Any,
    repo_path: Path,
    candidates: list[QuixBugsCandidate],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        buggy_source = (
            repo_path / "python_programs" / f"{candidate.program}.py"
        ).read_text(encoding="utf-8")
        prompt = _prompt_for_program(candidate.program, buggy_source)
        score = score_candidate_mean_nll(
            model,
            tokenizer,
            prompt,
            candidate.source_text,
        )
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "program": candidate.program,
                "source_text": candidate.source_text,
                "source_sha256": _sha256(candidate.source_text),
                "generator_label": RANKED_GENERATOR_LABEL,
                **score,
            }
        )
    return sorted(rows, key=lambda row: (row["program"], row["mean_nll"], row["candidate_id"]))


def _select_top_ranked(
    ranked: list[dict[str, Any]],
    top_candidates_per_program: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    programs = sorted({row["program"] for row in ranked})
    for program in programs:
        rows = [row for row in ranked if row["program"] == program]
        selected.extend(rows[:top_candidates_per_program])
    return selected
