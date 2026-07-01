from __future__ import annotations

import ast
import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weightlab.quixbugs_repair import (
    QuixBugsCandidate,
    QuixBugsCandidateConfig,
    evaluate_quixbugs_candidate_repairs,
)

GENERATOR_LABEL = "deterministic_ast_edit_baseline"


@dataclass(frozen=True)
class QuixBugsEditBaselineConfig:
    programs: tuple[str, ...] = ()
    timeout_seconds: int = 20
    seed: int = 123
    max_candidates_per_program: int = 8


def evaluate_quixbugs_edit_baseline(
    repo_path: Path,
    config: QuixBugsEditBaselineConfig | None = None,
) -> dict[str, Any]:
    config = config or QuixBugsEditBaselineConfig()
    programs = tuple(config.programs) or _discover_programs(Path(repo_path))
    candidates = build_quixbugs_edit_baseline_candidates(
        repo_path,
        programs=programs,
        max_candidates_per_program=config.max_candidates_per_program,
    )
    started = time.perf_counter()
    metrics = evaluate_quixbugs_candidate_repairs(
        repo_path,
        candidates,
        QuixBugsCandidateConfig(
            timeout_seconds=config.timeout_seconds,
            seed=config.seed,
        ),
    )
    metrics["benchmark_label"] = "quixbugs_python_edit_baseline_probe"
    metrics["generator"] = {
        "label": GENERATOR_LABEL,
        "candidate_count": len(candidates),
        "max_candidates_per_program": config.max_candidates_per_program,
        "edit_templates": [
            "swap_recursive_call_arguments",
            "yield_recursive_call_argument",
            "insert_empty_sequence_guard_before_star_unpack",
            "square_root_residual_condition",
        ],
    }
    metrics["limitations"] = [
        "not_model_generated",
        "hand_engineered_deterministic_edits",
        "does_not_read_oracle_correct_sources",
        "replacement_source_only",
        "python_subset_only",
        "not_public_leaderboard_comparison",
    ]
    metrics["final"]["baseline_runtime_ms"] = (time.perf_counter() - started) * 1000.0
    return metrics


def build_quixbugs_edit_baseline_candidates(
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
        source = _read_buggy_source(repo_path, program)
        candidate_sources = _generate_candidate_sources(source, program)
        for index, candidate_source in enumerate(
            candidate_sources[:max_candidates_per_program],
            start=1,
        ):
            candidates.append(
                QuixBugsCandidate(
                    candidate_id=f"{program}:edit_{index:02d}",
                    program=program,
                    source_text=candidate_source,
                    generator_label=GENERATOR_LABEL,
                )
            )
    if not candidates:
        raise ValueError("deterministic edit baseline generated no candidates")
    return candidates


def _generate_candidate_sources(source: str, program: str) -> list[str]:
    tree = ast.parse(source)
    candidates: list[str] = []
    seen = {source}
    for transformer in (
        _RecursiveCallArgumentSwap,
        _YieldRecursiveCallArgument,
        _EmptySequenceGuardBeforeStarUnpack,
        _SqrtResidualCondition,
    ):
        for mutated in transformer(program).mutations(tree):
            candidate_source = _to_source(mutated)
            if candidate_source not in seen:
                ast.parse(candidate_source)
                seen.add(candidate_source)
                candidates.append(candidate_source)
    return candidates


def _to_source(tree: ast.AST) -> str:
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _read_buggy_source(repo_path: Path, program: str) -> str:
    path = repo_path / "python_programs" / f"{program}.py"
    if not path.exists():
        raise FileNotFoundError(f"missing QuixBugs source file: {path}")
    return path.read_text(encoding="utf-8")


def _discover_programs(repo_path: Path) -> tuple[str, ...]:
    testcase_dir = repo_path / "python_testcases"
    return tuple(
        sorted(
            path.stem.removeprefix("test_")
            for path in testcase_dir.glob("test_*.py")
            if path.is_file()
        )
    )


class _SingleMutationTransformer(ast.NodeTransformer):
    def __init__(self, target_index: int) -> None:
        self.target_index = target_index
        self.seen = 0
        self.changed = False

    def _should_mutate(self) -> bool:
        if self.seen == self.target_index:
            self.changed = True
            return True
        self.seen += 1
        return False


class _MutationFactory:
    def __init__(self, program: str) -> None:
        self.program = program

    def mutations(self, tree: ast.Module) -> list[ast.Module]:
        mutations: list[ast.Module] = []
        for index in range(self._count_sites(tree)):
            cloned = copy.deepcopy(tree)
            transformer = self._build_transformer(index)
            mutated = transformer.visit(cloned)
            if transformer.changed:
                mutations.append(mutated)
        return mutations

    def _count_sites(self, tree: ast.Module) -> int:
        raise NotImplementedError

    def _build_transformer(self, index: int) -> _SingleMutationTransformer:
        raise NotImplementedError


class _RecursiveCallArgumentSwap(_MutationFactory):
    def _count_sites(self, tree: ast.Module) -> int:
        return sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == self.program
            and len(node.args) == 2
        )

    def _build_transformer(self, index: int) -> _SingleMutationTransformer:
        program = self.program

        class Transformer(_SingleMutationTransformer):
            def visit_Call(self, node: ast.Call) -> ast.AST:
                self.generic_visit(node)
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id == program
                    and len(node.args) == 2
                    and self._should_mutate()
                ):
                    node.args = [node.args[1], node.args[0]]
                return node

        return Transformer(index)


class _YieldRecursiveCallArgument(_MutationFactory):
    def _count_sites(self, tree: ast.Module) -> int:
        return sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Yield)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == self.program
            and len(node.value.args) == 1
        )

    def _build_transformer(self, index: int) -> _SingleMutationTransformer:
        program = self.program

        class Transformer(_SingleMutationTransformer):
            def visit_Yield(self, node: ast.Yield) -> ast.AST:
                self.generic_visit(node)
                if (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == program
                    and len(node.value.args) == 1
                    and self._should_mutate()
                ):
                    node.value = node.value.args[0]
                return node

        return Transformer(index)


class _EmptySequenceGuardBeforeStarUnpack(_MutationFactory):
    def _count_sites(self, tree: ast.Module) -> int:
        return sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], (ast.Tuple, ast.List))
            and any(isinstance(element, ast.Starred) for element in node.targets[0].elts)
            and isinstance(node.value, ast.Name)
        )

    def _build_transformer(self, index: int) -> _SingleMutationTransformer:
        class Transformer(_SingleMutationTransformer):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
                new_body: list[ast.stmt] = []
                for statement in node.body:
                    if (
                        isinstance(statement, ast.Assign)
                        and len(statement.targets) == 1
                        and isinstance(statement.targets[0], (ast.Tuple, ast.List))
                        and any(
                            isinstance(element, ast.Starred)
                            for element in statement.targets[0].elts
                        )
                        and isinstance(statement.value, ast.Name)
                        and self._should_mutate()
                    ):
                        new_body.append(
                            ast.If(
                                test=ast.UnaryOp(
                                    op=ast.Not(),
                                    operand=ast.Name(
                                        id=statement.value.id,
                                        ctx=ast.Load(),
                                    ),
                                ),
                                body=[ast.Return(value=ast.Constant(value=0))],
                                orelse=[],
                            )
                        )
                    new_body.append(statement)
                node.body = new_body
                return self.generic_visit(node)

        return Transformer(index)


class _SqrtResidualCondition(_MutationFactory):
    def _count_sites(self, tree: ast.Module) -> int:
        return sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Compare) and _is_abs_x_minus_approx(node.left)
        )

    def _build_transformer(self, index: int) -> _SingleMutationTransformer:
        class Transformer(_SingleMutationTransformer):
            def visit_Compare(self, node: ast.Compare) -> ast.AST:
                self.generic_visit(node)
                if _is_abs_x_minus_approx(node.left) and self._should_mutate():
                    node.left = ast.Call(
                        func=ast.Name(id="abs", ctx=ast.Load()),
                        args=[
                            ast.BinOp(
                                left=ast.Name(id="x", ctx=ast.Load()),
                                op=ast.Sub(),
                                right=ast.BinOp(
                                    left=ast.Name(id="approx", ctx=ast.Load()),
                                    op=ast.Mult(),
                                    right=ast.Name(id="approx", ctx=ast.Load()),
                                ),
                            )
                        ],
                        keywords=[],
                    )
                return node

        return Transformer(index)


def _is_abs_x_minus_approx(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "abs"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.BinOp)
        and isinstance(node.args[0].op, ast.Sub)
        and isinstance(node.args[0].left, ast.Name)
        and node.args[0].left.id == "x"
        and isinstance(node.args[0].right, ast.Name)
        and node.args[0].right.id == "approx"
    )
