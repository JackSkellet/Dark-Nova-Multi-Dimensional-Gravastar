from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from weightlab.dense_training import TokenizerLike
from weightlab.quixbugs_edit_baseline import build_quixbugs_edit_baseline_candidates
from weightlab.quixbugs_model_candidates import (
    _load_dense_checkpoint,
    _prompt_for_program,
    _sha256,
)
from weightlab.quixbugs_repair import (
    QuixBugsCandidate,
    QuixBugsCandidateConfig,
    evaluate_quixbugs_candidate_repairs,
)

GENERATOR_LABEL = "dense_ranked_ast_edit"


@dataclass(frozen=True)
class DenseQuixBugsEditRankConfig:
    device: str = "rocm"
    seed: int = 123
    timeout_seconds: int = 20
    max_candidates_per_program: int = 8
    top_candidates_per_program: int = 1


def evaluate_dense_ranked_quixbugs_edit_candidates(
    checkpoint_path: Path,
    repo_path: Path,
    *,
    programs: tuple[str, ...],
    config: DenseQuixBugsEditRankConfig | None = None,
) -> dict[str, Any]:
    config = config or DenseQuixBugsEditRankConfig()
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
    candidate_pool = build_quixbugs_edit_baseline_candidates(
        repo_path,
        programs=programs,
        max_candidates_per_program=config.max_candidates_per_program,
    )
    ranked = _rank_candidates(model, tokenizer, repo_path, candidate_pool)
    selected_rows = _select_top_ranked(ranked, config.top_candidates_per_program)
    selected_candidates = [
        QuixBugsCandidate(
            candidate_id=row["candidate_id"],
            program=row["program"],
            source_text=row["source_text"],
            generator_label=GENERATOR_LABEL,
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
    metrics["benchmark_label"] = "quixbugs_python_dense_ranked_edit_probe"
    metrics["model"] = {
        "checkpoint": str(checkpoint_path),
        **model_metadata,
    }
    metrics["candidate_pool"] = {
        "source": "deterministic_ast_edit_baseline",
        "generated_candidate_count": len(candidate_pool),
        "selected_candidate_count": len(selected_candidates),
        "max_candidates_per_program": config.max_candidates_per_program,
        "top_candidates_per_program": config.top_candidates_per_program,
    }
    metrics["ranking"] = [
        {
            key: value
            for key, value in row.items()
            if key not in {"source_text"}
        }
        for row in ranked
    ]
    metrics["limitations"] = [
        "local_model_ranked_candidate",
        "deterministic_candidate_pool",
        "not_free_form_generation",
        "replacement_source_only",
        "python_subset_only",
        "pytest_result_depends_on_local_environment",
        "ranking_uses_teacher_forced_candidate_likelihood",
    ]
    metrics["final"]["end_to_end_runtime_ms"] = (time.perf_counter() - started) * 1000.0
    return metrics


def score_candidate_mean_nll(
    model: torch.nn.Module,
    tokenizer: TokenizerLike,
    prompt_text: str,
    candidate_text: str,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt_text)[:-1]
    candidate_ids = tokenizer.encode(candidate_text)
    if not candidate_ids:
        raise ValueError("candidate_text produced no tokens")
    device = next(model.parameters()).device if any(True for _ in model.parameters()) else None
    if device is None:
        device = torch.device("cpu")
    context = list(prompt_ids)
    losses: list[float] = []
    with torch.no_grad():
        for token_id in candidate_ids:
            window = context[-int(model.seq_len) :]
            if not window:
                window = [tokenizer.eos_id]
            input_ids = torch.tensor([window], dtype=torch.long, device=device)
            logits = model(input_ids)[0, -1].float()
            target = torch.tensor([token_id], dtype=torch.long, device=device)
            loss = F.cross_entropy(logits.unsqueeze(0), target)
            losses.append(float(loss.detach().cpu()))
            context.append(token_id)
    mean_nll = sum(losses) / len(losses)
    return {
        "mean_nll": mean_nll,
        "total_nll": sum(losses),
        "token_count": len(candidate_ids),
    }


def _rank_candidates(
    model: torch.nn.Module,
    tokenizer: TokenizerLike,
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
                "generator_label": GENERATOR_LABEL,
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
        program_rows = [row for row in ranked if row["program"] == program]
        selected.extend(program_rows[:top_candidates_per_program])
    return selected
