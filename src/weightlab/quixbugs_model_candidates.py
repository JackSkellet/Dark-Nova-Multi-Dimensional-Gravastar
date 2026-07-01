from __future__ import annotations

import ast
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from weightlab.dense_functional_eval import _generate_ids
from weightlab.dense_training import (
    DenseDecoder,
    TokenizerLike,
    _load_model_state,
    _tokenizer_from_payload,
)
from weightlab.lookup import _resolve_torch_accelerator
from weightlab.quixbugs_repair import (
    QuixBugsCandidate,
    QuixBugsCandidateConfig,
    evaluate_quixbugs_candidate_repairs,
)


@dataclass(frozen=True)
class DenseQuixBugsCandidateConfig:
    device: str = "rocm"
    seed: int = 123
    max_new_tokens: int = 256
    timeout_seconds: int = 20
    samples_per_program: int = 1
    temperature: float = 0.0
    top_k: int = 0
    prefer_syntax_valid: bool = False


def evaluate_dense_quixbugs_candidate_repairs(
    checkpoint_path: Path,
    repo_path: Path,
    *,
    programs: tuple[str, ...],
    config: DenseQuixBugsCandidateConfig | None = None,
) -> dict[str, Any]:
    config = config or DenseQuixBugsCandidateConfig()
    if not programs:
        raise ValueError("at least one QuixBugs program is required")
    started = time.perf_counter()
    model, tokenizer, model_metadata = _load_dense_checkpoint(checkpoint_path, config.device)
    generated_candidates = []
    for program_index, program in enumerate(programs):
        generated_candidates.extend(
            _generate_program_candidates(
                model,
                tokenizer,
                repo_path,
                program,
                config,
                program_index=program_index,
            )
        )
    selected_candidates = _select_candidates_for_evaluation(generated_candidates, config)
    candidates = [
        QuixBugsCandidate(
            candidate_id=row["candidate_id"],
            program=row["program"],
            source_text=row["source_text"],
            generator_label=row["generator_label"],
        )
        for row in selected_candidates
    ]
    metrics = evaluate_quixbugs_candidate_repairs(
        repo_path,
        candidates,
        QuixBugsCandidateConfig(
            timeout_seconds=config.timeout_seconds,
            seed=config.seed,
        ),
    )
    metrics["benchmark_label"] = "quixbugs_python_dense_model_candidate_probe"
    metrics["model"] = {
        "checkpoint": str(checkpoint_path),
        **model_metadata,
    }
    metrics["generation"] = {
        "method": (
            "greedy_byte_generation"
            if config.samples_per_program == 1 and config.temperature <= 0
            else "sampled_byte_generation"
        ),
        "seed": config.seed,
        "max_new_tokens": config.max_new_tokens,
        "samples_per_program": config.samples_per_program,
        "temperature": config.temperature,
        "top_k": config.top_k,
        "prefer_syntax_valid": config.prefer_syntax_valid,
        "generated_candidate_count": len(generated_candidates),
        "evaluated_candidate_count": len(selected_candidates),
        "prompt_style": "buggy_source_then_fixed_source_marker",
    }
    metrics["syntax"] = _syntax_summary(generated_candidates, programs)
    metrics["generated_candidates"] = [
        {
            key: value
            for key, value in row.items()
            if key not in {"source_text", "prompt_text"}
        }
        for row in generated_candidates
    ]
    metrics["final"]["end_to_end_runtime_ms"] = (time.perf_counter() - started) * 1000.0
    metrics["limitations"] = [
        "local_model_generated_candidate",
        "prompted_source_replacement_from_general_lm",
        "replacement_source_only",
        "python_subset_only",
        "pytest_result_depends_on_local_environment",
    ]
    if config.samples_per_program == 1 and config.temperature <= 0:
        metrics["limitations"].append("greedy_byte_generation")
    else:
        metrics["limitations"].append("bounded_sampling_not_pass_at_k_estimate")
    if config.prefer_syntax_valid:
        metrics["limitations"].append("syntax_filter_falls_back_when_no_valid_candidate")
    return metrics


def _load_dense_checkpoint(
    checkpoint_path: Path,
    device_name: str,
) -> tuple[DenseDecoder, TokenizerLike, dict[str, Any]]:
    accelerator = _resolve_torch_accelerator(device_name)
    device = accelerator.device
    payload = torch.load(checkpoint_path, map_location=device)
    tokenizer = _tokenizer_from_payload(payload)
    model_config = payload["config"]
    model = DenseDecoder(
        tokenizer.vocab_size,
        int(model_config["seq_len"]),
        int(model_config["hidden_dim"]),
        int(model_config["layers"]),
        int(model_config["heads"]),
        str(model_config.get("attention_mask_mode", "additive_causal")),
        str(model_config.get("architecture_variant", "dense")),
        int(model_config.get("adapter_dim", 0)),
        str(model_config.get("block_impl", "torch_encoder")),
    ).to(device)
    _load_model_state(model, payload["model"])
    model.eval()
    return (
        model,
        tokenizer,
        {
            "device": str(device),
            "accelerator_backend": accelerator.backend,
            "checkpoint_step": int(payload.get("step", 0)),
            "checkpoint_type": str(payload.get("checkpoint_type", "training")),
            "tokenizer": tokenizer.to_jsonable(),
            "config": {
                "seq_len": int(model_config["seq_len"]),
                "hidden_dim": int(model_config["hidden_dim"]),
                "layers": int(model_config["layers"]),
                "heads": int(model_config["heads"]),
                "architecture_variant": str(
                    model_config.get("architecture_variant", "dense")
                ),
                "adapter_dim": int(model_config.get("adapter_dim", 0)),
                "attention_mask_mode": str(
                    model_config.get("attention_mask_mode", "")
                ),
                "block_impl": str(model_config.get("block_impl", "")),
            },
        },
    )


def _generate_program_candidates(
    model: DenseDecoder,
    tokenizer: TokenizerLike,
    repo_path: Path,
    program: str,
    config: DenseQuixBugsCandidateConfig,
    *,
    program_index: int,
) -> list[dict[str, Any]]:
    buggy_source = (repo_path / "python_programs" / f"{program}.py").read_text(
        encoding="utf-8"
    )
    prompt = _prompt_for_program(program, buggy_source)
    prompt_ids = tokenizer.encode(prompt)[:-1]
    rows: list[dict[str, Any]] = []
    for sample_index in range(config.samples_per_program):
        generated_ids = _generate_candidate_ids(
            model,
            prompt_ids,
            config,
            eos_id=tokenizer.eos_id,
            sample_seed=config.seed + program_index * 10_000 + sample_index,
        )
        generated_text = tokenizer.decode(generated_ids)
        source_text = _extract_source_candidate(generated_text)
        syntax = _python_syntax_check(source_text)
        generation_mode = "greedy" if config.temperature <= 0 else "sampled"
        rows.append(
            {
                "candidate_id": f"{program}:dense_{generation_mode}_{sample_index}",
                "program": program,
                "sample_index": sample_index,
                "source_text": source_text,
                "source_sha256": _sha256(source_text),
                "source_bytes": len(source_text.encode("utf-8", errors="ignore")),
                "prompt_sha256": _sha256(prompt),
                "prompt_bytes": len(prompt.encode("utf-8", errors="ignore")),
                "generated_sha256": _sha256(generated_text),
                "generated_bytes": len(generated_text.encode("utf-8", errors="ignore")),
                "generated_preview": generated_text[:160],
                "syntax_valid": syntax["valid"],
                "syntax_error": syntax["error"],
                "generator_label": f"dense_checkpoint_{generation_mode}",
                "selection_reason": "not_selected",
            }
        )
    return rows


def _generate_candidate_ids(
    model: DenseDecoder,
    prompt_ids: list[int],
    config: DenseQuixBugsCandidateConfig,
    *,
    eos_id: int,
    sample_seed: int,
) -> list[int]:
    device = next(model.parameters()).device
    if config.temperature <= 0:
        return _generate_ids(
            model,
            prompt_ids,
            config.max_new_tokens,
            device,
            model.seq_len,
        )
    ids = list(prompt_ids)
    generated: list[int] = []
    generator = torch.Generator(device=device).manual_seed(sample_seed)
    with torch.no_grad():
        for _ in range(config.max_new_tokens):
            window = ids[-model.seq_len :]
            input_ids = torch.tensor([window], dtype=torch.long, device=device)
            logits = model(input_ids)[0, -1].float() / config.temperature
            if config.top_k > 0:
                top_values, top_indices = torch.topk(
                    logits,
                    k=min(config.top_k, logits.shape[-1]),
                )
                probs = torch.softmax(top_values, dim=-1)
                sampled_index = int(torch.multinomial(probs, 1, generator=generator))
                next_id = int(top_indices[sampled_index].detach().cpu())
            else:
                probs = torch.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, 1, generator=generator))
            if next_id == eos_id:
                break
            ids.append(next_id)
            generated.append(next_id)
    return generated


def _select_candidates_for_evaluation(
    generated_candidates: list[dict[str, Any]],
    config: DenseQuixBugsCandidateConfig,
) -> list[dict[str, Any]]:
    if not config.prefer_syntax_valid:
        for row in generated_candidates:
            row["selection_reason"] = "all_candidates_evaluated"
        return generated_candidates
    selected: list[dict[str, Any]] = []
    programs = sorted({row["program"] for row in generated_candidates})
    for program in programs:
        rows = [row for row in generated_candidates if row["program"] == program]
        syntax_valid = [row for row in rows if row["syntax_valid"]]
        if syntax_valid:
            for row in syntax_valid:
                row["selection_reason"] = "syntax_valid_selected"
            selected.extend(syntax_valid)
        elif rows:
            rows[0]["selection_reason"] = "fallback_no_syntax_valid_candidate"
            selected.append(rows[0])
    return selected


def _syntax_summary(
    generated_candidates: list[dict[str, Any]],
    programs: tuple[str, ...],
) -> dict[str, Any]:
    syntax_valid = [row for row in generated_candidates if row["syntax_valid"]]
    programs_with_syntax_valid = {
        row["program"] for row in generated_candidates if row["syntax_valid"]
    }
    return {
        "generated_candidate_count": len(generated_candidates),
        "syntax_valid_candidate_count": len(syntax_valid),
        "syntax_valid_candidate_rate": len(syntax_valid) / len(generated_candidates)
        if generated_candidates
        else 0.0,
        "programs_with_syntax_valid_candidate": len(programs_with_syntax_valid),
        "program_syntax_valid_rate": len(programs_with_syntax_valid) / len(programs)
        if programs
        else 0.0,
    }


def _python_syntax_check(source_text: str) -> dict[str, Any]:
    try:
        ast.parse(source_text)
        return {"valid": True, "error": ""}
    except SyntaxError as exc:
        return {"valid": False, "error": f"{exc.__class__.__name__}: {exc.msg}"}


def _prompt_for_program(program: str, buggy_source: str) -> str:
    return (
        "# Repair this QuixBugs Python function.\n"
        f"# Program: {program}\n"
        "# Buggy source:\n"
        f"{buggy_source.rstrip()}\n\n"
        "# Fixed source:\n"
    )


def _extract_source_candidate(generated_text: str) -> str:
    stripped = generated_text.strip()
    if not stripped:
        return "# empty dense checkpoint generation\n"
    if "```" in stripped:
        parts = stripped.split("```")
        if len(parts) >= 2:
            stripped = parts[1]
            if stripped.startswith("python"):
                stripped = stripped.removeprefix("python").lstrip()
    return stripped.rstrip() + "\n"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
