from __future__ import annotations

import difflib
import hashlib
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from weightlab.dense_training import (
    ByteTokenizer,
    DenseDecoder,
    _load_model_state,
)
from weightlab.lookup import _resolve_torch_accelerator


@dataclass(frozen=True)
class FunctionalEvalConfig:
    device: str = "rocm"
    seed: int = 424242
    tasks_per_kind: int = 64
    prefix_tokens: int = 96
    infill_prefix_tokens: int = 64
    target_tokens: int = 32
    suffix_tokens: int = 32


def evaluate_dense_functional_checkpoint(
    checkpoint_path: Path,
    texts: list[str],
    split_name: str,
    config: FunctionalEvalConfig,
) -> dict[str, Any]:
    tokenizer = ByteTokenizer()
    accelerator = _resolve_torch_accelerator(config.device)
    device = accelerator.device
    payload = torch.load(checkpoint_path, map_location=device)
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

    rng = random.Random(config.seed)
    text_hash = hashlib.sha256("\n".join(texts).encode("utf-8", errors="ignore")).hexdigest()
    started = time.perf_counter()
    completion = _run_window_tasks(
        model,
        tokenizer,
        texts,
        rng,
        device,
        prefix_tokens=config.prefix_tokens,
        target_tokens=config.target_tokens,
        tasks=config.tasks_per_kind,
        task_kind="prefix_completion",
    )
    infill = _run_window_tasks(
        model,
        tokenizer,
        texts,
        rng,
        device,
        prefix_tokens=config.infill_prefix_tokens,
        target_tokens=config.target_tokens,
        suffix_tokens=config.suffix_tokens,
        tasks=config.tasks_per_kind,
        task_kind="causal_span_reconstruction",
    )
    comment_anchored = _run_comment_tasks(
        model,
        tokenizer,
        texts,
        rng,
        device,
        prefix_tokens=config.prefix_tokens,
        target_tokens=config.target_tokens,
        tasks=max(1, config.tasks_per_kind // 2),
    )
    elapsed_s = time.perf_counter() - started

    return {
        "benchmark_label": "d4_dense_functional_checkpoint_evaluation",
        "checkpoint": str(checkpoint_path),
        "split": split_name,
        "seed": config.seed,
        "device": str(device),
        "texts_sha256": text_hash,
        "document_count_used": len(texts),
        "elapsed_s": elapsed_s,
        "model": {
            "hidden_dim": int(model_config["hidden_dim"]),
            "layers": int(model_config["layers"]),
            "heads": int(model_config["heads"]),
            "seq_len": int(model_config["seq_len"]),
            "architecture_variant": str(model_config.get("architecture_variant", "dense")),
            "adapter_dim": int(model_config.get("adapter_dim", 0)),
            "attention_mask_mode": str(model_config.get("attention_mask_mode", "")),
            "block_impl": str(model_config.get("block_impl", "")),
        },
        "tasks": {
            "prefix_completion": completion,
            "causal_span_reconstruction": infill,
            "comment_anchored_source_completion": comment_anchored,
            "documentation_source_consistency": {
                "status": "not_measured",
                "reason": (
                    "D4 is a JavaScript source-only corpus with no paired documentation/source "
                    "consistency benchmark. Comment-anchored source completion is reported as a "
                    "held-out source-local proxy, not a documentation consistency claim."
                ),
            },
        },
        "limitations": [
            (
                "Greedy byte-level generation is deterministic and harsh; exact-string scores "
                "are expected to be low."
            ),
            (
                "The model is a causal decoder, so span reconstruction is not "
                "suffix-conditioned infilling."
            ),
            (
                "Functional tasks use held-out D4 rows only, but they are lightweight probes "
                "rather than executable JavaScript tests."
            ),
        ],
    }


def _run_window_tasks(
    model: DenseDecoder,
    tokenizer: ByteTokenizer,
    texts: list[str],
    rng: random.Random,
    device: torch.device,
    *,
    prefix_tokens: int,
    target_tokens: int,
    tasks: int,
    task_kind: str,
    suffix_tokens: int = 0,
) -> dict[str, Any]:
    candidates: list[tuple[int, list[int]]] = []
    required = prefix_tokens + target_tokens + suffix_tokens
    for index, text in enumerate(texts):
        ids = tokenizer.encode(text)[:-1]
        if len(ids) >= required:
            candidates.append((index, ids))
    rng.shuffle(candidates)
    examples: list[dict[str, Any]] = []
    for row_index, ids in candidates[:tasks]:
        max_start = len(ids) - required
        start = rng.randint(0, max_start) if max_start > 0 else 0
        prefix = ids[start: start + prefix_tokens]
        target = ids[start + prefix_tokens: start + prefix_tokens + target_tokens]
        suffix = ids[start + prefix_tokens + target_tokens: start + required]
        generated = _generate_ids(model, prefix, target_tokens, device, model.seq_len)
        examples.append(
            _score_example(
                tokenizer,
                row_index=row_index,
                start_token=start,
                task_kind=task_kind,
                prefix=prefix,
                target=target,
                generated=generated,
                suffix=suffix,
            )
        )
    return _summarize_examples(examples, requested_tasks=tasks, candidate_count=len(candidates))


def _run_comment_tasks(
    model: DenseDecoder,
    tokenizer: ByteTokenizer,
    texts: list[str],
    rng: random.Random,
    device: torch.device,
    *,
    prefix_tokens: int,
    target_tokens: int,
    tasks: int,
) -> dict[str, Any]:
    candidates: list[tuple[int, list[int], int]] = []
    comment_pattern = re.compile(r"/\*\*?|//")
    for index, text in enumerate(texts):
        match = comment_pattern.search(text)
        if match is None:
            continue
        ids = tokenizer.encode(text)[:-1]
        comment_start = len(text[: match.start()].encode("utf-8", errors="ignore"))
        if len(ids) - comment_start >= prefix_tokens + target_tokens:
            candidates.append((index, ids, comment_start))
    rng.shuffle(candidates)
    examples: list[dict[str, Any]] = []
    for row_index, ids, start in candidates[:tasks]:
        prefix = ids[start: start + prefix_tokens]
        target = ids[start + prefix_tokens: start + prefix_tokens + target_tokens]
        generated = _generate_ids(model, prefix, target_tokens, device, model.seq_len)
        examples.append(
            _score_example(
                tokenizer,
                row_index=row_index,
                start_token=start,
                task_kind="comment_anchored_source_completion",
                prefix=prefix,
                target=target,
                generated=generated,
                suffix=[],
            )
        )
    return _summarize_examples(examples, requested_tasks=tasks, candidate_count=len(candidates))


def _generate_ids(
    model: DenseDecoder,
    prefix: list[int],
    max_new_tokens: int,
    device: torch.device,
    seq_len: int,
) -> list[int]:
    ids = list(prefix)
    generated: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            window = ids[-seq_len:]
            input_ids = torch.tensor([window], dtype=torch.long, device=device)
            logits = model(input_ids)
            next_id = int(torch.argmax(logits[0, -1]).detach().cpu())
            if next_id == ByteTokenizer.eos_id:
                break
            ids.append(next_id)
            generated.append(next_id)
    if len(generated) < max_new_tokens:
        generated.extend([ByteTokenizer.eos_id] * (max_new_tokens - len(generated)))
    return generated[:max_new_tokens]


def _score_example(
    tokenizer: ByteTokenizer,
    *,
    row_index: int,
    start_token: int,
    task_kind: str,
    prefix: list[int],
    target: list[int],
    generated: list[int],
    suffix: list[int],
) -> dict[str, Any]:
    matches = sum(1 for got, expected in zip(generated, target, strict=False) if got == expected)
    target_text = tokenizer.decode(target)
    generated_text = tokenizer.decode(generated)
    return {
        "task_kind": task_kind,
        "row_index": row_index,
        "start_token": start_token,
        "prefix_sha256": _ids_sha256(prefix),
        "target_sha256": _ids_sha256(target),
        "suffix_sha256": _ids_sha256(suffix) if suffix else "",
        "generated_sha256": _ids_sha256(generated),
        "target_tokens": len(target),
        "generated_tokens": len(generated),
        "token_accuracy": matches / max(len(target), 1),
        "exact_match": generated == target,
        "edit_similarity": difflib.SequenceMatcher(None, generated_text, target_text).ratio(),
        "target_preview": target_text[:120],
        "generated_preview": generated_text[:120],
    }


def _summarize_examples(
    examples: list[dict[str, Any]],
    *,
    requested_tasks: int,
    candidate_count: int,
) -> dict[str, Any]:
    token_accuracies = [float(row["token_accuracy"]) for row in examples]
    edit_similarities = [float(row["edit_similarity"]) for row in examples]
    exact_matches = [bool(row["exact_match"]) for row in examples]
    return {
        "requested_tasks": requested_tasks,
        "candidate_count": candidate_count,
        "completed_tasks": len(examples),
        "token_accuracy_mean": _mean(token_accuracies),
        "token_accuracy_min": min(token_accuracies) if token_accuracies else math.nan,
        "token_accuracy_max": max(token_accuracies) if token_accuracies else math.nan,
        "exact_match_rate": _mean([1.0 if value else 0.0 for value in exact_matches]),
        "edit_similarity_mean": _mean(edit_similarities),
        "examples": examples[:5],
    }


def _ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(bytes(token for token in ids if 0 <= token < 256)).hexdigest()


def _mean(values: list[float]) -> float:
    if not values:
        return math.nan
    return float(sum(values) / len(values))
