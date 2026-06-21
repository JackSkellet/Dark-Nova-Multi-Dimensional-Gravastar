from __future__ import annotations

import difflib
import hashlib
import math
import random
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from weightlab.dense_functional_eval import _generate_ids
from weightlab.dense_training import (
    ByteTokenizer,
    DenseDecoder,
    _load_model_state,
)
from weightlab.lookup import _resolve_torch_accelerator


@dataclass(frozen=True)
class ExecutableJsEvalConfig:
    device: str = "rocm"
    seed: int = 424242
    tasks: int = 64
    prefix_chars: int = 16
    target_tokens: int = 48
    node_binary: str = "node"
    timeout_s: float = 2.0


def evaluate_dense_js_executable_checkpoint(
    checkpoint_path: Path,
    texts: list[str],
    split_name: str,
    config: ExecutableJsEvalConfig,
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

    node_probe = _node_check_javascript(
        "function __weightlab_probe__() { return 1; }\n",
        node_binary=config.node_binary,
        timeout_s=config.timeout_s,
    )
    rng = random.Random(config.seed)
    started = time.perf_counter()
    candidates = _line_completion_candidates(
        texts,
        prefix_chars=config.prefix_chars,
        target_chars=config.target_tokens,
        node_binary=config.node_binary,
        timeout_s=config.timeout_s,
        max_candidates=max(config.tasks * 16, config.tasks),
    )
    rng.shuffle(candidates)
    examples = [
        _run_line_completion_example(
            model,
            tokenizer,
            candidate,
            device,
            config,
        )
        for candidate in candidates[: config.tasks]
    ]
    elapsed_s = time.perf_counter() - started

    text_hash = hashlib.sha256("\n".join(texts).encode("utf-8", errors="ignore")).hexdigest()
    return {
        "benchmark_label": "d4_dense_js_executable_checkpoint_evaluation",
        "checkpoint": str(checkpoint_path),
        "split": split_name,
        "seed": config.seed,
        "device": str(device),
        "texts_sha256": text_hash,
        "document_count_used": len(texts),
        "elapsed_s": elapsed_s,
        "node": {
            "binary": config.node_binary,
            "available": bool(node_probe["available"]),
            "version": _node_version(config.node_binary),
            "probe": node_probe,
        },
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
            "line_completion_syntax": _summarize_executable_examples(
                examples,
                requested_tasks=config.tasks,
                candidate_count=len(candidates),
            )
        },
        "limitations": [
            "javascript_syntax_check_only_not_full_unit_tests",
            "line_completion_context_only_not_full_file_context",
            "greedy_byte_generation",
            "oracle_lines_filtered_by_node_check",
            "no_repair_or_api_reuse_task_yet",
        ],
    }


def _run_line_completion_example(
    model: DenseDecoder,
    tokenizer: ByteTokenizer,
    candidate: dict[str, Any],
    device: torch.device,
    config: ExecutableJsEvalConfig,
) -> dict[str, Any]:
    prefix = str(candidate["prefix"])
    target = str(candidate["target"])
    prefix_ids = tokenizer.encode(prefix)[:-1]
    target_ids = tokenizer.encode(target)[:-1]
    generated_ids = _generate_ids(
        model,
        prefix_ids,
        min(config.target_tokens, len(target_ids)),
        device,
        model.seq_len,
    )
    generated = tokenizer.decode(generated_ids)
    oracle_source = _wrap_js_line(prefix + target)
    generated_source = _wrap_js_line(prefix + generated)
    generated_check = _node_check_javascript(
        generated_source,
        node_binary=config.node_binary,
        timeout_s=config.timeout_s,
    )
    matches = sum(
        1 for got, expected in zip(generated_ids, target_ids, strict=False) if got == expected
    )
    return {
        "row_index": int(candidate["row_index"]),
        "line_index": int(candidate["line_index"]),
        "line_sha256": _text_sha256(str(candidate["line"])),
        "prefix_sha256": _text_sha256(prefix),
        "target_sha256": _text_sha256(target),
        "generated_sha256": _text_sha256(generated),
        "target_tokens": len(target_ids),
        "generated_tokens": len(generated_ids),
        "token_accuracy": matches / max(len(target_ids), 1),
        "exact_match": generated_ids == target_ids,
        "edit_similarity": difflib.SequenceMatcher(None, generated, target).ratio(),
        "oracle_node_check": candidate["oracle_node_check"],
        "generated_node_check": generated_check,
        "oracle_source_sha256": _text_sha256(oracle_source),
        "generated_source_sha256": _text_sha256(generated_source),
        "prefix_preview": prefix[:120],
        "target_preview": target[:120],
        "generated_preview": generated[:120],
    }


def _line_completion_candidates(
    texts: list[str],
    *,
    prefix_chars: int,
    target_chars: int,
    node_binary: str = "node",
    timeout_s: float = 2.0,
    max_candidates: int = 1024,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row_index, text in enumerate(texts):
        for line_index, raw_line in enumerate(text.splitlines()):
            line = raw_line.strip()
            if not _eligible_js_line(line, prefix_chars=prefix_chars):
                continue
            prefix = line[:prefix_chars]
            target = line[prefix_chars:]
            if not target or len(target.encode("utf-8", errors="ignore")) > target_chars:
                continue
            oracle_check = _node_check_javascript(
                _wrap_js_line(prefix + target),
                node_binary=node_binary,
                timeout_s=timeout_s,
            )
            if not oracle_check["passed"]:
                continue
            candidates.append(
                {
                    "row_index": row_index,
                    "line_index": line_index,
                    "line": line,
                    "prefix": prefix,
                    "target": target,
                    "oracle_node_check": oracle_check,
                }
            )
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def _eligible_js_line(line: str, *, prefix_chars: int) -> bool:
    if len(line) <= prefix_chars:
        return False
    stripped = line.strip()
    if not stripped:
        return False
    disallowed_prefixes = (
        "//",
        "*",
        "/*",
        "import ",
        "export ",
        "class ",
        "else",
        "case ",
        "default:",
    )
    if stripped.startswith(disallowed_prefixes):
        return False
    if any(token in stripped for token in ("await ", "yield ", "=>")):
        return False
    return stripped.endswith(";") or stripped.endswith("}")


def _wrap_js_line(line: str) -> str:
    return "function __weightlab_line_probe__() {\n" + line + "\n}\n"


def _node_check_javascript(
    source: str,
    *,
    node_binary: str = "node",
    timeout_s: float = 2.0,
) -> dict[str, Any]:
    if shutil.which(node_binary) is None:
        return {
            "available": False,
            "passed": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"{node_binary} not found",
        }
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".js",
        encoding="utf-8",
        delete=True,
    ) as handle:
        handle.write(source)
        handle.flush()
        try:
            completed = subprocess.run(
                [node_binary, "--check", handle.name],
                text=True,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "available": True,
                "passed": False,
                "returncode": None,
                "stdout": exc.stdout or "",
                "stderr": "node --check timed out",
            }
    return {
        "available": True,
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-500:],
        "stderr": completed.stderr[-500:],
    }


def _node_version(node_binary: str) -> str:
    if shutil.which(node_binary) is None:
        return ""
    completed = subprocess.run(
        [node_binary, "--version"],
        text=True,
        capture_output=True,
        timeout=2.0,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _summarize_executable_examples(
    examples: list[dict[str, Any]],
    *,
    requested_tasks: int,
    candidate_count: int,
) -> dict[str, Any]:
    token_accuracies = [float(row["token_accuracy"]) for row in examples]
    edit_similarities = [float(row["edit_similarity"]) for row in examples]
    exact_matches = [bool(row["exact_match"]) for row in examples]
    oracle_passes = [bool(row["oracle_node_check"]["passed"]) for row in examples]
    generated_passes = [bool(row["generated_node_check"]["passed"]) for row in examples]
    return {
        "requested_tasks": requested_tasks,
        "candidate_count": candidate_count,
        "completed_tasks": len(examples),
        "token_accuracy_mean": _mean(token_accuracies),
        "exact_match_rate": _mean([1.0 if value else 0.0 for value in exact_matches]),
        "edit_similarity_mean": _mean(edit_similarities),
        "oracle_node_syntax_pass_rate": _mean([1.0 if value else 0.0 for value in oracle_passes]),
        "generated_node_syntax_pass_rate": _mean(
            [1.0 if value else 0.0 for value in generated_passes]
        ),
        "examples": examples[:5],
    }


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _mean(values: list[float]) -> float:
    if not values:
        return math.nan
    return float(sum(values) / len(values))
