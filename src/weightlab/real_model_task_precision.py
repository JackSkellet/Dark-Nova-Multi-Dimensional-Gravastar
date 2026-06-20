from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from weightlab.importance import (
    _groupwise_quantize_matrix_rows,
    _to_bf16_like,
    _uniform_quantize_matrix,
)
from weightlab.real_model_precision import _load_state_dict


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for byte in range(2**8):
        if byte not in bs:
            bs.append(byte)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(codepoint) for codepoint in cs], strict=True))


def _load_gpt2_vocab(vocab_path: Path | str) -> dict[str, int]:
    return json.loads(Path(vocab_path).read_text(encoding="utf-8"))


def _load_gpt2_merges(merges_path: Path | str) -> dict[tuple[str, str], int]:
    merges: dict[tuple[str, str], int] = {}
    for line in Path(merges_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        merges[(parts[0], parts[1])] = len(merges)
    return merges


def _gpt2_text_chunks(text: str) -> list[str]:
    pattern = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")
    return pattern.findall(text)


def _symbol_pairs(symbols: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(zip(symbols, symbols[1:], strict=False))


def _bpe_symbols(
    token: str,
    merge_ranks: dict[tuple[str, str], int],
    cache: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    if token in cache:
        return cache[token]
    symbols = tuple(token)
    if len(symbols) <= 1:
        cache[token] = symbols
        return symbols

    while True:
        pairs = _symbol_pairs(symbols)
        if not pairs:
            break
        bigram = min(pairs, key=lambda pair: merge_ranks.get(pair, float("inf")))
        if bigram not in merge_ranks:
            break

        first, second = bigram
        merged: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                merged.append(first + second)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        symbols = tuple(merged)
        if len(symbols) == 1:
            break

    cache[token] = symbols
    return symbols


def _encode_gpt2_text(
    text: str,
    vocab: dict[str, int],
    merge_ranks: dict[tuple[str, str], int],
) -> list[int]:
    byte_encoder = _bytes_to_unicode()
    cache: dict[str, tuple[str, ...]] = {}
    token_ids: list[int] = []
    for chunk in _gpt2_text_chunks(text):
        token = "".join(byte_encoder[byte] for byte in chunk.encode("utf-8"))
        for piece in _bpe_symbols(token, merge_ranks, cache):
            if piece in vocab:
                token_ids.append(vocab[piece])
                continue
            for symbol in piece:
                if symbol not in vocab:
                    raise KeyError(f"GPT-2 tokenizer symbol {symbol!r} missing from vocab")
                token_ids.append(vocab[symbol])
    return token_ids


def gpt2_texts_to_token_ids(
    texts: list[str],
    vocab_path: Path | str,
    merges_path: Path | str,
    sequence_length: int,
) -> torch.Tensor:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2 for next-token evaluation")
    vocab = _load_gpt2_vocab(vocab_path)
    merge_ranks = _load_gpt2_merges(merges_path)
    if "<|endoftext|>" not in vocab:
        raise KeyError("GPT-2 vocab must include <|endoftext|> for padding")
    pad_id = int(vocab["<|endoftext|>"])

    rows: list[list[int]] = []
    for text in texts:
        token_ids = _encode_gpt2_text(text, vocab, merge_ranks)
        if not token_ids:
            token_ids = [pad_id]
        token_ids = token_ids[:sequence_length]
        if len(token_ids) < sequence_length:
            token_ids.extend([pad_id] * (sequence_length - len(token_ids)))
        rows.append(token_ids)
    return torch.tensor(rows, dtype=torch.long)


def _layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, normalized_shape=(x.shape[-1],), weight=weight, bias=bias)


def _linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x @ weight + bias


def _infer_layer_count(state_dict: dict[str, torch.Tensor]) -> int:
    layers = set()
    for name in state_dict:
        parts = name.split(".")
        if len(parts) > 3 and parts[:2] == ["transformer", "h"] and parts[2].isdigit():
            layers.add(int(parts[2]))
    return max(layers) + 1 if layers else 0


def _tiny_gpt2_logits(
    state_dict: dict[str, torch.Tensor],
    token_ids: torch.Tensor,
    n_layer: int | None = None,
    n_head: int = 2,
) -> torch.Tensor:
    token_ids = token_ids.to(torch.long)
    seq_len = token_ids.shape[1]
    hidden = state_dict["transformer.wte.weight"][token_ids]
    hidden = hidden + state_dict["transformer.wpe.weight"][:seq_len].unsqueeze(0)
    d_model = hidden.shape[-1]
    n_layer = _infer_layer_count(state_dict) if n_layer is None else n_layer
    head_dim = d_model // n_head
    if head_dim * n_head != d_model:
        raise ValueError(f"n_head={n_head} does not divide d_model={d_model}")

    for layer_idx in range(n_layer):
        prefix = f"transformer.h.{layer_idx}"
        residual = hidden
        hidden_ln = _layer_norm(
            hidden,
            state_dict[f"{prefix}.ln_1.weight"],
            state_dict[f"{prefix}.ln_1.bias"],
        )
        qkv = _linear(
            hidden_ln,
            state_dict[f"{prefix}.attn.c_attn.weight"],
            state_dict[f"{prefix}.attn.c_attn.bias"],
        )
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        batch, tokens, _ = q.shape
        q = q.view(batch, tokens, n_head, head_dim).transpose(1, 2)
        k = k.view(batch, tokens, n_head, head_dim).transpose(1, 2)
        v = v.view(batch, tokens, n_head, head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        mask = torch.triu(torch.ones(tokens, tokens, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.to(scores.device), -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn_out = (attn @ v).transpose(1, 2).contiguous().view(batch, tokens, d_model)
        attn_out = _linear(
            attn_out,
            state_dict[f"{prefix}.attn.c_proj.weight"],
            state_dict[f"{prefix}.attn.c_proj.bias"],
        )
        hidden = residual + attn_out

        residual = hidden
        hidden_ln = _layer_norm(
            hidden,
            state_dict[f"{prefix}.ln_2.weight"],
            state_dict[f"{prefix}.ln_2.bias"],
        )
        mlp = _linear(
            hidden_ln,
            state_dict[f"{prefix}.mlp.c_fc.weight"],
            state_dict[f"{prefix}.mlp.c_fc.bias"],
        )
        mlp = F.gelu(mlp)
        mlp = _linear(
            mlp,
            state_dict[f"{prefix}.mlp.c_proj.weight"],
            state_dict[f"{prefix}.mlp.c_proj.bias"],
        )
        hidden = residual + mlp

    hidden = _layer_norm(
        hidden,
        state_dict["transformer.ln_f.weight"],
        state_dict["transformer.ln_f.bias"],
    )
    head = state_dict.get("lm_head.weight", state_dict["transformer.wte.weight"])
    return hidden @ head.T


def _next_token_targets(token_ids: torch.Tensor) -> torch.Tensor:
    return token_ids[:, 1:].to(torch.long)


def _next_token_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, :-1, :]


def _cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)))


def _kl_divergence(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> float:
    ref = _next_token_logits(reference_logits)
    cand = _next_token_logits(candidate_logits)
    ref_probs = torch.softmax(ref, dim=-1)
    log_ref = torch.log_softmax(ref, dim=-1)
    log_cand = torch.log_softmax(cand, dim=-1)
    value = float(torch.mean(torch.sum(ref_probs * (log_ref - log_cand), dim=-1)))
    return max(0.0, value)


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=-1)
    return float(torch.mean((pred == targets).to(torch.float32)))


def _candidate_rows(token_ids: torch.Tensor, row_count: int) -> np.ndarray:
    rows = torch.unique(token_ids).cpu().numpy().astype(np.int64)
    return rows[(rows >= 0) & (rows < row_count)]


def _candidate_rows_for_strategy(
    token_ids: torch.Tensor,
    row_count: int,
    strategy: str,
) -> np.ndarray:
    if strategy == "token_ids":
        return _candidate_rows(token_ids, row_count)
    if strategy == "all_rows":
        return np.arange(row_count, dtype=np.int64)
    raise ValueError(f"unknown candidate_row_strategy: {strategy}")


def _with_tensor(
    state_dict: dict[str, torch.Tensor],
    tensor_name: str,
    tensor_value: np.ndarray,
) -> dict[str, torch.Tensor]:
    candidate = dict(state_dict)
    candidate[tensor_name] = torch.as_tensor(tensor_value, dtype=torch.float32)
    return candidate


def evaluate_tiny_gpt2_task_precision(
    checkpoint_path: Path | str,
    tensor_name: str,
    calibration_token_ids: torch.Tensor,
    heldout_token_ids: torch.Tensor,
    model_id: str = "unknown",
    model_commit: str = "unknown",
    seed: int = 0,
    protected_count: int = 8,
    n_layer: int | None = None,
    n_head: int = 2,
    candidate_row_strategy: str = "token_ids",
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    state_dict = _load_state_dict(path)
    if tensor_name not in state_dict:
        raise KeyError(f"tensor {tensor_name!r} not found in checkpoint")
    tensor = state_dict[tensor_name]
    if tensor.ndim != 2 or not torch.is_floating_point(tensor):
        raise ValueError(f"tensor {tensor_name!r} is not a 2D floating matrix")
    matrix = tensor.detach().cpu().to(torch.float32).numpy()

    uniform_matrix, uniform_bytes = _uniform_quantize_matrix(matrix, bits=4)
    groupwise_matrix, groupwise_bytes = _groupwise_quantize_matrix_rows(matrix, bits=4)
    calibration_rows = _candidate_rows_for_strategy(
        calibration_token_ids,
        matrix.shape[0],
        candidate_row_strategy,
    )
    protected_count = min(protected_count, len(calibration_rows))

    with torch.no_grad():
        calibration_reference = _tiny_gpt2_logits(
            state_dict,
            calibration_token_ids,
            n_layer=n_layer,
            n_head=n_head,
        )
        heldout_reference = _tiny_gpt2_logits(
            state_dict,
            heldout_token_ids,
            n_layer=n_layer,
            n_head=n_head,
        )

    row_scores = []
    for row_idx in calibration_rows:
        candidate_matrix = groupwise_matrix.copy()
        candidate_matrix[row_idx] = matrix[row_idx]
        candidate_state = _with_tensor(state_dict, tensor_name, candidate_matrix)
        with torch.no_grad():
            candidate_logits = _tiny_gpt2_logits(
                candidate_state,
                calibration_token_ids,
                n_layer=n_layer,
                n_head=n_head,
            )
        row_scores.append(_kl_divergence(calibration_reference, candidate_logits))
    row_scores_arr = np.asarray(row_scores, dtype=np.float64)
    selected_rows = set(calibration_rows[np.argsort(row_scores_arr)[:protected_count]])

    def protected_matrix(rows: set[int], mode: str) -> np.ndarray:
        candidate = groupwise_matrix.copy()
        for row_idx in rows:
            if mode == "bf16":
                candidate[row_idx] = _to_bf16_like(matrix[row_idx])
            elif mode == "fp32":
                candidate[row_idx] = matrix[row_idx]
            else:
                raise ValueError(f"unknown protected-row mode: {mode}")
        return candidate

    heldout_targets = _next_token_targets(heldout_token_ids)

    def policy_metrics(candidate_matrix: np.ndarray, total_bytes: int) -> dict[str, float]:
        candidate_state = _with_tensor(state_dict, tensor_name, candidate_matrix)
        with torch.no_grad():
            logits = _tiny_gpt2_logits(
                candidate_state,
                heldout_token_ids,
                n_layer=n_layer,
                n_head=n_head,
            )
        next_logits = _next_token_logits(logits)
        return {
            "heldout_kl_divergence": _kl_divergence(heldout_reference, logits),
            "heldout_nll": _cross_entropy(next_logits, heldout_targets),
            "heldout_accuracy": _accuracy(next_logits, heldout_targets),
            "total_bytes": float(total_bytes),
        }

    bf16_bytes = groupwise_bytes + len(selected_rows) * (matrix.shape[1] * 2 + 4)
    fp32_bytes = groupwise_bytes + len(selected_rows) * (matrix.shape[1] * 4 + 4)
    rng = np.random.default_rng(seed)
    random_bf16_kl = []
    random_fp32_kl = []
    random_trials = max(16, protected_count * 4)
    random_pool = calibration_rows
    for _ in range(random_trials):
        random_rows = set(rng.choice(random_pool, size=protected_count, replace=False))
        random_bf16 = policy_metrics(protected_matrix(random_rows, "bf16"), bf16_bytes)
        random_fp32 = policy_metrics(protected_matrix(random_rows, "fp32"), fp32_bytes)
        random_bf16_kl.append(random_bf16["heldout_kl_divergence"])
        random_fp32_kl.append(random_fp32["heldout_kl_divergence"])

    with torch.no_grad():
        full_next_logits = _next_token_logits(heldout_reference)

    policies = {
        "full_fp32": {
            "heldout_kl_divergence": 0.0,
            "heldout_nll": _cross_entropy(full_next_logits, heldout_targets),
            "heldout_accuracy": _accuracy(full_next_logits, heldout_targets),
            "total_bytes": float(matrix.size * 4),
        },
        "uniform_int4": policy_metrics(uniform_matrix, uniform_bytes),
        "groupwise_int4": policy_metrics(groupwise_matrix, groupwise_bytes),
        "output_error_bf16_protected": policy_metrics(
            protected_matrix(selected_rows, "bf16"),
            bf16_bytes,
        ),
        "output_error_fp32_protected": policy_metrics(
            protected_matrix(selected_rows, "fp32"),
            fp32_bytes,
        ),
        "random_bf16_protected_mean": {
            "heldout_kl_divergence": float(np.mean(random_bf16_kl)),
            "total_bytes": float(bf16_bytes),
        },
        "random_bf16_protected_best": {
            "heldout_kl_divergence": float(np.min(random_bf16_kl)),
            "total_bytes": float(bf16_bytes),
        },
        "random_fp32_protected_mean": {
            "heldout_kl_divergence": float(np.mean(random_fp32_kl)),
            "total_bytes": float(fp32_bytes),
        },
        "random_fp32_protected_best": {
            "heldout_kl_divergence": float(np.min(random_fp32_kl)),
            "total_bytes": float(fp32_bytes),
        },
    }

    return {
        "source": {
            "checkpoint_path": str(path),
            "model_id": model_id,
            "model_commit": model_commit,
        },
        "task": {
            "metric": "next_token_cross_entropy",
            "calibration_sequences": int(calibration_token_ids.shape[0]),
            "heldout_sequences": int(heldout_token_ids.shape[0]),
            "sequence_length": int(heldout_token_ids.shape[1]),
            "n_layer": int(_infer_layer_count(state_dict) if n_layer is None else n_layer),
            "n_head": int(n_head),
            "candidate_rows": int(len(calibration_rows)),
            "candidate_row_strategy": candidate_row_strategy,
        },
        "tensor": {
            "name": tensor_name,
            "shape": list(matrix.shape),
            "dtype": "float32",
            "parameter_count": int(matrix.size),
        },
        "protected_count": int(protected_count),
        "selected_rows": sorted(int(row) for row in selected_rows),
        "candidate_row_ids": [int(row) for row in calibration_rows],
        "row_calibration_kl_scores": row_scores_arr.tolist(),
        "random_trials": int(random_trials),
        "precision_policies": policies,
        "notes": (
            "Local tiny-GPT2 forward pass using held-out next-token loss/KL. This evaluates "
            "task behavior on deterministic token-id sequences, not a natural-language corpus."
        ),
    }


def evaluate_tiny_gpt2_natural_text_precision(
    checkpoint_path: Path | str,
    tensor_name: str,
    tokenizer_vocab_path: Path | str,
    tokenizer_merges_path: Path | str,
    calibration_texts: list[str],
    heldout_texts: list[str],
    sequence_length: int,
    model_id: str = "unknown",
    model_commit: str = "unknown",
    seed: int = 0,
    protected_count: int = 8,
    n_layer: int | None = None,
    n_head: int = 2,
    candidate_row_strategy: str = "token_ids",
) -> dict[str, Any]:
    calibration_token_ids = gpt2_texts_to_token_ids(
        texts=calibration_texts,
        vocab_path=tokenizer_vocab_path,
        merges_path=tokenizer_merges_path,
        sequence_length=sequence_length,
    )
    heldout_token_ids = gpt2_texts_to_token_ids(
        texts=heldout_texts,
        vocab_path=tokenizer_vocab_path,
        merges_path=tokenizer_merges_path,
        sequence_length=sequence_length,
    )
    result = evaluate_tiny_gpt2_task_precision(
        checkpoint_path=checkpoint_path,
        tensor_name=tensor_name,
        calibration_token_ids=calibration_token_ids,
        heldout_token_ids=heldout_token_ids,
        model_id=model_id,
        model_commit=model_commit,
        seed=seed,
        protected_count=protected_count,
        n_layer=n_layer,
        n_head=n_head,
        candidate_row_strategy=candidate_row_strategy,
    )
    result["source"]["tokenizer_vocab_path"] = str(tokenizer_vocab_path)
    result["source"]["tokenizer_merges_path"] = str(tokenizer_merges_path)
    result["task"]["input_kind"] = "natural_language_text"
    result["task"]["calibration_texts"] = int(len(calibration_texts))
    result["task"]["heldout_texts"] = int(len(heldout_texts))
    result["task"]["tokenizer"] = "byte_level_gpt2_bpe"
    result["notes"] = (
        "Local tiny-GPT2 forward pass using byte-level GPT-2 BPE tokenized natural-language "
        "code/repository prose. This remains a tiny pinned checkpoint smoke test, not a "
        "useful-scale language benchmark."
    )
    return result
