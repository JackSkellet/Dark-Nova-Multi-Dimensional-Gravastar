from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from weightlab.dense_training import (
    ByteTokenizer,
    DenseDecoder,
    DenseTrainingConfig,
    _load_model_state,
    _tokens_from_texts,
    _validation_metrics,
)
from weightlab.lookup import _resolve_torch_accelerator
from weightlab.metrics import ceil_div


@dataclass(frozen=True)
class QuantizationEvalConfig:
    device: str = "rocm"
    seed: int = 424243
    batches: int = 128
    protected_fraction: float = 0.01
    sparse_fraction: float = 0.01


def evaluate_checkpoint_quantization(
    checkpoint_path: Path,
    texts: list[str],
    split_name: str,
    config: QuantizationEvalConfig,
) -> dict[str, Any]:
    tokenizer = ByteTokenizer()
    accelerator = _resolve_torch_accelerator(config.device)
    device = accelerator.device
    payload = torch.load(checkpoint_path, map_location="cpu")
    model_config = payload["config"]
    training_config = DenseTrainingConfig(
        device=config.device,
        seq_len=int(model_config["seq_len"]),
        hidden_dim=int(model_config["hidden_dim"]),
        layers=int(model_config["layers"]),
        heads=int(model_config["heads"]),
        batch_size=2,
        validation_batches=config.batches,
        mixed_precision="fp32",
        learning_rate=float(model_config.get("learning_rate", 0.0)),
        attention_mask_mode=str(model_config.get("attention_mask_mode", "additive_causal")),
        optimizer_name=str(model_config.get("optimizer_name", "adamw")),
        architecture_variant=str(model_config.get("architecture_variant", "dense")),
        adapter_dim=int(model_config.get("adapter_dim", 0)),
        validation_seed=config.seed,
        block_impl=str(model_config.get("block_impl", "torch_encoder")),
    )
    tokens = _tokens_from_texts(texts, tokenizer)
    base_state = {}
    for key, value in payload["model"].items():
        detached = value.detach().cpu()
        base_state[key] = detached.float() if torch.is_floating_point(value) else detached
    state_fingerprint = _state_fingerprint(base_state)
    policies = _build_policy_states(
        base_state,
        protected_fraction=config.protected_fraction,
        sparse_fraction=config.sparse_fraction,
        seed=config.seed,
    )

    model = DenseDecoder(
        tokenizer.vocab_size,
        training_config.seq_len,
        training_config.hidden_dim,
        training_config.layers,
        training_config.heads,
        training_config.attention_mask_mode,
        training_config.architecture_variant,
        training_config.adapter_dim,
        training_config.block_impl,
    ).to(device)

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for policy in policies:
        _load_model_state(model, policy.pop("state"))
        metrics = _validation_metrics(
            model,
            tokens,
            tokenizer.vocab_size,
            training_config,
            torch.Generator(device="cpu").manual_seed(config.seed),
            device,
        )
        policy["loss"] = metrics["loss"]
        policy["batches"] = metrics["batches"]
        policy["tokens"] = metrics["tokens"]
        policy["sample_order_sha256"] = metrics["sample_order_sha256"]
        rows.append(policy)
    elapsed_s = time.perf_counter() - started

    return {
        "benchmark_label": "trained_dense_checkpoint_quantization",
        "checkpoint": str(checkpoint_path),
        "split": split_name,
        "seed": config.seed,
        "device": str(device),
        "document_count_used": len(texts),
        "state_sha256": state_fingerprint,
        "elapsed_s": elapsed_s,
        "model": {
            "architecture_variant": training_config.architecture_variant,
            "adapter_dim": training_config.adapter_dim,
            "hidden_dim": training_config.hidden_dim,
            "layers": training_config.layers,
            "heads": training_config.heads,
            "seq_len": training_config.seq_len,
            "attention_mask_mode": training_config.attention_mask_mode,
            "block_impl": training_config.block_impl,
        },
        "config": {
            "batches": config.batches,
            "protected_fraction": config.protected_fraction,
            "sparse_fraction": config.sparse_fraction,
            "runtime_buffers_policy": (
                "runtime_buffer_bytes counts the FP32 reconstructed tensors used by this "
                "PyTorch evaluation path; encoded_bytes excludes that runtime workspace."
            ),
        },
        "policies": rows,
        "limitations": [
            (
                "Quantized states are reconstructed into FP32 tensors for evaluation; "
                "no packed kernel is benchmarked."
            ),
            (
                "Importance-selected protection uses weight residual magnitude, not "
                "activation-aware calibration."
            ),
            "Sparse residuals store FP32 residual values and 32-bit flat indexes.",
        ],
    }


def _build_policy_states(
    state: dict[str, torch.Tensor],
    *,
    protected_fraction: float,
    sparse_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    return [
        _fp32_policy(state),
        _bf16_policy(state),
        _uniform_policy(state, bits=8),
        _uniform_policy(state, bits=4),
        _protected_policy(
            state,
            bits=4,
            protected_fraction=protected_fraction,
            value_dtype="bf16",
            selection="residual",
            seed=seed,
        ),
        _protected_policy(
            state,
            bits=4,
            protected_fraction=protected_fraction,
            value_dtype="fp32",
            selection="residual",
            seed=seed,
        ),
        _protected_policy(
            state,
            bits=4,
            protected_fraction=protected_fraction,
            value_dtype="bf16",
            selection="random",
            seed=seed,
        ),
        _protected_policy(
            state,
            bits=4,
            protected_fraction=protected_fraction,
            value_dtype="fp32",
            selection="random",
            seed=seed,
        ),
        _sparse_residual_policy(state, bits=4, sparse_fraction=sparse_fraction),
    ]


def _fp32_policy(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    encoded = _float_state_bytes(state, value_bytes=4)
    return _policy_row(
        name="fp32",
        state={key: value.clone() for key, value in state.items()},
        encoded_bytes=encoded,
        metadata_bytes=0,
        protected_values=0,
        runtime_buffer_bytes=_runtime_buffer_bytes(state),
    )


def _bf16_policy(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    quantized = {
        key: _to_bf16(value) if torch.is_floating_point(value) else value.clone()
        for key, value in state.items()
    }
    encoded = _float_state_bytes(state, value_bytes=2)
    return _policy_row(
        name="bf16_all",
        state=quantized,
        encoded_bytes=encoded,
        metadata_bytes=0,
        protected_values=0,
        runtime_buffer_bytes=_runtime_buffer_bytes(state),
    )


def _uniform_policy(state: dict[str, torch.Tensor], *, bits: int) -> dict[str, Any]:
    quantized: dict[str, torch.Tensor] = {}
    encoded = 0
    metadata = 0
    for key, value in state.items():
        if torch.is_floating_point(value):
            reconstructed, accounting = _quantize_tensor(value, bits)
            quantized[key] = reconstructed
            encoded += accounting["encoded_bytes"]
            metadata += accounting["metadata_bytes"]
        else:
            quantized[key] = value.clone()
            encoded += value.numel() * value.element_size()
    return _policy_row(
        name=f"uniform_int{bits}",
        state=quantized,
        encoded_bytes=encoded,
        metadata_bytes=metadata,
        protected_values=0,
        runtime_buffer_bytes=_runtime_buffer_bytes(state),
    )


def _protected_policy(
    state: dict[str, torch.Tensor],
    *,
    bits: int,
    protected_fraction: float,
    value_dtype: str,
    selection: str,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    quantized: dict[str, torch.Tensor] = {}
    encoded = 0
    metadata = 0
    protected_values = 0
    for tensor_index, (key, value) in enumerate(state.items()):
        if not torch.is_floating_point(value):
            quantized[key] = value.clone()
            encoded += value.numel() * value.element_size()
            continue
        base, accounting = _quantize_tensor(value, bits)
        flat_original = value.reshape(-1)
        flat_base = base.reshape(-1)
        keep = _protected_count(flat_original.numel(), protected_fraction)
        if keep:
            if selection == "residual":
                indexes = torch.topk((flat_original - flat_base).abs(), k=keep).indices
            elif selection == "random":
                local_rng = random.Random(rng.randint(0, 2**31 - 1) + tensor_index)
                indexes = torch.tensor(
                    local_rng.sample(range(flat_original.numel()), keep),
                    dtype=torch.long,
                )
            else:
                raise ValueError(f"unknown selection: {selection}")
            if value_dtype == "bf16":
                protected = _to_bf16(flat_original[indexes])
                value_bytes = 2
            elif value_dtype == "fp32":
                protected = flat_original[indexes]
                value_bytes = 4
            else:
                raise ValueError(f"unknown protected dtype: {value_dtype}")
            flat_base[indexes] = protected
            protected_values += keep
            encoded += keep * (4 + value_bytes)
            metadata += keep * 4
        quantized[key] = flat_base.reshape_as(value).clone()
        encoded += accounting["encoded_bytes"]
        metadata += accounting["metadata_bytes"]
    return _policy_row(
        name=f"{selection}_{value_dtype}_protected_int{bits}",
        state=quantized,
        encoded_bytes=encoded,
        metadata_bytes=metadata,
        protected_values=protected_values,
        runtime_buffer_bytes=_runtime_buffer_bytes(state),
    )


def _sparse_residual_policy(
    state: dict[str, torch.Tensor],
    *,
    bits: int,
    sparse_fraction: float,
) -> dict[str, Any]:
    quantized: dict[str, torch.Tensor] = {}
    encoded = 0
    metadata = 0
    protected_values = 0
    for key, value in state.items():
        if not torch.is_floating_point(value):
            quantized[key] = value.clone()
            encoded += value.numel() * value.element_size()
            continue
        base, accounting = _quantize_tensor(value, bits)
        residual = value.reshape(-1) - base.reshape(-1)
        keep = _protected_count(residual.numel(), sparse_fraction)
        flat_base = base.reshape(-1)
        if keep:
            indexes = torch.topk(residual.abs(), k=keep).indices
            flat_base[indexes] = flat_base[indexes] + residual[indexes]
            protected_values += keep
            encoded += keep * (4 + 4) + 8
            metadata += keep * 4 + 8
        quantized[key] = flat_base.reshape_as(value).clone()
        encoded += accounting["encoded_bytes"]
        metadata += accounting["metadata_bytes"]
    return _policy_row(
        name=f"sparse_fp32_residual_int{bits}",
        state=quantized,
        encoded_bytes=encoded,
        metadata_bytes=metadata,
        protected_values=protected_values,
        runtime_buffer_bytes=_runtime_buffer_bytes(state),
    )


def _quantize_tensor(tensor: torch.Tensor, bits: int) -> tuple[torch.Tensor, dict[str, int]]:
    source = tensor.float()
    qmax = (2 ** (bits - 1)) - 1
    scale = float(source.abs().max().item() / max(qmax, 1)) or 1.0
    q = torch.clamp(torch.round(source / scale), -qmax - 1, qmax)
    reconstructed = (q * scale).float()
    encoded_bytes = ceil_div(source.numel() * bits, 8) + 12
    return reconstructed, {"encoded_bytes": encoded_bytes, "metadata_bytes": 12}


def _protected_count(numel: int, fraction: float) -> int:
    if numel <= 0 or fraction <= 0:
        return 0
    return max(1, int(numel * fraction))


def _to_bf16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.bfloat16).to(torch.float32)


def _policy_row(
    *,
    name: str,
    state: dict[str, torch.Tensor],
    encoded_bytes: int,
    metadata_bytes: int,
    protected_values: int,
    runtime_buffer_bytes: int,
) -> dict[str, Any]:
    return {
        "policy": name,
        "state": state,
        "encoded_bytes": int(encoded_bytes),
        "metadata_bytes": int(metadata_bytes),
        "protected_values": int(protected_values),
        "runtime_buffer_bytes": int(runtime_buffer_bytes),
        "encoded_plus_runtime_bytes": int(encoded_bytes + runtime_buffer_bytes),
        "effective_encoded_bits_per_parameter": float(
            encoded_bytes * 8 / max(_floating_parameter_count(state), 1)
        ),
    }


def _runtime_buffer_bytes(state: dict[str, torch.Tensor]) -> int:
    return _float_state_bytes(state, value_bytes=4)


def _float_state_bytes(state: dict[str, torch.Tensor], *, value_bytes: int) -> int:
    total = 0
    for value in state.values():
        if torch.is_floating_point(value):
            total += value.numel() * value_bytes
        else:
            total += value.numel() * value.element_size()
    return int(total)


def _floating_parameter_count(state: dict[str, torch.Tensor]) -> int:
    return int(sum(value.numel() for value in state.values() if torch.is_floating_point(value)))


def _state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
