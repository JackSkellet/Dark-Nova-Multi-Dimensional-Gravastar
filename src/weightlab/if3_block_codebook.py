from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import torch

from weightlab.dense_training import (
    DenseDecoder,
    DenseTrainingConfig,
    _load_model_state,
    _tokenizer_from_payload,
    _tokens_from_texts,
    _validation_metrics,
)
from weightlab.lookup import _resolve_torch_accelerator


def run_if3_block_codebook_probe(
    checkpoint_path: Path,
    *,
    block_size: int = 256,
    codebook_size: int = 256,
    residual_fraction: float = 0.01,
    seed: int = 123,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    compression = block_codebook_compress_state(
        state,
        block_size=block_size,
        codebook_size=codebook_size,
        residual_fraction=residual_fraction,
        seed=seed,
    )
    return {
        "benchmark_label": "if3_block_codebook_checkpoint_probe",
        "candidate_id": "IF3",
        "checkpoint": {
            "path": str(checkpoint_path),
            "checkpoint_type": str(payload.get("checkpoint_type", "unknown"))
            if isinstance(payload, dict)
            else "state_dict",
            "step": int(payload.get("step", 0)) if isinstance(payload, dict) else 0,
        },
        "model_config": payload.get("config", {}) if isinstance(payload, dict) else {},
        "compression": compression,
        "parameter_evolution": "not_applicable",
        "packed_kernel_evaluated": False,
        "loss_evaluated": False,
        "limitations": [
            "reconstruction_proxy_only_no_language_model_loss",
            "no_packed_kernel_or_runtime_speed_measurement",
            "kmeans_runs_on_cpu_tensors",
            "random_control_uses_same_accounting_but_random_centroids",
        ],
    }


def block_codebook_compress_state(
    state: dict[str, torch.Tensor],
    *,
    block_size: int,
    codebook_size: int,
    residual_fraction: float,
    seed: int,
) -> dict[str, Any]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive")
    return _compression_artifacts(
        state,
        block_size=block_size,
        codebook_size=codebook_size,
        residual_fraction=residual_fraction,
        seed=seed,
        include_states=False,
    )["compression"]


def block_codebook_reconstruct_state(
    state: dict[str, torch.Tensor],
    *,
    block_size: int,
    codebook_size: int,
    residual_fraction: float,
    seed: int,
) -> dict[str, Any]:
    """Return learned and random reconstructed state dicts plus compression accounting."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive")
    return _compression_artifacts(
        state,
        block_size=block_size,
        codebook_size=codebook_size,
        residual_fraction=residual_fraction,
        seed=seed,
        include_states=True,
    )


def evaluate_if3_block_codebook_checkpoint(
    checkpoint_path: Path,
    texts: list[str],
    split_name: str,
    *,
    device: str = "rocm",
    seed: int = 424242,
    batches: int = 512,
    block_size: int = 256,
    codebook_size: int = 256,
    residual_fraction: float = 0.01,
) -> dict[str, Any]:
    accelerator = _resolve_torch_accelerator(device)
    torch_device = accelerator.device
    payload = torch.load(checkpoint_path, map_location=torch_device)
    tokenizer = _tokenizer_from_payload(payload)
    tokens = _tokens_from_texts(texts, tokenizer)
    checkpoint_config = dict(payload["config"])
    checkpoint_config["validation_batches"] = batches
    config = DenseTrainingConfig(**checkpoint_config)
    if len(tokens) < config.seq_len + 2:
        raise ValueError(f"not enough tokens for IF3 {split_name} evaluation")

    reconstruction = block_codebook_reconstruct_state(
        payload["model"],
        block_size=block_size,
        codebook_size=codebook_size,
        residual_fraction=residual_fraction,
        seed=seed,
    )
    fp32_metrics = _evaluate_state_dict(
        payload["model"],
        tokenizer.vocab_size,
        tokens,
        config,
        torch_device,
        seed,
    )
    learned_metrics = _evaluate_state_dict(
        reconstruction["states"]["learned_block_codebook"],
        tokenizer.vocab_size,
        tokens,
        config,
        torch_device,
        seed,
    )
    random_metrics = _evaluate_state_dict(
        reconstruction["states"]["random_block_codebook"],
        tokenizer.vocab_size,
        tokens,
        config,
        torch_device,
        seed,
    )
    return {
        "benchmark_label": "if3_block_codebook_validation_probe",
        "candidate_id": "IF3",
        "checkpoint": {
            "path": str(checkpoint_path),
            "checkpoint_type": str(payload.get("checkpoint_type", "unknown")),
            "step": int(payload.get("step", 0)),
            "bytes": checkpoint_path.stat().st_size,
        },
        "split": split_name,
        "seed": seed,
        "model": {
            "architecture": "causal_transformer_decoder",
            "parameter_count": int(sum(value.numel() for value in payload["model"].values())),
            "config": {**checkpoint_config},
        },
        "tokenizer": tokenizer.to_jsonable(),
        "device": {
            "requested": accelerator.requested_device,
            "resolved": str(torch_device),
            "accelerator_backend": accelerator.backend,
            "rocm_available": accelerator.rocm_available,
            "rocm_runtime_version": accelerator.rocm_runtime_version,
        },
        "compression": reconstruction["compression"],
        "policies": {
            "fp32": fp32_metrics,
            "learned_block_codebook": learned_metrics,
            "random_block_codebook": random_metrics,
        },
        "comparisons": {
            "learned_loss_delta_vs_fp32": float(learned_metrics["loss"] - fp32_metrics["loss"]),
            "random_loss_delta_vs_fp32": float(random_metrics["loss"] - fp32_metrics["loss"]),
            "learned_loss_delta_vs_random": float(
                learned_metrics["loss"] - random_metrics["loss"]
            ),
            "learned_beats_random_loss": learned_metrics["loss"] < random_metrics["loss"],
            "learned_mse_beats_random_mse": reconstruction["compression"]["learned_codebook"][
                "mse"
            ]
            < reconstruction["compression"]["random_codebook_control"]["mse"],
        },
        "loss_evaluated": True,
        "packed_kernel_evaluated": False,
        "limitations": [
            "reconstructed_into_fp32_pytorch_tensors_before_evaluation",
            "no_packed_codebook_kernel_or_runtime_speed_measurement",
            "same_fixed_validation_batches_used_for_all_policies",
        ],
    }


def _evaluate_state_dict(
    state: dict[str, torch.Tensor],
    vocab_size: int,
    tokens: torch.Tensor,
    config: DenseTrainingConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    model = DenseDecoder(
        vocab_size,
        config.seq_len,
        config.hidden_dim,
        config.layers,
        config.heads,
        config.attention_mask_mode,
        config.architecture_variant,
        config.adapter_dim,
        config.block_impl,
    ).to(device)
    state_on_device = {key: value.to(device) for key, value in state.items()}
    _load_model_state(model, state_on_device)
    return _validation_metrics(
        model,
        tokens,
        vocab_size,
        config,
        torch.Generator(device="cpu").manual_seed(seed),
        device,
    )


def _compression_artifacts(
    state: dict[str, torch.Tensor],
    *,
    block_size: int,
    codebook_size: int,
    residual_fraction: float,
    seed: int,
    include_states: bool,
) -> dict[str, Any]:
    floating_items = {
        key: value.detach().cpu().float()
        for key, value in state.items()
        if torch.is_floating_point(value)
    }
    blocks, block_slices = _state_to_blocks(floating_items, block_size)
    if blocks.numel() == 0:
        raise ValueError("state has no floating point tensors")
    learned_centroids, learned_ids = _fit_codebook(
        blocks,
        codebook_size=min(codebook_size, blocks.shape[0]),
        seed=seed,
    )
    random_centroids, random_ids = _random_codebook(
        blocks,
        codebook_size=min(codebook_size, blocks.shape[0]),
        seed=seed,
    )
    learned_blocks = _reconstruct_blocks(
        blocks,
        learned_centroids,
        learned_ids,
        residual_fraction=residual_fraction,
    )
    random_blocks = _reconstruct_blocks(
        blocks,
        random_centroids,
        random_ids,
        residual_fraction=residual_fraction,
    )
    learned = _policy_summary(
        blocks,
        learned_centroids,
        learned_ids,
        block_slices=block_slices,
        block_size=block_size,
        residual_fraction=residual_fraction,
        policy_name="learned_block_codebook",
        reconstructed=learned_blocks,
    )
    random_control = _policy_summary(
        blocks,
        random_centroids,
        random_ids,
        block_slices=block_slices,
        block_size=block_size,
        residual_fraction=residual_fraction,
        policy_name="random_block_codebook",
        reconstructed=random_blocks,
    )
    learned["beats_random_control"] = learned["mse"] < random_control["mse"]
    random_control["beats_learned_codebook"] = random_control["mse"] < learned["mse"]
    compression = {
        "benchmark_label": "if3_block_codebook_state_compression",
        "policy": {
            "block_size": block_size,
            "codebook_size": codebook_size,
            "residual_fraction": residual_fraction,
            "seed": seed,
        },
        "state_sha256": _state_fingerprint(floating_items),
        "floating_tensor_count": len(floating_items),
        "floating_parameter_count": int(sum(value.numel() for value in floating_items.values())),
        "block_count": int(blocks.shape[0]),
        "padded_value_count": int(blocks.numel()),
        "learned_codebook": learned,
        "random_codebook_control": random_control,
        "limitations": [
            "counts_padded_block_values",
            "residuals_store_fp32_values_and_32bit_indexes",
            "runtime_buffer_bytes_counts_fp32_reconstructed_state",
        ],
    }
    result: dict[str, Any] = {"compression": compression}
    if include_states:
        result["states"] = {
            "learned_block_codebook": _blocks_to_state(
                state,
                floating_items,
                block_slices,
                learned_blocks,
            ),
            "random_block_codebook": _blocks_to_state(
                state,
                floating_items,
                block_slices,
                random_blocks,
            ),
        }
    return result


def _state_to_blocks(
    state: dict[str, torch.Tensor],
    block_size: int,
) -> tuple[torch.Tensor, list[tuple[str, int, int]]]:
    block_rows: list[torch.Tensor] = []
    block_slices: list[tuple[str, int, int]] = []
    for key in sorted(state):
        flat = state[key].reshape(-1)
        for start in range(0, flat.numel(), block_size):
            chunk = flat[start : start + block_size]
            if chunk.numel() < block_size:
                padded = torch.zeros(block_size, dtype=torch.float32)
                padded[: chunk.numel()] = chunk
                chunk = padded
            block_rows.append(chunk.float())
            block_slices.append((key, start, min(start + block_size, flat.numel())))
    return torch.stack(block_rows) if block_rows else torch.empty(0, block_size), block_slices


def _fit_codebook(
    blocks: torch.Tensor,
    *,
    codebook_size: int,
    seed: int,
    iterations: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    del seed
    if blocks.shape[0] <= codebook_size:
        centroids = blocks.clone()
        ids = torch.arange(blocks.shape[0], dtype=torch.long)
        return centroids, ids
    initial = _farthest_first_indexes(blocks, codebook_size)
    centroids = blocks[initial].clone()
    ids = torch.zeros(blocks.shape[0], dtype=torch.long)
    for _ in range(iterations):
        distances = torch.cdist(blocks, centroids)
        ids = torch.argmin(distances, dim=1)
        for index in range(codebook_size):
            selected = blocks[ids == index]
            if selected.numel():
                centroids[index] = selected.mean(dim=0)
    return centroids, ids


def _farthest_first_indexes(blocks: torch.Tensor, codebook_size: int) -> torch.Tensor:
    norms = torch.linalg.vector_norm(blocks, dim=1)
    selected = [int(torch.argmin(norms).item())]
    while len(selected) < codebook_size:
        distances = torch.cdist(blocks, blocks[torch.tensor(selected)])
        nearest = torch.min(distances, dim=1).values
        nearest[torch.tensor(selected)] = -1.0
        selected.append(int(torch.argmax(nearest).item()))
    return torch.tensor(selected, dtype=torch.long)


def _random_codebook(
    blocks: torch.Tensor,
    *,
    codebook_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed + 10_003)
    mean = blocks.mean()
    std = blocks.std(unbiased=False)
    if float(std.item()) == 0.0:
        std = torch.tensor(1.0, dtype=torch.float32)
    centroids = torch.randn(
        (codebook_size, blocks.shape[1]),
        generator=generator,
        dtype=torch.float32,
    )
    centroids = centroids * std + mean
    ids = torch.argmin(torch.cdist(blocks, centroids), dim=1)
    return centroids, ids


def _policy_summary(
    blocks: torch.Tensor,
    centroids: torch.Tensor,
    ids: torch.Tensor,
    *,
    block_slices: list[tuple[str, int, int]],
    block_size: int,
    residual_fraction: float,
    policy_name: str,
    reconstructed: torch.Tensor | None = None,
) -> dict[str, Any]:
    if reconstructed is None:
        reconstructed = _reconstruct_blocks(
            blocks,
            centroids,
            ids,
            residual_fraction=residual_fraction,
        )
    residual_values = _residual_count(blocks.numel(), residual_fraction)
    mse = float(torch.mean((blocks - reconstructed) ** 2).item())
    max_abs = float(torch.max((blocks - reconstructed).abs()).item())
    codebook_bytes = int(centroids.numel() * 4)
    id_bits = max(1, math.ceil(math.log2(max(centroids.shape[0], 2))))
    id_bytes = int(math.ceil(blocks.shape[0] * id_bits / 8))
    scale_bytes = int(blocks.shape[0] * 4)
    residual_bytes = int(residual_values * (4 + 4))
    metadata_bytes = int(id_bytes + scale_bytes + residual_values * 4)
    encoded_bytes = int(codebook_bytes + id_bytes + scale_bytes + residual_bytes)
    runtime_buffer_bytes = int(blocks.numel() * 4)
    return {
        "policy": policy_name,
        "mse": mse,
        "max_abs_error": max_abs,
        "codebook_bytes": codebook_bytes,
        "id_bytes": id_bytes,
        "scale_bytes": scale_bytes,
        "residual_bytes": residual_bytes,
        "metadata_bytes": metadata_bytes,
        "encoded_bytes": encoded_bytes,
        "runtime_buffer_bytes": runtime_buffer_bytes,
        "encoded_plus_runtime_bytes": encoded_bytes + runtime_buffer_bytes,
        "effective_encoded_bits_per_padded_value": float(
            encoded_bytes * 8 / max(blocks.numel(), 1)
        ),
        "residual_value_count": residual_values,
        "block_count": int(blocks.shape[0]),
        "block_reference_count": len(block_slices),
        "beats_random_control": False,
    }


def _reconstruct_blocks(
    blocks: torch.Tensor,
    centroids: torch.Tensor,
    ids: torch.Tensor,
    *,
    residual_fraction: float,
) -> torch.Tensor:
    reconstructed = centroids[ids].clone()
    residual_values = _residual_count(blocks.numel(), residual_fraction)
    if residual_values:
        residual = blocks - reconstructed
        indexes = torch.topk(residual.reshape(-1).abs(), k=residual_values).indices
        flat_reconstructed = reconstructed.reshape(-1)
        flat_blocks = blocks.reshape(-1)
        flat_reconstructed[indexes] = flat_blocks[indexes]
    return reconstructed


def _blocks_to_state(
    original_state: dict[str, torch.Tensor],
    floating_items: dict[str, torch.Tensor],
    block_slices: list[tuple[str, int, int]],
    reconstructed_blocks: torch.Tensor,
) -> dict[str, torch.Tensor]:
    reconstructed_flats = {
        key: torch.empty(value.numel(), dtype=torch.float32)
        for key, value in floating_items.items()
    }
    for block_index, (key, start, end) in enumerate(block_slices):
        reconstructed_flats[key][start:end] = reconstructed_blocks[block_index, : end - start]
    reconstructed_state = {
        key: value.detach().cpu().clone()
        for key, value in original_state.items()
        if not torch.is_floating_point(value)
    }
    for key, flat in reconstructed_flats.items():
        reconstructed_state[key] = flat.reshape(original_state[key].shape).to(
            dtype=original_state[key].dtype
        )
    return reconstructed_state


def _residual_count(numel: int, fraction: float) -> int:
    if fraction <= 0 or numel <= 0:
        return 0
    return max(1, int(numel * fraction))


def _state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
