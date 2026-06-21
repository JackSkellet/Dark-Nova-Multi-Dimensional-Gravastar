from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import torch


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
    learned = _policy_summary(
        blocks,
        learned_centroids,
        learned_ids,
        block_slices=block_slices,
        block_size=block_size,
        residual_fraction=residual_fraction,
        policy_name="learned_block_codebook",
    )
    random_control = _policy_summary(
        blocks,
        random_centroids,
        random_ids,
        block_slices=block_slices,
        block_size=block_size,
        residual_fraction=residual_fraction,
        policy_name="random_block_codebook",
    )
    learned["beats_random_control"] = learned["mse"] < random_control["mse"]
    random_control["beats_learned_codebook"] = random_control["mse"] < learned["mse"]
    return {
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
) -> dict[str, Any]:
    reconstructed = centroids[ids].clone()
    residual = blocks - reconstructed
    residual_values = _residual_count(blocks.numel(), residual_fraction)
    if residual_values:
        indexes = torch.topk(residual.reshape(-1).abs(), k=residual_values).indices
        flat_reconstructed = reconstructed.reshape(-1)
        flat_blocks = blocks.reshape(-1)
        flat_reconstructed[indexes] = flat_blocks[indexes]
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
