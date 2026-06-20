from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from weightlab.importance import (
    _groupwise_quantize_matrix_rows,
    _to_bf16_like,
    _uniform_quantize_matrix,
)
from weightlab.metrics import mse, set_seed


def _load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(
        payload["state_dict"],
        dict,
    ):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint did not contain a state dict: {checkpoint_path}")
    return {str(key): value for key, value in payload.items() if torch.is_tensor(value)}


def _select_matrix(
    state_dict: dict[str, torch.Tensor],
    tensor_name: str | None,
    min_rows: int = 4,
    min_cols: int = 4,
) -> tuple[str, np.ndarray]:
    if tensor_name is not None:
        if tensor_name not in state_dict:
            raise KeyError(f"tensor {tensor_name!r} not found in checkpoint")
        tensor = state_dict[tensor_name]
        if tensor.ndim != 2 or not torch.is_floating_point(tensor):
            raise ValueError(f"tensor {tensor_name!r} is not a 2D floating matrix")
        return tensor_name, tensor.detach().cpu().to(torch.float32).numpy()

    candidates: list[tuple[str, torch.Tensor]] = []
    for name, tensor in state_dict.items():
        if (
            tensor.ndim == 2
            and torch.is_floating_point(tensor)
            and tensor.shape[0] >= min_rows
            and tensor.shape[1] >= min_cols
        ):
            candidates.append((name, tensor))
    if not candidates:
        raise ValueError("checkpoint did not contain a suitable 2D floating matrix")
    candidates.sort(key=lambda item: (item[1].numel(), item[0]), reverse=True)
    name, tensor = candidates[0]
    return name, tensor.detach().cpu().to(torch.float32).numpy()


def _row_error_scores(
    matrix: np.ndarray,
    groupwise_matrix: np.ndarray,
) -> np.ndarray:
    row_squared_error = np.sum(
        (matrix.astype(np.float64) - groupwise_matrix.astype(np.float64)) ** 2,
        axis=1,
    )
    total_squared_error = float(np.sum(row_squared_error))
    restored_squared_error = total_squared_error - row_squared_error
    return restored_squared_error / float(matrix.size)


def evaluate_real_model_matrix_precision(
    checkpoint_path: Path | str,
    tensor_name: str | None = None,
    model_id: str = "unknown",
    model_commit: str = "unknown",
    seed: int = 0,
    protected_count: int = 8,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    state_dict = _load_state_dict(path)
    selected_name, matrix = _select_matrix(state_dict, tensor_name=tensor_name)
    protected_count = min(protected_count, matrix.shape[0])

    uniform_matrix, uniform_bytes = _uniform_quantize_matrix(matrix, bits=4)
    groupwise_matrix, groupwise_bytes = _groupwise_quantize_matrix_rows(matrix, bits=4)
    row_scores = _row_error_scores(matrix, groupwise_matrix)
    selected_rows = set(np.argsort(row_scores)[:protected_count])

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

    def policy_metrics(candidate: np.ndarray, total_bytes: int) -> dict[str, float]:
        return {
            "matrix_mse": mse(matrix, candidate),
            "total_bytes": float(total_bytes),
        }

    bf16_bytes = groupwise_bytes + len(selected_rows) * (matrix.shape[1] * 2 + 4)
    fp32_bytes = groupwise_bytes + len(selected_rows) * (matrix.shape[1] * 4 + 4)

    rng = set_seed(seed)
    random_bf16_errors = []
    random_fp32_errors = []
    random_trials = max(16, protected_count * 4)
    for _ in range(random_trials):
        random_rows = set(
            rng.choice(np.arange(matrix.shape[0]), size=protected_count, replace=False)
        )
        random_bf16_errors.append(mse(matrix, protected_matrix(random_rows, mode="bf16")))
        random_fp32_errors.append(mse(matrix, protected_matrix(random_rows, mode="fp32")))

    policies = {
        "full_fp32": {
            "matrix_mse": 0.0,
            "total_bytes": float(matrix.size * 4),
        },
        "uniform_int4": policy_metrics(uniform_matrix, uniform_bytes),
        "groupwise_int4": policy_metrics(groupwise_matrix, groupwise_bytes),
        "output_error_bf16_protected": policy_metrics(
            protected_matrix(selected_rows, mode="bf16"),
            bf16_bytes,
        ),
        "output_error_fp32_protected": policy_metrics(
            protected_matrix(selected_rows, mode="fp32"),
            fp32_bytes,
        ),
        "random_bf16_protected_mean": {
            "matrix_mse": float(np.mean(random_bf16_errors)),
            "total_bytes": float(bf16_bytes),
        },
        "random_bf16_protected_best": {
            "matrix_mse": float(np.min(random_bf16_errors)),
            "total_bytes": float(bf16_bytes),
        },
        "random_fp32_protected_mean": {
            "matrix_mse": float(np.mean(random_fp32_errors)),
            "total_bytes": float(fp32_bytes),
        },
        "random_fp32_protected_best": {
            "matrix_mse": float(np.min(random_fp32_errors)),
            "total_bytes": float(fp32_bytes),
        },
    }

    return {
        "source": {
            "checkpoint_path": str(path),
            "model_id": model_id,
            "model_commit": model_commit,
        },
        "tensor": {
            "name": selected_name,
            "shape": list(matrix.shape),
            "dtype": "float32",
            "parameter_count": int(matrix.size),
        },
        "protected_count": protected_count,
        "selected_rows": sorted(int(row) for row in selected_rows),
        "row_output_error_scores": row_scores.tolist(),
        "random_trials": random_trials,
        "precision_policies": policies,
        "notes": (
            "Local checkpoint matrix reconstruction experiment. This uses a real open-model "
            "weight tensor, but it measures tensor reconstruction error rather than task loss."
        ),
    }
