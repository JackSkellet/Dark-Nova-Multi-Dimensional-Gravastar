from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from weightlab.metrics import ceil_div, mse


def _uniform_quantize(matrix: np.ndarray, bits: int) -> tuple[np.ndarray, int]:
    qmax = (2 ** (bits - 1)) - 1
    scale = float(np.max(np.abs(matrix)) / max(qmax, 1)) or 1.0
    quantized = np.clip(np.round(matrix / scale), -qmax - 1, qmax).astype(np.int8)
    reconstructed = quantized.astype(np.float32) * scale
    packed_value_bytes = ceil_div(matrix.size * bits, 8)
    metadata_bytes = 8 + 4
    return reconstructed, packed_value_bytes + metadata_bytes


def _low_rank(matrix: np.ndarray, rank: int) -> tuple[np.ndarray, int]:
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    rank = min(rank, len(s))
    reconstructed = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
    value_bytes = (u.shape[0] * rank + rank + rank * vt.shape[1]) * 4
    metadata_bytes = 12
    return reconstructed.astype(np.float32), value_bytes + metadata_bytes


def _low_rank_sparse_residual(
    matrix: np.ndarray, rank: int, residual_fraction: float = 0.1
) -> tuple[np.ndarray, int]:
    base, base_bytes = _low_rank(matrix, rank)
    residual = matrix - base
    keep = max(1, int(matrix.size * residual_fraction))
    flat = np.abs(residual).ravel()
    threshold_index = np.argpartition(flat, -keep)[-keep:]
    sparse = np.zeros(matrix.size, dtype=np.float32)
    sparse[threshold_index] = residual.ravel()[threshold_index]
    reconstructed = base + sparse.reshape(matrix.shape)
    sparse_bytes = keep * (4 + 4) + 8
    return reconstructed.astype(np.float32), base_bytes + sparse_bytes


def _rank1_outer_product(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    col = matrix[:, :1].astype(np.float32)
    row = matrix[:1, :].astype(np.float32)
    denom = float(matrix[0, 0]) if matrix[0, 0] != 0 else 1.0
    reconstructed = (col @ row) / denom
    return reconstructed.astype(np.float32), (col.size + row.size + 1) * 4


def _shared_basis_coefficients(matrix: np.ndarray, rank: int) -> tuple[np.ndarray, int]:
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    basis = vt[:rank, :]
    coeffs = u[:, :rank] * s[:rank]
    reconstructed = coeffs @ basis
    bytes_total = (basis.size + coeffs.size) * 4 + 12
    return reconstructed.astype(np.float32), bytes_total


def _tensorized_blocks(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = matrix.shape
    h2 = max(1, h // 2)
    top = matrix[:h2, :]
    bottom = matrix[h2:, :]
    top_mean = np.mean(top, axis=0, keepdims=True)
    bottom_mean = np.mean(bottom, axis=0, keepdims=True) if len(bottom) else top_mean
    reconstructed = np.vstack(
        [np.repeat(top_mean, len(top), axis=0), np.repeat(bottom_mean, len(bottom), axis=0)]
    )
    bytes_total = (top_mean.size + bottom_mean.size) * 4 + 16
    return reconstructed.astype(np.float32), bytes_total


def _product_quantized_rows(
    matrix: np.ndarray,
    subspaces: int = 4,
    codebook_size: int = 16,
    iterations: int = 8,
) -> tuple[np.ndarray, int]:
    rows, cols = matrix.shape
    subspaces = max(1, min(subspaces, cols))
    column_splits = np.array_split(np.arange(cols), subspaces)
    reconstructed = np.zeros_like(matrix, dtype=np.float32)
    codebook_value_count = 0
    assignment_count = 0

    for columns in column_splits:
        block = matrix[:, columns].astype(np.float32)
        centroids_count = min(codebook_size, rows)
        centroids = block[
            np.linspace(0, rows - 1, centroids_count, dtype=np.int64)
        ].copy()
        assignments = np.zeros(rows, dtype=np.int64)
        for _ in range(iterations):
            distances = np.sum((block[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            assignments = np.argmin(distances, axis=1)
            for idx in range(centroids_count):
                members = block[assignments == idx]
                if len(members):
                    centroids[idx] = np.mean(members, axis=0)
        reconstructed[:, columns] = centroids[assignments]
        codebook_value_count += centroids.size
        assignment_count += rows

    codebook_bytes = codebook_value_count * 4
    assignment_bits = max(1, math.ceil(math.log2(codebook_size)))
    assignment_bytes = ceil_div(assignment_count * assignment_bits, 8)
    metadata_bytes = 32 + len(column_splits) * 8
    return reconstructed.astype(np.float32), codebook_bytes + assignment_bytes + metadata_bytes


def _kronecker_rank1(
    matrix: np.ndarray,
    block_shape: tuple[int, int] = (8, 8),
) -> tuple[np.ndarray, int]:
    rows, cols = matrix.shape
    block_rows, block_cols = block_shape
    if rows % block_rows != 0 or cols % block_cols != 0:
        raise ValueError("matrix shape must be divisible by block_shape")
    outer_rows = rows // block_rows
    outer_cols = cols // block_cols

    rearranged = (
        matrix.reshape(outer_rows, block_rows, outer_cols, block_cols)
        .transpose(0, 2, 1, 3)
        .reshape(outer_rows * outer_cols, block_rows * block_cols)
    )
    u, s, vt = np.linalg.svd(rearranged, full_matrices=False)
    a = (u[:, 0] * math.sqrt(float(s[0]))).reshape(outer_rows, outer_cols)
    b = (vt[0, :] * math.sqrt(float(s[0]))).reshape(block_rows, block_cols)
    reconstructed = np.kron(a, b)
    value_bytes = (a.size + b.size) * 4
    metadata_bytes = 32
    return reconstructed.astype(np.float32), value_bytes + metadata_bytes


def _factor_pair(n: int) -> tuple[int, int]:
    for factor in range(int(math.sqrt(n)), 0, -1):
        if n % factor == 0:
            return factor, n // factor
    return 1, n


def _tensor_train_4d(matrix: np.ndarray, rank: int = 4) -> tuple[np.ndarray, int]:
    rows, cols = matrix.shape
    row_dims = _factor_pair(rows)
    col_dims = _factor_pair(cols)
    dims = [row_dims[0], row_dims[1], col_dims[0], col_dims[1]]
    tensor = matrix.reshape(dims).astype(np.float32)
    ranks = [1]
    cores: list[np.ndarray] = []
    unfolding = tensor

    for dim in dims[:-1]:
        left_rank = ranks[-1]
        unfolding = unfolding.reshape(left_rank * dim, -1)
        u, s, vt = np.linalg.svd(unfolding, full_matrices=False)
        next_rank = min(rank, len(s))
        cores.append(u[:, :next_rank].reshape(left_rank, dim, next_rank).astype(np.float32))
        ranks.append(next_rank)
        unfolding = (s[:next_rank, None] * vt[:next_rank, :]).astype(np.float32)

    cores.append(unfolding.reshape(ranks[-1], dims[-1], 1).astype(np.float32))
    reconstructed = cores[0]
    for core in cores[1:]:
        reconstructed = np.tensordot(reconstructed, core, axes=([-1], [0]))
    reconstructed = reconstructed.reshape(matrix.shape)
    value_bytes = sum(core.size for core in cores) * 4
    metadata_bytes = 48 + len(cores) * 12
    return reconstructed.astype(np.float32), value_bytes + metadata_bytes


def compare_compression_methods(matrix: np.ndarray, rank: int = 4) -> list[dict[str, Any]]:
    matrix = np.asarray(matrix, dtype=np.float32)
    methods = {
        "fp32": lambda: (matrix.copy(), matrix.nbytes, 0),
        "fp16": lambda: (matrix.astype(np.float16).astype(np.float32), matrix.size * 2, 0),
        "int8_uniform": lambda: (*_uniform_quantize(matrix, 8), 12),
        "int4_uniform": lambda: (*_uniform_quantize(matrix, 4), 12),
        "low_rank_svd": lambda: (*_low_rank(matrix, rank), 12),
        "low_rank_sparse_residual": lambda: (*_low_rank_sparse_residual(matrix, rank), 20),
        "rank1_outer_product": lambda: (*_rank1_outer_product(matrix), 4),
        "shared_basis_coefficients": lambda: (*_shared_basis_coefficients(matrix, rank), 12),
        "tensorized_two_block": lambda: (*_tensorized_blocks(matrix), 16),
        "product_quantized_rows": lambda: (*_product_quantized_rows(matrix), 64),
        "kronecker_rank1": lambda: (*_kronecker_rank1(matrix), 32),
        "tensor_train_4d": lambda: (*_tensor_train_4d(matrix, rank=rank), 96),
    }
    rows: list[dict[str, Any]] = []
    for name, fn in methods.items():
        start = time.perf_counter()
        reconstructed, total_bytes, metadata_bytes = fn()
        encode_decode_ms = (time.perf_counter() - start) * 1000.0
        rows.append(
            {
                "method": name,
                "total_bytes": int(math.ceil(total_bytes)),
                "metadata_bytes": int(metadata_bytes),
                "reconstruction_mse": mse(matrix, reconstructed),
                "encode_decode_ms": encode_decode_ms,
                "effective_bits_per_parameter": float(total_bytes * 8 / matrix.size),
            }
        )
    return rows
