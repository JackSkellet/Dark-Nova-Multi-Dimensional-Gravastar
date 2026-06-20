from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from weightlab.metrics import p50_p95_ms, set_seed


@dataclass(frozen=True)
class TorchAccelerator:
    device: torch.device
    backend: str
    cuda_available: bool
    rocm_available: bool
    rocm_runtime_version: str
    requested_device: str


def _normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9
    return x / denom


def _latency_samples(fn, queries: np.ndarray) -> tuple[list[int], list[float]]:
    predictions: list[int] = []
    times: list[float] = []
    for query in queries:
        start = time.perf_counter()
        predictions.append(int(fn(query)))
        times.append(time.perf_counter() - start)
    return predictions, times


def benchmark_lookup_methods(
    seed: int = 0,
    bank_sizes: list[int] | None = None,
    n_queries: int = 256,
    dim: int = 32,
) -> dict[str, dict[str, dict[str, float]]]:
    if bank_sizes is None:
        bank_sizes = [16, 128, 1024]
    rng = set_seed(seed)
    output: dict[str, dict[str, dict[str, float]]] = {}

    for bank_size in bank_sizes:
        bank = _normalize(rng.normal(size=(bank_size, dim)).astype(np.float32))
        true_ids = rng.integers(0, bank_size, size=n_queries // 2)
        true_ids = np.concatenate([true_ids, true_ids[: n_queries - len(true_ids)]])
        queries = _normalize(bank[true_ids] + rng.normal(scale=0.03, size=(len(true_ids), dim)))
        exact_truth = np.argmax(queries @ bank.T, axis=1)

        centroid_count = max(2, int(np.sqrt(bank_size)))
        assignments = np.arange(bank_size) % centroid_count
        centroids = np.vstack([bank[assignments == i].mean(axis=0) for i in range(centroid_count)])
        centroids = _normalize(centroids)
        projections = rng.normal(size=(dim, 8)).astype(np.float32)
        bank_hashes = ((bank @ projections) > 0).astype(np.uint8)

        cache: dict[str, int] = {}
        cache_hits = {"count": 0}

        def dense_topk(query: np.ndarray, bank: np.ndarray = bank) -> int:
            return int(np.argmax(query @ bank.T))

        def exact_vector(query: np.ndarray, bank: np.ndarray = bank) -> int:
            scores = np.einsum("d,nd->n", query, bank)
            return int(np.argmax(scores))

        def tree_centroid(
            query: np.ndarray,
            bank: np.ndarray = bank,
            centroids: np.ndarray = centroids,
            assignments: np.ndarray = assignments,
        ) -> int:
            centroid = int(np.argmax(query @ centroids.T))
            members = np.where(assignments == centroid)[0]
            return int(members[np.argmax(query @ bank[members].T)])

        def hash_routing(
            query: np.ndarray,
            bank: np.ndarray = bank,
            projections: np.ndarray = projections,
            bank_hashes: np.ndarray = bank_hashes,
        ) -> int:
            qhash = ((query @ projections) > 0).astype(np.uint8)
            distances = np.count_nonzero(bank_hashes != qhash, axis=1)
            candidates = np.where(distances == distances.min())[0]
            return int(candidates[np.argmax(query @ bank[candidates].T)])

        def cached_previous(
            query: np.ndarray,
            cache: dict[str, int] = cache,
            cache_hits: dict[str, int] = cache_hits,
        ) -> int:
            key = hashlib.sha1(np.round(query, 2).tobytes()).hexdigest()
            if key in cache:
                cache_hits["count"] += 1
                return cache[key]
            value = exact_vector(query)
            cache[key] = value
            return value

        methods = {
            "dense_topk": dense_topk,
            "exact_vector": exact_vector,
            "tree_centroid": tree_centroid,
            "hash_routing": hash_routing,
            "cached_previous": cached_previous,
        }
        output[str(bank_size)] = {}
        for name, fn in methods.items():
            cache_hits["count"] = 0
            preds, times = _latency_samples(fn, queries)
            recall = float(np.mean(np.asarray(preds) == exact_truth))
            metrics: dict[str, Any] = {
                "recall_at_1": recall,
                "index_memory_bytes": float(
                    bank.nbytes
                    if name in {"dense_topk", "exact_vector", "cached_previous"}
                    else centroids.nbytes
                    if name == "tree_centroid"
                    else bank_hashes.nbytes + projections.nbytes
                ),
                **p50_p95_ms(times),
            }
            if name == "cached_previous":
                metrics["cache_hit_rate"] = cache_hits["count"] / len(queries)
            else:
                metrics["cache_hit_rate"] = 0.0
            output[str(bank_size)][name] = metrics
    return output


def _make_repeated_queries(
    rng: np.random.Generator,
    bank: np.ndarray,
    n_queries: int,
    repeat_fraction: float,
    candidate_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    source_ids = np.arange(len(bank)) if candidate_ids is None else candidate_ids
    unique_count = max(1, int(n_queries * (1.0 - repeat_fraction)))
    unique_ids = rng.choice(source_ids, size=unique_count, replace=True)
    unique_queries = _normalize(
        bank[unique_ids] + rng.normal(scale=0.025, size=(unique_count, bank.shape[1]))
    )
    repeated_ids = np.resize(unique_ids, n_queries)
    repeated_query_indices = np.resize(np.arange(unique_count), n_queries)
    order = np.arange(n_queries)
    rng.shuffle(order)
    repeated_ids = repeated_ids[order]
    repeated_query_indices = repeated_query_indices[order]
    return unique_queries[repeated_query_indices].astype(np.float32), repeated_ids


def _summarize_stage(samples: list[dict[str, float]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in [
        "authorization",
        "lookup",
        "cache",
        "transfer",
        "reconstruction",
        "dispatch",
        "end_to_end",
    ]:
        values = [sample[key] for sample in samples]
        prefix = f"stage_{key}" if key != "end_to_end" else "end_to_end"
        output[f"{prefix}_ms_p50"] = float(np.percentile(values, 50) * 1000.0)
        output[f"{prefix}_ms_p95"] = float(np.percentile(values, 95) * 1000.0)
    return output


def _resolve_torch_accelerator(device: str | None = None) -> TorchAccelerator:
    requested = (device or "auto").lower()
    cuda_available = torch.cuda.is_available()
    rocm_available = bool(cuda_available and getattr(torch.version, "hip", None))

    if requested in {"rocm", "hip"}:
        if rocm_available:
            return TorchAccelerator(
                device=torch.device("cuda"),
                backend="rocm",
                cuda_available=cuda_available,
                rocm_available=rocm_available,
                rocm_runtime_version=str(getattr(torch.version, "hip", "") or ""),
                requested_device=requested,
            )
        return TorchAccelerator(
            device=torch.device("cpu"),
            backend="cpu",
            cuda_available=cuda_available,
            rocm_available=rocm_available,
            rocm_runtime_version=str(getattr(torch.version, "hip", "") or ""),
            requested_device=requested,
        )

    if requested == "cpu":
        return TorchAccelerator(
            device=torch.device("cpu"),
            backend="cpu",
            cuda_available=cuda_available,
            rocm_available=rocm_available,
            rocm_runtime_version=str(getattr(torch.version, "hip", "") or ""),
            requested_device=requested,
        )

    if requested == "cuda":
        if cuda_available:
            return TorchAccelerator(
                device=torch.device("cuda"),
                backend="rocm" if rocm_available else "cuda",
                cuda_available=cuda_available,
                rocm_available=rocm_available,
                rocm_runtime_version=str(getattr(torch.version, "hip", "") or ""),
                requested_device=requested,
            )
        return TorchAccelerator(
            device=torch.device("cpu"),
            backend="cpu",
            cuda_available=cuda_available,
            rocm_available=rocm_available,
            rocm_runtime_version=str(getattr(torch.version, "hip", "") or ""),
            requested_device=requested,
        )

    if requested != "auto":
        candidate = torch.device(requested)
        return TorchAccelerator(
            device=candidate,
            backend=candidate.type,
            cuda_available=cuda_available,
            rocm_available=rocm_available,
            rocm_runtime_version=str(getattr(torch.version, "hip", "") or ""),
            requested_device=requested,
        )

    if cuda_available:
        return TorchAccelerator(
            device=torch.device("cuda"),
            backend="rocm" if rocm_available else "cuda",
            cuda_available=cuda_available,
            rocm_available=rocm_available,
            rocm_runtime_version=str(getattr(torch.version, "hip", "") or ""),
            requested_device=requested,
        )
    return TorchAccelerator(
        device=torch.device("cpu"),
        backend="cpu",
        cuda_available=cuda_available,
        rocm_available=rocm_available,
        rocm_runtime_version=str(getattr(torch.version, "hip", "") or ""),
        requested_device=requested,
    )


def _sync_if_accelerator(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _p50_p95(values: list[float]) -> tuple[float, float]:
    return float(np.percentile(values, 50)), float(np.percentile(values, 95))


def benchmark_rocm_transfer_scaling(
    payload_bytes: list[int] | None = None,
    iterations: int = 8,
    warmup_iterations: int = 2,
    device: str | None = "rocm",
) -> dict[str, dict[str, dict[str, float | str]]]:
    if payload_bytes is None:
        payload_bytes = [1 << 20, 8 << 20, 32 << 20]
    accelerator = _resolve_torch_accelerator(device)
    torch_device = accelerator.device
    results: dict[str, dict[str, dict[str, float | str]]] = {}

    for payload in payload_bytes:
        element_count = max(1, int(payload) // 4)
        source = torch.linspace(0.0, 1.0, element_count, dtype=torch.float32)

        h2d_samples: list[float] = []
        dispatch_samples: list[float] = []
        d2h_samples: list[float] = []

        for iteration in range(warmup_iterations + iterations):
            start = time.perf_counter()
            device_tensor = source.clone().to(torch_device)
            _sync_if_accelerator(torch_device)
            h2d_s = time.perf_counter() - start

            start = time.perf_counter()
            dispatched = device_tensor.mul(1.0001).add_(0.125)
            _sync_if_accelerator(torch_device)
            dispatch_s = time.perf_counter() - start

            start = time.perf_counter()
            host_tensor = dispatched.cpu().clone()
            _sync_if_accelerator(torch_device)
            d2h_s = time.perf_counter() - start

            if float(host_tensor[0]) < 0.0:
                raise AssertionError("unreachable guard to keep transfer result live")
            if iteration >= warmup_iterations:
                h2d_samples.append(h2d_s)
                dispatch_samples.append(dispatch_s)
                d2h_samples.append(d2h_s)

        h2d_p50, h2d_p95 = _p50_p95(h2d_samples)
        dispatch_p50, dispatch_p95 = _p50_p95(dispatch_samples)
        d2h_p50, d2h_p95 = _p50_p95(d2h_samples)
        metrics = {
            "requested_device": accelerator.requested_device,
            "device": torch_device.type,
            "accelerator_backend": accelerator.backend,
            "cuda_available": float(accelerator.cuda_available),
            "torch_cuda_api_available": float(accelerator.cuda_available),
            "rocm_available": float(accelerator.rocm_available),
            "rocm_runtime_version": accelerator.rocm_runtime_version,
            "payload_bytes": float(payload),
            "elements": float(element_count),
            "iterations": float(iterations),
            "warmup_iterations": float(warmup_iterations),
            "host_to_device_ms_p50": h2d_p50 * 1000.0,
            "host_to_device_ms_p95": h2d_p95 * 1000.0,
            "device_dispatch_ms_p50": dispatch_p50 * 1000.0,
            "device_dispatch_ms_p95": dispatch_p95 * 1000.0,
            "device_to_host_ms_p50": d2h_p50 * 1000.0,
            "device_to_host_ms_p95": d2h_p95 * 1000.0,
            "host_to_device_bandwidth_gb_s_p50": float(
                payload / max(h2d_p50, 1e-12) / 1e9
            ),
            "device_to_host_bandwidth_gb_s_p50": float(
                payload / max(d2h_p50, 1e-12) / 1e9
            ),
            "uses_accelerator_transfer": float(torch_device.type == "cuda"),
            "uses_rocm_transfer": float(accelerator.backend == "rocm"),
            "uses_cuda_transfer": float(accelerator.backend == "cuda"),
            "measures_occupancy": 0.0,
            "measures_kernel_fusion": 0.0,
            "measures_power": 0.0,
        }
        results[str(payload)] = {"rocm_transfer_scaling": metrics}
    return results


def benchmark_torch_batched_routed_execution(
    seed: int = 0,
    bank_sizes: list[int] | None = None,
    n_queries: int = 256,
    dim: int = 32,
    component_dim: int = 64,
    batch_size: int = 32,
    device: str | None = None,
) -> dict[str, dict[str, dict[str, float | str]]]:
    if bank_sizes is None:
        bank_sizes = [512, 4096]
    rng = set_seed(seed)
    accelerator = _resolve_torch_accelerator(device)
    torch_device = accelerator.device

    results: dict[str, dict[str, dict[str, float | str]]] = {}
    for bank_size in bank_sizes:
        bank_np = _normalize(rng.normal(size=(bank_size, dim)).astype(np.float32))
        authorized_np = np.zeros(bank_size, dtype=bool)
        authorized_np[: max(1, int(bank_size * 0.8))] = True
        authorized_ids = np.flatnonzero(authorized_np)
        true_ids = rng.choice(authorized_ids, size=n_queries, replace=True)
        queries_np = _normalize(
            bank_np[true_ids] + rng.normal(scale=0.02, size=(n_queries, dim))
        ).astype(np.float32)

        bank = torch.from_numpy(bank_np).to(torch_device)
        queries_cpu = torch.from_numpy(queries_np)
        authorized = torch.from_numpy(authorized_np).to(torch_device)
        components_cpu = torch.from_numpy(
            rng.normal(size=(bank_size, component_dim, component_dim)).astype(np.float32)
        )
        codebook = torch.from_numpy(
            rng.normal(size=(16, component_dim)).astype(np.float32)
        ).to(torch_device)
        truth = torch.argmax(
            torch.where(
                authorized[None, :],
                torch.from_numpy(queries_np).to(torch_device) @ bank.T,
                torch.full((n_queries, bank_size), -torch.inf, device=torch_device),
            ),
            dim=1,
        ).cpu()

        stage_samples: list[dict[str, float]] = []
        predictions: list[torch.Tensor] = []
        total_start = time.perf_counter()
        for start_idx in range(0, n_queries, batch_size):
            batch_queries_cpu = queries_cpu[start_idx : start_idx + batch_size]
            batch_start = time.perf_counter()

            start = time.perf_counter()
            batch_queries = batch_queries_cpu.to(torch_device)
            _sync_if_accelerator(torch_device)
            transfer_query_s = time.perf_counter() - start

            start = time.perf_counter()
            scores = batch_queries @ bank.T
            scores = torch.where(
                authorized[None, :],
                scores,
                torch.full_like(scores, -torch.inf),
            )
            _sync_if_accelerator(torch_device)
            authorization_lookup_s = time.perf_counter() - start

            start = time.perf_counter()
            selected = torch.argmax(scores, dim=1)
            _sync_if_accelerator(torch_device)
            lookup_s = time.perf_counter() - start

            start = time.perf_counter()
            selected_components = components_cpu[selected.cpu()].to(torch_device)
            _sync_if_accelerator(torch_device)
            transfer_component_s = time.perf_counter() - start

            start = time.perf_counter()
            reconstructed = selected_components + 0.001 * codebook[
                torch.remainder(selected, codebook.shape[0])
            ][:, None, :]
            _sync_if_accelerator(torch_device)
            reconstruction_s = time.perf_counter() - start

            start = time.perf_counter()
            usable_dim = min(dim, component_dim)
            dispatch_input = batch_queries[:, :usable_dim].unsqueeze(1)
            _ = torch.bmm(dispatch_input, reconstructed[:, :usable_dim, :]).squeeze(1)
            _sync_if_accelerator(torch_device)
            dispatch_s = time.perf_counter() - start

            predictions.append(selected.cpu())
            stage_samples.append(
                {
                    "authorization": authorization_lookup_s,
                    "lookup": lookup_s,
                    "cache": 0.0,
                    "transfer": transfer_query_s + transfer_component_s,
                    "reconstruction": reconstruction_s,
                    "dispatch": dispatch_s,
                    "end_to_end": time.perf_counter() - batch_start,
                }
            )
        total_elapsed = time.perf_counter() - total_start

        predictions_tensor = torch.cat(predictions)
        metrics = _summarize_stage(stage_samples)
        metrics.update(
            {
                "device": torch_device.type,
                "accelerator_backend": accelerator.backend,
                "requested_device": accelerator.requested_device,
                "cuda_available": float(accelerator.cuda_available),
                "torch_cuda_api_available": float(accelerator.cuda_available),
                "rocm_available": float(accelerator.rocm_available),
                "rocm_runtime_version": accelerator.rocm_runtime_version,
                "bank_size": float(bank_size),
                "batch_size": float(batch_size),
                "n_queries": float(n_queries),
                "recall_at_1": float(torch.mean((predictions_tensor == truth).float()).item()),
                "authorized_candidates": float(np.count_nonzero(authorized_np)),
                "component_bytes_per_query": float(components_cpu[0].numel() * 4),
                "index_memory_bytes": float(bank.numel() * 4 + authorized.numel()),
                "throughput_queries_per_s": float(n_queries / max(total_elapsed, 1e-9)),
                "uses_real_torch_dispatch": 1.0,
                "uses_accelerator_transfer": float(torch_device.type == "cuda"),
                "uses_cuda_transfer": float(accelerator.backend == "cuda"),
                "uses_rocm_transfer": float(accelerator.backend == "rocm"),
            }
        )
        results[str(bank_size)] = {"torch_exact_batched": metrics}
    return results


def benchmark_routed_execution(
    seed: int = 0,
    bank_sizes: list[int] | None = None,
    n_queries: int = 512,
    dim: int = 32,
    component_dim: int = 64,
    repeat_fraction: float = 0.35,
) -> dict[str, dict[str, dict[str, float]]]:
    if bank_sizes is None:
        bank_sizes = [128, 2048, 8192]
    rng = set_seed(seed)
    results: dict[str, dict[str, dict[str, float]]] = {}

    for bank_size in bank_sizes:
        bank = _normalize(rng.normal(size=(bank_size, dim)).astype(np.float32))
        components = rng.normal(size=(bank_size, component_dim, component_dim)).astype(np.float32)
        codebook = rng.normal(size=(16, component_dim)).astype(np.float32)
        authorized = np.zeros(bank_size, dtype=bool)
        authorized[: max(1, int(bank_size * 0.8))] = True
        queries, true_ids = _make_repeated_queries(
            rng,
            bank,
            n_queries,
            repeat_fraction,
            candidate_ids=np.flatnonzero(authorized),
        )
        first_authorized = int(np.flatnonzero(authorized)[0])
        true_ids = np.asarray(
            [idx if authorized[idx] else first_authorized for idx in true_ids],
            dtype=np.int64,
        )
        projections = rng.normal(size=(dim, 12)).astype(np.float32)
        bank_hashes = ((bank @ projections) > 0).astype(np.uint8)

        authorized_bank = bank[authorized]
        authorized_indices = np.flatnonzero(authorized)

        def dense_router(
            query: np.ndarray,
            bank_size: int = bank_size,
            bank: np.ndarray = bank,
            authorized: np.ndarray = authorized,
        ) -> int:
            scores = np.full(bank_size, -np.inf, dtype=np.float32)
            scores[authorized] = query @ bank[authorized].T
            return int(np.argmax(scores))

        def exact_index(
            query: np.ndarray,
            authorized_indices: np.ndarray = authorized_indices,
            authorized_bank: np.ndarray = authorized_bank,
        ) -> int:
            return int(authorized_indices[np.argmax(query @ authorized_bank.T)])

        def hash_index(
            query: np.ndarray,
            bank: np.ndarray = bank,
            bank_hashes: np.ndarray = bank_hashes,
            authorized: np.ndarray = authorized,
            authorized_indices: np.ndarray = authorized_indices,
            projections: np.ndarray = projections,
        ) -> int:
            qhash = ((query @ projections) > 0).astype(np.uint8)
            distances = np.count_nonzero(bank_hashes[authorized] != qhash, axis=1)
            candidates = authorized_indices[np.where(distances == distances.min())[0]]
            return int(candidates[np.argmax(query @ bank[candidates].T)])

        methods = {
            "dense_router": dense_router,
            "exact_index": exact_index,
            "cached_exact_index": exact_index,
            "hash_index": hash_index,
        }
        results[str(bank_size)] = {}

        for method_name, lookup_fn in methods.items():
            cache: dict[str, tuple[int, np.ndarray]] = {}
            cache_hits = 0
            predictions: list[int] = []
            stage_samples: list[dict[str, float]] = []

            for query in queries:
                total_start = time.perf_counter()

                start = time.perf_counter()
                allowed_count = int(np.count_nonzero(authorized))
                authorization_s = time.perf_counter() - start

                key = hashlib.sha1(np.round(query, 2).tobytes()).hexdigest()
                start = time.perf_counter()
                if method_name == "cached_exact_index" and key in cache:
                    selected, reconstructed = cache[key]
                    cache_hits += 1
                    lookup_s = 0.0
                    cache_s = time.perf_counter() - start
                    transfer_s = 0.0
                    reconstruction_s = 0.0
                else:
                    selected = lookup_fn(query)
                    lookup_s = time.perf_counter() - start
                    start = time.perf_counter()
                    component = components[selected].copy()
                    transfer_s = time.perf_counter() - start
                    start = time.perf_counter()
                    reconstructed = component + 0.001 * codebook[selected % len(codebook)][None, :]
                    reconstruction_s = time.perf_counter() - start
                    start = time.perf_counter()
                    if method_name == "cached_exact_index":
                        cache[key] = (selected, reconstructed)
                    cache_s = time.perf_counter() - start

                start = time.perf_counter()
                _ = query[: min(dim, component_dim)] @ reconstructed[: min(dim, component_dim), :]
                dispatch_s = time.perf_counter() - start
                predictions.append(selected)
                stage_samples.append(
                    {
                        "authorization": authorization_s,
                        "lookup": lookup_s,
                        "cache": cache_s,
                        "transfer": transfer_s,
                        "reconstruction": reconstruction_s,
                        "dispatch": dispatch_s,
                        "end_to_end": time.perf_counter() - total_start,
                    }
                )

            metrics = _summarize_stage(stage_samples)
            predictions_arr = np.asarray(predictions)
            metrics.update(
                {
                    "recall_at_1": float(np.mean(predictions_arr == true_ids)),
                    "cache_hit_rate": cache_hits / len(queries),
                    "authorized_candidates": float(allowed_count),
                    "component_bytes": float(components[0].nbytes),
                    "index_memory_bytes": float(
                        bank.nbytes
                        if method_name in {"dense_router", "exact_index", "cached_exact_index"}
                        else bank_hashes.nbytes + projections.nbytes
                    ),
                    "cache_entries": float(len(cache)),
                }
            )
            results[str(bank_size)][method_name] = metrics
    return results
