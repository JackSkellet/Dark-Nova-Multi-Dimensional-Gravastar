import torch

from weightlab.config import default_config_path, read_accelerator_backend
from weightlab.lookup import (
    _resolve_torch_accelerator,
    benchmark_lookup_methods,
    benchmark_rocm_transfer_scaling,
    benchmark_routed_execution,
    benchmark_torch_batched_routed_execution,
)


def test_lookup_benchmark_reports_complete_methods_and_cache_hits():
    result = benchmark_lookup_methods(seed=13, bank_sizes=[16, 128], n_queries=64, dim=12)

    for size in ["16", "128"]:
        methods = result[size]
        assert {
            "dense_topk",
            "exact_vector",
            "tree_centroid",
            "hash_routing",
            "cached_previous",
        } <= set(methods)
        assert methods["exact_vector"]["recall_at_1"] == 1.0
        assert methods["cached_previous"]["cache_hit_rate"] >= 0.0
        assert methods["dense_topk"]["latency_ms_p50"] >= 0.0


def test_routed_execution_benchmark_accounts_for_complete_path_and_cache():
    result = benchmark_routed_execution(
        seed=19,
        bank_sizes=[128, 2048],
        n_queries=96,
        dim=16,
        component_dim=32,
        repeat_fraction=0.5,
    )

    for size in ["128", "2048"]:
        methods = result[size]
        assert {"dense_router", "exact_index", "cached_exact_index", "hash_index"} <= set(methods)
        for metrics in methods.values():
            assert metrics["authorized_candidates"] > 0
            assert metrics["stage_authorization_ms_p50"] >= 0.0
            assert metrics["stage_lookup_ms_p50"] >= 0.0
            assert metrics["stage_transfer_ms_p50"] >= 0.0
            assert metrics["stage_reconstruction_ms_p50"] >= 0.0
            assert metrics["stage_dispatch_ms_p50"] >= 0.0
            assert metrics["end_to_end_ms_p95"] >= metrics["end_to_end_ms_p50"]
            assert metrics["index_memory_bytes"] >= 0.0
        assert methods["cached_exact_index"]["cache_hit_rate"] > 0.0
        assert methods["exact_index"]["recall_at_1"] == 1.0


def test_torch_batched_routed_execution_reports_device_and_complete_path():
    result = benchmark_torch_batched_routed_execution(
        seed=23,
        bank_sizes=[128],
        n_queries=64,
        dim=16,
        component_dim=24,
        batch_size=16,
        device="cpu",
    )

    metrics = result["128"]["torch_exact_batched"]

    assert metrics["device"] == "cpu"
    assert metrics["accelerator_backend"] == "cpu"
    assert metrics["cuda_available"] in {0.0, 1.0}
    assert metrics["torch_cuda_api_available"] in {0.0, 1.0}
    assert metrics["rocm_available"] in {0.0, 1.0}
    assert isinstance(metrics["rocm_runtime_version"], str)
    assert metrics["recall_at_1"] == 1.0
    assert metrics["batch_size"] == 16.0
    assert metrics["throughput_queries_per_s"] > 0.0
    assert metrics["stage_authorization_ms_p50"] >= 0.0
    assert metrics["stage_lookup_ms_p50"] >= 0.0
    assert metrics["stage_transfer_ms_p50"] >= 0.0
    assert metrics["stage_reconstruction_ms_p50"] >= 0.0
    assert metrics["stage_dispatch_ms_p50"] >= 0.0
    assert metrics["end_to_end_ms_p95"] >= metrics["end_to_end_ms_p50"]
    assert metrics["component_bytes_per_query"] > 0.0
    assert metrics["uses_accelerator_transfer"] == 0.0


def test_rocm_transfer_scaling_records_backend_and_limitations_on_cpu_path():
    result = benchmark_rocm_transfer_scaling(
        payload_bytes=[4096],
        iterations=2,
        warmup_iterations=1,
        device="cpu",
    )

    metrics = result["4096"]["rocm_transfer_scaling"]

    assert metrics["requested_device"] == "cpu"
    assert metrics["accelerator_backend"] == "cpu"
    assert metrics["payload_bytes"] == 4096.0
    assert metrics["iterations"] == 2.0
    assert metrics["warmup_iterations"] == 1.0
    assert metrics["host_to_device_ms_p50"] >= 0.0
    assert metrics["device_to_host_ms_p50"] >= 0.0
    assert metrics["device_dispatch_ms_p50"] >= 0.0
    assert metrics["host_to_device_bandwidth_gb_s_p50"] >= 0.0
    assert metrics["measures_occupancy"] == 0.0
    assert metrics["uses_rocm_transfer"] == 0.0


def test_rocm_backend_selection_uses_torch_cuda_device_with_rocm_label(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.1.0", raising=False)

    accelerator = _resolve_torch_accelerator("rocm")

    assert accelerator.device.type == "cuda"
    assert accelerator.backend == "rocm"
    assert accelerator.rocm_available is True
    assert accelerator.cuda_available is True


def test_rocm_request_falls_back_to_cpu_without_rocm_runtime(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    accelerator = _resolve_torch_accelerator("rocm")

    assert accelerator.device.type == "cpu"
    assert accelerator.backend == "cpu"
    assert accelerator.rocm_available is False


def test_experiment_config_reads_rocm_backend(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "seed: 123",
                "accelerator:",
                "  backend: rocm",
                "  allow_cpu_fallback: true",
            ]
        )
        + "\n"
    )

    assert read_accelerator_backend(config) == "rocm"


def test_default_config_prefers_root_config_yaml(tmp_path):
    root_config = tmp_path / "config.yaml"
    smoke_config = tmp_path / "configs" / "smoke.yaml"
    smoke_config.parent.mkdir()
    root_config.write_text("accelerator_backend: rocm\n")
    smoke_config.write_text("accelerator_backend: cpu\n")

    assert default_config_path(tmp_path) == root_config


def test_default_config_falls_back_to_smoke_yaml(tmp_path):
    smoke_config = tmp_path / "configs" / "smoke.yaml"
    smoke_config.parent.mkdir()
    smoke_config.write_text("accelerator_backend: rocm\n")

    assert default_config_path(tmp_path) == smoke_config
