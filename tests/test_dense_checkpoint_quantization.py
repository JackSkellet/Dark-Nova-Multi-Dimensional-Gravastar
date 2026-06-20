from __future__ import annotations

import torch

from weightlab.dense_checkpoint_quantization import (
    _build_policy_states,
    _quantize_tensor,
)


def test_quantize_tensor_accounts_for_packed_values_and_metadata() -> None:
    tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    reconstructed, accounting = _quantize_tensor(tensor, bits=4)

    assert reconstructed.shape == tensor.shape
    assert accounting["encoded_bytes"] == 8 + 12
    assert accounting["metadata_bytes"] == 12


def test_checkpoint_quantization_builds_random_and_selected_controls() -> None:
    state = {
        "weight": torch.linspace(-1.0, 1.0, steps=128, dtype=torch.float32).reshape(16, 8),
        "bias": torch.zeros(8, dtype=torch.float32),
    }

    policies = _build_policy_states(state, protected_fraction=0.1, sparse_fraction=0.1, seed=7)
    by_name = {row["policy"]: row for row in policies}

    assert "uniform_int8" in by_name
    assert "uniform_int4" in by_name
    assert "residual_bf16_protected_int4" in by_name
    assert "residual_fp32_protected_int4" in by_name
    assert "random_bf16_protected_int4" in by_name
    assert "random_fp32_protected_int4" in by_name
    assert "sparse_fp32_residual_int4" in by_name
    assert by_name["residual_fp32_protected_int4"]["protected_values"] == by_name[
        "random_fp32_protected_int4"
    ]["protected_values"]
    assert by_name["residual_bf16_protected_int4"]["encoded_bytes"] < by_name[
        "residual_fp32_protected_int4"
    ]["encoded_bytes"]
    assert by_name["uniform_int4"]["metadata_bytes"] > 0
