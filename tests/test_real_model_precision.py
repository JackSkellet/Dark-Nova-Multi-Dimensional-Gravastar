import importlib
from pathlib import Path

import torch


def test_real_model_matrix_precision_uses_local_checkpoint_and_random_controls(
    tmp_path: Path,
):
    real_model_precision = importlib.import_module("weightlab.real_model_precision")
    checkpoint_path = tmp_path / "model.bin"
    torch.manual_seed(0)
    matrix = torch.randn(24, 12)
    matrix[:4] *= 12.0
    torch.save(
        {
            "transformer.h.0.mlp.c_fc.weight": matrix,
            "transformer.h.0.attn.bias": torch.zeros(1, 1, 4, 4),
        },
        checkpoint_path,
    )

    result = real_model_precision.evaluate_real_model_matrix_precision(
        checkpoint_path=checkpoint_path,
        tensor_name="transformer.h.0.mlp.c_fc.weight",
        model_id="local/test-model",
        model_commit="local-test",
        seed=101,
        protected_count=4,
    )

    policies = result["precision_policies"]

    assert result["source"]["checkpoint_path"] == str(checkpoint_path)
    assert result["source"]["model_id"] == "local/test-model"
    assert result["source"]["model_commit"] == "local-test"
    assert result["tensor"]["name"] == "transformer.h.0.mlp.c_fc.weight"
    assert result["tensor"]["shape"] == [24, 12]
    assert len(result["selected_rows"]) == 4
    assert policies["groupwise_int4"]["matrix_mse"] < policies["uniform_int4"]["matrix_mse"]
    assert policies["output_error_bf16_protected"]["matrix_mse"] < policies[
        "random_bf16_protected_mean"
    ]["matrix_mse"]
    assert policies["output_error_fp32_protected"]["matrix_mse"] < policies[
        "random_fp32_protected_mean"
    ]["matrix_mse"]
    assert policies["output_error_bf16_protected"]["total_bytes"] == policies[
        "random_bf16_protected_mean"
    ]["total_bytes"]
    assert policies["output_error_fp32_protected"]["total_bytes"] == policies[
        "random_fp32_protected_mean"
    ]["total_bytes"]
    assert policies["output_error_bf16_protected"]["total_bytes"] < policies[
        "output_error_fp32_protected"
    ]["total_bytes"]
