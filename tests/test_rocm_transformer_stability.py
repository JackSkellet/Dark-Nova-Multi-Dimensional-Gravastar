from weightlab.rocm_transformer_stability import (
    StabilityCase,
    run_transformer_stability_microbench,
)


def test_transformer_stability_microbench_records_component_rows():
    result = run_transformer_stability_microbench(
        device="cpu",
        cases=[
            StabilityCase(
                component="embedding",
                dtype="fp32",
                operation_stage="forward",
                batch_size=2,
                seq_len=8,
            ),
            StabilityCase(
                component="norm",
                dtype="fp32",
                operation_stage="loss",
                batch_size=2,
                seq_len=8,
            ),
            StabilityCase(
                component="mlp",
                dtype="fp32",
                operation_stage="backward",
                batch_size=2,
                seq_len=8,
            ),
            StabilityCase(
                component="attention",
                dtype="fp32",
                mask_mode="bool_causal",
                operation_stage="optimizer",
                sdp_kernel="math",
                batch_size=2,
                seq_len=8,
            ),
            StabilityCase(
                component="transformer",
                dtype="fp32",
                mask_mode="additive_causal",
                operation_stage="optimizer",
                optimizer_foreach=False,
                compile_model=False,
                batch_size=2,
                seq_len=8,
            ),
        ],
        steps=1,
        hidden_dim=16,
        heads=4,
        vocab_size=64,
        seed=123,
    )

    assert result["benchmark_label"] == "rocm_transformer_stability_microbench"
    assert result["requested_device"] == "cpu"
    assert result["accelerator_backend"] == "cpu"
    assert result["case_count"] == 5
    assert {row["component"] for row in result["case_results"]} == {
        "embedding",
        "norm",
        "mlp",
        "attention",
        "transformer",
    }
    assert result["failure_count"] == 0
    assert result["first_failure"] is None
    for row in result["case_results"]:
        assert row["status"] == "ok"
        assert row["finite_loss"] is True
        assert row["tokens_per_second"] > 0.0
        assert row["loss"] > 0.0
        assert row["parameters"] > 0
        assert row["operation_stage"] in {"forward", "loss", "backward", "optimizer"}
        assert row["compile_model"] is False


def test_transformer_stability_microbench_rejects_invalid_case():
    result = run_transformer_stability_microbench(
        device="cpu",
        cases=[
            StabilityCase(
                component="attention",
                dtype="fp32",
                mask_mode="unsupported",
                operation_stage="optimizer",
                batch_size=2,
                seq_len=8,
            ),
        ],
        steps=1,
        hidden_dim=16,
        heads=4,
        vocab_size=64,
        seed=123,
    )

    assert result["failure_count"] == 1
    assert result["first_failure"]["status"] == "failed"
    assert result["first_failure"]["component"] == "attention"
    assert "unsupported_mask_mode" in result["first_failure"]["error"]
