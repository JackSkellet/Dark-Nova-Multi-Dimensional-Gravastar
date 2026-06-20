from weightlab.rocm_validation import validate_training_runtime


def test_training_runtime_validation_records_precision_throughput_and_resume(tmp_path):
    checkpoint_path = tmp_path / "resume.pt"

    result = validate_training_runtime(
        device="cpu",
        batch_sizes=[1, 2],
        seq_len=8,
        hidden_dim=16,
        vocab_size=32,
        steps_per_batch=1,
        checkpoint_path=checkpoint_path,
    )

    assert result["benchmark_label"] == "training_runtime_validation"
    assert result["requested_device"] == "cpu"
    assert result["accelerator_backend"] == "cpu"
    assert result["checkpoint_resume"]["path"] == str(checkpoint_path)
    assert result["checkpoint_resume"]["resume_ok"] is True
    assert result["precision_support"]["fp32_forward_backward"] is True
    assert "bf16_forward_backward" in result["precision_support"]
    assert "fp16_forward_backward" in result["precision_support"]
    assert len(result["batch_results"]) == 2
    assert result["stable_batch_size"] in {1, 2}
    assert result["max_stable_tokens"] >= 8
    for row in result["batch_results"]:
        assert row["status"] == "ok"
        assert row["tokens_per_second"] > 0.0
        assert row["loss"] > 0.0
    assert result["limitations"]
    assert checkpoint_path.exists()


def test_training_runtime_validation_handles_failed_batch(tmp_path):
    result = validate_training_runtime(
        device="cpu",
        batch_sizes=[0, 1],
        seq_len=8,
        hidden_dim=16,
        vocab_size=32,
        steps_per_batch=1,
        checkpoint_path=tmp_path / "resume.pt",
    )

    by_batch = {row["batch_size"]: row for row in result["batch_results"]}
    assert by_batch[0]["status"] == "failed"
    assert by_batch[1]["status"] == "ok"
    assert result["stable_batch_size"] == 1
