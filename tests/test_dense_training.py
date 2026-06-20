from weightlab.dense_training import (
    DenseTrainingConfig,
    debug_dense_step_stability,
    train_dense_decoder,
)


def test_dense_decoder_training_smoke_records_metrics_and_checkpoint(tmp_path):
    texts = [
        "def parse_config(text): return text.strip()",
        "README: parse_config loads local configuration.",
        "def render_changelog(entries): return '\\n'.join(entries)",
    ]
    config = DenseTrainingConfig(
        device="cpu",
        seq_len=16,
        hidden_dim=32,
        layers=1,
        heads=4,
        batch_size=2,
        steps=2,
        validation_batches=1,
        gradient_accumulation_steps=1,
        mixed_precision="fp32",
        progress_interval=1,
        checkpoint_interval=1,
    )

    assert config.attention_mask_mode == "additive_causal"
    assert config.optimizer_name == "adamw"

    result = train_dense_decoder(
        texts=texts,
        config=config,
        output_dir=tmp_path,
        seed=123,
    )

    assert result["benchmark_label"] == "dense_decoder_training_smoke"
    assert result["status"] == "completed"
    assert result["model"]["parameter_count"] > 0
    assert result["model"]["config"]["attention_mask_mode"] == "additive_causal"
    assert result["model"]["config"]["optimizer_name"] == "adamw"
    assert result["tokenizer"]["name"] == "byte_level"
    assert result["training"]["train_tokens"] > 0
    assert len(result["training"]["loss_curve"]) == 2
    assert result["validation"]["loss"] > 0.0
    assert result["generation_samples"]
    assert result["checkpoint"]["path"]
    assert result["checkpoint"]["resume_ok"] is True
    assert (tmp_path / "dense_decoder_last.pt").exists()
    assert result["progress"]["records"] == 2
    assert result["progress"]["latest"]["step"] == 2
    assert result["progress"]["latest_checkpoint"]["exists"] is True
    assert (tmp_path / "dense_decoder_latest.pt").exists()


def test_dense_step_debug_probe_records_phase_tensor_health():
    texts = [
        "def parse_config(text): return text.strip()",
        "README: parse_config loads local configuration.",
        "def render_changelog(entries): return '\\n'.join(entries)",
    ]
    config = DenseTrainingConfig(
        device="cpu",
        seq_len=16,
        hidden_dim=32,
        layers=1,
        heads=4,
        batch_size=2,
        steps=2,
        validation_batches=1,
        gradient_accumulation_steps=1,
        mixed_precision="fp32",
        optimizer_name="sgd",
        learning_rate=0.0,
    )

    result = debug_dense_step_stability(texts=texts, config=config, seed=123, steps=2)

    assert result["benchmark_label"] == "dense_step_stability_debug"
    assert result["first_nonfinite_phase"] is None
    assert result["model"]["parameter_count"] > 0
    assert [row["step"] for row in result["step_results"]] == [1, 2]
    assert result["initial_parameters"]["finite"] is True
    for row in result["step_results"]:
        assert row["input_ids"]["finite"] is True
        assert row["logits"]["finite"] is True
        assert row["loss"]["finite"] is True
        assert row["gradients"]["finite"] is True
        assert row["parameters_after_optimizer"]["finite"] is True
