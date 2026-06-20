from weightlab.dense_training import DenseTrainingConfig, train_dense_decoder


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
