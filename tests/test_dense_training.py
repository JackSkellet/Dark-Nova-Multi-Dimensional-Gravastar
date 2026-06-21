import torch

from weightlab.dense_training import (
    DenseDecoder,
    DenseTrainingConfig,
    _causal_mask,
    debug_dense_step_stability,
    evaluate_dense_checkpoint,
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
    assert result["model"]["config"]["architecture_variant"] == "dense"
    assert result["model"]["trainable_parameter_count"] == result["model"]["parameter_count"]
    assert result["model"]["active_parameter_count"] == result["model"]["parameter_count"]
    assert result["tokenizer"]["name"] == "byte_level"
    assert result["training"]["train_tokens"] > 0
    assert len(result["training"]["loss_curve"]) == 2
    assert result["training"]["gradient_norms"]["summary"]["count"] == 2
    assert result["validation"]["loss"] > 0.0
    assert result["validation"]["batches"] == 1
    assert result["validation"]["tokens"] == 34
    assert len(result["validation"]["sample_order_sha256"]) == 64
    assert result["generation_samples"]
    assert result["checkpoint"]["path"]
    assert result["checkpoint"]["model_only_bytes"] < result["checkpoint"]["bytes"]
    assert result["checkpoint"]["optimizer_state_bytes"] > 0
    assert result["checkpoint"]["resume_ok"] is True
    assert result["best_checkpoint"]["path"].endswith("dense_decoder_best.pt")
    assert result["best_checkpoint"]["model_only_bytes"] < result["best_checkpoint"]["bytes"]
    assert result["best_checkpoint"]["validation"]["sample_order_sha256"] == result["validation"][
        "sample_order_sha256"
    ]
    assert (tmp_path / "dense_decoder_last.pt").exists()
    assert (tmp_path / "dense_decoder_last_model_only.pt").exists()
    assert (tmp_path / "dense_decoder_best.pt").exists()
    assert (tmp_path / "dense_decoder_best_model_only.pt").exists()
    assert result["progress"]["records"] == 2
    assert result["progress"]["latest"]["step"] == 2
    assert len(result["progress"]["checkpoint_validations"]) == 2
    assert result["progress"]["latest_checkpoint"]["exists"] is True
    assert (tmp_path / "dense_decoder_latest.pt").exists()
    assert result["memory"]["peak_allocated_bytes"] == 0
    assert result["adapter"]["architecture_variant"] == "dense"

    resumed = train_dense_decoder(
        texts=texts,
        config=DenseTrainingConfig(
            device="cpu",
            seq_len=16,
            hidden_dim=32,
            layers=1,
            heads=4,
            batch_size=2,
            steps=3,
            validation_batches=1,
            mixed_precision="fp32",
            progress_interval=1,
            checkpoint_interval=1,
        ),
        output_dir=tmp_path,
        seed=123,
        resume_checkpoint=tmp_path / "dense_decoder_latest.pt",
    )

    assert resumed["status"] == "completed"
    assert resumed["training"]["start_step"] == 2
    assert resumed["training"]["completed_steps_this_invocation"] == 1
    assert resumed["training"]["resumed_from"].endswith("dense_decoder_latest.pt")


def test_dense_decoder_failed_run_reports_actual_steps_and_tokens(tmp_path, monkeypatch):
    texts = [
        "def parse_config(text): return text.strip()",
        "README: parse_config loads local configuration.",
    ]
    config = DenseTrainingConfig(
        device="cpu",
        seq_len=16,
        hidden_dim=32,
        layers=1,
        heads=4,
        batch_size=2,
        steps=3,
        validation_batches=1,
        mixed_precision="fp32",
    )

    def infinite_loss(*args, **kwargs):
        return torch.tensor(float("inf"), requires_grad=True)

    monkeypatch.setattr(torch.nn.functional, "cross_entropy", infinite_loss)

    result = train_dense_decoder(texts, config, tmp_path, seed=123)

    assert result["status"] == "failed_nonfinite_loss"
    assert result["failure"] == "nonfinite_loss_at_step_1"
    assert result["training"]["failure_step"] == 1
    assert result["training"]["completed_steps_this_invocation"] == 0
    assert result["training"]["last_completed_step"] == 0
    assert result["training"]["train_tokens"] == 0
    assert result["training"]["planned_train_tokens"] == 96
    assert result["checkpoint"]["step"] == 0
    assert result["checkpoint"]["resume_ok"] is True


def test_dense_decoder_records_nonfinite_gradient_norm_without_failing(tmp_path, monkeypatch):
    texts = [
        "def parse_config(text): return text.strip()",
        "README: parse_config loads local configuration.",
    ]
    config = DenseTrainingConfig(
        device="cpu",
        seq_len=16,
        hidden_dim=32,
        layers=1,
        heads=4,
        batch_size=2,
        steps=1,
        validation_batches=1,
        mixed_precision="fp32",
    )

    def infinite_grad_norm(*args, **kwargs):
        return torch.tensor(float("inf"))

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", infinite_grad_norm)

    result = train_dense_decoder(texts, config, tmp_path, seed=123)

    assert result["status"] == "completed"
    assert result["training"]["gradient_norms"]["summary"]["nonfinite_count"] == 1


def test_dense_decoder_uses_separate_validation_texts_when_provided(tmp_path):
    train_texts = [
        "function trainA() { return 1 }",
        "function trainB() { return 2 }",
        "function trainC() { return 3 }",
    ]
    validation_texts = [
        "function validationA() { return 4 }",
        "function validationB() { return 5 }",
    ]
    config = DenseTrainingConfig(
        device="cpu",
        seq_len=8,
        hidden_dim=16,
        layers=1,
        heads=4,
        batch_size=2,
        steps=1,
        validation_batches=2,
        mixed_precision="fp32",
        optimizer_name="sgd",
        learning_rate=0.0,
    )

    result = train_dense_decoder(
        train_texts,
        config,
        tmp_path,
        seed=123,
        validation_texts=validation_texts,
    )

    assert result["status"] == "completed"
    assert result["validation"]["source"] == "provided_validation_texts"
    assert result["validation"]["heldout_texts_provided"] is True
    assert result["validation"]["tokens"] == 36


def test_adapter_decoder_training_smoke_records_variant(tmp_path):
    texts = [
        "function usable(value) { return value + 1 }",
        "README: usable increments a value.",
        "test('usable', () => expect(usable(1)).toBe(2))",
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
        mixed_precision="fp32",
        architecture_variant="adapter",
        adapter_dim=8,
    )

    result = train_dense_decoder(
        texts=texts,
        config=config,
        output_dir=tmp_path,
        seed=123,
    )

    assert result["status"] == "completed"
    assert result["model"]["config"]["architecture_variant"] == "adapter"
    assert result["model"]["config"]["adapter_dim"] == 8
    assert result["model"]["parameter_count"] > 0
    assert result["adapter"]["architecture_variant"] == "adapter"
    assert result["adapter"]["layers"][0]["adapter_update_norm"] >= 0.0


def test_adapter_decoder_starts_with_identity_residual_adapters():
    torch.manual_seed(123)
    model = DenseDecoder(
        vocab_size=257,
        seq_len=16,
        hidden_dim=32,
        layers=1,
        heads=4,
        architecture_variant="adapter",
        adapter_dim=8,
    )
    hidden = torch.randn(2, 16, 32)

    adapted = model.adapters[0](hidden)

    assert torch.equal(adapted, hidden)


def test_explicit_causal_block_forward_shape():
    model = DenseDecoder(
        vocab_size=257,
        seq_len=16,
        hidden_dim=32,
        layers=1,
        heads=4,
        attention_mask_mode="additive_causal",
        block_impl="explicit_causal",
    )

    logits = model(torch.randint(0, 257, (2, 16)))

    assert logits.shape == (2, 16, 257)
    assert model.block_impl == "explicit_causal"


def test_finite_causal_mask_uses_large_finite_negative_values():
    mask = _causal_mask("finite_causal", 4, torch.device("cpu"))

    assert mask is not None
    assert torch.isfinite(mask).all()
    assert mask[0, 1].item() == -1.0e4
    assert mask[0, 0].item() == 0.0


def test_evaluate_dense_checkpoint_uses_dedicated_texts_and_seed(tmp_path):
    train_texts = [
        "function trainA() { return 1 }",
        "function trainB() { return 2 }",
        "function trainC() { return 3 }",
    ]
    heldout_texts = [
        "function heldoutA() { return 4 }",
        "function heldoutB() { return 5 }",
    ]
    config = DenseTrainingConfig(
        device="cpu",
        seq_len=8,
        hidden_dim=16,
        layers=1,
        heads=4,
        batch_size=2,
        steps=1,
        validation_batches=1,
        mixed_precision="fp32",
        optimizer_name="sgd",
        learning_rate=0.0,
    )
    train_dense_decoder(train_texts, config, tmp_path, seed=123)

    first = evaluate_dense_checkpoint(
        checkpoint_path=tmp_path / "dense_decoder_last.pt",
        texts=heldout_texts,
        split_name="validation",
        device="cpu",
        seed=999,
        batches=3,
    )
    second = evaluate_dense_checkpoint(
        checkpoint_path=tmp_path / "dense_decoder_last.pt",
        texts=heldout_texts,
        split_name="validation",
        device="cpu",
        seed=999,
        batches=3,
    )

    assert first["benchmark_label"] == "dense_checkpoint_evaluation"
    assert first["split"] == "validation"
    assert first["batches"] == 3
    assert first["tokens"] == 54
    assert first["sample_order_sha256"] == second["sample_order_sha256"]
    assert first["loss"] == second["loss"]
    assert first["checkpoint"]["step"] == 1
    assert first["model"]["config"]["seq_len"] == 8

    with_batch_losses = evaluate_dense_checkpoint(
        checkpoint_path=tmp_path / "dense_decoder_last.pt",
        texts=heldout_texts,
        split_name="validation",
        device="cpu",
        seed=999,
        batches=3,
        include_batch_losses=True,
    )

    assert len(with_batch_losses["batch_loss_records"]) == 3
    assert with_batch_losses["batch_loss_records"][0]["batch_index"] == 0
    assert with_batch_losses["batch_loss_records"][0]["tokens"] == 18
    assert with_batch_losses["batch_loss_records"][0]["sample_sha256"]
    assert with_batch_losses["batch_loss_records"][0]["loss"] > 0.0
    assert "batch_loss_records" not in first


def test_evaluate_dense_checkpoint_loads_legacy_transformer_encoder_keys(tmp_path):
    train_texts = [
        "function trainA() { return 1 }",
        "function trainB() { return 2 }",
    ]
    heldout_texts = [
        "function heldoutA() { return 4 }",
        "function heldoutB() { return 5 }",
    ]
    config = DenseTrainingConfig(
        device="cpu",
        seq_len=8,
        hidden_dim=16,
        layers=1,
        heads=4,
        batch_size=2,
        steps=1,
        validation_batches=1,
        mixed_precision="fp32",
        optimizer_name="sgd",
        learning_rate=0.0,
    )
    train_dense_decoder(train_texts, config, tmp_path, seed=123)
    checkpoint_path = tmp_path / "dense_decoder_last.pt"
    payload = torch.load(checkpoint_path, map_location="cpu")
    payload["model"] = {
        key.replace("blocks.0.", "blocks.layers.0."): value
        for key, value in payload["model"].items()
    }
    legacy_path = tmp_path / "legacy_dense_decoder_last.pt"
    torch.save(payload, legacy_path)

    result = evaluate_dense_checkpoint(
        checkpoint_path=legacy_path,
        texts=heldout_texts,
        split_name="validation",
        device="cpu",
        seed=999,
        batches=1,
    )

    assert result["benchmark_label"] == "dense_checkpoint_evaluation"
    assert result["loss"] > 0.0


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


def test_masked_decoder_blocks_disable_autocast_for_stable_backward(monkeypatch):
    autocast_states = []

    def recording_forward(self, *args, **kwargs):
        autocast_states.append(torch.is_autocast_enabled("cpu"))
        return args[0]

    monkeypatch.setattr(torch.nn.TransformerEncoderLayer, "forward", recording_forward)
    model = DenseDecoder(
        vocab_size=257,
        seq_len=16,
        hidden_dim=32,
        layers=1,
        heads=4,
        attention_mask_mode="additive_causal",
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model(torch.randint(0, 257, (2, 16)))

    assert logits.shape == (2, 16, 257)
    assert autocast_states == [False]
