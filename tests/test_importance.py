import weightlab.importance as importance
from weightlab.importance import (
    evaluate_carrier_importance,
    evaluate_output_error_selected_precision,
    evaluate_tiny_transformer_precision,
    evaluate_trained_tiny_transformer_bf16_vs_fp32_carriers,
    evaluate_trained_tiny_transformer_internal_matrix_precision,
    evaluate_trained_tiny_transformer_precision,
)


def test_causal_importance_beats_random_protection_control():
    result = evaluate_carrier_importance(seed=11, n_samples=256, n_features=24, protected_count=4)

    assert result["causal_topk_overlap"] > result["random_topk_overlap"]
    assert (
        result["precision_policies"]["causal_fp32_protected"]["mse"]
        < result["precision_policies"]["random_fp32_protected"]["mse"]
    )
    assert (
        result["precision_policies"]["sparse_fp32_residual"]["total_bytes"]
        < result["precision_policies"]["full_fp32"]["total_bytes"]
    )


def test_output_error_selected_groupwise_precision_beats_random_control():
    result = evaluate_output_error_selected_precision(
        seed=23,
        n_train=256,
        n_validation=256,
        n_features=32,
        protected_count=4,
        group_size=8,
    )

    policies = result["precision_policies"]

    assert policies["groupwise_int4"]["mse"] < policies["uniform_int4"]["mse"]
    assert (
        policies["output_error_fp32_protected"]["mse"]
        < policies["random_fp32_protected_mean"]["mse"]
    )
    assert (
        policies["output_error_fp32_protected"]["total_bytes"]
        == policies["random_fp32_protected_mean"]["total_bytes"]
    )
    assert (
        policies["sparse_fp32_residual"]["total_bytes"]
        < policies["full_fp32"]["total_bytes"]
    )


def test_tiny_transformer_precision_uses_held_out_prompts_and_beats_random_control():
    result = evaluate_tiny_transformer_precision(
        seed=31,
        n_calibration_prompts=32,
        n_heldout_prompts=32,
        protected_count=4,
    )

    policies = result["precision_policies"]

    assert result["model"]["architecture"] == "tiny_transformer_lm"
    assert result["calibration_prompts"] == 32
    assert result["heldout_prompts"] == 32
    assert policies["groupwise_int4"]["heldout_logit_mse"] < policies["uniform_int4"][
        "heldout_logit_mse"
    ]
    assert (
        policies["output_error_fp32_protected"]["heldout_logit_mse"]
        < policies["random_fp32_protected_mean"]["heldout_logit_mse"]
    )
    assert (
        policies["output_error_fp32_protected"]["total_bytes"]
        == policies["random_fp32_protected_mean"]["total_bytes"]
    )
    assert policies["full_fp32"]["heldout_kl_divergence"] == 0.0
    assert policies["output_error_fp32_protected"]["heldout_kl_divergence"] >= 0.0


def test_trained_tiny_transformer_precision_uses_trained_head_and_held_out_targets():
    result = evaluate_trained_tiny_transformer_precision(
        seed=37,
        n_train_prompts=48,
        n_heldout_prompts=32,
        protected_count=4,
    )

    policies = result["precision_policies"]

    assert result["model"]["architecture"] == "trained_tiny_transformer_lm"
    assert result["training"]["trained_heldout_nll"] < result["training"]["untrained_heldout_nll"]
    assert policies["groupwise_int4"]["heldout_nll"] <= policies["uniform_int4"]["heldout_nll"]
    assert (
        policies["output_error_fp32_protected"]["heldout_logit_mse"]
        < policies["random_fp32_protected_mean"]["heldout_logit_mse"]
    )
    assert (
        policies["output_error_fp32_protected"]["total_bytes"]
        == policies["random_fp32_protected_mean"]["total_bytes"]
    )


def test_bf16_carriers_are_compared_with_fp32_and_random_controls():
    result = evaluate_trained_tiny_transformer_bf16_vs_fp32_carriers(
        seed=41,
        n_train_prompts=48,
        n_heldout_prompts=32,
        protected_count=4,
    )

    policies = result["precision_policies"]

    assert result["model"]["architecture"] == "trained_tiny_transformer_lm"
    assert policies["output_error_bf16_protected"]["total_bytes"] < policies[
        "output_error_fp32_protected"
    ]["total_bytes"]
    assert policies["output_error_fp32_protected"]["heldout_logit_mse"] <= policies[
        "output_error_bf16_protected"
    ]["heldout_logit_mse"]
    assert policies["output_error_bf16_protected"]["heldout_logit_mse"] < policies[
        "groupwise_int4"
    ]["heldout_logit_mse"]
    assert policies["output_error_bf16_protected"]["heldout_logit_mse"] < policies[
        "random_bf16_protected_mean"
    ]["heldout_logit_mse"]
    assert policies["output_error_bf16_protected"]["total_bytes"] == policies[
        "random_bf16_protected_mean"
    ]["total_bytes"]


def test_internal_matrix_precision_uses_held_out_prompts_and_random_control():
    result = evaluate_trained_tiny_transformer_internal_matrix_precision(
        seed=43,
        n_train_prompts=48,
        n_heldout_prompts=32,
        protected_count=6,
    )

    policies = result["precision_policies"]

    assert result["model"]["quantized_tensor"] == "w_mlp_out_rows"
    assert result["training"]["heldout_prompts"] == 32
    assert policies["groupwise_int4"]["heldout_logit_mse"] <= policies["uniform_int4"][
        "heldout_logit_mse"
    ]
    assert (
        policies["output_error_fp32_protected"]["heldout_logit_mse"]
        < policies["random_fp32_protected_mean"]["heldout_logit_mse"]
    )
    assert (
        policies["output_error_fp32_protected"]["total_bytes"]
        == policies["random_fp32_protected_mean"]["total_bytes"]
    )


def test_trained_internal_layer_precision_uses_trained_matrix_and_bf16_control():
    assert hasattr(importance, "evaluate_trained_internal_layer_precision")
    result = importance.evaluate_trained_internal_layer_precision(
        seed=47,
        n_train_prompts=48,
        n_heldout_prompts=32,
        protected_count=6,
    )

    policies = result["precision_policies"]

    assert result["model"]["trained_component"] == "w_mlp_out_ridge_regression"
    assert result["model"]["quantized_tensor"] == "trained_w_mlp_out_rows"
    assert result["training"]["trained_internal_heldout_nll"] < result["training"][
        "untrained_internal_heldout_nll"
    ]
    assert policies["groupwise_int4"]["heldout_logit_mse"] <= policies["uniform_int4"][
        "heldout_logit_mse"
    ]
    assert policies["output_error_bf16_protected"]["total_bytes"] < policies[
        "output_error_fp32_protected"
    ]["total_bytes"]
    assert policies["output_error_bf16_protected"]["heldout_logit_mse"] >= 0.0
    assert policies["output_error_fp32_protected"]["heldout_logit_mse"] >= 0.0
    assert policies["output_error_bf16_protected"]["total_bytes"] == policies[
        "random_bf16_protected_mean"
    ]["total_bytes"]
    assert policies["output_error_fp32_protected"]["heldout_logit_mse"] < policies[
        "random_fp32_protected_mean"
    ]["heldout_logit_mse"]
    assert policies["output_error_fp32_protected"]["total_bytes"] == policies[
        "random_fp32_protected_mean"
    ]["total_bytes"]
