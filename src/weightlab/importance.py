from __future__ import annotations

import numpy as np

from weightlab.compression import _uniform_quantize
from weightlab.metrics import ceil_div, mse, set_seed


def _quantize_with_protection(
    weights: np.ndarray, protected: set[int], bits: int = 4
) -> tuple[np.ndarray, int]:
    quantized, base_bytes = _uniform_quantize(weights.reshape(1, -1), bits)
    result = quantized.ravel()
    for idx in protected:
        result[idx] = weights[idx]
    protected_bytes = len(protected) * (4 + 4)
    return result.astype(np.float32), base_bytes + protected_bytes


def _uniform_quantize_vector(weights: np.ndarray, bits: int = 4) -> tuple[np.ndarray, int]:
    quantized, total_bytes = _uniform_quantize(weights.reshape(1, -1), bits)
    return quantized.ravel().astype(np.float32), total_bytes


def _groupwise_quantize_vector(
    weights: np.ndarray,
    bits: int = 4,
    group_size: int = 8,
) -> tuple[np.ndarray, int]:
    pieces: list[np.ndarray] = []
    total_bytes = 0
    for start in range(0, weights.size, group_size):
        group = weights[start : start + group_size]
        quantized, group_bytes = _uniform_quantize(group.reshape(1, -1), bits)
        pieces.append(quantized.ravel().astype(np.float32))
        total_bytes += group_bytes
    return np.concatenate(pieces).astype(np.float32), total_bytes


def _uniform_quantize_matrix(matrix: np.ndarray, bits: int = 4) -> tuple[np.ndarray, int]:
    quantized, total_bytes = _uniform_quantize(np.asarray(matrix, dtype=np.float32), bits)
    return quantized.astype(np.float32), total_bytes


def _groupwise_quantize_matrix_rows(
    matrix: np.ndarray,
    bits: int = 4,
) -> tuple[np.ndarray, int]:
    pieces: list[np.ndarray] = []
    total_bytes = 0
    for row in np.asarray(matrix, dtype=np.float32):
        quantized, row_bytes = _uniform_quantize(row.reshape(1, -1), bits)
        pieces.append(quantized.ravel().astype(np.float32))
        total_bytes += row_bytes
    return np.vstack(pieces).astype(np.float32), total_bytes


def _to_bf16_like(values: np.ndarray) -> np.ndarray:
    fp32 = np.asarray(values, dtype=np.float32).copy()
    bits = fp32.view(np.uint32)
    rounded = (bits + np.uint32(0x8000)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _kl_divergence(reference_logits: np.ndarray, candidate_logits: np.ndarray) -> float:
    reference_probs = _softmax(reference_logits)
    candidate_probs = np.clip(_softmax(candidate_logits), 1e-9, 1.0)
    reference_probs = np.clip(reference_probs, 1e-9, 1.0)
    return float(
        np.mean(
            np.sum(
                reference_probs * (np.log(reference_probs) - np.log(candidate_probs)),
                axis=-1,
            )
        )
    )


def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + eps)


def _tiny_transformer_mlp_inputs(
    prompts: np.ndarray,
    params: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    token_embeddings = params["token_embeddings"][prompts]
    seq_len = prompts.shape[1]
    hidden = token_embeddings + params["position_embeddings"][:seq_len]

    q = hidden @ params["w_q"]
    k = hidden @ params["w_k"]
    v = hidden @ params["w_v"]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    scores = np.where(causal_mask[None, :, :], -1e9, scores)
    attention = _softmax(scores)
    hidden = _layer_norm(hidden + attention @ v @ params["w_o"])
    mlp_features = np.maximum(hidden @ params["w_mlp_in"], 0.0)
    return hidden, mlp_features


def _tiny_transformer_hidden(
    prompts: np.ndarray,
    params: dict[str, np.ndarray],
) -> np.ndarray:
    hidden, mlp_features = _tiny_transformer_mlp_inputs(prompts, params)
    hidden = _layer_norm(hidden + mlp_features @ params["w_mlp_out"])
    return hidden


def _tiny_transformer_logits(
    prompts: np.ndarray,
    params: dict[str, np.ndarray],
    output_weight: np.ndarray | None = None,
) -> np.ndarray:
    hidden = _tiny_transformer_hidden(prompts, params)
    head = params["lm_head"] if output_weight is None else output_weight
    return hidden @ head.T


def _make_tiny_transformer(
    seed: int, vocab_size: int, seq_len: int, d_model: int
) -> dict[str, np.ndarray]:
    rng = set_seed(seed)
    scale = 1.0 / np.sqrt(d_model)
    params = {
        "token_embeddings": rng.normal(scale=scale, size=(vocab_size, d_model)).astype(np.float32),
        "position_embeddings": rng.normal(scale=scale, size=(seq_len, d_model)).astype(np.float32),
        "w_q": rng.normal(scale=scale, size=(d_model, d_model)).astype(np.float32),
        "w_k": rng.normal(scale=scale, size=(d_model, d_model)).astype(np.float32),
        "w_v": rng.normal(scale=scale, size=(d_model, d_model)).astype(np.float32),
        "w_o": rng.normal(scale=scale, size=(d_model, d_model)).astype(np.float32),
        "w_mlp_in": rng.normal(scale=scale, size=(d_model, d_model * 2)).astype(np.float32),
        "w_mlp_out": rng.normal(scale=scale, size=(d_model * 2, d_model)).astype(np.float32),
        "lm_head": rng.normal(scale=scale, size=(vocab_size, d_model)).astype(np.float32),
    }
    carrier_rows = np.arange(min(8, vocab_size))
    params["lm_head"][carrier_rows] *= np.linspace(3.0, 1.7, len(carrier_rows), dtype=np.float32)[
        :, None
    ]
    return params


def _make_prompts(
    seed: int,
    n_prompts: int,
    seq_len: int,
    vocab_size: int,
) -> np.ndarray:
    rng = set_seed(seed)
    prompts = rng.integers(8, vocab_size, size=(n_prompts, seq_len), dtype=np.int64)
    pattern_count = max(1, n_prompts // 4)
    for row in range(pattern_count):
        prompts[row, :4] = np.array([0, 1, 2, 3], dtype=np.int64)
    return prompts


def _policy_metrics(
    reference_logits: np.ndarray,
    candidate_logits: np.ndarray,
    total_bytes: int,
) -> dict[str, float]:
    return {
        "heldout_logit_mse": mse(reference_logits, candidate_logits),
        "heldout_kl_divergence": _kl_divergence(reference_logits, candidate_logits),
        "total_bytes": float(total_bytes),
    }


def _next_token_targets(prompts: np.ndarray, vocab_size: int) -> np.ndarray:
    return ((prompts + 1) % vocab_size).astype(np.int64)


def _cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    probs = _softmax(logits)
    flat_probs = probs.reshape(-1, probs.shape[-1])
    flat_targets = targets.reshape(-1)
    return float(
        -np.mean(np.log(np.clip(flat_probs[np.arange(flat_targets.size), flat_targets], 1e-9, 1.0)))
    )


def _accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.argmax(logits, axis=-1) == targets))


def _train_output_head_ridge(
    hidden: np.ndarray,
    targets: np.ndarray,
    vocab_size: int,
    ridge: float = 1e-3,
    target_logit: float = 4.0,
) -> np.ndarray:
    features = hidden.reshape(-1, hidden.shape[-1]).astype(np.float64)
    labels = targets.reshape(-1)
    desired = np.full((features.shape[0], vocab_size), -target_logit / vocab_size, dtype=np.float64)
    desired[np.arange(labels.size), labels] = target_logit
    gram = features.T @ features + ridge * np.eye(features.shape[1], dtype=np.float64)
    weights = np.linalg.solve(gram, features.T @ desired)
    return weights.T.astype(np.float32)


def _normalize_rows(matrix: np.ndarray, target_norm: float) -> np.ndarray:
    rows = np.asarray(matrix, dtype=np.float32).copy()
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return (rows / np.maximum(norms, 1e-6) * target_norm).astype(np.float32)


def _train_internal_mlp_out_ridge(
    prompts: np.ndarray,
    params: dict[str, np.ndarray],
    targets: np.ndarray,
    ridge: float = 1e-3,
) -> np.ndarray:
    residual_hidden, mlp_features = _tiny_transformer_mlp_inputs(prompts, params)
    features = mlp_features.reshape(-1, mlp_features.shape[-1]).astype(np.float64)
    residual = residual_hidden.reshape(-1, residual_hidden.shape[-1]).astype(np.float64)
    target_hidden = params["lm_head"][targets.reshape(-1)].astype(np.float64)
    desired_delta = target_hidden - residual
    gram = features.T @ features + ridge * np.eye(features.shape[1], dtype=np.float64)
    weights = np.linalg.solve(gram, features.T @ desired_delta)
    return weights.astype(np.float32)


def evaluate_carrier_importance(
    seed: int = 0,
    n_samples: int = 512,
    n_features: int = 32,
    protected_count: int = 4,
) -> dict[str, object]:
    rng = set_seed(seed)
    x = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    weights = rng.normal(scale=0.15, size=n_features).astype(np.float32)
    true_carriers = np.arange(protected_count)
    weights[true_carriers] = np.linspace(2.5, 1.8, protected_count, dtype=np.float32)
    y = x @ weights

    causal_scores = []
    activation_scores = np.mean(np.abs(x), axis=0)
    gradient_scores = np.abs(weights) * np.mean(np.abs(x), axis=0)
    for idx in range(n_features):
        ablated = weights.copy()
        ablated[idx] = 0.0
        causal_scores.append(mse(y, x @ ablated))
    causal_scores_arr = np.asarray(causal_scores)

    causal_top = set(np.argsort(causal_scores_arr)[-protected_count:])
    activation_top = set(np.argsort(activation_scores)[-protected_count:])
    gradient_top = set(np.argsort(gradient_scores)[-protected_count:])
    random_top = set(
        rng.choice(np.arange(protected_count, n_features), size=protected_count, replace=False)
    )
    true_set = set(int(i) for i in true_carriers)

    policies: dict[str, dict[str, float]] = {}
    for name, protected in {
        "uniform_int4": set(),
        "random_fp32_protected": random_top,
        "activation_fp32_protected": activation_top,
        "causal_fp32_protected": causal_top,
        "gradient_fp32_protected": gradient_top,
    }.items():
        q_weights, total_bytes = _quantize_with_protection(weights, protected)
        policies[name] = {"mse": mse(y, x @ q_weights), "total_bytes": float(total_bytes)}

    full_fp32_bytes = weights.size * 4
    sparse_residual_weights, sparse_bytes = _quantize_with_protection(weights, causal_top)
    policies["full_fp32"] = {"mse": 0.0, "total_bytes": float(full_fp32_bytes)}
    policies["sparse_fp32_residual"] = {
        "mse": mse(y, x @ sparse_residual_weights),
        "total_bytes": float(ceil_div(weights.size * 4, 8) + 12 + len(causal_top) * 8),
    }

    return {
        "causal_topk_overlap": len(causal_top & true_set) / protected_count,
        "activation_topk_overlap": len(activation_top & true_set) / protected_count,
        "gradient_topk_overlap": len(gradient_top & true_set) / protected_count,
        "random_topk_overlap": len(random_top & true_set) / protected_count,
        "causal_scores": causal_scores_arr.tolist(),
        "precision_policies": policies,
    }


def evaluate_output_error_selected_precision(
    seed: int = 0,
    n_train: int = 512,
    n_validation: int = 512,
    n_features: int = 48,
    protected_count: int = 6,
    group_size: int = 8,
) -> dict[str, object]:
    rng = set_seed(seed)
    x_train = rng.normal(size=(n_train, n_features)).astype(np.float32)
    x_validation = rng.normal(size=(n_validation, n_features)).astype(np.float32)

    weights = rng.normal(scale=0.12, size=n_features).astype(np.float32)
    carrier_count = max(protected_count, n_features // 8)
    carriers = np.arange(carrier_count)
    weights[carriers] += np.linspace(2.4, 1.2, carrier_count, dtype=np.float32)

    y_train = x_train @ weights
    y_validation = x_validation @ weights

    uniform_weights, uniform_bytes = _uniform_quantize_vector(weights, bits=4)
    groupwise_weights, groupwise_bytes = _groupwise_quantize_vector(
        weights,
        bits=4,
        group_size=group_size,
    )

    output_error_scores = []
    for idx in range(n_features):
        candidate = groupwise_weights.copy()
        candidate[idx] = weights[idx]
        output_error_scores.append(mse(y_train, x_train @ candidate))
    output_error_scores_arr = np.asarray(output_error_scores)
    output_error_top = set(np.argsort(output_error_scores_arr)[:protected_count])

    random_policy_errors = []
    random_trials = max(16, protected_count * 4)
    random_total_bytes = None
    candidate_pool = np.arange(n_features)
    for _ in range(random_trials):
        random_top = set(rng.choice(candidate_pool, size=protected_count, replace=False))
        random_weights = groupwise_weights.copy()
        for idx in random_top:
            random_weights[idx] = weights[idx]
        random_total_bytes = groupwise_bytes + len(random_top) * 8
        random_policy_errors.append(mse(y_validation, x_validation @ random_weights))

    output_weights = groupwise_weights.copy()
    for idx in output_error_top:
        output_weights[idx] = weights[idx]

    output_bytes = groupwise_bytes + len(output_error_top) * 8
    full_fp32_bytes = weights.size * 4

    policies = {
        "uniform_int4": {
            "mse": mse(y_validation, x_validation @ uniform_weights),
            "total_bytes": float(uniform_bytes),
        },
        "groupwise_int4": {
            "mse": mse(y_validation, x_validation @ groupwise_weights),
            "total_bytes": float(groupwise_bytes),
        },
        "output_error_fp32_protected": {
            "mse": mse(y_validation, x_validation @ output_weights),
            "total_bytes": float(output_bytes),
        },
        "random_fp32_protected_mean": {
            "mse": float(np.mean(random_policy_errors)),
            "total_bytes": float(random_total_bytes or output_bytes),
        },
        "random_fp32_protected_best": {
            "mse": float(np.min(random_policy_errors)),
            "total_bytes": float(random_total_bytes or output_bytes),
        },
        "full_fp32": {
            "mse": 0.0,
            "total_bytes": float(full_fp32_bytes),
        },
        "sparse_fp32_residual": {
            "mse": mse(y_validation, x_validation @ output_weights),
            "total_bytes": float(output_bytes),
        },
    }

    return {
        "selected_indices": sorted(int(i) for i in output_error_top),
        "true_carrier_overlap": len(output_error_top & set(int(i) for i in carriers))
        / protected_count,
        "group_size": group_size,
        "random_trials": random_trials,
        "output_error_scores": output_error_scores_arr.tolist(),
        "precision_policies": policies,
    }


def evaluate_tiny_transformer_precision(
    seed: int = 0,
    n_calibration_prompts: int = 64,
    n_heldout_prompts: int = 64,
    protected_count: int = 6,
    vocab_size: int = 32,
    seq_len: int = 12,
    d_model: int = 24,
) -> dict[str, object]:
    params = _make_tiny_transformer(
        seed=seed,
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=d_model,
    )
    calibration_prompts = _make_prompts(
        seed=seed + 1,
        n_prompts=n_calibration_prompts,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )
    heldout_prompts = _make_prompts(
        seed=seed + 2,
        n_prompts=n_heldout_prompts,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )

    full_head = params["lm_head"]
    calibration_reference = _tiny_transformer_logits(calibration_prompts, params)
    heldout_reference = _tiny_transformer_logits(heldout_prompts, params)

    uniform_head, uniform_bytes = _uniform_quantize_matrix(full_head, bits=4)
    groupwise_head, groupwise_bytes = _groupwise_quantize_matrix_rows(full_head, bits=4)

    row_scores = []
    for row_idx in range(vocab_size):
        candidate_head = groupwise_head.copy()
        candidate_head[row_idx] = full_head[row_idx]
        candidate_logits = _tiny_transformer_logits(
            calibration_prompts,
            params,
            output_weight=candidate_head,
        )
        row_scores.append(mse(calibration_reference, candidate_logits))
    row_scores_arr = np.asarray(row_scores)
    selected_rows = set(np.argsort(row_scores_arr)[:protected_count])

    output_head = groupwise_head.copy()
    for row_idx in selected_rows:
        output_head[row_idx] = full_head[row_idx]
    protected_bytes = groupwise_bytes + len(selected_rows) * (d_model * 4 + 4)

    rng = set_seed(seed + 3)
    random_errors = []
    random_kls = []
    random_trials = max(16, protected_count * 4)
    random_total_bytes = protected_bytes
    for _ in range(random_trials):
        random_rows = set(rng.choice(np.arange(vocab_size), size=protected_count, replace=False))
        random_head = groupwise_head.copy()
        for row_idx in random_rows:
            random_head[row_idx] = full_head[row_idx]
        random_logits = _tiny_transformer_logits(heldout_prompts, params, output_weight=random_head)
        random_errors.append(mse(heldout_reference, random_logits))
        random_kls.append(_kl_divergence(heldout_reference, random_logits))

    uniform_logits = _tiny_transformer_logits(heldout_prompts, params, output_weight=uniform_head)
    groupwise_logits = _tiny_transformer_logits(
        heldout_prompts, params, output_weight=groupwise_head
    )
    output_logits = _tiny_transformer_logits(heldout_prompts, params, output_weight=output_head)

    full_fp32_bytes = int(full_head.size * 4)
    policies = {
        "full_fp32": {
            "heldout_logit_mse": 0.0,
            "heldout_kl_divergence": 0.0,
            "total_bytes": float(full_fp32_bytes),
        },
        "uniform_int4": _policy_metrics(heldout_reference, uniform_logits, uniform_bytes),
        "groupwise_int4": _policy_metrics(heldout_reference, groupwise_logits, groupwise_bytes),
        "output_error_fp32_protected": _policy_metrics(
            heldout_reference,
            output_logits,
            protected_bytes,
        ),
        "random_fp32_protected_mean": {
            "heldout_logit_mse": float(np.mean(random_errors)),
            "heldout_kl_divergence": float(np.mean(random_kls)),
            "total_bytes": float(random_total_bytes),
        },
        "random_fp32_protected_best": {
            "heldout_logit_mse": float(np.min(random_errors)),
            "heldout_kl_divergence": float(np.min(random_kls)),
            "total_bytes": float(random_total_bytes),
        },
    }

    return {
        "model": {
            "architecture": "tiny_transformer_lm",
            "vocab_size": vocab_size,
            "sequence_length": seq_len,
            "d_model": d_model,
            "quantized_tensor": "lm_head_rows",
        },
        "calibration_prompts": n_calibration_prompts,
        "heldout_prompts": n_heldout_prompts,
        "protected_count": protected_count,
        "selected_rows": sorted(int(i) for i in selected_rows),
        "row_output_error_scores": row_scores_arr.tolist(),
        "precision_policies": policies,
        "notes": (
            "Random initialized tiny transformer-style LM. Carrier rows are selected on "
            "calibration prompts and evaluated on disjoint held-out prompts."
        ),
    }


def evaluate_trained_tiny_transformer_precision(
    seed: int = 0,
    n_train_prompts: int = 96,
    n_heldout_prompts: int = 64,
    protected_count: int = 6,
    vocab_size: int = 32,
    seq_len: int = 12,
    d_model: int = 24,
) -> dict[str, object]:
    params = _make_tiny_transformer(
        seed=seed,
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=d_model,
    )
    train_prompts = _make_prompts(
        seed=seed + 11,
        n_prompts=n_train_prompts,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )
    heldout_prompts = _make_prompts(
        seed=seed + 12,
        n_prompts=n_heldout_prompts,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )
    train_targets = _next_token_targets(train_prompts, vocab_size)
    heldout_targets = _next_token_targets(heldout_prompts, vocab_size)

    train_hidden = _tiny_transformer_hidden(train_prompts, params)
    trained_head = _train_output_head_ridge(train_hidden, train_targets, vocab_size)
    untrained_head = params["lm_head"].copy()
    params["lm_head"] = trained_head

    train_reference = _tiny_transformer_logits(train_prompts, params)
    heldout_reference = _tiny_transformer_logits(heldout_prompts, params)
    untrained_heldout = _tiny_transformer_logits(
        heldout_prompts,
        params,
        output_weight=untrained_head,
    )

    uniform_head, uniform_bytes = _uniform_quantize_matrix(trained_head, bits=4)
    groupwise_head, groupwise_bytes = _groupwise_quantize_matrix_rows(trained_head, bits=4)

    row_scores = []
    for row_idx in range(vocab_size):
        candidate_head = groupwise_head.copy()
        candidate_head[row_idx] = trained_head[row_idx]
        candidate_logits = _tiny_transformer_logits(
            train_prompts,
            params,
            output_weight=candidate_head,
        )
        row_scores.append(mse(train_reference, candidate_logits))
    row_scores_arr = np.asarray(row_scores)
    selected_rows = set(np.argsort(row_scores_arr)[:protected_count])

    output_head = groupwise_head.copy()
    for row_idx in selected_rows:
        output_head[row_idx] = trained_head[row_idx]
    protected_bytes = groupwise_bytes + len(selected_rows) * (d_model * 4 + 4)

    rng = set_seed(seed + 13)
    random_errors = []
    random_nll = []
    random_trials = max(16, protected_count * 4)
    for _ in range(random_trials):
        random_rows = set(rng.choice(np.arange(vocab_size), size=protected_count, replace=False))
        random_head = groupwise_head.copy()
        for row_idx in random_rows:
            random_head[row_idx] = trained_head[row_idx]
        random_logits = _tiny_transformer_logits(heldout_prompts, params, output_weight=random_head)
        random_errors.append(mse(heldout_reference, random_logits))
        random_nll.append(_cross_entropy(random_logits, heldout_targets))

    def metrics_for(head: np.ndarray, total_bytes: int) -> dict[str, float]:
        logits = _tiny_transformer_logits(heldout_prompts, params, output_weight=head)
        return {
            "heldout_logit_mse": mse(heldout_reference, logits),
            "heldout_kl_divergence": _kl_divergence(heldout_reference, logits),
            "heldout_nll": _cross_entropy(logits, heldout_targets),
            "heldout_accuracy": _accuracy(logits, heldout_targets),
            "total_bytes": float(total_bytes),
        }

    full_fp32_bytes = int(trained_head.size * 4)
    policies = {
        "full_fp32": {
            "heldout_logit_mse": 0.0,
            "heldout_kl_divergence": 0.0,
            "heldout_nll": _cross_entropy(heldout_reference, heldout_targets),
            "heldout_accuracy": _accuracy(heldout_reference, heldout_targets),
            "total_bytes": float(full_fp32_bytes),
        },
        "uniform_int4": metrics_for(uniform_head, uniform_bytes),
        "groupwise_int4": metrics_for(groupwise_head, groupwise_bytes),
        "output_error_fp32_protected": metrics_for(output_head, protected_bytes),
        "random_fp32_protected_mean": {
            "heldout_logit_mse": float(np.mean(random_errors)),
            "heldout_nll": float(np.mean(random_nll)),
            "total_bytes": float(protected_bytes),
        },
        "random_fp32_protected_best": {
            "heldout_logit_mse": float(np.min(random_errors)),
            "heldout_nll": float(np.min(random_nll)),
            "total_bytes": float(protected_bytes),
        },
    }

    return {
        "model": {
            "architecture": "trained_tiny_transformer_lm",
            "vocab_size": vocab_size,
            "sequence_length": seq_len,
            "d_model": d_model,
            "trained_component": "lm_head_ridge_regression",
            "quantized_tensor": "lm_head_rows",
        },
        "training": {
            "train_prompts": n_train_prompts,
            "heldout_prompts": n_heldout_prompts,
            "target_rule": "next_token_is_current_token_plus_one_mod_vocab",
            "untrained_heldout_nll": _cross_entropy(untrained_heldout, heldout_targets),
            "trained_heldout_nll": _cross_entropy(heldout_reference, heldout_targets),
            "trained_heldout_accuracy": _accuracy(heldout_reference, heldout_targets),
        },
        "protected_count": protected_count,
        "selected_rows": sorted(int(i) for i in selected_rows),
        "row_output_error_scores": row_scores_arr.tolist(),
        "precision_policies": policies,
        "notes": (
            "Fixed tiny transformer features with ridge-trained output head. Precision "
            "selection uses train prompts and evaluation uses disjoint held-out prompts."
        ),
    }


def evaluate_trained_tiny_transformer_bf16_vs_fp32_carriers(
    seed: int = 0,
    n_train_prompts: int = 96,
    n_heldout_prompts: int = 64,
    protected_count: int = 6,
    vocab_size: int = 32,
    seq_len: int = 12,
    d_model: int = 24,
) -> dict[str, object]:
    params = _make_tiny_transformer(
        seed=seed,
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=d_model,
    )
    train_prompts = _make_prompts(
        seed=seed + 21,
        n_prompts=n_train_prompts,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )
    heldout_prompts = _make_prompts(
        seed=seed + 22,
        n_prompts=n_heldout_prompts,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )
    train_targets = _next_token_targets(train_prompts, vocab_size)
    heldout_targets = _next_token_targets(heldout_prompts, vocab_size)

    train_hidden = _tiny_transformer_hidden(train_prompts, params)
    trained_head = _train_output_head_ridge(train_hidden, train_targets, vocab_size)
    params["lm_head"] = trained_head

    train_reference = _tiny_transformer_logits(train_prompts, params)
    heldout_reference = _tiny_transformer_logits(heldout_prompts, params)

    groupwise_head, groupwise_bytes = _groupwise_quantize_matrix_rows(trained_head, bits=4)
    row_scores = []
    for row_idx in range(vocab_size):
        candidate_head = groupwise_head.copy()
        candidate_head[row_idx] = trained_head[row_idx]
        candidate_logits = _tiny_transformer_logits(
            train_prompts,
            params,
            output_weight=candidate_head,
        )
        row_scores.append(mse(train_reference, candidate_logits))
    row_scores_arr = np.asarray(row_scores)
    selected_rows = set(np.argsort(row_scores_arr)[:protected_count])

    def protected_head(rows: set[int], mode: str) -> np.ndarray:
        head = groupwise_head.copy()
        for row_idx in rows:
            if mode == "bf16":
                head[row_idx] = _to_bf16_like(trained_head[row_idx])
            elif mode == "fp32":
                head[row_idx] = trained_head[row_idx]
            else:
                raise ValueError(f"unknown protected-row mode: {mode}")
        return head

    def metrics_for(head: np.ndarray, total_bytes: int) -> dict[str, float]:
        logits = _tiny_transformer_logits(heldout_prompts, params, output_weight=head)
        return {
            "heldout_logit_mse": mse(heldout_reference, logits),
            "heldout_kl_divergence": _kl_divergence(heldout_reference, logits),
            "heldout_nll": _cross_entropy(logits, heldout_targets),
            "heldout_accuracy": _accuracy(logits, heldout_targets),
            "total_bytes": float(total_bytes),
        }

    bf16_bytes = groupwise_bytes + len(selected_rows) * (d_model * 2 + 4)
    fp32_bytes = groupwise_bytes + len(selected_rows) * (d_model * 4 + 4)
    selected_bf16_head = protected_head(selected_rows, mode="bf16")
    selected_fp32_head = protected_head(selected_rows, mode="fp32")

    rng = set_seed(seed + 23)
    random_bf16_errors = []
    random_fp32_errors = []
    random_trials = max(16, protected_count * 4)
    for _ in range(random_trials):
        random_rows = set(rng.choice(np.arange(vocab_size), size=protected_count, replace=False))
        random_bf16_logits = _tiny_transformer_logits(
            heldout_prompts,
            params,
            output_weight=protected_head(random_rows, mode="bf16"),
        )
        random_fp32_logits = _tiny_transformer_logits(
            heldout_prompts,
            params,
            output_weight=protected_head(random_rows, mode="fp32"),
        )
        random_bf16_errors.append(mse(heldout_reference, random_bf16_logits))
        random_fp32_errors.append(mse(heldout_reference, random_fp32_logits))

    full_fp32_bytes = int(trained_head.size * 4)
    policies = {
        "full_fp32": {
            "heldout_logit_mse": 0.0,
            "heldout_kl_divergence": 0.0,
            "heldout_nll": _cross_entropy(heldout_reference, heldout_targets),
            "heldout_accuracy": _accuracy(heldout_reference, heldout_targets),
            "total_bytes": float(full_fp32_bytes),
        },
        "groupwise_int4": metrics_for(groupwise_head, groupwise_bytes),
        "output_error_bf16_protected": metrics_for(selected_bf16_head, bf16_bytes),
        "output_error_fp32_protected": metrics_for(selected_fp32_head, fp32_bytes),
        "random_bf16_protected_mean": {
            "heldout_logit_mse": float(np.mean(random_bf16_errors)),
            "total_bytes": float(bf16_bytes),
        },
        "random_bf16_protected_best": {
            "heldout_logit_mse": float(np.min(random_bf16_errors)),
            "total_bytes": float(bf16_bytes),
        },
        "random_fp32_protected_mean": {
            "heldout_logit_mse": float(np.mean(random_fp32_errors)),
            "total_bytes": float(fp32_bytes),
        },
        "random_fp32_protected_best": {
            "heldout_logit_mse": float(np.min(random_fp32_errors)),
            "total_bytes": float(fp32_bytes),
        },
    }

    return {
        "model": {
            "architecture": "trained_tiny_transformer_lm",
            "vocab_size": vocab_size,
            "sequence_length": seq_len,
            "d_model": d_model,
            "trained_component": "lm_head_ridge_regression",
            "quantized_tensor": "lm_head_rows",
        },
        "training": {
            "train_prompts": n_train_prompts,
            "heldout_prompts": n_heldout_prompts,
            "target_rule": "next_token_is_current_token_plus_one_mod_vocab",
            "trained_heldout_nll": _cross_entropy(heldout_reference, heldout_targets),
            "trained_heldout_accuracy": _accuracy(heldout_reference, heldout_targets),
        },
        "protected_count": protected_count,
        "selected_rows": sorted(int(i) for i in selected_rows),
        "row_output_error_scores": row_scores_arr.tolist(),
        "precision_policies": policies,
        "notes": (
            "Fixed tiny transformer features with ridge-trained output head. Selected "
            "carrier rows are stored as simulated BF16 or FP32 and compared against "
            "random protected rows with matched byte accounting."
        ),
    }


def evaluate_trained_tiny_transformer_internal_matrix_precision(
    seed: int = 0,
    n_train_prompts: int = 96,
    n_heldout_prompts: int = 64,
    protected_count: int = 8,
    vocab_size: int = 32,
    seq_len: int = 12,
    d_model: int = 24,
) -> dict[str, object]:
    params = _make_tiny_transformer(
        seed=seed,
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=d_model,
    )
    train_prompts = _make_prompts(
        seed=seed + 31,
        n_prompts=n_train_prompts,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )
    heldout_prompts = _make_prompts(
        seed=seed + 32,
        n_prompts=n_heldout_prompts,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )
    train_targets = _next_token_targets(train_prompts, vocab_size)
    heldout_targets = _next_token_targets(heldout_prompts, vocab_size)

    train_hidden = _tiny_transformer_hidden(train_prompts, params)
    trained_head = _train_output_head_ridge(train_hidden, train_targets, vocab_size)
    params["lm_head"] = trained_head
    full_internal = params["w_mlp_out"].copy()
    train_reference = _tiny_transformer_logits(train_prompts, params)
    heldout_reference = _tiny_transformer_logits(heldout_prompts, params)

    uniform_internal, uniform_bytes = _uniform_quantize_matrix(full_internal, bits=4)
    groupwise_internal, groupwise_bytes = _groupwise_quantize_matrix_rows(
        full_internal,
        bits=4,
    )

    def logits_with_internal(prompts: np.ndarray, internal: np.ndarray) -> np.ndarray:
        candidate_params = dict(params)
        candidate_params["w_mlp_out"] = internal
        return _tiny_transformer_logits(prompts, candidate_params)

    row_scores = []
    for row_idx in range(full_internal.shape[0]):
        candidate_internal = groupwise_internal.copy()
        candidate_internal[row_idx] = full_internal[row_idx]
        candidate_logits = logits_with_internal(train_prompts, candidate_internal)
        row_scores.append(mse(train_reference, candidate_logits))
    row_scores_arr = np.asarray(row_scores)
    selected_rows = set(np.argsort(row_scores_arr)[:protected_count])

    selected_internal = groupwise_internal.copy()
    for row_idx in selected_rows:
        selected_internal[row_idx] = full_internal[row_idx]
    protected_bytes = groupwise_bytes + len(selected_rows) * (full_internal.shape[1] * 4 + 4)

    rng = set_seed(seed + 33)
    random_errors = []
    random_nll = []
    random_trials = max(16, protected_count * 4)
    for _ in range(random_trials):
        random_rows = set(
            rng.choice(np.arange(full_internal.shape[0]), size=protected_count, replace=False)
        )
        random_internal = groupwise_internal.copy()
        for row_idx in random_rows:
            random_internal[row_idx] = full_internal[row_idx]
        random_logits = logits_with_internal(heldout_prompts, random_internal)
        random_errors.append(mse(heldout_reference, random_logits))
        random_nll.append(_cross_entropy(random_logits, heldout_targets))

    def metrics_for(internal: np.ndarray, total_bytes: int) -> dict[str, float]:
        logits = logits_with_internal(heldout_prompts, internal)
        return {
            "heldout_logit_mse": mse(heldout_reference, logits),
            "heldout_kl_divergence": _kl_divergence(heldout_reference, logits),
            "heldout_nll": _cross_entropy(logits, heldout_targets),
            "heldout_accuracy": _accuracy(logits, heldout_targets),
            "total_bytes": float(total_bytes),
        }

    full_fp32_bytes = int(full_internal.size * 4)
    policies = {
        "full_fp32": {
            "heldout_logit_mse": 0.0,
            "heldout_kl_divergence": 0.0,
            "heldout_nll": _cross_entropy(heldout_reference, heldout_targets),
            "heldout_accuracy": _accuracy(heldout_reference, heldout_targets),
            "total_bytes": float(full_fp32_bytes),
        },
        "uniform_int4": metrics_for(uniform_internal, uniform_bytes),
        "groupwise_int4": metrics_for(groupwise_internal, groupwise_bytes),
        "output_error_fp32_protected": metrics_for(selected_internal, protected_bytes),
        "random_fp32_protected_mean": {
            "heldout_logit_mse": float(np.mean(random_errors)),
            "heldout_nll": float(np.mean(random_nll)),
            "total_bytes": float(protected_bytes),
        },
        "random_fp32_protected_best": {
            "heldout_logit_mse": float(np.min(random_errors)),
            "heldout_nll": float(np.min(random_nll)),
            "total_bytes": float(protected_bytes),
        },
    }

    return {
        "model": {
            "architecture": "trained_tiny_transformer_lm",
            "vocab_size": vocab_size,
            "sequence_length": seq_len,
            "d_model": d_model,
            "trained_component": "lm_head_ridge_regression",
            "quantized_tensor": "w_mlp_out_rows",
        },
        "training": {
            "train_prompts": n_train_prompts,
            "heldout_prompts": n_heldout_prompts,
            "target_rule": "next_token_is_current_token_plus_one_mod_vocab",
            "trained_heldout_nll": _cross_entropy(heldout_reference, heldout_targets),
            "trained_heldout_accuracy": _accuracy(heldout_reference, heldout_targets),
        },
        "protected_count": protected_count,
        "selected_rows": sorted(int(i) for i in selected_rows),
        "row_output_error_scores": row_scores_arr.tolist(),
        "precision_policies": policies,
        "notes": (
            "Fixed tiny transformer features with ridge-trained output head. Precision "
            "selection quantizes the internal MLP output matrix rather than the output "
            "head, then evaluates held-out logits and targets."
        ),
    }


def evaluate_trained_internal_layer_precision(
    seed: int = 0,
    n_train_prompts: int = 96,
    n_heldout_prompts: int = 64,
    protected_count: int = 8,
    vocab_size: int = 32,
    seq_len: int = 12,
    d_model: int = 24,
) -> dict[str, object]:
    params = _make_tiny_transformer(
        seed=seed,
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=d_model,
    )
    params["lm_head"] = _normalize_rows(params["lm_head"], target_norm=float(np.sqrt(d_model)))
    original_internal = params["w_mlp_out"].copy()

    train_prompts = _make_prompts(
        seed=seed + 41,
        n_prompts=n_train_prompts,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )
    heldout_prompts = _make_prompts(
        seed=seed + 42,
        n_prompts=n_heldout_prompts,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )
    train_targets = _next_token_targets(train_prompts, vocab_size)
    heldout_targets = _next_token_targets(heldout_prompts, vocab_size)

    trained_internal = _train_internal_mlp_out_ridge(train_prompts, params, train_targets)
    params["w_mlp_out"] = trained_internal
    train_reference = _tiny_transformer_logits(train_prompts, params)
    heldout_reference = _tiny_transformer_logits(heldout_prompts, params)

    untrained_params = dict(params)
    untrained_params["w_mlp_out"] = original_internal
    untrained_heldout = _tiny_transformer_logits(heldout_prompts, untrained_params)

    uniform_internal, uniform_bytes = _uniform_quantize_matrix(trained_internal, bits=4)
    groupwise_internal, groupwise_bytes = _groupwise_quantize_matrix_rows(
        trained_internal,
        bits=4,
    )

    def logits_with_internal(prompts: np.ndarray, internal: np.ndarray) -> np.ndarray:
        candidate_params = dict(params)
        candidate_params["w_mlp_out"] = internal
        return _tiny_transformer_logits(prompts, candidate_params)

    row_scores = []
    for row_idx in range(trained_internal.shape[0]):
        candidate_internal = groupwise_internal.copy()
        candidate_internal[row_idx] = trained_internal[row_idx]
        candidate_logits = logits_with_internal(train_prompts, candidate_internal)
        row_scores.append(mse(train_reference, candidate_logits))
    row_scores_arr = np.asarray(row_scores)
    selected_rows = set(np.argsort(row_scores_arr)[:protected_count])

    def protected_internal(rows: set[int], mode: str) -> np.ndarray:
        internal = groupwise_internal.copy()
        for row_idx in rows:
            if mode == "bf16":
                internal[row_idx] = _to_bf16_like(trained_internal[row_idx])
            elif mode == "fp32":
                internal[row_idx] = trained_internal[row_idx]
            else:
                raise ValueError(f"unknown protected-row mode: {mode}")
        return internal

    def metrics_for(internal: np.ndarray, total_bytes: int) -> dict[str, float]:
        logits = logits_with_internal(heldout_prompts, internal)
        return {
            "heldout_logit_mse": mse(heldout_reference, logits),
            "heldout_kl_divergence": _kl_divergence(heldout_reference, logits),
            "heldout_nll": _cross_entropy(logits, heldout_targets),
            "heldout_accuracy": _accuracy(logits, heldout_targets),
            "total_bytes": float(total_bytes),
        }

    bf16_bytes = groupwise_bytes + len(selected_rows) * (trained_internal.shape[1] * 2 + 4)
    fp32_bytes = groupwise_bytes + len(selected_rows) * (trained_internal.shape[1] * 4 + 4)
    selected_bf16_internal = protected_internal(selected_rows, mode="bf16")
    selected_fp32_internal = protected_internal(selected_rows, mode="fp32")

    rng = set_seed(seed + 43)
    random_bf16_errors = []
    random_fp32_errors = []
    random_trials = max(16, protected_count * 4)
    for _ in range(random_trials):
        random_rows = set(
            rng.choice(np.arange(trained_internal.shape[0]), size=protected_count, replace=False)
        )
        random_bf16_logits = logits_with_internal(
            heldout_prompts,
            protected_internal(random_rows, mode="bf16"),
        )
        random_fp32_logits = logits_with_internal(
            heldout_prompts,
            protected_internal(random_rows, mode="fp32"),
        )
        random_bf16_errors.append(mse(heldout_reference, random_bf16_logits))
        random_fp32_errors.append(mse(heldout_reference, random_fp32_logits))

    full_fp32_bytes = int(trained_internal.size * 4)
    policies = {
        "full_fp32": {
            "heldout_logit_mse": 0.0,
            "heldout_kl_divergence": 0.0,
            "heldout_nll": _cross_entropy(heldout_reference, heldout_targets),
            "heldout_accuracy": _accuracy(heldout_reference, heldout_targets),
            "total_bytes": float(full_fp32_bytes),
        },
        "uniform_int4": metrics_for(uniform_internal, uniform_bytes),
        "groupwise_int4": metrics_for(groupwise_internal, groupwise_bytes),
        "output_error_bf16_protected": metrics_for(selected_bf16_internal, bf16_bytes),
        "output_error_fp32_protected": metrics_for(selected_fp32_internal, fp32_bytes),
        "random_bf16_protected_mean": {
            "heldout_logit_mse": float(np.mean(random_bf16_errors)),
            "total_bytes": float(bf16_bytes),
        },
        "random_bf16_protected_best": {
            "heldout_logit_mse": float(np.min(random_bf16_errors)),
            "total_bytes": float(bf16_bytes),
        },
        "random_fp32_protected_mean": {
            "heldout_logit_mse": float(np.mean(random_fp32_errors)),
            "total_bytes": float(fp32_bytes),
        },
        "random_fp32_protected_best": {
            "heldout_logit_mse": float(np.min(random_fp32_errors)),
            "total_bytes": float(fp32_bytes),
        },
    }

    return {
        "model": {
            "architecture": "trained_internal_tiny_transformer_lm",
            "vocab_size": vocab_size,
            "sequence_length": seq_len,
            "d_model": d_model,
            "trained_component": "w_mlp_out_ridge_regression",
            "quantized_tensor": "trained_w_mlp_out_rows",
        },
        "training": {
            "train_prompts": n_train_prompts,
            "heldout_prompts": n_heldout_prompts,
            "target_rule": "next_token_is_current_token_plus_one_mod_vocab",
            "untrained_internal_heldout_nll": _cross_entropy(
                untrained_heldout,
                heldout_targets,
            ),
            "trained_internal_heldout_nll": _cross_entropy(
                heldout_reference,
                heldout_targets,
            ),
            "trained_internal_heldout_accuracy": _accuracy(
                heldout_reference,
                heldout_targets,
            ),
        },
        "protected_count": protected_count,
        "selected_rows": sorted(int(i) for i in selected_rows),
        "row_output_error_scores": row_scores_arr.tolist(),
        "precision_policies": policies,
        "notes": (
            "Fixed tiny transformer attention and input MLP weights with an internal "
            "MLP output matrix trained by ridge regression. Precision selection "
            "quantizes that trained internal matrix and evaluates disjoint held-out "
            "prompts against BF16, FP32, and random protected-row controls."
        ),
    }
