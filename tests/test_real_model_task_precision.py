import importlib
import string
from pathlib import Path

import torch


def _synthetic_tiny_gpt2_state(
    vocab_size: int = 32,
    n_positions: int = 8,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(3)
    d_model = 4
    hidden = 16
    state = {
        "transformer.wte.weight": torch.randn(vocab_size, d_model) * 0.05,
        "transformer.wpe.weight": torch.randn(n_positions, d_model) * 0.02,
        "transformer.h.0.ln_1.weight": torch.ones(d_model),
        "transformer.h.0.ln_1.bias": torch.zeros(d_model),
        "transformer.h.0.attn.c_attn.weight": torch.randn(d_model, d_model * 3) * 0.03,
        "transformer.h.0.attn.c_attn.bias": torch.zeros(d_model * 3),
        "transformer.h.0.attn.c_proj.weight": torch.eye(d_model),
        "transformer.h.0.attn.c_proj.bias": torch.zeros(d_model),
        "transformer.h.0.ln_2.weight": torch.ones(d_model),
        "transformer.h.0.ln_2.bias": torch.zeros(d_model),
        "transformer.h.0.mlp.c_fc.weight": torch.randn(d_model, hidden) * 0.03,
        "transformer.h.0.mlp.c_fc.bias": torch.zeros(hidden),
        "transformer.h.0.mlp.c_proj.weight": torch.randn(hidden, d_model) * 0.03,
        "transformer.h.0.mlp.c_proj.bias": torch.zeros(d_model),
        "transformer.ln_f.weight": torch.ones(d_model),
        "transformer.ln_f.bias": torch.zeros(d_model),
    }
    for row in range(6):
        state["transformer.wte.weight"][row] = torch.tensor(
            [row + 1.0, -row - 0.5, row * 0.5 + 0.25, -row * 0.25 - 0.125]
        )
    state["lm_head.weight"] = state["transformer.wte.weight"].clone()
    return state


def _write_character_tokenizer(tmp_path: Path, texts: list[str]) -> tuple[Path, Path]:
    tokenizer = importlib.import_module("weightlab.real_model_task_precision")
    vocab: dict[str, int] = {}
    for text in texts:
        for byte in text.encode("utf-8"):
            symbol = tokenizer._bytes_to_unicode()[byte]
            if symbol not in vocab:
                vocab[symbol] = len(vocab)
    for char in string.ascii_letters + string.digits + " .,;:_-":
        symbol = tokenizer._bytes_to_unicode()[ord(char)]
        if symbol not in vocab:
            vocab[symbol] = len(vocab)
    vocab["<|endoftext|>"] = len(vocab)
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"
    vocab_path.write_text(__import__("json").dumps(vocab), encoding="utf-8")
    merges_path.write_text("#version: 0.2\n", encoding="utf-8")
    return vocab_path, merges_path


def test_tiny_gpt2_task_precision_uses_heldout_loss_and_random_controls(tmp_path: Path):
    task_precision = importlib.import_module("weightlab.real_model_task_precision")
    checkpoint_path = tmp_path / "tiny-gpt2.bin"
    torch.save(_synthetic_tiny_gpt2_state(), checkpoint_path)
    calibration = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7],
        ],
        dtype=torch.long,
    )
    heldout = torch.tensor(
        [
            [0, 2, 4, 6, 8, 10],
            [1, 3, 5, 7, 9, 11],
            [2, 4, 6, 8, 10, 12],
        ],
        dtype=torch.long,
    )

    result = task_precision.evaluate_tiny_gpt2_task_precision(
        checkpoint_path=checkpoint_path,
        tensor_name="transformer.wte.weight",
        calibration_token_ids=calibration,
        heldout_token_ids=heldout,
        model_id="local/synthetic-gpt2",
        model_commit="local-test",
        seed=17,
        protected_count=4,
        n_layer=1,
        n_head=2,
    )

    policies = result["precision_policies"]

    assert result["task"]["metric"] == "next_token_cross_entropy"
    assert result["tensor"]["name"] == "transformer.wte.weight"
    assert result["tensor"]["shape"] == [32, 4]
    assert len(result["selected_rows"]) == 4
    assert policies["full_fp32"]["heldout_kl_divergence"] == 0.0
    assert policies["groupwise_int4"]["heldout_kl_divergence"] < policies["uniform_int4"][
        "heldout_kl_divergence"
    ]
    assert policies["output_error_bf16_protected"]["heldout_kl_divergence"] < policies[
        "random_bf16_protected_mean"
    ]["heldout_kl_divergence"]
    assert policies["output_error_fp32_protected"]["heldout_kl_divergence"] < policies[
        "random_fp32_protected_mean"
    ]["heldout_kl_divergence"]
    assert policies["output_error_bf16_protected"]["total_bytes"] == policies[
        "random_bf16_protected_mean"
    ]["total_bytes"]


def test_gpt2_bpe_tokenizer_builds_fixed_natural_text_sequences(tmp_path: Path):
    task_precision = importlib.import_module("weightlab.real_model_task_precision")
    texts = [
        "parse config returns offline docs.",
        "audit logs stay local.",
    ]
    vocab_path, merges_path = _write_character_tokenizer(tmp_path, texts)

    token_ids = task_precision.gpt2_texts_to_token_ids(
        texts=texts,
        vocab_path=vocab_path,
        merges_path=merges_path,
        sequence_length=12,
    )

    assert token_ids.shape == (2, 12)
    assert token_ids.dtype == torch.long
    assert int(torch.max(token_ids)) < len(task_precision._load_gpt2_vocab(vocab_path))
    assert len(torch.unique(token_ids)) > 6


def test_tiny_gpt2_natural_text_precision_records_text_task_and_tokenizer(tmp_path: Path):
    task_precision = importlib.import_module("weightlab.real_model_task_precision")
    calibration_texts = [
        "parse config returns offline docs.",
        "the local runtime keeps audit logs.",
        "repository symbols answer code questions.",
    ]
    heldout_texts = [
        "offline docs describe parse config.",
        "audit logs stay local for review.",
        "code questions use repository symbols.",
    ]
    vocab_path, merges_path = _write_character_tokenizer(
        tmp_path,
        calibration_texts + heldout_texts,
    )
    checkpoint_path = tmp_path / "tiny-gpt2.bin"
    state = _synthetic_tiny_gpt2_state(vocab_size=128, n_positions=12)
    torch.save(state, checkpoint_path)

    result = task_precision.evaluate_tiny_gpt2_natural_text_precision(
        checkpoint_path=checkpoint_path,
        tensor_name="transformer.wte.weight",
        tokenizer_vocab_path=vocab_path,
        tokenizer_merges_path=merges_path,
        calibration_texts=calibration_texts,
        heldout_texts=heldout_texts,
        sequence_length=12,
        model_id="local/synthetic-gpt2",
        model_commit="local-test",
        seed=19,
        protected_count=5,
        n_layer=1,
        n_head=2,
    )

    policies = result["precision_policies"]

    assert result["task"]["input_kind"] == "natural_language_text"
    assert result["task"]["calibration_texts"] == 3
    assert result["task"]["heldout_texts"] == 3
    assert result["source"]["tokenizer_vocab_path"] == str(vocab_path)
    assert result["source"]["tokenizer_merges_path"] == str(merges_path)
    assert policies["full_fp32"]["heldout_kl_divergence"] == 0.0
    assert policies["groupwise_int4"]["total_bytes"] > 0
    assert result["protected_count"] == 5


def test_tiny_gpt2_natural_text_precision_can_select_all_rows_for_internal_tensor(
    tmp_path: Path,
):
    task_precision = importlib.import_module("weightlab.real_model_task_precision")
    calibration_texts = [
        "parse config returns offline docs.",
        "the local runtime keeps audit logs.",
        "repository symbols answer code questions.",
    ]
    heldout_texts = [
        "offline docs describe parse config.",
        "audit logs stay local for review.",
        "code questions use repository symbols.",
    ]
    vocab_path, merges_path = _write_character_tokenizer(
        tmp_path,
        calibration_texts + heldout_texts,
    )
    checkpoint_path = tmp_path / "tiny-gpt2.bin"
    state = _synthetic_tiny_gpt2_state(vocab_size=128, n_positions=12)
    torch.save(state, checkpoint_path)

    result = task_precision.evaluate_tiny_gpt2_natural_text_precision(
        checkpoint_path=checkpoint_path,
        tensor_name="transformer.h.0.mlp.c_fc.weight",
        tokenizer_vocab_path=vocab_path,
        tokenizer_merges_path=merges_path,
        calibration_texts=calibration_texts,
        heldout_texts=heldout_texts,
        sequence_length=12,
        model_id="local/synthetic-gpt2",
        model_commit="local-test",
        seed=23,
        protected_count=2,
        n_layer=1,
        n_head=2,
        candidate_row_strategy="all_rows",
    )

    assert result["tensor"]["name"] == "transformer.h.0.mlp.c_fc.weight"
    assert result["tensor"]["shape"] == [4, 16]
    assert result["task"]["candidate_row_strategy"] == "all_rows"
    assert result["task"]["candidate_rows"] == 4
    assert len(result["selected_rows"]) == 2
    for policy in result["precision_policies"].values():
        if "heldout_kl_divergence" in policy:
            assert policy["heldout_kl_divergence"] >= 0.0
