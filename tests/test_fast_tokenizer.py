from __future__ import annotations

from weightlab.fast_tokenizer import (
    load_fast_bpe_tokenizer,
    train_fast_bpe_tokenizer,
    write_fast_bpe_tokenizer,
)


def test_fast_bpe_tokenizer_trains_encodes_and_serializes():
    tokenizer = train_fast_bpe_tokenizer(
        [
            "function parseConfig(value) { return value.trim(); }\n",
            "function renderConfig(value) { return value.trim(); }\n",
            "def parse_config(value):\n    return value.strip()\n",
        ],
        vocab_size=320,
    )

    encoded = tokenizer.encode("function parseConfig(value) { return value.trim(); }\n")

    assert tokenizer.vocab_size > 257
    assert len(tokenizer.checksum) == 64
    assert encoded
    assert all(isinstance(token, int) for token in encoded)
    assert tokenizer.to_jsonable()["type"] == "hf_tokenizers_bpe_bytelevel"


def test_fast_bpe_tokenizer_artifact_round_trips_with_checksum(tmp_path):
    tokenizer = train_fast_bpe_tokenizer(
        [
            "export function parseConfig(value) { return value.trim(); }\n",
            "export function renderConfig(value) { return value.trim(); }\n",
        ],
        vocab_size=320,
        min_frequency=1,
    )
    artifact_path = tmp_path / "tokenizer.json"

    write_fast_bpe_tokenizer(
        artifact_path,
        tokenizer,
        training_config={
            "vocab_size": 320,
            "min_frequency": 1,
            "source": "unit_test",
        },
    )
    loaded = load_fast_bpe_tokenizer(artifact_path)

    text = "export function parseConfig(value) { return value.trim(); }\n"
    assert loaded.checksum == tokenizer.checksum
    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.decode(loaded.encode(text)) == text
    artifact = loaded.to_jsonable(include_tokenizer_json=True)
    assert artifact["checksum"] == tokenizer.checksum
    assert "tokenizer_json" in artifact


def test_fast_bpe_tokenizer_reuses_loaded_tokenizer_instance():
    tokenizer = train_fast_bpe_tokenizer(
        [
            "function parseConfig(value) { return value.trim(); }\n",
            "function renderConfig(value) { return value.trim(); }\n",
        ],
        vocab_size=320,
        min_frequency=1,
    )

    assert tokenizer._tokenizer is tokenizer._tokenizer
