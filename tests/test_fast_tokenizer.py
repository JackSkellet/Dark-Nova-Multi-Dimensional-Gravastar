from __future__ import annotations

from weightlab.fast_tokenizer import train_fast_bpe_tokenizer


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
