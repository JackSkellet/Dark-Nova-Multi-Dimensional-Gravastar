from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

UNK_TOKEN = "<unk>"
EOS_TOKEN = "<eos>"


@dataclass(frozen=True)
class FastBpeTokenizer:
    tokenizer_json: str
    vocab_size: int
    eos_id: int
    checksum: str
    name: str = "hf_tokenizers_bpe_bytelevel"

    @cached_property
    def _tokenizer(self) -> Tokenizer:
        return Tokenizer.from_str(self.tokenizer_json)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode([token for token in ids if token != self.eos_id])

    def to_jsonable(self, *, include_tokenizer_json: bool = False) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "type": self.name,
            "library": "tokenizers",
            "vocab_size": self.vocab_size,
            "eos_token": EOS_TOKEN,
            "eos_id": self.eos_id,
            "checksum": self.checksum,
        }
        if include_tokenizer_json:
            payload["tokenizer_json"] = self.tokenizer_json
        return payload


def train_fast_bpe_tokenizer(
    texts: list[str],
    *,
    vocab_size: int = 8192,
    min_frequency: int = 2,
) -> FastBpeTokenizer:
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[UNK_TOKEN, EOS_TOKEN],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator((text for text in texts if text), trainer=trainer)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_id is None:
        raise ValueError("trained tokenizer is missing EOS token")
    tokenizer_json = tokenizer.to_str()
    checksum = hashlib.sha256(
        json.dumps(
            {
                "type": "hf_tokenizers_bpe_bytelevel",
                "vocab_size": vocab_size,
                "min_frequency": min_frequency,
                "eos_token": EOS_TOKEN,
                "tokenizer_json": json.loads(tokenizer_json),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return FastBpeTokenizer(
        tokenizer_json=tokenizer_json,
        vocab_size=tokenizer.get_vocab_size(),
        eos_id=eos_id,
        checksum=checksum,
    )


def write_fast_bpe_tokenizer(
    path: Path,
    tokenizer: FastBpeTokenizer,
    *,
    training_config: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = tokenizer.to_jsonable(include_tokenizer_json=True)
    payload["training_config"] = training_config or {}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_fast_bpe_tokenizer(path: Path) -> FastBpeTokenizer:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tokenizer_json = str(payload["tokenizer_json"])
    checksum = str(payload["checksum"])
    training_config = payload.get("training_config", {})
    if "vocab_size" in training_config and "min_frequency" in training_config:
        expected = hashlib.sha256(
            json.dumps(
                {
                    "type": "hf_tokenizers_bpe_bytelevel",
                    "vocab_size": training_config["vocab_size"],
                    "min_frequency": training_config["min_frequency"],
                    "eos_token": EOS_TOKEN,
                    "tokenizer_json": json.loads(tokenizer_json),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if checksum != expected:
            raise ValueError(f"tokenizer checksum mismatch for {path}")
    return FastBpeTokenizer(
        tokenizer_json=tokenizer_json,
        vocab_size=int(payload["vocab_size"]),
        eos_id=int(payload["eos_id"]),
        checksum=checksum,
    )
