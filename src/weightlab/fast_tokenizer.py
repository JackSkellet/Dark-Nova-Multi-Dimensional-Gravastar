from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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

    def encode(self, text: str) -> list[int]:
        tokenizer = Tokenizer.from_str(self.tokenizer_json)
        return tokenizer.encode(text).ids + [self.eos_id]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": "hf_tokenizers_bpe_bytelevel",
            "library": "tokenizers",
            "vocab_size": self.vocab_size,
            "eos_token": EOS_TOKEN,
            "eos_id": self.eos_id,
            "checksum": self.checksum,
        }


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
