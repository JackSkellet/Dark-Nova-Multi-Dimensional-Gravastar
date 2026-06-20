from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

EOS_ID = 256


@dataclass(frozen=True)
class BytePairTokenizer:
    merges: tuple[tuple[int, int, int], ...]
    vocab_size: int

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8", errors="ignore")) + [EOS_ID]
        for left, right, merged in self.merges:
            tokens = _apply_merge(tokens, left, right, merged)
        return tokens

    def checksum(self) -> str:
        payload = {
            "type": "byte_pair",
            "eos_id": EOS_ID,
            "vocab_size": self.vocab_size,
            "merges": self.merges,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def train_byte_pair_tokenizer(
    texts: list[str],
    *,
    target_vocab_size: int = 1024,
    max_texts: int | None = None,
) -> BytePairTokenizer:
    if target_vocab_size <= EOS_ID + 1:
        return BytePairTokenizer(merges=(), vocab_size=EOS_ID + 1)
    sequences = [
        list(text.encode("utf-8", errors="ignore")) + [EOS_ID]
        for text in texts[:max_texts]
        if text
    ]
    merges: list[tuple[int, int, int]] = []
    next_id = EOS_ID + 1
    while next_id < target_vocab_size:
        pair_counts = _pair_counts(sequences)
        if not pair_counts:
            break
        (left, right), count = pair_counts.most_common(1)[0]
        if count < 2:
            break
        sequences = [_apply_merge(sequence, left, right, next_id) for sequence in sequences]
        merges.append((left, right, next_id))
        next_id += 1
    return BytePairTokenizer(merges=tuple(merges), vocab_size=next_id)


def compare_tokenizers(
    train_texts: list[str],
    eval_splits: dict[str, list[str]],
    *,
    target_vocab_size: int = 1024,
    max_train_texts: int | None = None,
) -> dict[str, Any]:
    tokenizer = train_byte_pair_tokenizer(
        train_texts,
        target_vocab_size=target_vocab_size,
        max_texts=max_train_texts,
    )
    byte_metrics = {
        split: _token_count_metrics(texts, _byte_encode)
        for split, texts in eval_splits.items()
    }
    bpe_metrics = {
        split: _token_count_metrics(texts, tokenizer.encode)
        for split, texts in eval_splits.items()
    }
    return {
        "benchmark_label": "tokenizer_efficiency_comparison",
        "train_document_count": len(train_texts),
        "max_train_texts": max_train_texts,
        "tokenizers": {
            "byte_level": {
                "type": "byte_level_utf8_plus_eos",
                "vocab_size": EOS_ID + 1,
                "eos_id": EOS_ID,
            },
            "byte_pair": {
                "type": "byte_pair_on_utf8_bytes",
                "target_vocab_size": target_vocab_size,
                "vocab_size": tokenizer.vocab_size,
                "merge_count": len(tokenizer.merges),
                "checksum": tokenizer.checksum(),
            },
        },
        "splits": {
            split: {
                "byte_level": byte_metrics[split],
                "byte_pair": bpe_metrics[split],
                "token_reduction_ratio": _ratio(
                    byte_metrics[split]["token_count"],
                    bpe_metrics[split]["token_count"],
                ),
            }
            for split in eval_splits
        },
        "limitations": [
            "token_efficiency_only",
            "no_model_training_with_bpe_yet",
            "simple_byte_pair_trainer_not_sentencepiece_unigram",
        ],
    }


def _byte_encode(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="ignore")) + [EOS_ID]


def _pair_counts(sequences: list[list[int]]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for sequence in sequences:
        for left, right in zip(sequence, sequence[1:], strict=False):
            if left == EOS_ID or right == EOS_ID:
                continue
            counts[(left, right)] += 1
    return counts


def _apply_merge(sequence: list[int], left: int, right: int, merged: int) -> list[int]:
    output: list[int] = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and sequence[index] == left and sequence[index + 1] == right:
            output.append(merged)
            index += 2
        else:
            output.append(sequence[index])
            index += 1
    return output


def _token_count_metrics(texts: list[str], encode) -> dict[str, Any]:
    token_count = 0
    byte_count = 0
    for text in texts:
        token_count += len(encode(text))
        byte_count += len(text.encode("utf-8", errors="ignore"))
    return {
        "document_count": len(texts),
        "byte_count": byte_count,
        "token_count": token_count,
        "bytes_per_token": _ratio(byte_count, token_count),
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
