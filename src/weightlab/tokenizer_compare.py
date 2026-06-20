from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable
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
    context_lengths: tuple[int, ...] = (128, 512, 1024),
) -> dict[str, Any]:
    tokenizer = train_byte_pair_tokenizer(
        train_texts,
        target_vocab_size=target_vocab_size,
        max_texts=max_train_texts,
    )
    byte_metrics = {
        split: _tokenizer_eval_metrics(texts, _byte_encode, context_lengths)
        for split, texts in eval_splits.items()
    }
    bpe_metrics = {
        split: _tokenizer_eval_metrics(texts, tokenizer.encode, context_lengths)
        for split, texts in eval_splits.items()
    }
    return {
        "benchmark_label": "tokenizer_efficiency_comparison",
        "train_document_count": len(train_texts),
        "max_train_texts": max_train_texts,
        "context_lengths": list(context_lengths),
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


def _tokenizer_eval_metrics(
    texts: list[str],
    encode: Callable[[str], list[int]],
    context_lengths: tuple[int, ...],
) -> dict[str, Any]:
    encoded_lengths: list[int] = []
    byte_lengths: list[int] = []
    start = time.perf_counter()
    for text in texts:
        encoded_lengths.append(len(encode(text)))
        byte_lengths.append(len(text.encode("utf-8", errors="ignore")))
    elapsed_s = max(time.perf_counter() - start, 1e-12)
    token_count = sum(encoded_lengths)
    byte_count = sum(byte_lengths)
    return {
        "document_count": len(texts),
        "byte_count": byte_count,
        "token_count": token_count,
        "bytes_per_token": _ratio(byte_count, token_count),
        "throughput": {
            "elapsed_s": elapsed_s,
            "documents_per_second": _ratio(len(texts), elapsed_s),
            "bytes_per_second": _ratio(byte_count, elapsed_s),
            "tokens_per_second": _ratio(token_count, elapsed_s),
        },
        "context_coverage": {
            str(context_len): _context_coverage(
                byte_lengths,
                encoded_lengths,
                context_len,
            )
            for context_len in context_lengths
        },
    }


def _context_coverage(
    byte_lengths: list[int],
    token_lengths: list[int],
    context_len: int,
) -> dict[str, Any]:
    if not byte_lengths:
        return {
            "context_tokens": context_len,
            "full_document_fit_count": 0,
            "full_document_fit_ratio": 0.0,
            "mean_bytes_covered": 0.0,
            "mean_fraction_bytes_covered": 0.0,
        }
    fit_count = sum(1 for token_len in token_lengths if token_len <= context_len)
    bytes_covered = [
        byte_len * min(1.0, _ratio(context_len, token_len))
        for byte_len, token_len in zip(byte_lengths, token_lengths, strict=True)
    ]
    fractions = [
        min(1.0, _ratio(context_len, token_len))
        for token_len in token_lengths
    ]
    return {
        "context_tokens": context_len,
        "full_document_fit_count": fit_count,
        "full_document_fit_ratio": _ratio(fit_count, len(token_lengths)),
        "mean_bytes_covered": _ratio(sum(bytes_covered), len(bytes_covered)),
        "mean_fraction_bytes_covered": _ratio(sum(fractions), len(fractions)),
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
