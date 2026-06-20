from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from weightlab.tokenizer_compare import compare_tokenizers, train_byte_pair_tokenizer


def test_byte_pair_tokenizer_reduces_repeated_code_tokens():
    texts = [
        "function parseConfig(value) { return value.trim(); }\n",
        "function renderConfig(value) { return value.trim(); }\n",
        "function parseConfig(input) { return input.trim(); }\n",
    ]

    tokenizer = train_byte_pair_tokenizer(texts, target_vocab_size=280)
    comparison = compare_tokenizers(
        texts,
        {"train": texts},
        target_vocab_size=280,
    )

    assert tokenizer.vocab_size > 257
    assert len(tokenizer.checksum()) == 64
    assert comparison["tokenizers"]["byte_pair"]["merge_count"] > 0
    assert comparison["splits"]["train"]["token_reduction_ratio"] > 1.0


def test_compare_tokenizers_reports_throughput_and_context_coverage():
    texts = [
        "function parseConfig(value) { return value.trim(); }\n" * 20,
        "function renderConfig(value) { return value.trim(); }\n" * 20,
        "function parseConfig(input) { return input.trim(); }\n" * 20,
    ]

    comparison = compare_tokenizers(
        texts,
        {"validation": texts},
        target_vocab_size=300,
        context_lengths=(64, 128),
    )
    split = comparison["splits"]["validation"]

    assert split["byte_level"]["throughput"]["bytes_per_second"] > 0
    assert split["byte_pair"]["throughput"]["tokens_per_second"] > 0
    assert split["byte_level"]["context_coverage"]["64"]["mean_fraction_bytes_covered"] > 0
    assert (
        split["byte_pair"]["context_coverage"]["64"]["mean_fraction_bytes_covered"]
        > split["byte_level"]["context_coverage"]["64"]["mean_fraction_bytes_covered"]
    )


def test_compare_tokenizers_cli_writes_machine_readable_record(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        "\n".join(
            [
                json.dumps({
                    "text": "function alpha(value) { return value + 1; }",
                    "split": "train",
                }),
                json.dumps({
                    "text": "function beta(value) { return value + 2; }",
                    "split": "train",
                }),
                json.dumps({
                    "text": "function validate(value) { return value > 0; }",
                    "split": "validation",
                }),
                json.dumps({
                    "text": "function testValue(value) { return value === 1; }",
                    "split": "test",
                }),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "tokenizers.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/compare_tokenizers.py",
            "--corpus-jsonl",
            str(corpus_path),
            "--target-vocab-size",
            "280",
            "--max-train-texts",
            "2",
            "--output",
            str(output_path),
            "--experiment-id",
            "tokenizer_cli_test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "tokenizer_cli_test"
    assert record["metrics"]["benchmark_label"] == "tokenizer_efficiency_comparison"
    assert record["metrics"]["tokenizers"]["byte_pair"]["vocab_size"] > 257
    assert record["metrics"]["splits"]["validation"]["byte_pair"]["token_count"] > 0
    assert "--target-vocab-size 280" in record["command"]
