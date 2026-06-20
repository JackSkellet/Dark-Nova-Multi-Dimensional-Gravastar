from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_train_dense_decoder_cli_accepts_attention_mask_mode(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        "\n".join(
            [
                json.dumps({"text": "def add(a, b): return a + b"}),
                json.dumps({"text": "README: add returns the sum of two values."}),
                json.dumps({"text": "assert add(1, 2) == 3"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "result.json"
    output_dir = tmp_path / "artifacts"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_dense_decoder.py",
            "--corpus-jsonl",
            str(corpus_path),
            "--device",
            "cpu",
            "--seq-len",
            "16",
            "--hidden-dim",
            "32",
            "--layers",
            "1",
            "--heads",
            "4",
            "--batch-size",
            "2",
            "--steps",
            "1",
            "--validation-batches",
            "1",
            "--mixed-precision",
            "fp32",
            "--optimizer-name",
            "sgd",
            "--learning-rate",
            "0.0",
            "--attention-mask-mode",
            "bool_causal",
            "--max-documents",
            "0",
            "--output-dir",
            str(output_dir),
            "--output",
            str(output_path),
            "--experiment-id",
            "cli_attention_mask_mode",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["metrics"]["model"]["config"]["attention_mask_mode"] == "bool_causal"
