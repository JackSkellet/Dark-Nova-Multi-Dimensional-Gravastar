from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_COMPARISON_CONFIG = Path("configs/d4_adamw_fp32_matched_comparison.json")


def load_comparison_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_runs(config: dict[str, Any], labels: list[str]) -> list[dict[str, Any]]:
    runs = list(config["runs"])
    if not labels or labels == ["all"]:
        return runs
    wanted = set(labels)
    selected = [run for run in runs if run["label"] in wanted]
    missing = wanted - {run["label"] for run in selected}
    if missing:
        raise ValueError(f"unknown run label(s): {', '.join(sorted(missing))}")
    return selected


def build_train_command(config: dict[str, Any], run: dict[str, Any]) -> list[str]:
    protocol = config["training_protocol"]
    corpus = config["corpus"]
    return [
        sys.executable,
        "scripts/train_dense_decoder.py",
        "--corpus-jsonl",
        corpus["jsonl_path"],
        "--corpus-record",
        corpus["record"],
        "--device",
        protocol["device"],
        "--seq-len",
        str(protocol["seq_len"]),
        "--hidden-dim",
        str(run["hidden_dim"]),
        "--layers",
        str(protocol["layers"]),
        "--heads",
        str(protocol["heads"]),
        "--batch-size",
        str(protocol["batch_size"]),
        "--steps",
        str(protocol["steps"]),
        "--validation-batches",
        str(protocol["checkpoint_validation_batches"]),
        "--gradient-accumulation-steps",
        str(protocol["gradient_accumulation_steps"]),
        "--mixed-precision",
        protocol["mixed_precision"],
        "--learning-rate",
        str(protocol["learning_rate"]),
        "--optimizer-name",
        protocol["optimizer_name"],
        "--attention-mask-mode",
        protocol["attention_mask_mode"],
        "--block-impl",
        protocol["block_impl"],
        "--progress-interval",
        str(protocol["progress_interval"]),
        "--checkpoint-interval",
        str(protocol["checkpoint_interval"]),
        "--architecture-variant",
        run["architecture_variant"],
        "--adapter-dim",
        str(run["adapter_dim"]),
        "--validation-seed",
        str(config["data_order"]["validation_seed"]),
        "--max-documents",
        str(protocol["max_documents"]),
        "--output-dir",
        run["output_dir"],
        "--output",
        run["output"],
        "--experiment-id",
        run["experiment_id"],
        "--seed",
        str(config["data_order"]["train_seed"]),
    ]


def build_eval_commands(config: dict[str, Any], run: dict[str, Any]) -> list[list[str]]:
    protocol = config["training_protocol"]
    corpus = config["corpus"]
    output_dir = Path(run["output_dir"])
    commands: list[list[str]] = []
    for checkpoint_label, checkpoint_name in [
        ("final", "dense_decoder_last.pt"),
        ("best", "dense_decoder_best.pt"),
    ]:
        for split, seed in [
            ("validation", config["data_order"]["validation_seed"]),
            ("test", config["data_order"]["test_seed"]),
        ]:
            commands.append(
                [
                    sys.executable,
                    "scripts/evaluate_dense_checkpoint.py",
                    "--checkpoint",
                    str(output_dir / checkpoint_name),
                    "--corpus-jsonl",
                    corpus["jsonl_path"],
                    "--split",
                    split,
                    "--device",
                    protocol["device"],
                    "--seed",
                    str(seed),
                    "--batches",
                    str(protocol["final_eval_batches"]),
                    "--output",
                    f"results/{run['experiment_id']}_{checkpoint_label}_{split}_eval.json",
                    "--experiment-id",
                    f"{run['experiment_id']}_{checkpoint_label}_{split}_eval",
                ]
            )
    return commands
