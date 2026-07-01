from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from weightlab.metrics import write_json
from weightlab.quixbugs_model_candidates import _load_dense_checkpoint


def _load_tasks(path: Path, max_tasks: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= max_tasks:
                break
    return rows


def _generate_ids(
    model: torch.nn.Module,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    eos_id: int,
) -> list[int]:
    device = next(model.parameters()).device
    ids = list(prompt_ids)
    generated: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            window = ids[-model.seq_len :]
            input_ids = torch.tensor([window], dtype=torch.long, device=device)
            logits = model(input_ids)
            next_id = int(torch.argmax(logits[0, -1]).detach().cpu())
            if next_id == eos_id:
                break
            ids.append(next_id)
            generated.append(next_id)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate saved local dense outputs for the GLM public harness."
    )
    parser.add_argument("--tasks-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="rocm")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-tasks", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-label", default="T11c_dense528_adamw_fp32_50m")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer, model_metadata = _load_dense_checkpoint(args.checkpoint, args.device)
    tasks = _load_tasks(args.tasks_jsonl, args.max_tasks)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        prompt = str(task["prompt"])
        prompt_ids = tokenizer.encode(prompt)[:-1]
        started = time.perf_counter()
        generated_ids = _generate_ids(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            eos_id=int(tokenizer.eos_id),
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        rows.append(
            {
                "task_id": str(task["task_id"]),
                "model": args.model_label,
                "prompt": prompt,
                "output": tokenizer.decode(generated_ids),
                "context_length": len(prompt_ids),
                "output_length": len(generated_ids),
                "thinking_effort": "none",
                "tool_access": "none",
                "temperature": 0.0,
                "sampling": {
                    "strategy": "greedy",
                    "max_new_tokens": args.max_new_tokens,
                },
                "latency_ms": latency_ms,
                "tokens_in": len(prompt_ids),
                "tokens_out": len(generated_ids),
                "cost_usd": 0.0,
                "checkpoint": str(args.checkpoint),
                "model_metadata": model_metadata,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    metadata_path = args.output.with_suffix(".metadata.json")
    write_json(
        metadata_path,
        {
            "tasks_jsonl": str(args.tasks_jsonl),
            "checkpoint": str(args.checkpoint),
            "device": args.device,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "max_tasks": args.max_tasks,
            "output": str(args.output),
            "model_label": args.model_label,
            "prediction_count": len(rows),
            "model_metadata": model_metadata,
        },
    )


if __name__ == "__main__":
    main()
