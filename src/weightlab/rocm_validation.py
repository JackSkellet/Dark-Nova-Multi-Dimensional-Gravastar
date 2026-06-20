from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from weightlab.lookup import _resolve_torch_accelerator


class TinyDecoder(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(self.embedding(input_ids))
        return self.head(hidden)


def validate_training_runtime(
    device: str = "rocm",
    batch_sizes: list[int] | None = None,
    seq_len: int = 128,
    hidden_dim: int = 256,
    vocab_size: int = 4096,
    steps_per_batch: int = 3,
    checkpoint_path: Path | str = Path("artifacts/rocm_validation/resume.pt"),
    seed: int = 123,
) -> dict[str, Any]:
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32]

    accelerator = _resolve_torch_accelerator(device)
    torch_device = accelerator.device
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    precision_support = {
        "fp32_forward_backward": _precision_probe(
            torch.float32,
            torch_device,
            vocab_size=min(vocab_size, 256),
            hidden_dim=min(hidden_dim, 64),
            generator=generator,
        ),
        "bf16_forward_backward": _precision_probe(
            torch.bfloat16,
            torch_device,
            vocab_size=min(vocab_size, 256),
            hidden_dim=min(hidden_dim, 64),
            generator=generator,
        ),
        "fp16_forward_backward": _precision_probe(
            torch.float16,
            torch_device,
            vocab_size=min(vocab_size, 256),
            hidden_dim=min(hidden_dim, 64),
            generator=generator,
        ),
    }

    batch_results: list[dict[str, Any]] = []
    stable_batch_size = 0
    max_stable_tokens = 0
    for batch_size in batch_sizes:
        row = _run_training_batch_probe(
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            steps=steps_per_batch,
            device=torch_device,
            generator=generator,
        )
        batch_results.append(row)
        if row["status"] == "ok":
            stable_batch_size = int(row["batch_size"])
            max_stable_tokens = int(row["tokens_per_step"])

    checkpoint_resume = _checkpoint_resume_probe(
        checkpoint,
        torch_device,
        vocab_size=min(vocab_size, 512),
        hidden_dim=min(hidden_dim, 128),
        generator=generator,
    )

    return {
        "benchmark_label": "training_runtime_validation",
        "requested_device": accelerator.requested_device,
        "device": str(torch_device),
        "accelerator_backend": accelerator.backend,
        "cuda_available": accelerator.cuda_available,
        "rocm_available": accelerator.rocm_available,
        "rocm_runtime_version": accelerator.rocm_runtime_version,
        "torch_version": torch.__version__,
        "device_properties": _device_properties(torch_device),
        "memory": _memory_snapshot(torch_device),
        "precision_support": precision_support,
        "batch_results": batch_results,
        "stable_batch_size": stable_batch_size,
        "max_stable_tokens": max_stable_tokens,
        "checkpoint_resume": checkpoint_resume,
        "limitations": [
            "tiny_decoder_training_probe_only",
            "not_full_model_training",
            "not_occupancy_measurement",
            "throughput_depends_on_python_training_loop",
        ],
    }


def _precision_probe(
    dtype: torch.dtype,
    device: torch.device,
    vocab_size: int,
    hidden_dim: int,
    generator: torch.Generator,
) -> bool:
    try:
        model = TinyDecoder(vocab_size, hidden_dim).to(device)
        if dtype != torch.float32:
            model = model.to(dtype=dtype)
        input_ids = torch.randint(0, vocab_size, (2, 8), generator=generator).to(device)
        target = torch.randint(0, vocab_size, (2, 8), generator=generator).to(device)
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, vocab_size),
            target.reshape(-1),
        )
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return bool(torch.isfinite(loss.detach().cpu()))
    except Exception:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return False


def _run_training_batch_probe(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    vocab_size: int,
    steps: int,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, Any]:
    if batch_size <= 0:
        return {
            "batch_size": int(batch_size),
            "tokens_per_step": 0,
            "status": "failed",
            "error": "batch_size_must_be_positive",
            "tokens_per_second": 0.0,
            "loss": 0.0,
        }

    try:
        model = TinyDecoder(vocab_size, hidden_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        start = time.perf_counter()
        loss_value = 0.0
        for _ in range(steps):
            input_ids = torch.randint(
                0,
                vocab_size,
                (batch_size, seq_len),
                generator=generator,
            ).to(device)
            target = torch.randint(
                0,
                vocab_size,
                (batch_size, seq_len),
                generator=generator,
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size),
                target.reshape(-1),
            )
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = max(time.perf_counter() - start, 1e-9)
        tokens = batch_size * seq_len * steps
        return {
            "batch_size": int(batch_size),
            "tokens_per_step": int(batch_size * seq_len),
            "status": "ok",
            "error": "",
            "steps": int(steps),
            "elapsed_s": elapsed,
            "tokens_per_second": float(tokens / elapsed),
            "loss": loss_value,
        }
    except Exception as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "batch_size": int(batch_size),
            "tokens_per_step": int(max(batch_size, 0) * seq_len),
            "status": "failed",
            "error": type(exc).__name__,
            "tokens_per_second": 0.0,
            "loss": 0.0,
        }


def _checkpoint_resume_probe(
    checkpoint_path: Path,
    device: torch.device,
    vocab_size: int,
    hidden_dim: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    model = TinyDecoder(vocab_size, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    row = _run_training_batch_probe(
        batch_size=1,
        seq_len=8,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        steps=1,
        device=device,
        generator=generator,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": 1,
        },
        checkpoint_path,
    )
    resumed_model = TinyDecoder(vocab_size, hidden_dim).to(device)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    payload = torch.load(checkpoint_path, map_location=device)
    resumed_model.load_state_dict(payload["model"])
    resumed_optimizer.load_state_dict(payload["optimizer"])
    return {
        "path": str(checkpoint_path),
        "resume_ok": int(payload.get("step", 0)) == 1 and row["status"] == "ok",
        "step": int(payload.get("step", 0)),
        "bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else 0,
    }


def _device_properties(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"name": "cpu"}
    props = torch.cuda.get_device_properties(device)
    return {
        "name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "multi_processor_count": int(props.multi_processor_count),
    }


def _memory_snapshot(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"total_bytes": 0, "free_bytes": 0, "allocated_bytes": 0}
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "total_bytes": int(total_bytes),
        "free_bytes": int(free_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }
