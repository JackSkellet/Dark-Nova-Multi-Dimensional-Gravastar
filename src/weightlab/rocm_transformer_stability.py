from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from weightlab.lookup import _resolve_torch_accelerator
from weightlab.rocm_validation import _device_properties, _memory_snapshot

Component = Literal["embedding", "norm", "mlp", "attention", "transformer"]
DTypeName = Literal["fp32", "bf16", "fp16"]
OperationStage = Literal["forward", "loss", "backward", "optimizer"]


@dataclass(frozen=True)
class StabilityCase:
    component: Component
    dtype: DTypeName = "fp32"
    mask_mode: str = "none"
    operation_stage: OperationStage = "optimizer"
    batch_size: int = 4
    seq_len: int = 64
    hidden_dim: int | None = None
    heads: int | None = None
    optimizer_foreach: bool = False
    compile_model: bool = False
    sdp_kernel: str = "default"


class EmbeddingProbe(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        return self.head(hidden)


class NormProbe(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(self.embedding(input_ids))
        return self.head(hidden)


class MlpProbe(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        hidden = self.norm(hidden + self.mlp(hidden))
        return self.head(hidden)


class AttentionProbe(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, heads: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.attention = torch.nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        attended, _ = self.attention(hidden, hidden, hidden, attn_mask=mask, need_weights=False)
        return self.head(self.norm(hidden + attended))


class TransformerProbe(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, heads: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.block = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        hidden = self.block(hidden, src_mask=mask, is_causal=mask is not None)
        return self.head(self.norm(hidden))


def run_transformer_stability_microbench(
    device: str = "rocm",
    cases: list[StabilityCase] | None = None,
    steps: int = 3,
    hidden_dim: int = 64,
    heads: int = 4,
    vocab_size: int = 257,
    learning_rate: float = 3e-4,
    seed: int = 123,
) -> dict[str, Any]:
    if cases is None:
        cases = default_stability_cases()

    accelerator = _resolve_torch_accelerator(device)
    torch_device = accelerator.device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)

    case_results = [
        _run_case(
            case=case,
            default_hidden_dim=hidden_dim,
            default_heads=heads,
            vocab_size=vocab_size,
            steps=steps,
            learning_rate=learning_rate,
            device=torch_device,
            generator=generator,
        )
        for case in cases
    ]
    failures = [row for row in case_results if row["status"] != "ok"]
    return {
        "benchmark_label": "rocm_transformer_stability_microbench",
        "requested_device": accelerator.requested_device,
        "device": str(torch_device),
        "accelerator_backend": accelerator.backend,
        "cuda_available": accelerator.cuda_available,
        "rocm_available": accelerator.rocm_available,
        "rocm_runtime_version": accelerator.rocm_runtime_version,
        "torch_version": torch.__version__,
        "device_properties": _device_properties(torch_device),
        "memory": _memory_snapshot(torch_device),
        "case_count": len(case_results),
        "failure_count": len(failures),
        "first_failure": failures[0] if failures else None,
        "case_results": case_results,
        "limitations": [
            "microbench_only",
            "not_full_dense_training",
            "single_process_probe",
            "does_not_measure_kernel_occupancy",
        ],
    }


def default_stability_cases() -> list[StabilityCase]:
    return [
        StabilityCase("embedding", "fp32", "none", "forward", 4, 64),
        StabilityCase("embedding", "fp32", "none", "loss", 4, 64),
        StabilityCase("embedding", "fp32", "none", "backward", 4, 64),
        StabilityCase("embedding", "fp32", "none", "optimizer", 4, 64),
        StabilityCase("norm", "fp32", "none", "optimizer", 4, 64),
        StabilityCase("mlp", "fp32", "none", "optimizer", 4, 64),
        StabilityCase("attention", "fp32", "none", "optimizer", 4, 64),
        StabilityCase("attention", "fp32", "bool_causal", "optimizer", 4, 64),
        StabilityCase("attention", "fp32", "additive_causal", "optimizer", 4, 64),
        StabilityCase("attention", "fp32", "bool_causal", "optimizer", 4, 64, sdp_kernel="math"),
        StabilityCase("transformer", "fp32", "bool_causal", "optimizer", 4, 64),
        StabilityCase("transformer", "fp32", "additive_causal", "optimizer", 4, 64),
        StabilityCase("transformer", "fp16", "bool_causal", "optimizer", 4, 64),
        StabilityCase("transformer", "bf16", "bool_causal", "optimizer", 4, 64),
        StabilityCase("transformer", "bf16", "bool_causal", "optimizer", 4, 64, sdp_kernel="math"),
        StabilityCase("transformer", "fp32", "bool_causal", "optimizer", 8, 128, 128, 4),
        StabilityCase("transformer", "bf16", "bool_causal", "optimizer", 8, 128, 128, 4),
    ]


def _run_case(
    case: StabilityCase,
    default_hidden_dim: int,
    default_heads: int,
    vocab_size: int,
    steps: int,
    learning_rate: float,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, Any]:
    case_hidden_dim = int(case.hidden_dim or default_hidden_dim)
    case_heads = int(case.heads or default_heads)
    base = {
        **asdict(case),
        "hidden_dim": case_hidden_dim,
        "heads": case_heads,
        "steps": int(steps),
        "tokens_per_step": int(case.batch_size * case.seq_len),
    }
    try:
        if case.batch_size <= 0 or case.seq_len <= 1:
            raise ValueError("batch_size_must_be_positive_and_seq_len_must_exceed_1")
        if case_hidden_dim % case_heads != 0:
            raise ValueError("hidden_dim_must_be_divisible_by_heads")

        dtype = _dtype_from_name(case.dtype)
        use_autocast = device.type == "cuda" and dtype is not None
        model = _build_model(case.component, vocab_size, case_hidden_dim, case_heads).to(device)
        if case.compile_model:
            model = torch.compile(model)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            foreach=case.optimizer_foreach,
        )
        mask = _build_mask(case.mask_mode, case.seq_len - 1, device)
        start = time.perf_counter()
        loss_value = 0.0
        for _ in range(steps):
            batch = torch.randint(
                0,
                vocab_size,
                (case.batch_size, case.seq_len),
                generator=generator,
            ).to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            optimizer.zero_grad(set_to_none=True)
            with (
                _sdp_kernel_context(case.sdp_kernel, device),
                torch.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast),
            ):
                logits = _forward_model(model, case.component, inputs, mask)
                if case.operation_stage == "forward":
                    loss = logits.float().square().mean()
                else:
                    loss = torch.nn.functional.cross_entropy(
                        logits.float().reshape(-1, vocab_size),
                        targets.reshape(-1),
                    )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite_loss")
            if case.operation_stage in {"backward", "optimizer"}:
                loss.backward()
            if case.operation_stage == "optimizer":
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            loss_value = float(loss.detach().cpu())
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = max(time.perf_counter() - start, 1e-9)
        return {
            **base,
            "status": "ok",
            "error": "",
            "finite_loss": True,
            "loss": loss_value,
            "elapsed_s": elapsed,
            "tokens_per_second": float(case.batch_size * case.seq_len * steps / elapsed),
            "parameters": int(sum(param.numel() for param in model.parameters())),
        }
    except Exception as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            **base,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "finite_loss": False,
            "loss": 0.0,
            "elapsed_s": 0.0,
            "tokens_per_second": 0.0,
            "parameters": 0,
        }


def _build_model(
    component: str,
    vocab_size: int,
    hidden_dim: int,
    heads: int,
) -> torch.nn.Module:
    if component == "embedding":
        return EmbeddingProbe(vocab_size, hidden_dim)
    if component == "norm":
        return NormProbe(vocab_size, hidden_dim)
    if component == "mlp":
        return MlpProbe(vocab_size, hidden_dim)
    if component == "attention":
        return AttentionProbe(vocab_size, hidden_dim, heads)
    if component == "transformer":
        return TransformerProbe(vocab_size, hidden_dim, heads)
    raise ValueError(f"unsupported_component: {component}")


def _forward_model(
    model: torch.nn.Module,
    component: str,
    inputs: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if component in {"attention", "transformer"}:
        return model(inputs, mask)
    return model(inputs)


def _build_mask(mask_mode: str, seq_len: int, device: torch.device) -> torch.Tensor | None:
    if mask_mode == "none":
        return None
    if mask_mode == "bool_causal":
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )
    if mask_mode == "additive_causal":
        mask = torch.zeros(seq_len, seq_len, dtype=torch.float32, device=device)
        return mask.masked_fill(
            torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1),
            float("-inf"),
        )
    raise ValueError(f"unsupported_mask_mode: {mask_mode}")


@contextmanager
def _sdp_kernel_context(sdp_kernel: str, device: torch.device) -> Iterator[None]:
    if device.type != "cuda" or sdp_kernel == "default":
        with nullcontext():
            yield
        return
    if sdp_kernel != "math":
        raise ValueError(f"unsupported_sdp_kernel: {sdp_kernel}")
    backends = getattr(torch.backends, "cuda", None)
    if backends is None:
        with nullcontext():
            yield
        return
    flash_enabled = backends.flash_sdp_enabled()
    mem_efficient_enabled = backends.mem_efficient_sdp_enabled()
    math_enabled = backends.math_sdp_enabled()
    try:
        backends.enable_flash_sdp(False)
        backends.enable_mem_efficient_sdp(False)
        backends.enable_math_sdp(True)
        yield
    finally:
        backends.enable_flash_sdp(flash_enabled)
        backends.enable_mem_efficient_sdp(mem_efficient_enabled)
        backends.enable_math_sdp(math_enabled)


def _dtype_from_name(dtype: str) -> torch.dtype | None:
    if dtype == "fp32":
        return None
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    raise ValueError(f"unsupported_dtype: {dtype}")
