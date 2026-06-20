from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from weightlab.lookup import _resolve_torch_accelerator


@dataclass(frozen=True)
class DenseTrainingConfig:
    device: str = "rocm"
    seq_len: int = 128
    hidden_dim: int = 256
    layers: int = 4
    heads: int = 4
    batch_size: int = 8
    steps: int = 100
    validation_batches: int = 4
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    learning_rate: float = 3e-4
    attention_mask_mode: str = "additive_causal"
    optimizer_name: str = "adamw"
    progress_interval: int = 0
    checkpoint_interval: int = 0
    architecture_variant: str = "dense"
    adapter_dim: int = 0
    validation_seed: int = 424242
    block_impl: str = "torch_encoder"


class ByteTokenizer:
    name = "byte_level"
    vocab_size = 257
    eos_id = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="ignore")) + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        byte_values = bytes(token for token in ids if 0 <= token < 256)
        return byte_values.decode("utf-8", errors="ignore")


class DenseDecoder(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        attention_mask_mode: str = "additive_causal",
        architecture_variant: str = "dense",
        adapter_dim: int = 0,
        block_impl: str = "torch_encoder",
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.attention_mask_mode = attention_mask_mode
        self.architecture_variant = architecture_variant
        self.block_impl = block_impl
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = torch.nn.Embedding(seq_len, hidden_dim)
        if block_impl == "torch_encoder":
            self.blocks = torch.nn.ModuleList(
                [
                    torch.nn.TransformerEncoderLayer(
                        d_model=hidden_dim,
                        nhead=heads,
                        dim_feedforward=hidden_dim * 4,
                        dropout=0.0,
                        batch_first=True,
                        activation="gelu",
                    )
                    for _ in range(layers)
                ]
            )
        elif block_impl == "explicit_causal":
            self.blocks = torch.nn.ModuleList(
                [_ExplicitCausalTransformerLayer(hidden_dim, heads) for _ in range(layers)]
            )
        else:
            raise ValueError(f"unknown block_impl: {block_impl}")
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, vocab_size)
        if architecture_variant == "dense":
            if adapter_dim != 0:
                raise ValueError("dense architecture requires adapter_dim=0")
            self.adapters = torch.nn.ModuleList([torch.nn.Identity() for _ in range(layers)])
        elif architecture_variant == "adapter":
            if adapter_dim <= 0:
                raise ValueError("adapter architecture requires adapter_dim > 0")
            self.adapters = torch.nn.ModuleList(
                [_ResidualAdapter(hidden_dim, adapter_dim) for _ in range(layers)]
            )
        else:
            raise ValueError(f"unknown architecture_variant: {architecture_variant}")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        mask = _causal_mask(self.attention_mask_mode, seq_len, input_ids.device)
        for block, adapter in zip(self.blocks, self.adapters, strict=True):
            hidden = _run_transformer_block(
                block,
                hidden,
                mask,
                device_type=input_ids.device.type,
            )
            hidden = adapter(hidden)
        return self.head(self.norm(hidden))


class _ResidualAdapter(torch.nn.Module):
    def __init__(self, hidden_dim: int, adapter_dim: int) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.down = torch.nn.Linear(hidden_dim, adapter_dim)
        self.up = torch.nn.Linear(adapter_dim, hidden_dim)
        self.activation = torch.nn.GELU()
        torch.nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.down.bias)
        torch.nn.init.zeros_(self.up.weight)
        torch.nn.init.zeros_(self.up.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.up(self.activation(self.down(self.norm(hidden))))


class _ExplicitCausalTransformerLayer(torch.nn.Module):
    def __init__(self, hidden_dim: int, heads: int) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.in_proj = torch.nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = torch.nn.Linear(hidden_dim * 4, hidden_dim)
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)
        self.activation = torch.nn.GELU()

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        del is_causal
        batch, seq_len, _hidden = src.shape
        qkv = self.in_proj(src).view(batch, seq_len, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        if src_mask is not None:
            if src_mask.dtype == torch.bool:
                scores = scores.masked_fill(
                    src_mask.view(1, 1, seq_len, seq_len),
                    float("-inf"),
                )
            else:
                scores = scores + src_mask.view(1, 1, seq_len, seq_len).float()
        attention = torch.softmax(scores, dim=-1).to(value.dtype)
        context = torch.matmul(attention, value)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_dim)
        src = self.norm1(src + self.out_proj(context))
        feedforward = self.linear2(self.activation(self.linear1(src)))
        return self.norm2(src + feedforward)


def train_dense_decoder(
    texts: list[str],
    config: DenseTrainingConfig,
    output_dir: Path,
    seed: int = 123,
    resume_checkpoint: Path | None = None,
    validation_texts: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteTokenizer()
    tokens = _tokens_from_texts(texts, tokenizer)
    if len(tokens) < config.seq_len + 2:
        raise ValueError("not enough tokens for dense decoder training")
    validation_source = "training_texts"
    validation_tokens_tensor = tokens
    if validation_texts is not None:
        validation_tokens_tensor = _tokens_from_texts(validation_texts, tokenizer)
        validation_source = "provided_validation_texts"
        if len(validation_tokens_tensor) < config.seq_len + 2:
            raise ValueError("not enough tokens for dense decoder validation")

    accelerator = _resolve_torch_accelerator(config.device)
    device = accelerator.device
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    _reset_peak_memory_stats(device)
    model = DenseDecoder(
        tokenizer.vocab_size,
        config.seq_len,
        config.hidden_dim,
        config.layers,
        config.heads,
        config.attention_mask_mode,
        config.architecture_variant,
        config.adapter_dim,
        config.block_impl,
    ).to(device)
    optimizer = _make_optimizer(model, config)
    dtype = _autocast_dtype(config.mixed_precision)
    use_autocast = device.type == "cuda" and dtype is not None
    start_step = 0
    resumed_from = ""
    if resume_checkpoint is not None:
        payload = torch.load(resume_checkpoint, map_location=device)
        _load_model_state(model, payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if "generator_state" in payload:
            generator.set_state(payload["generator_state"].cpu())
        start_step = int(payload.get("step", 0))
        resumed_from = str(resume_checkpoint)

    loss_curve: list[dict[str, float]] = []
    progress_path = output_dir / "training_progress.jsonl"
    if config.progress_interval > 0 and progress_path.exists() and resume_checkpoint is None:
        progress_path.unlink()
    latest_checkpoint_path = output_dir / "dense_decoder_latest.pt"
    best_checkpoint_path = output_dir / "dense_decoder_best.pt"
    best_model_only_path = output_dir / "dense_decoder_best_model_only.pt"
    checkpoint_flushes: list[dict[str, Any]] = []
    checkpoint_validations: list[dict[str, Any]] = []
    best_checkpoint: dict[str, Any] = {}
    gradient_norms: list[dict[str, Any]] = []
    train_start = time.perf_counter()
    status = "completed"
    failure = ""
    failure_step = 0
    last_completed_step = start_step
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, config.steps + 1):
        step_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            batch = _sample_batch(tokens, config.batch_size, config.seq_len, generator).to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
                logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, tokenizer.vocab_size),
                    targets.reshape(-1),
                )
                scaled_loss = loss / config.gradient_accumulation_steps
            if not torch.isfinite(loss):
                status = "failed_nonfinite_loss"
                failure = f"nonfinite_loss_at_step_{step}"
                failure_step = step
                break
            scaled_loss.backward()
            step_loss += float(loss.detach().cpu())
        if status != "completed":
            break
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        should_record_grad_norm = (
            _should_flush(step, config.progress_interval)
            or _should_flush(step, config.checkpoint_interval)
            or step == config.steps
        )
        grad_norm = math.nan
        if should_record_grad_norm:
            grad_norm = float(grad_norm_tensor.detach().cpu())
            gradient_norms.append(
                {
                    "step": step,
                    "global_norm_before_clip": grad_norm,
                    "finite": math.isfinite(grad_norm),
                }
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        last_completed_step = step
        loss_curve.append(
            {"step": float(step), "loss": step_loss / config.gradient_accumulation_steps}
        )
        step_loss_value = step_loss / config.gradient_accumulation_steps
        elapsed_so_far = max(time.perf_counter() - train_start, 1e-9)
        if _should_flush(step, config.progress_interval):
            _append_progress(
                progress_path,
                {
                    "step": step,
                    "loss": step_loss_value,
                    "train_tokens": step
                    * config.gradient_accumulation_steps
                    * config.batch_size
                    * config.seq_len,
                    "elapsed_s": elapsed_so_far,
                    "tokens_per_second": (
                        step
                        * config.gradient_accumulation_steps
                        * config.batch_size
                        * config.seq_len
                    )
                    / elapsed_so_far,
                    "gradient_norm_before_clip": grad_norm,
                },
            )
        if _should_flush(step, config.checkpoint_interval):
            checkpoint_validation = _fixed_validation_metrics(
                model,
                validation_tokens_tensor,
                tokenizer.vocab_size,
                config,
                device,
            )
            checkpoint_validation["source"] = validation_source
            checkpoint_validation["heldout_texts_provided"] = validation_texts is not None
            _save_training_checkpoint(
                latest_checkpoint_path,
                model,
                optimizer,
                config,
                step=step,
                latest_loss=step_loss_value,
                generator=generator,
                validation=checkpoint_validation,
            )
            is_best = (
                not best_checkpoint
                or float(checkpoint_validation["loss"])
                < float(best_checkpoint["validation"]["loss"])
            )
            if is_best:
                _save_training_checkpoint(
                    best_checkpoint_path,
                    model,
                    optimizer,
                    config,
                    step=step,
                    latest_loss=step_loss_value,
                    generator=generator,
                    validation=checkpoint_validation,
                )
                _save_model_only_checkpoint(best_model_only_path, model, config, step=step)
                best_checkpoint = {
                    "path": str(best_checkpoint_path),
                    "model_only_path": str(best_model_only_path),
                    "step": step,
                    "bytes": best_checkpoint_path.stat().st_size,
                    "model_only_bytes": best_model_only_path.stat().st_size,
                    "optimizer_state_bytes": (
                        best_checkpoint_path.stat().st_size - best_model_only_path.stat().st_size
                    ),
                    "validation": checkpoint_validation,
                }
            validation_row = {
                "step": step,
                "path": str(latest_checkpoint_path),
                "validation": checkpoint_validation,
                "is_best": is_best,
            }
            checkpoint_validations.append(validation_row)
            checkpoint_flushes.append(
                {
                    "step": step,
                    "path": str(latest_checkpoint_path),
                    "bytes": latest_checkpoint_path.stat().st_size,
                    "validation_loss": checkpoint_validation["loss"],
                    "is_best": is_best,
                }
            )

    elapsed = max(time.perf_counter() - train_start, 1e-9)
    completed_steps_this_invocation = len(loss_curve)
    train_tokens = completed_steps_this_invocation * _tokens_per_step(config)
    planned_train_tokens = (config.steps - start_step) * _tokens_per_step(config)
    total_completed_tokens = last_completed_step * _tokens_per_step(config)
    validation = (
        _validation_metrics(
            model,
            validation_tokens_tensor,
            tokenizer.vocab_size,
            config,
            torch.Generator(device="cpu").manual_seed(config.validation_seed),
            device,
        )
        if status == "completed"
        else {
            "loss": math.nan,
            "batches": 0,
            "tokens": 0,
            "sample_order_sha256": "",
        }
    )
    validation["source"] = validation_source
    validation["heldout_texts_provided"] = validation_texts is not None
    sample = _generate_sample(model, tokenizer, "def ", config.seq_len, device)
    checkpoint_path = output_dir / "dense_decoder_last.pt"
    model_only_checkpoint_path = output_dir / "dense_decoder_last_model_only.pt"
    parameter_count = sum(param.numel() for param in model.parameters())
    trainable_parameter_count = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "step": last_completed_step,
            "generator_state": generator.get_state(),
        },
        checkpoint_path,
    )
    _save_model_only_checkpoint(
        model_only_checkpoint_path,
        model,
        config,
        step=last_completed_step,
    )
    if status == "completed" and not best_checkpoint:
        _save_training_checkpoint(
            best_checkpoint_path,
            model,
            optimizer,
            config,
            step=last_completed_step,
            latest_loss=float(loss_curve[-1]["loss"]) if loss_curve else math.nan,
            generator=generator,
            validation=validation,
        )
        _save_model_only_checkpoint(best_model_only_path, model, config, step=last_completed_step)
        best_checkpoint = {
            "path": str(best_checkpoint_path),
            "model_only_path": str(best_model_only_path),
            "step": last_completed_step,
            "bytes": best_checkpoint_path.stat().st_size,
            "model_only_bytes": best_model_only_path.stat().st_size,
            "optimizer_state_bytes": (
                best_checkpoint_path.stat().st_size - best_model_only_path.stat().st_size
            ),
            "validation": validation,
        }
    resume_ok = _resume_check(
        checkpoint_path,
        tokenizer.vocab_size,
        config,
        device,
        expected_step=last_completed_step,
    )
    return {
        "benchmark_label": "dense_decoder_training_smoke",
        "status": status,
        "failure": failure,
        "accelerator_backend": accelerator.backend,
        "rocm_available": accelerator.rocm_available,
        "rocm_runtime_version": accelerator.rocm_runtime_version,
        "tokenizer": {"name": tokenizer.name, "vocab_size": tokenizer.vocab_size},
        "model": {
            "architecture": "causal_transformer_decoder",
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "active_parameter_count": trainable_parameter_count,
            "config": asdict(config),
        },
        "training": {
            "train_tokens": train_tokens,
            "planned_train_tokens": planned_train_tokens,
            "total_completed_tokens": total_completed_tokens,
            "steps": config.steps,
            "start_step": start_step,
            "last_completed_step": last_completed_step,
            "completed_steps_this_invocation": completed_steps_this_invocation,
            "failure_step": failure_step,
            "resumed_from": resumed_from,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "elapsed_s": elapsed,
            "tokens_per_second": train_tokens / elapsed if train_tokens else 0.0,
            "loss_curve": loss_curve,
            "gradient_norms": {
                "records": gradient_norms,
                "summary": _gradient_norm_summary(gradient_norms),
            },
        },
        "memory": _memory_summary(device),
        "progress": {
            "path": str(progress_path),
            "records": _count_jsonl_records(progress_path),
            "latest": _last_jsonl_record(progress_path),
            "checkpoint_flushes": checkpoint_flushes,
            "checkpoint_validations": checkpoint_validations,
            "latest_checkpoint": {
                "path": str(latest_checkpoint_path),
                "exists": latest_checkpoint_path.exists(),
                "bytes": latest_checkpoint_path.stat().st_size
                if latest_checkpoint_path.exists()
                else 0,
            },
        },
        "validation": validation,
        "adapter": _adapter_activation_stats(
            model,
            validation_tokens_tensor,
            config,
            device,
        ),
        "generation_samples": [{"prompt": "def ", "text": sample}],
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "model_only_path": str(model_only_checkpoint_path),
            "model_only_bytes": model_only_checkpoint_path.stat().st_size,
            "optimizer_state_bytes": (
                checkpoint_path.stat().st_size - model_only_checkpoint_path.stat().st_size
            ),
            "step": last_completed_step,
            "resume_ok": resume_ok,
        },
        "best_checkpoint": best_checkpoint,
        "limitations": [
            "training_smoke_only",
            "byte_tokenizer_baseline",
            "not_50m_token_run",
            "no_functional_coding_evaluation_yet",
        ]
        + (
            ["not_10m_parameter_model"]
            if parameter_count < 10_000_000
            else ["10m_parameter_floor_reached"]
        ),
    }


def debug_dense_step_stability(
    texts: list[str],
    config: DenseTrainingConfig,
    seed: int = 123,
    steps: int = 2,
) -> dict[str, Any]:
    tokenizer = ByteTokenizer()
    tokens = _tokens_from_texts(texts, tokenizer)
    if len(tokens) < config.seq_len + 2:
        raise ValueError("not enough tokens for dense decoder debug probe")

    accelerator = _resolve_torch_accelerator(config.device)
    device = accelerator.device
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = DenseDecoder(
        tokenizer.vocab_size,
        config.seq_len,
        config.hidden_dim,
        config.layers,
        config.heads,
        config.attention_mask_mode,
        config.architecture_variant,
        config.adapter_dim,
        config.block_impl,
    ).to(device)
    optimizer = _make_optimizer(model, config)
    dtype = _autocast_dtype(config.mixed_precision)
    use_autocast = device.type == "cuda" and dtype is not None

    first_nonfinite_phase: str | None = None
    step_results: list[dict[str, Any]] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    initial_parameters = _module_tensor_summary(model.named_parameters())

    for step in range(1, steps + 1):
        batch = _sample_batch(tokens, config.batch_size, config.seq_len, generator).to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        row: dict[str, Any] = {
            "step": step,
            "input_ids": _tensor_stats(inputs),
            "targets": _tensor_stats(targets),
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, tokenizer.vocab_size),
                targets.reshape(-1),
            )
        row["logits"] = _tensor_stats(logits)
        row["loss"] = _tensor_stats(loss)
        if first_nonfinite_phase is None and not row["logits"]["finite"]:
            first_nonfinite_phase = f"step_{step}_logits"
        if first_nonfinite_phase is None and not row["loss"]["finite"]:
            first_nonfinite_phase = f"step_{step}_loss"
        if row["loss"]["finite"]:
            loss.backward()
            row["gradients"] = _module_tensor_summary(
                (name, parameter.grad)
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            )
            if first_nonfinite_phase is None and not row["gradients"]["finite"]:
                first_nonfinite_phase = f"step_{step}_gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            row["parameters_after_clip"] = _module_tensor_summary(model.named_parameters())
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            row["parameters_after_optimizer"] = _module_tensor_summary(model.named_parameters())
            if first_nonfinite_phase is None and not row["parameters_after_optimizer"]["finite"]:
                first_nonfinite_phase = f"step_{step}_parameters_after_optimizer"
        else:
            row["gradients"] = _empty_tensor_summary()
            row["parameters_after_clip"] = _module_tensor_summary(model.named_parameters())
            row["parameters_after_optimizer"] = _module_tensor_summary(model.named_parameters())
        step_results.append(row)
        if first_nonfinite_phase is not None:
            break

    return {
        "benchmark_label": "dense_step_stability_debug",
        "requested_device": accelerator.requested_device,
        "device": str(device),
        "accelerator_backend": accelerator.backend,
        "rocm_available": accelerator.rocm_available,
        "rocm_runtime_version": accelerator.rocm_runtime_version,
        "torch_version": torch.__version__,
        "first_nonfinite_phase": first_nonfinite_phase,
        "initial_parameters": initial_parameters,
        "step_results": step_results,
        "tokenizer": {"name": tokenizer.name, "vocab_size": tokenizer.vocab_size},
        "model": {
            "architecture": "causal_transformer_decoder",
            "parameter_count": sum(param.numel() for param in model.parameters()),
            "config": asdict(config),
        },
        "limitations": [
            "debug_probe_only",
            "short_two_step_window",
            "not_training_quality_evaluation",
        ],
    }


def evaluate_dense_checkpoint(
    checkpoint_path: Path,
    texts: list[str],
    split_name: str,
    device: str = "rocm",
    seed: int = 123,
    batches: int | None = None,
) -> dict[str, Any]:
    tokenizer = ByteTokenizer()
    tokens = _tokens_from_texts(texts, tokenizer)
    accelerator = _resolve_torch_accelerator(device)
    torch_device = accelerator.device
    payload = torch.load(checkpoint_path, map_location=torch_device)
    checkpoint_config = dict(payload["config"])
    if batches is not None:
        checkpoint_config["validation_batches"] = batches
    config = DenseTrainingConfig(**checkpoint_config)
    if len(tokens) < config.seq_len + 2:
        raise ValueError(f"not enough tokens for dense decoder {split_name} evaluation")
    model = DenseDecoder(
        tokenizer.vocab_size,
        config.seq_len,
        config.hidden_dim,
        config.layers,
        config.heads,
        config.attention_mask_mode,
        config.architecture_variant,
        config.adapter_dim,
        config.block_impl,
    ).to(torch_device)
    _load_model_state(model, payload["model"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    metrics = _validation_metrics(
        model,
        tokens,
        tokenizer.vocab_size,
        config,
        generator,
        torch_device,
    )
    return {
        "benchmark_label": "dense_checkpoint_evaluation",
        "split": split_name,
        "seed": seed,
        "checkpoint": {
            "path": str(checkpoint_path),
            "step": int(payload.get("step", 0)),
            "bytes": checkpoint_path.stat().st_size,
        },
        "model": {
            "architecture": "causal_transformer_decoder",
            "parameter_count": sum(param.numel() for param in model.parameters()),
            "config": asdict(config),
        },
        "tokenizer": {"name": tokenizer.name, "vocab_size": tokenizer.vocab_size},
        "device": {
            "requested": accelerator.requested_device,
            "resolved": str(torch_device),
            "accelerator_backend": accelerator.backend,
            "rocm_available": accelerator.rocm_available,
            "rocm_runtime_version": accelerator.rocm_runtime_version,
        },
        **metrics,
    }


def _tokens_from_texts(texts: list[str], tokenizer: ByteTokenizer) -> torch.Tensor:
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text))
    return torch.tensor(ids, dtype=torch.long)


def _load_model_state(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        migrated = _migrate_legacy_transformer_encoder_keys(state_dict)
        if migrated == state_dict:
            raise exc
        model.load_state_dict(migrated)


def _migrate_legacy_transformer_encoder_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    migrated: dict[str, torch.Tensor] = {}
    changed = False
    for key, value in state_dict.items():
        if key.startswith("blocks.layers."):
            new_key = "blocks." + key.removeprefix("blocks.layers.")
            changed = True
        else:
            new_key = key
        migrated[new_key] = value
    return migrated if changed else state_dict


def _tokens_per_step(config: DenseTrainingConfig) -> int:
    return config.gradient_accumulation_steps * config.batch_size * config.seq_len


def _fixed_validation_metrics(
    model: DenseDecoder,
    tokens: torch.Tensor,
    vocab_size: int,
    config: DenseTrainingConfig,
    device: torch.device,
) -> dict[str, Any]:
    return _validation_metrics(
        model,
        tokens,
        vocab_size,
        config,
        torch.Generator(device="cpu").manual_seed(config.validation_seed),
        device,
    )


def _sample_batch(
    tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    generator: torch.Generator,
) -> torch.Tensor:
    max_start = len(tokens) - (seq_len + 1)
    starts = torch.randint(0, max_start, (batch_size,), generator=generator)
    return torch.stack([tokens[start: start + seq_len + 1] for start in starts])


def _validation_metrics(
    model: DenseDecoder,
    tokens: torch.Tensor,
    vocab_size: int,
    config: DenseTrainingConfig,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    sample_order = hashlib.sha256()
    validation_tokens = 0
    with torch.no_grad():
        for _ in range(config.validation_batches):
            batch = _sample_batch(tokens, config.batch_size, config.seq_len, generator).to(device)
            batch_cpu = batch.detach().cpu().contiguous()
            sample_order.update(batch_cpu.numpy().tobytes())
            validation_tokens += int(batch_cpu.numel())
            logits = model(batch[:, :-1])
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size),
                batch[:, 1:].reshape(-1),
            )
            losses.append(float(loss.detach().cpu()))
    model.train()
    return {
        "loss": float(sum(losses) / max(len(losses), 1)),
        "batches": len(losses),
        "tokens": validation_tokens,
        "sample_order_sha256": sample_order.hexdigest(),
    }


def _generate_sample(
    model: DenseDecoder,
    tokenizer: ByteTokenizer,
    prompt: str,
    seq_len: int,
    device: torch.device,
    max_new_tokens: int = 32,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)[:-1]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            window = ids[-seq_len:]
            input_ids = torch.tensor([window], dtype=torch.long, device=device)
            logits = model(input_ids)
            next_id = int(torch.argmax(logits[0, -1]).detach().cpu())
            ids.append(next_id)
            if next_id == tokenizer.eos_id:
                break
    return tokenizer.decode(ids)


def _resume_check(
    checkpoint_path: Path,
    vocab_size: int,
    config: DenseTrainingConfig,
    device: torch.device,
    *,
    expected_step: int,
) -> bool:
    payload = torch.load(checkpoint_path, map_location=device)
    model = DenseDecoder(
        vocab_size,
        config.seq_len,
        config.hidden_dim,
        config.layers,
        config.heads,
        config.attention_mask_mode,
        config.architecture_variant,
        config.adapter_dim,
        config.block_impl,
    ).to(device)
    optimizer = _make_optimizer(model, config)
    _load_model_state(model, payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("step", 0)) == expected_step


def _should_flush(step: int, interval: int) -> bool:
    return interval > 0 and step > 0 and step % interval == 0


def _append_progress(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _save_training_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: DenseTrainingConfig,
    *,
    step: int,
    latest_loss: float,
    generator: torch.Generator,
    validation: dict[str, Any] | None = None,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "step": step,
            "latest_loss": latest_loss,
            "generator_state": generator.get_state(),
            "validation": validation or {},
        },
        checkpoint_path,
    )


def _save_model_only_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    config: DenseTrainingConfig,
    *,
    step: int,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "step": step,
            "checkpoint_type": "model_only",
        },
        checkpoint_path,
    )


def _gradient_norm_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    finite_values = [
        float(row["global_norm_before_clip"])
        for row in records
        if row.get("finite") and math.isfinite(float(row["global_norm_before_clip"]))
    ]
    return {
        "count": len(records),
        "finite_count": len(finite_values),
        "nonfinite_count": len(records) - len(finite_values),
        "first": finite_values[0] if finite_values else math.nan,
        "last": finite_values[-1] if finite_values else math.nan,
        "min": min(finite_values) if finite_values else math.nan,
        "max": max(finite_values) if finite_values else math.nan,
    }


def _reset_peak_memory_stats(device: torch.device) -> None:
    if device.type != "cuda":
        return
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        return


def _memory_summary(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "device": str(device),
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "free_bytes": 0,
            "total_bytes": 0,
        }
    free_bytes = 0
    total_bytes = 0
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except Exception:
        pass
    return {
        "device": str(device),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


def _adapter_activation_stats(
    model: DenseDecoder,
    tokens: torch.Tensor,
    config: DenseTrainingConfig,
    device: torch.device,
) -> dict[str, Any]:
    if config.architecture_variant != "adapter":
        return {"architecture_variant": config.architecture_variant, "layers": []}
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(config.validation_seed)
    batch = _sample_batch(tokens, config.batch_size, config.seq_len, generator).to(device)
    input_ids = batch[:, :-1]
    layers: list[dict[str, Any]] = []
    with torch.no_grad():
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        hidden = model.token_embedding(input_ids) + model.position_embedding(positions)
        mask = _causal_mask(model.attention_mask_mode, seq_len, device)
        for index, (block, adapter) in enumerate(zip(model.blocks, model.adapters, strict=True)):
            hidden = _run_transformer_block(block, hidden, mask, device_type=input_ids.device.type)
            if isinstance(adapter, _ResidualAdapter):
                normalized = adapter.norm(hidden)
                down = adapter.down(normalized)
                activated = adapter.activation(down)
                update = adapter.up(activated)
                update_norm = float(update.float().norm().detach().cpu())
                hidden_norm = float(hidden.float().norm().detach().cpu())
                layers.append(
                    {
                        "layer": index,
                        "hidden_norm": hidden_norm,
                        "adapter_update_norm": update_norm,
                        "adapter_to_hidden_norm_ratio": (
                            update_norm / hidden_norm if hidden_norm else math.nan
                        ),
                        "down_activation": _tensor_stats(down),
                        "up_update": _tensor_stats(update),
                    }
                )
                hidden = hidden + update
            else:
                layers.append({"layer": index, "adapter": "identity"})
    model.train()
    return {
        "architecture_variant": config.architecture_variant,
        "layers": layers,
    }


def _count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _last_jsonl_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    last = ""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            last = line
    return json.loads(last) if last else {}


def _autocast_dtype(mixed_precision: str) -> torch.dtype | None:
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "fp32":
        return None
    raise ValueError(f"unknown mixed precision mode: {mixed_precision}")


def _run_transformer_block(
    block: torch.nn.TransformerEncoderLayer,
    hidden: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    device_type: str,
) -> torch.Tensor:
    if mask is not None and _is_autocast_enabled(device_type):
        with torch.autocast(device_type=device_type, enabled=False):
            return block(hidden.float(), src_mask=mask, is_causal=True)
    return block(hidden, src_mask=mask, is_causal=mask is not None)


def _is_autocast_enabled(device_type: str) -> bool:
    try:
        return bool(torch.is_autocast_enabled(device_type))
    except TypeError:
        return bool(torch.is_autocast_enabled())


def _make_optimizer(
    model: torch.nn.Module,
    config: DenseTrainingConfig,
) -> torch.optim.Optimizer:
    if config.optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, foreach=False)
    if config.optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    raise ValueError(f"unknown optimizer: {config.optimizer_name}")


def _causal_mask(
    attention_mask_mode: str,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    if attention_mask_mode == "none":
        return None
    if attention_mask_mode == "bool_causal":
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )
    if attention_mask_mode == "additive_causal":
        mask = torch.zeros(seq_len, seq_len, dtype=torch.float32, device=device)
        return mask.masked_fill(
            torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1),
            float("-inf"),
        )
    if attention_mask_mode == "finite_causal":
        mask = torch.zeros(seq_len, seq_len, dtype=torch.float32, device=device)
        return mask.masked_fill(
            torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1),
            -1.0e4,
        )
    raise ValueError(f"unknown attention mask mode: {attention_mask_mode}")


def _tensor_stats(tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None:
        return _empty_tensor_summary()
    detached = tensor.detach()
    if detached.dtype == torch.bool:
        numeric = detached.to(dtype=torch.float32)
    elif detached.is_floating_point() or detached.is_complex():
        numeric = detached.float()
    else:
        numeric = detached.to(dtype=torch.float32)
    finite_mask = torch.isfinite(numeric)
    finite = bool(finite_mask.all().detach().cpu())
    finite_values = numeric[finite_mask]
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "finite": finite,
        "numel": int(detached.numel()),
        "finite_count": int(finite_mask.sum().detach().cpu()),
        "nan_count": int(torch.isnan(numeric).sum().detach().cpu()),
        "posinf_count": int(torch.isposinf(numeric).sum().detach().cpu()),
        "neginf_count": int(torch.isneginf(numeric).sum().detach().cpu()),
        "abs_max": float(finite_values.abs().max().detach().cpu())
        if finite_values.numel()
        else math.nan,
        "mean": float(finite_values.mean().detach().cpu()) if finite_values.numel() else math.nan,
    }


def _module_tensor_summary(
    named_tensors: Any,
    max_examples: int = 8,
) -> dict[str, Any]:
    total_numel = 0
    finite_count = 0
    nonfinite_tensors: list[dict[str, Any]] = []
    max_abs = 0.0
    tensor_count = 0
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        tensor_count += 1
        stats = _tensor_stats(tensor)
        total_numel += int(stats["numel"])
        finite_count += int(stats["finite_count"])
        if math.isfinite(float(stats["abs_max"])):
            max_abs = max(max_abs, float(stats["abs_max"]))
        if not stats["finite"] and len(nonfinite_tensors) < max_examples:
            nonfinite_tensors.append({"name": name, **stats})
    return {
        "finite": finite_count == total_numel,
        "tensor_count": tensor_count,
        "numel": total_numel,
        "finite_count": finite_count,
        "nonfinite_count": total_numel - finite_count,
        "abs_max": max_abs,
        "nonfinite_tensors": nonfinite_tensors,
    }


def _empty_tensor_summary() -> dict[str, Any]:
    return {
        "finite": True,
        "tensor_count": 0,
        "numel": 0,
        "finite_count": 0,
        "nonfinite_count": 0,
        "abs_max": 0.0,
        "nonfinite_tensors": [],
    }
