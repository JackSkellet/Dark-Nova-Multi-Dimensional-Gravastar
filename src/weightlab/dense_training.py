from __future__ import annotations

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
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = torch.nn.Embedding(seq_len, hidden_dim)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = torch.nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        hidden = self.blocks(hidden, mask=mask, is_causal=True)
        return self.head(self.norm(hidden))


def train_dense_decoder(
    texts: list[str],
    config: DenseTrainingConfig,
    output_dir: Path,
    seed: int = 123,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteTokenizer()
    tokens = _tokens_from_texts(texts, tokenizer)
    if len(tokens) < config.seq_len + 2:
        raise ValueError("not enough tokens for dense decoder training")

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
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    dtype = _autocast_dtype(config.mixed_precision)
    use_autocast = device.type == "cuda" and dtype is not None

    loss_curve: list[dict[str, float]] = []
    train_start = time.perf_counter()
    status = "completed"
    failure = ""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, config.steps + 1):
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
                break
            scaled_loss.backward()
            step_loss += float(loss.detach().cpu())
        if status != "completed":
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        loss_curve.append(
            {"step": float(step), "loss": step_loss / config.gradient_accumulation_steps}
        )

    elapsed = max(time.perf_counter() - train_start, 1e-9)
    train_tokens = (
        config.steps
        * config.gradient_accumulation_steps
        * config.batch_size
        * config.seq_len
    )
    validation_loss = (
        _validation_loss(model, tokens, tokenizer.vocab_size, config, generator, device)
        if status == "completed"
        else math.nan
    )
    sample = _generate_sample(model, tokenizer, "def ", config.seq_len, device)
    checkpoint_path = output_dir / "dense_decoder_last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "step": config.steps,
        },
        checkpoint_path,
    )
    resume_ok = _resume_check(checkpoint_path, tokenizer.vocab_size, config, device)
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
            "parameter_count": sum(param.numel() for param in model.parameters()),
            "config": asdict(config),
        },
        "training": {
            "train_tokens": train_tokens,
            "steps": config.steps,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "elapsed_s": elapsed,
            "tokens_per_second": train_tokens / elapsed,
            "loss_curve": loss_curve,
        },
        "validation": {"loss": validation_loss},
        "generation_samples": [{"prompt": "def ", "text": sample}],
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "resume_ok": resume_ok,
        },
        "limitations": [
            "training_smoke_only",
            "byte_tokenizer_baseline",
            "not_10m_parameter_model",
            "not_50m_token_run",
            "no_functional_coding_evaluation_yet",
        ],
    }


def _tokens_from_texts(texts: list[str], tokenizer: ByteTokenizer) -> torch.Tensor:
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text))
    return torch.tensor(ids, dtype=torch.long)


def _sample_batch(
    tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    generator: torch.Generator,
) -> torch.Tensor:
    max_start = len(tokens) - (seq_len + 1)
    starts = torch.randint(0, max_start, (batch_size,), generator=generator)
    return torch.stack([tokens[start: start + seq_len + 1] for start in starts])


def _validation_loss(
    model: DenseDecoder,
    tokens: torch.Tensor,
    vocab_size: int,
    config: DenseTrainingConfig,
    generator: torch.Generator,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(config.validation_batches):
            batch = _sample_batch(tokens, config.batch_size, config.seq_len, generator).to(device)
            logits = model(batch[:, :-1])
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size),
                batch[:, 1:].reshape(-1),
            )
            losses.append(float(loss.detach().cpu()))
    model.train()
    return float(sum(losses) / max(len(losses), 1))


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
) -> bool:
    payload = torch.load(checkpoint_path, map_location=device)
    model = DenseDecoder(
        vocab_size,
        config.seq_len,
        config.hidden_dim,
        config.layers,
        config.heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("step", 0)) == config.steps


def _autocast_dtype(mixed_precision: str) -> torch.dtype | None:
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "fp32":
        return None
    raise ValueError(f"unknown mixed precision mode: {mixed_precision}")
