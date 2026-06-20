from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def p50_p95_ms(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"latency_ms_p50": 0.0, "latency_ms_p95": 0.0}
    arr = np.asarray(samples, dtype=np.float64) * 1000.0
    return {
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
    }


def timed(callable_obj, repeats: int = 1) -> tuple[Any, list[float]]:
    samples: list[float] = []
    result: Any = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = callable_obj()
        samples.append(time.perf_counter() - start)
    return result, samples


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def ceil_div(a: int, b: int) -> int:
    return int(math.ceil(a / b))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "uncommitted"


def git_status_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--short"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def hardware_summary() -> dict[str, str]:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "processor": platform.processor()[:80],
    }


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    hypothesis: str
    seed: int
    command: str
    metrics: dict[str, Any]
    status: str = "completed"
    notes: str = ""

    def to_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        data["git_commit"] = git_commit()
        status_short = git_status_short()
        data["git_dirty"] = bool(status_short)
        data["git_status_short"] = status_short
        data["hardware"] = hardware_summary()
        data["recorded_at_unix"] = time.time()
        return data


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
