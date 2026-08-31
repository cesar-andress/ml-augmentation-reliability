"""GPU / process memory helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class TimedResult:
    seconds: float
    peak_gpu_memory_mb: float | None


def peak_gpu_memory_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated() / (1024**2))
    except Exception:
        return None


def reset_peak_gpu_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    except Exception:
        pass


@contextmanager
def timed_gpu(reset: bool = True) -> Iterator[TimedResult]:
    if reset:
        reset_peak_gpu_memory()
    t0 = time.perf_counter()
    out = TimedResult(seconds=0.0, peak_gpu_memory_mb=None)
    try:
        yield out
    finally:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        out.seconds = time.perf_counter() - t0
        out.peak_gpu_memory_mb = peak_gpu_memory_mb()
