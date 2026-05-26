"""Canonical runtime benchmarker with CUDA-event timing and CPU fallback.

This file owns the runtime timing protocol, memory reporting, warmup policy,
and throughput fields used for fair inference benchmarking.

Related files:
- Runtime policy comes from `configs/runtime/canonical.yaml`.
- `main.py` calls `benchmark_inference` for runtime tasks.
- Model inference is accessed through `models/base.py::BaseModelAdapter.infer`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class RuntimeBenchmarkConfig:
    precision: str = "bf16"
    batch_size: int = 1
    warmup_steps: int = 20
    benchmark_steps: int = 100
    execution_modes: tuple[str, ...] = ("eager",)
    throughput_units: tuple[str, ...] = (
        "tokens_per_sec",
        "vertices_per_sec",
        "faces_per_sec",
        "samples_per_sec",
    )
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("Runtime precision must be fp32, fp16, or bf16.")
        if self.batch_size <= 0:
            raise ValueError("Runtime batch_size must be positive.")
        if self.warmup_steps < 0:
            raise ValueError("Runtime warmup_steps must be non-negative.")
        if self.benchmark_steps <= 0:
            raise ValueError("Runtime benchmark_steps must be positive.")


def _sync(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _memory(torch) -> dict[str, int]:
    if not torch.cuda.is_available():
        return {"max_memory_allocated": 0, "max_memory_reserved": 0}
    return {
        "max_memory_allocated": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved": int(torch.cuda.max_memory_reserved()),
    }


def benchmark_inference(
    infer_fn: Callable[[Any], Any],
    batch: Any,
    config: RuntimeBenchmarkConfig,
    *,
    throughput_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    config.validate()
    throughput_counts = throughput_counts or {}
    try:
        import torch
    except Exception as exc:
        start = time.perf_counter()
        for _ in range(config.benchmark_steps):
            infer_fn(batch)
        elapsed = time.perf_counter() - start
        latency_ms = elapsed * 1000.0 / config.benchmark_steps
        return {
            "timing_backend": "cpu_perf_counter",
            "torch_import_error": repr(exc),
            "latency_ms_mean": latency_ms,
            "benchmark_steps": config.benchmark_steps,
            **asdict(config),
        }

    for _ in range(config.warmup_steps):
        infer_fn(batch)

    if not torch.cuda.is_available():
        start = time.perf_counter()
        for _ in range(config.benchmark_steps):
            infer_fn(batch)
        elapsed = time.perf_counter() - start
        latency_ms = elapsed * 1000.0 / config.benchmark_steps
        result = {
            "timing_backend": "cpu_perf_counter",
            "latency_ms_mean": latency_ms,
            "benchmark_steps": config.benchmark_steps,
            **_memory(torch),
            **asdict(config),
        }
    else:
        torch.cuda.reset_peak_memory_stats()
        _sync(torch)
        latencies = []
        for _ in range(config.benchmark_steps):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            infer_fn(batch)
            end_event.record()
            _sync(torch)
            latencies.append(float(start_event.elapsed_time(end_event)))
        result = {
            "timing_backend": "cuda_event",
            "latency_ms_mean": sum(latencies) / len(latencies),
            "latency_ms_min": min(latencies),
            "latency_ms_max": max(latencies),
            "benchmark_steps": config.benchmark_steps,
            **_memory(torch),
            **asdict(config),
        }

    seconds_per_step = result["latency_ms_mean"] / 1000.0
    if seconds_per_step > 0:
        for key, count in throughput_counts.items():
            result[f"{key}_per_sec"] = float(count) / seconds_per_step
    return result
