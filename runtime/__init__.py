"""Runtime benchmarking package exports.

This package exposes the shared timing protocol used for model inference
benchmarking.

Related files:
- Implementation: `runtime/benchmark.py`.
- Runtime config: `configs/runtime/canonical.yaml`.
"""

from .benchmark import RuntimeBenchmarkConfig, benchmark_inference

__all__ = ["RuntimeBenchmarkConfig", "benchmark_inference"]
