"""Tests for runtime benchmark behavior and CPU fallback.

Related files:
- Exercises `runtime/benchmark.py`.
- Protects `main.py task=runtime` behavior in non-CUDA shells.
"""

import unittest

from runtime import RuntimeBenchmarkConfig, benchmark_inference


class RuntimeTest(unittest.TestCase):
    def test_runtime_cpu_fallback_shape(self):
        cfg = RuntimeBenchmarkConfig(warmup_steps=0, benchmark_steps=2)
        result = benchmark_inference(lambda batch: batch, {"x": 1}, cfg)
        self.assertIn("latency_ms_mean", result)
        self.assertEqual(result["benchmark_steps"], 2)


if __name__ == "__main__":
    unittest.main()
