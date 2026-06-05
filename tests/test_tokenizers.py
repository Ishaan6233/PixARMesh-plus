"""Tests for mesh quantization/dequantization round-trip.

Related files:
- Exercises `src/data/utils.py::quantize_points` and `dequantize_points`.
- Protects the quantization error bound (1 / num_pos_tokens).
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.utils import dequantize_points, quantize_points


class QuantizationRoundTripTest(unittest.TestCase):
    def _roundtrip_error(self, num_tokens: int, n: int = 1000) -> float:
        rng = np.random.default_rng(42)
        pts = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
        q = quantize_points(pts, num_tokens)
        rec = dequantize_points(q, num_tokens).astype(np.float32)
        return float(np.abs(pts - rec).max())

    def test_roundtrip_128_tokens(self):
        max_err = self._roundtrip_error(128)
        self.assertLessEqual(max_err, 1.0 / 128 + 1e-5)

    def test_roundtrip_512_tokens(self):
        max_err = self._roundtrip_error(512)
        self.assertLessEqual(max_err, 1.0 / 512 + 1e-5)

    def test_quantized_range(self):
        pts = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
        q = quantize_points(pts, 128)
        self.assertTrue(np.all(q >= 0))
        self.assertTrue(np.all(q < 128))

    def test_boundary_values(self):
        pts = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        q = quantize_points(pts, 64)
        # -1 maps to bin 0, +1 maps to bin 63 (clipped)
        self.assertEqual(int(q[0, 0]), 0)
        self.assertEqual(int(q[1, 0]), 63)


if __name__ == "__main__":
    unittest.main()
