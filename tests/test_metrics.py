"""Tests for centralized benchmark metric implementations.

Related files:
- Exercises `metrics/fscore.py` and, when Torch is available, `metrics/chamfer.py`.
- Protects `evaluators/canonical.py` metric dependencies.
"""

import unittest

import numpy as np

from metrics.fscore import fscore


class MetricsTest(unittest.TestCase):
    def test_fscore_identical_clouds(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        self.assertEqual(fscore(points, points, threshold=0.01), 1.0)

    def test_chamfer_identical_clouds_when_torch_available(self):
        try:
            import torch
        except Exception:
            self.skipTest("torch is unavailable in this environment")
        from metrics.chamfer import chamfer_distance

        points = torch.zeros(1, 4, 3)
        self.assertEqual(float(chamfer_distance(points, points).item()), 0.0)


if __name__ == "__main__":
    unittest.main()
