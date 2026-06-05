"""Numerical tests for canonical evaluation metrics.

Related files:
- Exercises `metrics/fscore.py` and `metrics/chamfer.py`.
- Verifies known analytical results, not just identical-cloud trivialities.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics.fscore import fscore


class FScoreNumericalTest(unittest.TestCase):
    def test_no_overlap_returns_zero(self):
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0]])
        # threshold=0.01, distance=1.0 >> threshold → F=0
        self.assertAlmostEqual(fscore(a, b, threshold=0.01), 0.0)

    def test_all_within_threshold_returns_one(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(-0.001, 0.001, size=(50, 3))
        b = rng.uniform(-0.001, 0.001, size=(50, 3))
        # all points within 0.01 of each other → F=1
        self.assertAlmostEqual(fscore(a, b, threshold=0.01), 1.0)

    def test_half_overlap_precision_recall(self):
        # a = 4 points: 2 within threshold of b, 2 outside
        a = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0],
                      [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        b = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
        # precision(a→b): 2/4=0.5 within tau²  |  recall(b→a): 2/2=1.0
        f = fscore(a, b, threshold=0.01**2, squared=False)
        expected = 2 * 0.5 * 1.0 / (0.5 + 1.0)
        self.assertAlmostEqual(f, expected, places=5)

    def test_symmetric(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal((30, 3))
        b = rng.standard_normal((30, 3))
        self.assertAlmostEqual(fscore(a, b, threshold=0.5),
                               fscore(b, a, threshold=0.5), places=10)

    def test_invalid_threshold_raises(self):
        pts = np.zeros((2, 3))
        with self.assertRaises(ValueError):
            fscore(pts, pts, threshold=0.0)


class ChamferNumericalTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            self.torch = torch
        except ImportError:
            self.torch = None

    def test_chamfer_known_distance(self):
        if self.torch is None:
            self.skipTest("torch not available")
        from metrics.chamfer import chamfer_distance

        # Two single-point clouds separated by distance d along x
        d = 3.0
        a = self.torch.tensor([[[0.0, 0.0, 0.0]]])
        b = self.torch.tensor([[[d, 0.0, 0.0]]])
        # Bidirectional squared Chamfer = d² + d² (mean of each direction, then mean over batch)
        # per_batch = a_to_b.mean + b_to_a.mean = d² + d² = 18.0
        result = float(chamfer_distance(a, b, squared=True))
        self.assertAlmostEqual(result, 2 * d**2, places=4)

    def test_chamfer_identical_is_zero(self):
        if self.torch is None:
            self.skipTest("torch not available")
        from metrics.chamfer import chamfer_distance

        pts = self.torch.randn(1, 20, 3)
        self.assertAlmostEqual(float(chamfer_distance(pts, pts)), 0.0, places=5)

    def test_chamfer_single_directional(self):
        if self.torch is None:
            self.skipTest("torch not available")
        from metrics.chamfer import chamfer_distance

        d = 2.0
        a = self.torch.tensor([[[0.0, 0.0, 0.0]]])
        b = self.torch.tensor([[[d, 0.0, 0.0]]])
        result = float(chamfer_distance(a, b, squared=True, single_directional=True))
        self.assertAlmostEqual(result, d**2, places=4)


if __name__ == "__main__":
    unittest.main()
