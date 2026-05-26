"""Tests for deterministic benchmark setup.

Related files:
- Exercises `utils/determinism.py`.
- Protects the reproducibility behavior used by `main.py`.
"""

import os
import unittest

from utils.determinism import setup_determinism


class DeterminismTest(unittest.TestCase):
    def test_sets_cublas_workspace(self):
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        report = setup_determinism(123, strict=False)
        self.assertEqual(report.seed, 123)
        self.assertEqual(report.cublas_workspace_config, ":4096:8")


if __name__ == "__main__":
    unittest.main()
