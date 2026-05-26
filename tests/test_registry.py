"""Tests for model registry governance.

Related files:
- Exercises `models/registry.py`.
- Protects config-driven model swapping for `configs/model/*.yaml`.
"""

import unittest

from models.registry import MODEL_REGISTRY, get_model_adapter_class


class RegistryTest(unittest.TestCase):
    def test_required_models_registered(self):
        self.assertEqual(
            {"pixarmesh", "bpt", "edgerunner", "meshxl"},
            set(MODEL_REGISTRY),
        )

    def test_unknown_model_rejected(self):
        with self.assertRaises(KeyError):
            get_model_adapter_class("depr")


if __name__ == "__main__":
    unittest.main()
