"""Tests for canonical dataset and fairness-contract validation.

Related files:
- Exercises `utils/specs.py`.
- Protects `configs/dataset/canonical_3d_front.yaml` assumptions.
"""

import unittest

from utils.specs import DatasetSpec


class DatasetContractTest(unittest.TestCase):
    def test_canonical_spec_validates(self):
        DatasetSpec().validate()

    def test_rejects_coordinate_mismatch(self):
        with self.assertRaises(ValueError):
            DatasetSpec(axis_convention="z_up").validate()

    def test_rejects_normalization_bound_mismatch(self):
        with self.assertRaises(ValueError):
            DatasetSpec.from_mapping(
                {
                    "normalization": {
                        "mode": "bbox",
                        "bound": 1.0,
                        "center": True,
                        "preserve_aspect_ratio": True,
                    }
                }
            )


if __name__ == "__main__":
    unittest.main()
