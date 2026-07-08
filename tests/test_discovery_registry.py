"""Tests for the MV instance-discovery method registry.

Related files:
- Exercises `src/models/discovery/__init__.py` (the segmentation-tournament
  plug-in boundary).
- Protects the contract that `EdgeRunner.get_mv_inputs_with_cond` and
  `scripts/eval/eval_voxels.py` rely on when dispatching by
  `mv_discovery_method` / `--method`.
"""

import inspect
import unittest

from src.models.discovery import (
    available_methods,
    get_discovery_fn,
    register_discovery,
)
from src.models.frozen_geo_encoder import discover_instance_points_mv


class DiscoveryRegistryTest(unittest.TestCase):
    def test_consensus_baseline_registered(self):
        self.assertIn("consensus", available_methods())
        # The default path must remain byte-identical to the existing baseline.
        self.assertIs(get_discovery_fn("consensus"), discover_instance_points_mv)

    def test_none_defaults_to_consensus(self):
        self.assertIs(get_discovery_fn(None), discover_instance_points_mv)

    def test_unknown_method_rejected(self):
        with self.assertRaises(KeyError):
            get_discovery_fn("does-not-exist")

    def test_duplicate_registration_rejected(self):
        with self.assertRaises(ValueError):

            @register_discovery("consensus")
            def _dup(*args, **kwargs):  # pragma: no cover - registration fails first
                raise AssertionError("should not be reached")

    def test_contract_signature(self):
        """Every method must accept every kwarg the call sites pass.

        This is the full union over the real call sites — EdgeRunner /
        BPT `get_mv_inputs_with_cond` and the eval_voxels / diagnose_pool_misses
        harnesses — so a method that passes this test cannot TypeError at a
        call site on an unexpected keyword.
        """
        required = {
            "local_points", "scene_transforms", "panoptic_masks", "K_per_view",
            "view_mask", "seed_pcs", "num_obj_voxels", "num_ctx_voxels", "conf",
            "pool_size", "min_views", "depth_rtol", "adaptive_fallback",
            "mask_seeded_pool", "boundary_bias_alpha", "intra_obj_register",
            "register_iters", "return_target_ids", "return_diagnostics",
            "voxel_sampling",
        }
        for name in available_methods():
            params = set(inspect.signature(get_discovery_fn(name)).parameters)
            missing = required - params
            self.assertFalse(missing, f"method '{name}' missing kwargs: {missing}")


if __name__ == "__main__":
    unittest.main()
