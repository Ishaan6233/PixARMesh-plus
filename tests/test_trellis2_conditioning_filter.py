"""Tests for the degenerate-conditioning fixes in the Trellis2-MV loader (WS2).

Covers: object-blind (support-0) empty selection in `_select_diverse_views`, the
shared `covis_object_supports` helper, and the `load_conditioning_filter` sidecar
reader.

Related files:
- src/data/trellis2_mv.py
- scripts/data/build_conditioning_filter.py
"""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.trellis2_mv import (
    _select_diverse_views,
    covis_object_supports,
    load_conditioning_filter,
)


def _identity_cam():
    """OpenCV camera at the origin looking down +z."""
    return np.eye(4, dtype=np.float32)


def _backwards_cam():
    """Camera rotated 180 deg about y — scene points are behind it."""
    w2c = np.eye(4, dtype=np.float32)
    w2c[0, 0] = w2c[2, 2] = -1.0
    return w2c


_K = np.array([[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]], dtype=np.float32)
_IMG_HW = (64, 64)


class SelectDiverseViewsTest(unittest.TestCase):
    def _pts(self, n=20):
        rng = np.random.RandomState(0)
        return (rng.uniform(-0.1, 0.1, (n, 3)) + [0, 0, 2.0]).astype(np.float32)

    def test_object_blind_returns_empty(self):
        pts = self._pts()
        w2c = np.stack([_backwards_cam()] * 3)
        ks = np.stack([_K] * 3)
        support = np.zeros(3, dtype=np.float32)
        self.assertEqual(
            _select_diverse_views(pts, w2c, ks, _IMG_HW, support, k_max=8), []
        )

    def test_subthreshold_support_still_selects(self):
        # Positive-but-below-min_support_pts supports must keep the old best-effort
        # behavior (all views, sorted by support), NOT the new empty return.
        pts = self._pts()
        w2c = np.stack([_identity_cam()] * 3)
        ks = np.stack([_K] * 3)
        support = np.array([3.0, 7.0, 1.0], dtype=np.float32)
        sel = _select_diverse_views(pts, w2c, ks, _IMG_HW, support, k_max=8,
                                    min_support_pts=50)
        self.assertEqual(sel, [1, 0, 2])  # sorted by support, none dropped


class CovisObjectSupportsTest(unittest.TestCase):
    def _cond(self, n_pts=30):
        rng = np.random.RandomState(1)
        pts = (rng.uniform(-0.1, 0.1, (n_pts, 3)) + [0, 0, 2.0]).astype(np.float32)
        corners = np.array(
            [[x, y, z] for x in (-0.15, 0.15) for y in (-0.15, 0.15)
             for z in (1.85, 2.15)], dtype=np.float32)
        t_obj = np.eye(4, dtype=np.float32)
        t_obj[:3, 3] = corners.mean(0)
        return {
            "T_output_from_norm": np.eye(4, dtype=np.float32),
            "scene_point_clouds": pts,
            "bboxes": corners[None],
            "object_to_norm_transforms": t_obj,
        }

    def test_supports_front_vs_behind(self):
        cond = self._cond()
        w2c = np.stack([_identity_cam(), _backwards_cam()])
        ks = np.stack([_K, _K])
        obj_pts, support = covis_object_supports(cond, w2c, ks, _IMG_HW)
        self.assertEqual(len(obj_pts), 30)  # bbox crop keeps all synthetic points
        self.assertEqual(support[0], 30.0)  # all project in-frame for the front cam
        self.assertEqual(support[1], 0.0)   # all behind the backwards cam

    def test_empty_bbox_crop_is_object_blind(self):
        # Object bbox far from every scene point: the crop is empty, so supports
        # must come out all-zero (no whole-scene fallback masking a degenerate
        # instance as healthy).
        cond = self._cond()
        cond["bboxes"] = cond["bboxes"] + 50.0
        cond["object_to_norm_transforms"][:3, 3] += 50.0
        w2c = np.stack([_identity_cam()])
        ks = np.stack([_K])
        obj_pts, support = covis_object_supports(cond, w2c, ks, _IMG_HW)
        self.assertEqual(len(obj_pts), 0)
        self.assertTrue((support == 0).all())

    def test_frame_correction_keeps_crop_in_norm_frame(self):
        # A rotated norm->world frame makes the world-axis AABB looser than the
        # object's norm-frame crop. The support helper must crop first in norm
        # space, then transform selected points for projection.
        cond = self._cond(n_pts=1)
        cond["scene_point_clouds"] = np.array(
            [[0.0, 0.0, 2.0], [0.29, 0.0, 2.0]], dtype=np.float32
        )
        theta = np.pi / 4.0
        c, s = np.cos(theta), np.sin(theta)
        T_n2w = np.array(
            [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        w2c = np.stack([_identity_cam()])
        ks = np.stack([_K])
        obj_pts, support = covis_object_supports(
            cond, w2c, ks, _IMG_HW, T_norm_to_world=T_n2w
        )
        self.assertEqual(len(obj_pts), 1)
        self.assertEqual(support[0], 1.0)

    def test_rng_makes_subsample_deterministic(self):
        cond = self._cond(n_pts=2000)  # > _OBJ_PTS_SAMPLE forces the subsample
        w2c = np.stack([_identity_cam()])
        ks = np.stack([_K])
        p1, s1 = covis_object_supports(cond, w2c, ks, _IMG_HW,
                                       rng=np.random.RandomState(7))
        p2, s2 = covis_object_supports(cond, w2c, ks, _IMG_HW,
                                       rng=np.random.RandomState(7))
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(s1, s2)


class LoadConditioningFilterTest(unittest.TestCase):
    def test_reads_keep_set(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "conditioning_filter.csv"
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["sha256", "keep"])
                w.writeheader()
                w.writerow({"sha256": "a", "keep": "True"})
                w.writerow({"sha256": "b", "keep": "False"})
                w.writerow({"sha256": "c", "keep": "true"})
            self.assertEqual(load_conditioning_filter(td), {"a", "c"})

    def test_missing_sidecar_raises_with_hint(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError) as ctx:
                load_conditioning_filter(td)
            self.assertIn("build_conditioning_filter.py", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
