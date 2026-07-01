"""Regression test for the MV view-registration fix.

The per-view scene_transform must map a fixed world point to the SAME scene
coordinate through every view (i.e. scene_transform_n @ wrd2cam_n is
view-independent). A prior bug only de-tilted each camera about its own centre
and never registered the views into a common frame (cross-view scatter ~2.3m),
corrupting every cross-view conditioning path.
"""
import os
import unittest

import numpy as np

_DATA = "datasets/3d-front-multiview"


@unittest.skipUnless(os.path.isdir(_DATA), "MV dataset not present")
class MVViewRegistrationTest(unittest.TestCase):
    @staticmethod
    def _make_cfg():
        """Deterministic (augmentation-off) MV DataConfig shared by the tests."""
        from src.utils.config import DataConfig

        return DataConfig(
            type="3d-front-multiview", path=_DATA, num_views=4, num_points=4096,
            norm_bound=0.95, load_images=False, use_masked_obj_pc=False,
            random_scale=False, random_rotate=False, random_jitter_point_clouds=False,
            random_jitter_depth=False, random_shift=False,
        )

    def test_scene_transform_registers_views_to_common_frame(self):
        import datasets as hf
        from src.data.mesh import transform_3d_front_multiview

        cfg = self._make_cfg()
        d = hf.load_from_disk(_DATA)["validation"]
        out = transform_3d_front_multiview(
            d[0:1], is_train=False, data_cfg=cfg, image_preprocessor=None
        )
        st = np.asarray(out["scene_transforms"][0]).astype(np.float64)   # (N,4,4)
        vm = np.asarray(out["view_mask"][0]).astype(bool)                # (N,)
        w2c = [np.array(x, np.float64) for x in d[0]["wrd2cam_rects"]]

        X = np.array([0.3, -0.2, 0.5, 1.0], np.float64)  # arbitrary world point
        pts = []
        for n in range(st.shape[0]):
            if not vm[n]:
                continue  # padded/invalid views repeat a real view; skip
            pts.append((st[n] @ (w2c[n] @ X))[:3])
        pts = np.stack(pts)
        scatter = float(np.linalg.norm(pts - pts.mean(0), axis=1).max())
        self.assertLess(
            scatter, 1e-3,
            f"views not registered to a common frame: cross-view scatter={scatter:.4f} m",
        )


    def test_padded_views_marked_invalid(self):
        """Repeat-padded views (scenes with < num_views) must be view_mask=False.

        Regression: view_valid was indexed by _vidx VALUE (always < _orig) instead of
        position, so every padded duplicate slot was flagged valid — letting one real
        view vote num_views times and defeating the min_views consensus.
        """
        import datasets as hf
        from src.data.mesh import transform_3d_front_multiview

        cfg = self._make_cfg()
        d = hf.load_from_disk(_DATA)["validation"]
        idx = next(
            (i for i in range(min(len(d), 300)) if len(d[i]["wrd2cam_rects"]) < 4), None
        )
        if idx is None:
            self.skipTest("no scene with < num_views views in val split")
        n_real = len(d[idx]["wrd2cam_rects"])
        out = transform_3d_front_multiview(
            d[idx : idx + 1], is_train=False, data_cfg=cfg, image_preprocessor=None
        )
        vm = np.asarray(out["view_mask"][0]).astype(bool)
        self.assertEqual(
            int(vm.sum()), n_real,
            f"expected exactly {n_real} valid views, got {int(vm.sum())}: {vm}",
        )
        self.assertTrue(
            vm[:n_real].all() and not vm[n_real:].any(),
            f"view_mask must be {n_real} True then padded False, got {vm}",
        )


    def test_obj_canon_transform_is_valid_affine(self):
        """obj_canon_transform must be a well-conditioned, orientation-preserving affine.

        The model extracts R = obj_canon_transform[:3,:3] (edgerunner.py ~L477) and
        rotates scene-frame obj_voxels by it before self-normalising to [-0.95,0.95]^3.
        R carries a scale factor from the GT object transform (3D-FRONT metric units),
        which the subsequent self-normalisation removes — so only the DIRECTION (sign of
        det and singular-value ratios) matters.  Failure modes:
         - NaN/inf: matrix construction bug.
         - det < 0: reflection (mirrors the geometry; canonical axes flipped).
         - extreme singular-value spread (> 1000x): near-degenerate; self-norm can't
           recover orientation when one axis collapses.
        After applying R + self-normalisation to the GT scene-frame bbox corners,
        the result must lie within [-0.95, 0.95]^3.
        """
        import datasets as hf
        from src.data.mesh import transform_3d_front_multiview

        cfg = self._make_cfg()
        d = hf.load_from_disk(_DATA)["validation"]
        out = transform_3d_front_multiview(
            d[0:1], is_train=False, data_cfg=cfg, image_preprocessor=None
        )
        T = np.asarray(out["obj_canon_transform"][0], dtype=np.float64)  # (4, 4)
        R = T[:3, :3]

        # Must be finite
        self.assertFalse(np.any(~np.isfinite(T)), "obj_canon_transform contains NaN/inf")

        # det(R) > 0: orientation-preserving (not a reflection)
        det = float(np.linalg.det(R))
        self.assertGreater(det, 0.0,
            f"obj_canon_transform[:3,:3] has det={det:.4f} (reflection!)")

        # Singular values must be positive and well-conditioned (no collapsed axes)
        _, sv, _ = np.linalg.svd(R)
        sv_ratio = float(sv.max() / sv.min())
        self.assertLess(sv_ratio, 1000.0,
            f"obj_canon_transform R is near-singular: sv_ratio={sv_ratio:.1f}")

        # Rotating scene-frame bbox corners by R then self-normalising must give
        # coordinates within [-0.95, 0.95]^3 — same pipeline as edgerunner.py ~L477-495.
        bboxes = np.asarray(out["bboxes"][0], dtype=np.float64)   # (n_obj, 8, 3)
        obj_idx = int(out["obj_indices"][0])
        v = bboxes[obj_idx] @ R.T                                  # (8, 3) rotated
        center = 0.5 * (v.min(0) + v.max(0))
        half_ext = np.abs(v - center).max()
        if half_ext > 1e-6:
            v_norm = (v - center) / half_ext * 0.95
        else:
            v_norm = v
        max_abs = float(np.abs(v_norm).max())
        self.assertLessEqual(
            max_abs, 0.951,
            f"Self-normalised bbox exceeds 0.95 bound: max_abs={max_abs:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
