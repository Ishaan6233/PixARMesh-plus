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
    def test_scene_transform_registers_views_to_common_frame(self):
        import datasets as hf
        from src.data.mesh import transform_3d_front_multiview
        from src.utils.config import DataConfig

        cfg = DataConfig(
            type="3d-front-multiview", path=_DATA, num_views=4, num_points=4096,
            norm_bound=0.95, load_images=False, use_masked_obj_pc=False,
            random_scale=False, random_rotate=False, random_jitter_point_clouds=False,
            random_jitter_depth=False, random_shift=False,
        )
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


if __name__ == "__main__":
    unittest.main()
