"""Regression tests for the Trellis2-MV y-up camera adapter."""

import os
import unittest

import numpy as np
import torch

from src.data.trellis2_mv import (
    _CAM_YUP_TO_OPENCV_4,
    _project_world_points_to_normalized_pixels,
)
from src.models.frozen_geo_encoder import _project_pts_to_views


_TRELLIS2_DATA = (
    "datasets/mesh_datasets/datasets/"
    "3d-front-trellis2-slat-mv-da3-aug-srcperturb-r5-qfcat-obj015-light-bgtex-20260629"
)
_HF_DATA = "datasets/3d-front-multiview-full"


def _norm_to_pixel(xy_norm: np.ndarray, h: int, w: int) -> np.ndarray:
    px = np.empty_like(xy_norm, dtype=np.float64)
    px[:, 0] = (xy_norm[:, 0] + 1.0) * float(w) / 2.0 - 0.5
    px[:, 1] = (xy_norm[:, 1] + 1.0) * float(h) / 2.0 - 0.5
    return px


class Trellis2CameraConventionAlgebraTest(unittest.TestCase):
    def test_scene_transform_maps_opencv_camera_to_norm_frame(self):
        theta = 0.37
        c, s = np.cos(theta), np.sin(theta)
        wrd2cam = np.array(
            [
                [c, 0.0, s, 0.4],
                [0.0, 1.0, 0.0, -0.2],
                [-s, 0.0, c, 1.3],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        T_world_to_norm = np.array(
            [
                [1.7, 0.0, 0.0, -0.3],
                [0.0, 1.7, 0.0, 0.5],
                [0.0, 0.0, 1.7, 0.1],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        S = np.array(
            [
                [0.8, 0.0, 0.0, 0.2],
                [0.0, 0.8, 0.0, -0.1],
                [0.0, 0.0, 0.8, 0.3],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        scene_transform = (
            S @ T_world_to_norm @ np.linalg.inv(wrd2cam) @ _CAM_YUP_TO_OPENCV_4
        ).astype(np.float32)

        X_world = np.array([0.6, -0.4, 2.2, 1.0], dtype=np.float32)
        X_cam_opencv = _CAM_YUP_TO_OPENCV_4 @ (wrd2cam @ X_world)

        got = scene_transform @ X_cam_opencv
        expected = S @ T_world_to_norm @ X_world
        np.testing.assert_allclose(got, expected, atol=1e-5)

    def test_projection_helper_matches_k_times_adapter_times_wrd2cam(self):
        wrd2cam = np.eye(4, dtype=np.float32)
        K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]],
                     dtype=np.float32)
        X = np.array([[0.2, -0.1, 2.0]], dtype=np.float32)

        xy = _project_world_points_to_normalized_pixels(X, wrd2cam, K, out_h=48, out_w=64)
        uv = _norm_to_pixel(xy.astype(np.float64), h=48, w=64)[0]

        X_yup = (wrd2cam @ np.array([0.2, -0.1, 2.0, 1.0], dtype=np.float32))[:3]
        X_ocv = np.array([-X_yup[0], -X_yup[1], X_yup[2]], dtype=np.float32)
        expected = (K @ X_ocv)[:2] / X_ocv[2]
        raw = (K @ X_yup)[:2] / X_yup[2]

        np.testing.assert_allclose(uv, expected, atol=1e-5)
        self.assertGreater(float(np.linalg.norm(uv - raw)), 20.0)


@unittest.skipUnless(
    os.path.isdir(_TRELLIS2_DATA) and os.path.isdir(_HF_DATA),
    "Trellis2-MV and HF datasets not present",
)
class Trellis2CameraConventionDatasetTest(unittest.TestCase):
    def test_loader_seed_pixels_round_trip_through_model_projection(self):
        from src.data.trellis2_mv import Trellis2MVDataset
        from src.utils.config import DataConfig

        cfg = DataConfig(
            type="3d-front-trellis2-mv",
            path=_TRELLIS2_DATA,
            trellis2_hf_path=_HF_DATA,
            num_views=8,
            num_points=4096,
            norm_bound=0.95,
            load_images=False,
            use_masked_obj_pc=False,
            random_scale=False,
            random_rotate=False,
            random_jitter_point_clouds=False,
            random_jitter_depth=False,
            random_shift=False,
            mv_frame_correction=True,
            mv_filter_degenerate=False,
        )
        ds = Trellis2MVDataset(_TRELLIS2_DATA, _HF_DATA, cfg, image_preprocessor=None,
                               is_train=False)

        item = None
        for i in range(min(len(ds), 50)):
            cand = ds[i]
            if cand.get("point_clouds_valid", False):
                item = cand
                break
        self.assertIsNotNone(item, "no valid Trellis2-MV item found in first 50 val items")

        pts = torch.as_tensor(item["point_clouds"], dtype=torch.float32)
        st = torch.as_tensor(item["scene_transforms"], dtype=torch.float32)
        K = torch.as_tensor(item["K_per_view"], dtype=torch.float32)
        ref = int(item["ref_view"])
        h = int(round(float(item["K_per_view"][ref][1, 2]) * 2.0))
        w = int(round(float(item["K_per_view"][ref][0, 2]) * 2.0))

        pix_coords, _ = _project_pts_to_views(pts, st, K, h, w)
        got = _norm_to_pixel(pix_coords[:, ref].cpu().numpy().astype(np.float64), h, w)
        expected = _norm_to_pixel(np.asarray(item["point_clouds_2d"], dtype=np.float64), h, w)
        err = np.linalg.norm(got - expected, axis=1)

        self.assertLess(float(np.max(err)), 0.05)


if __name__ == "__main__":
    unittest.main()
