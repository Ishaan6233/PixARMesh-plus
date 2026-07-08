import unittest

import torch

from src.models.frozen_geo_encoder import (
    _adaptive_voxel_fps_sample,
    _batched_obj_fps,
    _voxel_grid_sample,
)


class VoxelSamplingTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _cloud(self):
        pts = torch.tensor([
            [0.00, 0.00, 0.00],
            [0.02, 0.00, 0.00],
            [1.00, 0.00, 0.00],
            [0.00, 1.00, 0.00],
            [0.00, 0.00, 1.00],
            [1.00, 1.00, 0.00],
            [1.00, 0.00, 1.00],
            [0.00, 1.00, 1.00],
        ], device=self.device)
        scores = torch.tensor(
            [0.1, 0.9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], device=self.device
        )
        geom = torch.arange(pts.shape[0], device=self.device).float().unsqueeze(1).expand(-1, 3)
        return pts, scores, geom

    def test_grid_sampling_returns_indices_for_geometry_twin(self):
        pts, scores, geom = self._cloud()
        expected_pts, _expected_scores, expected_idx = _voxel_grid_sample(
            pts, scores, n_sample=5, grid_res=4
        )

        obj, geom_out = _batched_obj_fps(
            [pts], [scores], [geom], n_sample=5, device=self.device,
            out_dtype=torch.float32, max_pts=32, sampling_mode="grid",
        )

        torch.testing.assert_close(obj[0], expected_pts)
        torch.testing.assert_close(geom_out[0], geom[expected_idx])

    def test_adaptive_sampling_returns_indices_for_geometry_twin(self):
        pts, scores, geom = self._cloud()
        expected_pts, _expected_scores, expected_idx = _adaptive_voxel_fps_sample(
            pts, scores, n_sample=5
        )

        obj, geom_out = _batched_obj_fps(
            [pts], [scores], [geom], n_sample=5, device=self.device,
            out_dtype=torch.float32, max_pts=32, sampling_mode="adaptive",
        )

        torch.testing.assert_close(obj[0], expected_pts)
        torch.testing.assert_close(geom_out[0], geom[expected_idx])

    def test_unknown_sampling_mode_raises(self):
        pts, scores, geom = self._cloud()
        with self.assertRaises(ValueError):
            _batched_obj_fps(
                [pts], [scores], [geom], n_sample=5, device=self.device,
                out_dtype=torch.float32, max_pts=32, sampling_mode="not-a-mode",
            )


if __name__ == "__main__":
    unittest.main()
