"""Tests for the Pi3X depth-vs-GT diagnostic (scripts/eval/eval_pi3x_depth.py).

Covers the pure metric/mask/crop helpers and the aggregation path on synthetic
records. GPU inference (process_scene / run_shard) is exercised by the live
smoke run, not here.
"""

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location(
    "eval_pi3x_depth", _REPO / "scripts" / "eval" / "eval_pi3x_depth.py"
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


class DecodeMaskCropTest(unittest.TestCase):
    def test_decode_gt_depth_inverse_encoding(self):
        raw = np.array([[0, 255, 51]], dtype=np.uint8)
        depth = m.decode_gt_depth(raw)
        np.testing.assert_allclose(depth, [[10.0, 0.0, 8.0]], atol=1e-6)

    def test_valid_mask_excludes_far_plane_and_invalid(self):
        raw = np.array([0, 1, 254, 255], dtype=np.uint8)
        pred = np.ones(4, dtype=np.float32)
        np.testing.assert_array_equal(
            m.make_valid_mask(raw, pred), [False, True, True, False]
        )
        np.testing.assert_array_equal(
            m.make_valid_mask(raw, pred, include_far_plane=True),
            [True, True, True, False],
        )

    def test_valid_mask_excludes_bad_pred(self):
        raw = np.full(3, 100, dtype=np.uint8)
        pred = np.array([1.0, 0.0, np.nan], dtype=np.float32)
        np.testing.assert_array_equal(m.make_valid_mask(raw, pred), [True, False, False])

    def test_crop_padding_matches_mesh_py_geometry(self):
        # 484x648 padded to 504x672 with pad_top=10, pad_left=12 (mesh.py:615-617)
        arr = np.zeros((2, 504, 672), dtype=np.float32)
        arr[:, 10:494, 12:660] = 7.0
        out = m.crop_padding(arr, 10, 12, (484, 648))
        self.assertEqual(out.shape, (2, 484, 648))
        self.assertTrue((out == 7.0).all())

    def test_crop_padding_asserts_on_wrong_window(self):
        arr = np.zeros((1, 504, 672), dtype=np.float32)
        with self.assertRaises(AssertionError):
            m.crop_padding(arr, 30, 30, (484, 648))  # window runs off the array


class SeedSubsetTest(unittest.TestCase):
    def test_stable_seed_deterministic_and_distinct(self):
        self.assertEqual(m.stable_seed("scene", 3), m.stable_seed("scene", 3))
        self.assertNotEqual(m.stable_seed("scene", 3), m.stable_seed("scene", 4))

    def test_strided_view_subset(self):
        self.assertEqual(m.strided_view_subset(5, 8), [0, 1, 2, 3, 4])
        sub = m.strided_view_subset(21, 8)
        self.assertEqual(len(sub), 8)
        self.assertEqual(sub[0], 0)
        self.assertEqual(sub[-1], 20)
        self.assertEqual(sub, sorted(set(sub)))


class ViewMetricsTest(unittest.TestCase):
    def _gt(self, rng, shape=(64, 64), lo=2.0, hi=6.0):
        return rng.uniform(lo, hi, size=shape).astype(np.float32)

    def test_pure_scale_error_recovered(self):
        rng = np.random.RandomState(0)
        gt = self._gt(rng)
        s_true = 1.3  # pred = gt / s -> s_v should recover s_true
        pred = gt / s_true
        mask = np.ones_like(gt, dtype=bool)
        conf = np.full_like(gt, 0.9)
        r = m.compute_view_metrics(pred, gt, mask, conf, seed=1)
        self.assertAlmostEqual(r["s_v"], s_true, places=5)
        self.assertAlmostEqual(r["s_v_log"], s_true, places=5)
        self.assertLess(r["absRel_pv_scale"], 1e-6)
        self.assertLess(r["silog_raw"], 1e-4)  # silog is scale-invariant
        expected_raw = abs(1.0 / s_true - 1.0)  # |p-g|/g = |1/s - 1|
        self.assertAlmostEqual(r["absRel_raw"], expected_raw, places=5)
        # conf uniformly above threshold -> conf variant equals the full metric
        self.assertAlmostEqual(r["absRel_raw_conf"], r["absRel_raw"], places=6)
        self.assertEqual(r["n_valid"], gt.size)

    def test_scale_shift_alignment_recovers_affine(self):
        rng = np.random.RandomState(1)
        gt = self._gt(rng)
        a_true, b_true = 2.0, 0.5  # pred = (gt - b)/a -> aligned = a*pred + b = gt
        pred = (gt - b_true) / a_true
        mask = np.ones_like(gt, dtype=bool)
        conf = np.full_like(gt, 0.9)
        r = m.compute_view_metrics(pred, gt, mask, conf, seed=2)
        self.assertAlmostEqual(r["ss_a"], a_true, places=4)
        self.assertAlmostEqual(r["ss_b"], b_true, places=4)
        self.assertLess(r["absRel_ss"], 1e-6)

    def test_metrics_deterministic_across_calls(self):
        rng = np.random.RandomState(2)
        gt = self._gt(rng)
        pred = gt / 1.1 + rng.normal(0, 0.05, gt.shape).astype(np.float32)
        pred = np.clip(pred, 0.1, None)
        mask = np.ones_like(gt, dtype=bool)
        conf = np.full_like(gt, 0.9)
        r1 = m.compute_view_metrics(pred, gt, mask, conf, seed=7, align_subsample=500)
        r2 = m.compute_view_metrics(pred, gt, mask, conf, seed=7, align_subsample=500)
        self.assertEqual(r1, r2)

    def test_low_conf_yields_nan_conf_variant(self):
        rng = np.random.RandomState(3)
        gt = self._gt(rng)
        mask = np.ones_like(gt, dtype=bool)
        conf = np.full_like(gt, 0.1)  # all below default 0.5 threshold
        r = m.compute_view_metrics(gt.copy(), gt, mask, conf, seed=4)
        self.assertEqual(r["n_valid_conf"], 0)
        self.assertTrue(np.isnan(r["absRel_raw_conf"]))


class SceneScaleTest(unittest.TestCase):
    def test_identical_scales_zero_inconsistency(self):
        r = m.scene_scale_metrics(np.array([1.2, 1.2, 1.2]))
        self.assertAlmostEqual(r["s_scene"], 1.2)
        self.assertAlmostEqual(r["r_std"], 0.0)
        self.assertAlmostEqual(r["r_spread"], 0.0)

    def test_known_spread(self):
        r = m.scene_scale_metrics(np.array([0.9, 1.0, 1.1]))
        self.assertAlmostEqual(r["s_scene"], 1.0)
        self.assertAlmostEqual(r["r_std"], np.std([0.9, 1.0, 1.1]), places=6)
        self.assertAlmostEqual(r["r_spread"], 1.1 / 0.9 - 1.0, places=6)
        self.assertAlmostEqual(r["r_mean_abs_dev"], 0.2 / 3, places=6)

    def test_single_view_and_empty(self):
        r1 = m.scene_scale_metrics(np.array([1.5]))
        self.assertAlmostEqual(r1["s_scene"], 1.5)
        self.assertTrue(np.isnan(r1["r_std"]))
        r0 = m.scene_scale_metrics(np.array([]))
        self.assertTrue(np.isnan(r0["s_scene"]))


class IoHelpersTest(unittest.TestCase):
    def test_sanitize_numpy_and_nan(self):
        rec = {"a": np.float32(1.5), "b": np.int64(3), "c": float("nan"),
               "d": np.bool_(True), "e": "x"}
        out = m._sanitize(rec)
        self.assertEqual(out, {"a": 1.5, "b": 3, "c": None, "d": True, "e": "x"})
        json.dumps(out)  # must be strictly JSON-serializable

    def test_load_done_keys_skips_junk_lines(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "scenes_shard0.jsonl"
            p.write_text(
                json.dumps({"key": "a"}) + "\n"
                + "not json\n"
                + json.dumps({"nokey": 1}) + "\n"
                + json.dumps({"key": "b"}) + "\n"
            )
            self.assertEqual(m.load_done_keys(p), {"a", "b"})
        self.assertEqual(m.load_done_keys(Path(td) / "missing.jsonl"), set())


class AggregateTest(unittest.TestCase):
    def _fake_records(self, out_dir: Path):
        rng = np.random.RandomState(0)
        for shard in (0, 1):
            views, scenes = [], []
            for si in range(3):
                key = f"scene{shard}{si}"
                s_v = 1.0 + rng.normal(0, 0.1, size=4)
                for vi in range(4):
                    views.append({
                        "key": key, "uid": key, "scene_id": key, "mode": "all-views",
                        "instance_uid": None, "subset": None, "view_idx": vi,
                        "n_views_scene": 4, "n_views_used": 4, "n_valid": 50000,
                        "skipped": False,
                        "absRel_raw": float(abs(rng.normal(0.2, 0.02))),
                        "rmse_raw": 0.5, "silog_raw": 8.0,
                        "s_v": float(s_v[vi]), "s_v_log": float(s_v[vi]),
                        "absRel_pv_scale": 0.05, "rmse_pv_scale": 0.1,
                        "ss_a": 1.0, "ss_b": 0.0, "absRel_ss": 0.04, "rmse_ss": 0.09,
                        "mean_conf": 0.8, "n_valid_conf": 40000,
                        "absRel_raw_conf": 0.18, "absRel_pv_scale_conf": 0.045,
                        "absRel_scene_scale": 0.08, "rmse_scene_scale": 0.2,
                    })
                st = m.scene_scale_metrics(s_v)
                scenes.append(dict(
                    {"key": key, "uid": key, "scene_id": key, "mode": "all-views",
                     "instance_uid": None, "subset": None, "n_views": 4,
                     "n_views_used": 4, "n_views_scored": 4, "wall_time_s": 1.0,
                     "mean_absRel_raw": 0.2, "mean_absRel_pv_scale": 0.05,
                     "mean_absRel_scene_scale": 0.08, "mean_absRel_ss": 0.04,
                     "mean_silog_raw": 8.0},
                    **st,
                ))
            with open(out_dir / f"views_shard{shard}.jsonl", "w") as f:
                f.writelines(json.dumps(m._sanitize(r)) + "\n" for r in views)
            with open(out_dir / f"scenes_shard{shard}.jsonl", "w") as f:
                f.writelines(json.dumps(m._sanitize(r)) + "\n" for r in scenes)

    def test_aggregate_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            self._fake_records(out)
            # duplicate one scene across shards to exercise dedupe (keep=last)
            dup_scene = (out / "scenes_shard0.jsonl").read_text().splitlines()[0]
            with open(out / "scenes_shard1.jsonl", "a") as f:
                f.write(dup_scene + "\n")
            args = argparse.Namespace(out=str(out), compare_to=None)
            m.aggregate(args)

            self.assertTrue((out / "views.parquet").exists())
            self.assertTrue((out / "scenes.parquet").exists())
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["n_scenes"], 6)  # dedupe removed the duplicate
            self.assertEqual(summary["n_views"], 24)
            self.assertIn("cross_view_inconsistency", summary)
            self.assertGreater(summary["cross_view_inconsistency"]["r_std"]["n"], 0)
            md = (out / "summary.md").read_text()
            self.assertIn("Cross-view scale inconsistency", md)
            self.assertTrue((out / "plots" / "r_std_hist_cdf.png").exists())
            self.assertTrue((out / "plots" / "absrel_cdfs.png").exists())

    def test_aggregate_compare_to_join(self):
        with tempfile.TemporaryDirectory() as td:
            out_a, out_b = Path(td) / "a", Path(td) / "b"
            out_a.mkdir()
            out_b.mkdir()
            self._fake_records(out_a)
            self._fake_records(out_b)
            m.aggregate(argparse.Namespace(out=str(out_a), compare_to=None))
            m.aggregate(argparse.Namespace(out=str(out_b), compare_to=str(out_a)))
            summary = json.loads((out_b / "summary.json").read_text())
            self.assertIn("compare_to", summary)
            self.assertEqual(summary["compare_to"]["n_joined_views"], 24)


if __name__ == "__main__":
    unittest.main()
