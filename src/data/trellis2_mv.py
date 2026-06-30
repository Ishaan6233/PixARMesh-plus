"""Dataset loader for the trellis2-MV per-object file format.

Each instance is stored as two files:
  mesh_dumps/{sha256}.pickle  — {'objects': [{'vertices': (V,3), 'faces': (F,3)}]}
  mv_cond/{sha256}.pt         — {'cond': {T_norm_from_output, scene_point_clouds, bboxes, ...}}

Images and camera parameters are cross-referenced from a local HuggingFace
3d-front-multiview[-full] dataset by scene_id.  The pre-computed
T_norm_from_output (world→gravity-aligned norm frame) is combined with the HF
dataset's wrd2cam_rects to produce scene_transforms without needing the
original 21-view source renders.

Reference view is selected by covisibility: the local camera whose visible
scene points overlap most with all other local cameras.
"""

import os
import sys
import csv
import pickle
import types
import random

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

import datasets as hf_datasets

from src.utils.config import DataConfig
from src.data import utils
from src.data.mesh import subsample_point_clouds, get_instance_mesh

# ── Numpy 2.x compatibility shim ────────────────────────────────────────────
# mesh_dumps pickles were serialised with numpy 2.x (numpy._core); our env has
# numpy 1.26.4 (numpy.core).  Register the alias once at import time.
import numpy.core as _np_core

_np_mod = types.ModuleType("numpy._core")
_np_mod.numeric = _np_core.numeric
sys.modules.setdefault("numpy._core", _np_mod)
sys.modules.setdefault("numpy._core.numeric", _np_core.numeric)
# ─────────────────────────────────────────────────────────────────────────────

_COVIS_SAMPLE = 2048  # downsample scene pts before covisibility loop (speed)


def _fps_np(pts: np.ndarray, n: int) -> np.ndarray:
    """Greedy farthest-point sampling returning n indices into pts."""
    if len(pts) <= n:
        return np.arange(len(pts))
    chosen = [0]
    dists = np.full(len(pts), np.inf)
    for _ in range(n - 1):
        last = pts[chosen[-1]]
        d = np.sum((pts - last) ** 2, axis=1)
        dists = np.minimum(dists, d)
        chosen.append(int(np.argmax(dists)))
    return np.array(chosen, dtype=np.int64)


def _covisibility_ref_view(
    pts_world: np.ndarray,
    wrd2cams: np.ndarray,
    Ks: np.ndarray,
    img_hw: tuple,
) -> int:
    """Return the index of the local view with highest covisibility.

    For each candidate view n we count how many of its visible scene points
    also project inside the image of every other view m.  The view with the
    highest total co-visible count is the reference.

    pts_world : (M, 3)  world-frame points (sampled subset for speed)
    wrd2cams  : (N, 4, 4)  world→camera transforms
    Ks        : (N, 3, 3)  camera intrinsics
    img_hw    : (H, W) raw image resolution
    """
    H, W = img_hw
    N = len(wrd2cams)
    scores = np.zeros(N, dtype=np.int64)
    pts_h = np.concatenate([pts_world, np.ones((len(pts_world), 1), dtype=np.float32)], axis=1)  # (M,4)
    for n in range(N):
        pts_cam_n = (wrd2cams[n] @ pts_h.T).T[:, :3]  # (M, 3)
        valid_n = pts_cam_n[:, 2] > 1e-4
        if not valid_n.any():
            continue
        sub = pts_cam_n[valid_n]  # (M_n, 3) — camera-n frame
        # back-project cam-n → world before projecting into other views
        cam2wrd_n = np.linalg.inv(wrd2cams[n])
        sub_h = np.concatenate([sub, np.ones((len(sub), 1), dtype=np.float32)], axis=1)
        sub_world_h = (cam2wrd_n @ sub_h.T).T  # (M_n, 4) world frame
        for m in range(N):
            if m == n:
                continue
            pts_cam_m = (wrd2cams[m] @ sub_world_h.T).T[:, :3]  # (M_n, 3)
            z_m = pts_cam_m[:, 2]
            valid_m = z_m > 1e-4
            if not valid_m.any():
                continue
            uvw = (Ks[m] @ pts_cam_m[valid_m].T)  # (3, k)
            u = uvw[0] / uvw[2]
            v = uvw[1] / uvw[2]
            in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            scores[n] += int(in_bounds.sum())
    return int(np.argmax(scores))


class Trellis2MVDataset(Dataset):
    """PyTorch Dataset over the trellis2 per-object file format.

    Emits per-item dicts that exactly match the output of
    transform_3d_front_multiview, so the existing Front3DCollator and model
    forward pass work unchanged.
    """

    def __init__(
        self,
        trellis2_dir: str,
        hf_path: str,
        data_cfg: DataConfig,
        image_preprocessor=None,
        is_train: bool = True,
    ):
        self.root = Path(trellis2_dir)
        self.data_cfg = data_cfg
        self.image_preprocessor = image_preprocessor
        self.is_train = is_train
        self.norm_bound = data_cfg.norm_bound

        # ── Load instance list ───────────────────────────────────────────────
        meta_path = self.root / "metadata.csv"
        split_tag = "train" if is_train else "val"
        self.instances: list[str] = []  # sha256 keys
        with open(meta_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # metadata.csv uses 'train' for both train and val (no explicit val split);
                # fall back: treat all rows as train, use a deterministic 95/5 split by index
                self.instances.append(row["sha256"])
        # Simple deterministic train/val split (no explicit val column in this dataset)
        all_idx = list(range(len(self.instances)))
        rng = np.random.RandomState(42)
        rng.shuffle(all_idx)
        n_val = max(1, int(0.05 * len(all_idx)))
        if is_train:
            keep = sorted(all_idx[n_val:])
        else:
            keep = sorted(all_idx[:n_val])
        self.instances = [self.instances[i] for i in keep]

        # ── Load HF dataset and build scene_id → row index ──────────────────
        hf_path_abs = Path(hf_path).absolute().as_posix()
        try:
            hf_data = hf_datasets.load_from_disk(hf_path_abs)
        except Exception:
            hf_data = hf_datasets.load_dataset(hf_path_abs)

        if isinstance(hf_data, hf_datasets.Dataset):
            hf_split = hf_data
        else:
            split_key = next(
                (k for k in ("train", "validation", "val", "test") if k in hf_data),
                next(iter(hf_data)),
            )
            hf_split = hf_data[split_key]

        self._hf = hf_split
        self._scene_id_to_idx: dict[str, int] = {}
        for i, sid in enumerate(hf_split["scene_id"]):
            self._scene_id_to_idx[sid] = i

        # ── Cache column existence flags ─────────────────────────────────────
        self._hf_has_panoptic = (
            "panoptic_masks" in hf_split.features
            or "panoptic_mask" in hf_split.features
        )
        self._pan_key = (
            "panoptic_masks" if "panoptic_masks" in hf_split.features
            else ("panoptic_mask" if "panoptic_mask" in hf_split.features else None)
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _load_mesh(self, sha256: str):
        path = self.root / "mesh_dumps" / f"{sha256}.pickle"
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = d["objects"][0]
        return np.array(obj["vertices"], dtype=np.float32), np.array(obj["faces"], dtype=np.int64)

    def _load_cond(self, sha256: str) -> dict:
        path = self.root / "mv_cond" / f"{sha256}.pt"
        d = torch.load(path, map_location="cpu", weights_only=False)
        return d["cond"]

    # ── main item ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, i: int) -> dict:
        sha256 = self.instances[i]
        data_cfg = self.data_cfg
        is_train = self.is_train
        norm_bound = self.norm_bound
        num_points = data_cfg.num_points
        with_normals = data_cfg.with_normals
        has_pc = num_points > 0
        load_images = data_cfg.load_images and self.image_preprocessor is not None

        # ── 1. Mesh ──────────────────────────────────────────────────────────
        raw_verts, raw_faces = self._load_mesh(sha256)
        vertices, faces = get_instance_mesh(raw_verts, raw_faces)
        if vertices is not None and faces is not None:
            vertices = utils.normalize_vertices(vertices, bound=norm_bound)
            faces = np.array(faces)

        # ── 2. Pre-computed conditioning ─────────────────────────────────────
        cond = self._load_cond(sha256)
        T_norm_from_output = np.array(cond["T_norm_from_output"], dtype=np.float32)  # world→gravity_norm
        T_output_from_norm = np.array(cond["T_output_from_norm"], dtype=np.float32)  # gravity_norm→world
        sc_pts_norm = np.array(cond["scene_point_clouds"], dtype=np.float32)  # (M, 3) gravity_norm frame
        sc_pts_world = (T_output_from_norm[:3, :3] @ sc_pts_norm.T + T_output_from_norm[:3, 3:]).T  # (M,3)
        bboxes_norm = np.array(cond["bboxes"], dtype=np.float32)  # (1,8,3) gravity_norm frame
        T_obj_to_norm = np.array(cond["object_to_norm_transforms"], dtype=np.float32)  # (4,4)
        uid = cond["uid"]
        scene_id = cond["scene_id"]

        # ── 3. Local HF row ──────────────────────────────────────────────────
        hf_idx = self._scene_id_to_idx.get(scene_id)
        if hf_idx is None:
            return self._make_empty(uid, vertices, faces)

        row = self._hf[hf_idx]
        n_avail = len(row["wrd2cam_rects"])
        _nv = getattr(data_cfg, "num_views", n_avail) or n_avail
        # Pad when n_avail < _nv by repeating the last valid view
        if n_avail >= _nv:
            _vidx = list(range(_nv))
        else:
            _vidx = list(range(n_avail)) + [n_avail - 1] * (_nv - n_avail)
        view_valid = np.array([k < n_avail for k in range(_nv)], dtype=bool)
        N_views = _nv

        wrd2cam_rects_n = [np.array(row["wrd2cam_rects"][v], dtype=np.float32) for v in _vidx]
        Ks_raw_n = [np.array(row["Ks"][v], dtype=np.float32) for v in _vidx]
        images_n = [row["images"][v] for v in _vidx]
        raw_img_hw = (np.array(images_n[0]).shape[0], np.array(images_n[0]).shape[1])

        # ── 4. Per-view images, K_adj, and raw scene_transforms ─────────────
        # scene_trans_raw_n[n] = T_norm_from_output @ inv(wrd2cam_n) maps
        # camera_n → gravity_norm frame (no bbox-normalization yet, same as
        # M_rot_4d @ cam_to_ref in transform_3d_front_multiview).
        scene_trans_raw_n = []
        K_adj_n = []
        pad_info_n = []
        pv_n_list = []
        for n in range(N_views):
            scene_trans_raw_n.append(
                (T_norm_from_output @ np.linalg.inv(wrd2cam_rects_n[n])).astype(np.float32)
            )
            K_n = Ks_raw_n[n].copy()
            pad_left = pad_top = 0
            if load_images:
                img_arr = np.array(images_n[n])
                proc = self.image_preprocessor(images=img_arr, return_tensors="pt")
                pv_n = proc["pixel_values"]
                pv_n_list.append(pv_n)
                out_h, out_w = pv_n.shape[2], pv_n.shape[3]
                pad_left = (out_w - img_arr.shape[1]) // 2
                pad_top = (out_h - img_arr.shape[0]) // 2
            else:
                out_h, out_w = raw_img_hw
            K_adj = K_n.copy()
            K_adj[0, 2] += pad_left
            K_adj[1, 2] += pad_top
            K_adj_n.append(K_adj.astype(np.float32))
            pad_info_n.append({"pad_top": pad_top, "pad_left": pad_left, "out_h": out_h, "out_w": out_w})

        # ── 4b. Scene normalization ───────────────────────────────────────────
        # The gravity_norm frame uses unit_median_scale (not bounded [-1,1]).
        # Fit bboxes + scene pts together into [-bound, bound] — same step as
        # normalize_bboxes_with_point_clouds in transform_3d_front_multiview.
        # The resulting matrix S = M_shift @ normalize_matrix composes on the
        # left of each raw scene_transform to give the final scene_transform.
        if is_train and getattr(data_cfg, "random_scale", False):
            bound = float(np.random.uniform(
                getattr(data_cfg, "random_scale_min", 0.75), norm_bound
            ))
        else:
            bound = norm_bound

        bboxes_scene, sc_pts_scene, normalize_matrix = utils.normalize_bboxes_with_point_clouds(
            bboxes_norm, sc_pts_norm, bound=bound, return_matrix=True
        )
        M_shift = np.eye(4, dtype=np.float32)
        if is_train and getattr(data_cfg, "random_shift", False):
            bboxes_scene, sc_pts_scene, M_shift = utils.random_shift_bboxes_with_point_clouds(
                bboxes_scene, sc_pts_scene,
                max_shift=getattr(data_cfg, "random_shift_max", 0.2),
                bound=bound, return_matrix=True,
            )

        S = (M_shift @ normalize_matrix).astype(np.float32)  # gravity_norm → scene frame
        S_inv = np.linalg.inv(S).astype(np.float32)

        # camera_n → final scene frame
        scene_transforms_n = [(S @ st).astype(np.float32) for st in scene_trans_raw_n]

        # scene → object-canonical (only rotation is metric-safe; model re-scales by voxel extent)
        obj_canon_transform = (np.linalg.inv(T_obj_to_norm) @ S_inv).astype(np.float32)

        # Used later to project scene-frame pts back to world for 2D pixel coords
        T_scene_to_world = (T_output_from_norm @ S_inv).astype(np.float32)

        # ── 5. Reference view by covisibility (valid views only) ─────────────
        n_valid = int(view_valid.sum())
        if len(sc_pts_world) > _COVIS_SAMPLE:
            samp_idx = _fps_np(sc_pts_world, _COVIS_SAMPLE)
            sc_pts_sample = sc_pts_world[samp_idx]
        else:
            sc_pts_sample = sc_pts_world

        ref_view = _covisibility_ref_view(
            sc_pts_sample,
            np.stack(wrd2cam_rects_n[:n_valid], axis=0),
            np.stack(K_adj_n[:n_valid], axis=0),
            (raw_img_hw[0], raw_img_hw[1]),
        )

        # ── 6. Object voxel seed (cond_pcs) ─────────────────────────────────
        if has_pc:
            obj_bbox_scene = bboxes_scene[0]  # (8,3) corners in scene frame
            bbox_min = obj_bbox_scene.min(axis=0) - 0.05
            bbox_max = obj_bbox_scene.max(axis=0) + 0.05
            in_bbox = (
                (sc_pts_scene[:, 0] >= bbox_min[0]) & (sc_pts_scene[:, 0] <= bbox_max[0]) &
                (sc_pts_scene[:, 1] >= bbox_min[1]) & (sc_pts_scene[:, 1] <= bbox_max[1]) &
                (sc_pts_scene[:, 2] >= bbox_min[2]) & (sc_pts_scene[:, 2] <= bbox_max[2])
            )
            obj_pts_scene = sc_pts_scene[in_bbox]  # (K, 3) scene frame
            if len(obj_pts_scene) == 0:
                obj_pts_scene = sc_pts_scene  # fallback

            dummy_2d = np.zeros((len(obj_pts_scene), 2), dtype=np.float32)
            sampled_pc, sampled_pc_2d, pc_valid, sample_inds = subsample_point_clouds(
                obj_pts_scene, dummy_2d, num_points, is_train, data_cfg, with_normals
            )

            # cond_pcs_2d: project sampled pts (scene frame) into ref view
            if sample_inds is not None:
                pts3d = obj_pts_scene[sample_inds]  # (P, 3) scene frame
            else:
                pts3d = obj_pts_scene[:num_points]

            pts3d_h = np.concatenate([pts3d, np.ones((len(pts3d), 1), dtype=np.float32)], axis=1)
            pts3d_world = (T_scene_to_world @ pts3d_h.T).T[:, :3]  # (P, 3) world frame
            pts3d_cam_h = np.concatenate([pts3d_world, np.ones((len(pts3d_world), 1), dtype=np.float32)], axis=1)
            pts3d_cam = (wrd2cam_rects_n[ref_view] @ pts3d_cam_h.T).T[:, :3]
            K_ref = K_adj_n[ref_view]
            z = pts3d_cam[:, 2]
            px = np.where(z > 1e-4, pts3d_cam[:, 0] / z, 0.0)
            py = np.where(z > 1e-4, pts3d_cam[:, 1] / z, 0.0)
            u = K_ref[0, 0] * px + K_ref[0, 2]
            v = K_ref[1, 1] * py + K_ref[1, 2]
            out_h_ref = pad_info_n[ref_view]["out_h"]
            out_w_ref = pad_info_n[ref_view]["out_w"]
            u_norm = (u + 0.5) / out_w_ref * 2 - 1
            v_norm = (v + 0.5) / out_h_ref * 2 - 1
            sampled_pc_2d = np.stack([u_norm, v_norm], axis=1).astype(np.float32)

        # ── 7. Panoptic masks ────────────────────────────────────────────────
        pan_stack = None
        if self._pan_key is not None:
            raw_masks = row[self._pan_key]
            if isinstance(raw_masks, list):
                raw_masks = [raw_masks[v] for v in _vidx]

            def _decode_pan(m):
                arr = np.array(m, dtype=np.uint32)
                if arr.ndim == 3:
                    return (arr[..., 0] * 65536 + arr[..., 1] * 256 + arr[..., 2]).astype(np.int32)
                return arr.astype(np.int32)

            def _pad_pan(dec, n):
                pvd = pad_info_n[n]
                oh, ow = pvd["out_h"], pvd["out_w"]
                pt, pl = pvd["pad_top"], pvd["pad_left"]
                h, w = dec.shape
                if (h, w) == (oh, ow):
                    return dec
                out = np.zeros((oh, ow), dtype=dec.dtype)
                out[pt:pt + h, pl:pl + w] = dec
                return out

            if isinstance(raw_masks, list):
                pan_stack = np.stack(
                    [_pad_pan(_decode_pan(m), n) for n, m in enumerate(raw_masks)], axis=0
                )
            else:
                pan_stack = np.stack(
                    [_pad_pan(_decode_pan(raw_masks), n) for n in range(N_views)], axis=0
                )

        # ── 8. Assemble output ───────────────────────────────────────────────
        ret = {
            "uid": uid,
            "bboxes": bboxes_scene,   # (1, 8, 3) in [-bound, bound]
            "obj_indices": 0,
            "vertices": vertices,
            "faces": faces,
        }
        if has_pc:
            ret["point_clouds"] = sampled_pc          # (P, 3) scene frame
            ret["point_clouds_2d"] = sampled_pc_2d    # (P, 2)
            ret["point_clouds_valid"] = pc_valid
            ret["scene_transforms"] = np.stack(scene_transforms_n)  # (N, 4, 4)
            ret["K_per_view"] = np.stack(K_adj_n)                    # (N, 3, 3)
            ret["view_mask"] = view_valid                             # (N,)
            ret["ref_view"] = ref_view                               # int
            ret["obj_canon_transform"] = obj_canon_transform         # (4, 4)

        if load_images:
            pv_stack = torch.cat(pv_n_list, dim=0)  # (N, C, H, W)
            ret["pixel_values"] = pv_stack

        if pan_stack is not None:
            ret["panoptic_masks"] = pan_stack  # (N, H, W) int32

        return ret

    def _make_empty(self, uid: str, vertices, faces) -> dict:
        """Fallback item when scene_id is not found in HF dataset."""
        n = getattr(self.data_cfg, "num_views", 4) or 4
        return {
            "uid": uid,
            "bboxes": np.zeros((1, 8, 3), dtype=np.float32),
            "obj_indices": 0,
            "vertices": vertices,
            "faces": faces,
            "point_clouds": np.zeros((self.data_cfg.num_points, 3), dtype=np.float32),
            "point_clouds_2d": np.zeros((self.data_cfg.num_points, 2), dtype=np.float32),
            "point_clouds_valid": False,
            "scene_transforms": np.eye(4, dtype=np.float32)[None].repeat(n, axis=0),
            "K_per_view": np.eye(3, dtype=np.float32)[None].repeat(n, axis=0),
            "view_mask": np.zeros(n, dtype=bool),
            "ref_view": 0,
            "obj_canon_transform": np.eye(4, dtype=np.float32),
        }
