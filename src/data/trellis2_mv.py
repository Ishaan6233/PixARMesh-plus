"""Trellis2-MV per-object dataset loader.

The runtime reference-view policy is intentionally simple: score every HF view by
how many target-object scene points project in-frame, select a diverse support
set, and use the highest-support selected view as the seed/reference view. If
panoptic masks are available and the target instance id can be resolved, the
reference must also have a visible target mask and a few seed-point hits.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
import types
import warnings
from pathlib import Path

import datasets as hf_datasets
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data import utils
from src.data.mesh import get_instance_mesh, subsample_point_clouds
from src.utils.config import DataConfig

if np.__version__ < "2":
    import numpy.core as _np_core

    _np_mod = types.ModuleType("numpy._core")
    _np_mod.numeric = _np_core.numeric
    sys.modules.setdefault("numpy._core", _np_mod)
    sys.modules.setdefault("numpy._core.numeric", _np_core.numeric)


_OBJ_PTS_SAMPLE = 512
MV_FEATURE_CACHE_VERSION = 2
MV_FEATURE_CACHE_KEYS = (
    "cache_version",
    "local_points",
    "conf",
    "dino_feats",
    "view_indices",
    "view_mask",
    "ref_view",
)
_CAM_YUP_TO_OPENCV_4 = np.diag(np.array([-1, -1, 1, 1], dtype=np.float32))
_CAM_YUP_TO_OPENCV_3 = np.diag(np.array([-1, -1, 1], dtype=np.float32))


def _stable_seed(*parts) -> int:
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest()[:4], "little", signed=False)


def load_conditioning_filter(root, expected_frame_correction: bool | None = None) -> set:
    path = Path(root) / "conditioning_filter.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"mv_filter_degenerate=True but {path} does not exist; build it with "
            "scripts/data/build_conditioning_filter.py."
        )
    meta_path = Path(root) / "conditioning_filter.meta.json"
    if expected_frame_correction is not None:
        if not meta_path.exists():
            raise ValueError(
                f"{path} has no conditioning_filter.meta.json; rebuild the sidecar "
                "so its frame-correction mode is explicit."
            )
        built_fc = bool(json.loads(meta_path.read_text()).get("frame_correction"))
        if built_fc != bool(expected_frame_correction):
            raise ValueError(
                f"conditioning_filter.csv was built with frame_correction={built_fc} "
                f"but the loader runs with mv_frame_correction={expected_frame_correction}."
            )
    keep = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if str(row["keep"]).strip().lower() in ("true", "1"):
                keep.add(row["sha256"])
    return keep


def _decode_pan_arr(mask) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.uint32)
    if arr.ndim == 3:
        return (arr[..., 0] * 65536 + arr[..., 1] * 256 + arr[..., 2]).astype(np.int32)
    return arr.astype(np.int32)


def _pad_pan_arr(decoded: np.ndarray, pad_info: dict) -> np.ndarray:
    out_h, out_w = int(pad_info["out_h"]), int(pad_info["out_w"])
    pad_top, pad_left = int(pad_info["pad_top"]), int(pad_info["pad_left"])
    h, w = decoded.shape
    if (h, w) == (out_h, out_w):
        return decoded
    out = np.zeros((out_h, out_w), dtype=decoded.dtype)
    out[pad_top : pad_top + h, pad_left : pad_left + w] = decoded
    return out


def _world_points_to_opencv_camera(
    points_world: np.ndarray,
    wrd2cam_yup: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    pts_cam_yup = (np.asarray(wrd2cam_yup, dtype=np.float32) @ pts_h.T).T[:, :3]
    return (pts_cam_yup @ _CAM_YUP_TO_OPENCV_3.T).astype(np.float32)


def _project_world_points_to_pixels(
    points_world: np.ndarray,
    wrd2cam_yup: np.ndarray,
    K: np.ndarray,
    img_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = img_hw
    pts_cam = _world_points_to_opencv_camera(points_world, wrd2cam_yup)
    z = pts_cam[:, 2]
    valid_z = z > 1e-4
    u = np.full(len(points_world), -1.0, dtype=np.float32)
    v = np.full(len(points_world), -1.0, dtype=np.float32)
    if valid_z.any():
        K = np.asarray(K, dtype=np.float32)
        u[valid_z] = K[0, 0] * pts_cam[valid_z, 0] / z[valid_z] + K[0, 2]
        v[valid_z] = K[1, 1] * pts_cam[valid_z, 1] / z[valid_z] + K[1, 2]
    in_frame = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, in_frame


def _project_world_points_to_normalized_pixels(
    points_world: np.ndarray,
    wrd2cam_yup: np.ndarray,
    K: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    u, v, _ = _project_world_points_to_pixels(points_world, wrd2cam_yup, K, (out_h, out_w))
    u_norm = (u + 0.5) / float(out_w) * 2.0 - 1.0
    v_norm = (v + 0.5) / float(out_h) * 2.0 - 1.0
    return np.stack([u_norm, v_norm], axis=1).astype(np.float32)


def _covisibility_scores(
    obj_pts_world: np.ndarray,
    wrd2cams: np.ndarray,
    Ks: np.ndarray,
    img_hw: tuple[int, int],
) -> np.ndarray:
    scores = np.zeros(len(wrd2cams), dtype=np.float32)
    if len(obj_pts_world) == 0:
        return scores
    for view_idx in range(len(wrd2cams)):
        _, _, in_frame = _project_world_points_to_pixels(
            obj_pts_world, wrd2cams[view_idx], Ks[view_idx], img_hw
        )
        scores[view_idx] = float(in_frame.sum())
    return scores


def norm_to_world_transform(cond: dict, hf_obj_transform: np.ndarray) -> np.ndarray:
    T_obj_to_norm = np.asarray(cond["object_to_norm_transforms"], dtype=np.float32)
    return (np.asarray(hf_obj_transform, dtype=np.float32) @ np.linalg.inv(T_obj_to_norm)).astype(
        np.float32
    )


def target_box_index(bboxes_norm: np.ndarray, T_obj_to_norm: np.ndarray) -> int:
    centroids = np.asarray(bboxes_norm, dtype=np.float32).mean(axis=1)
    target_center = np.asarray(T_obj_to_norm, dtype=np.float32)[:3, 3]
    return int(np.linalg.norm(centroids - target_center, axis=1).argmin())


def covis_object_supports(
    cond: dict,
    wrd2cams: np.ndarray,
    Ks: np.ndarray,
    img_hw: tuple[int, int],
    rng=None,
    T_norm_to_world: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    T_out = (
        np.asarray(T_norm_to_world, dtype=np.float32)
        if T_norm_to_world is not None
        else np.asarray(cond["T_output_from_norm"], dtype=np.float32)
    )
    scene_pts_norm = np.asarray(cond["scene_point_clouds"], dtype=np.float32)
    bboxes_norm = np.asarray(cond["bboxes"], dtype=np.float32)
    T_obj_to_norm = np.asarray(cond["object_to_norm_transforms"], dtype=np.float32)
    obj_idx = target_box_index(bboxes_norm, T_obj_to_norm)
    corners = bboxes_norm[obj_idx]
    in_bbox = (
        (scene_pts_norm >= corners.min(axis=0) - 0.1)
        & (scene_pts_norm <= corners.max(axis=0) + 0.1)
    ).all(axis=1)
    obj_pts_norm = scene_pts_norm[in_bbox]
    obj_pts_world = (T_out[:3, :3] @ obj_pts_norm.T + T_out[:3, 3:]).T
    if len(obj_pts_world) > _OBJ_PTS_SAMPLE:
        chooser = rng if rng is not None else np.random
        obj_pts_world = obj_pts_world[
            chooser.choice(len(obj_pts_world), _OBJ_PTS_SAMPLE, replace=False)
        ]
    return obj_pts_world.astype(np.float32), _covisibility_scores(
        obj_pts_world, wrd2cams, Ks, img_hw
    )


def _select_diverse_views(
    obj_pts_world: np.ndarray,
    wrd2cams: np.ndarray,
    Ks: np.ndarray,
    img_hw: tuple[int, int],
    pixel_support: np.ndarray,
    k_max: int,
    min_support_pts: int = 50,
) -> list[int]:
    """Select up to k_max views; the first is always highest object support."""
    if len(pixel_support) == 0 or float(np.max(pixel_support)) <= 0:
        return []
    if float(np.max(pixel_support)) < float(min_support_pts):
        return []

    strong = [i for i in range(len(pixel_support)) if pixel_support[i] >= min_support_pts]
    strong = sorted(strong, key=lambda i: -float(pixel_support[i]))
    if len(strong) <= k_max:
        weak = [
            i
            for i in sorted(range(len(pixel_support)), key=lambda j: -float(pixel_support[j]))
            if i not in strong and pixel_support[i] > 0
        ]
        return (strong + weak)[:k_max]

    h, w = img_hw
    cand_inframe = {}
    for idx in strong:
        pts_cam = _world_points_to_opencv_camera(obj_pts_world, wrd2cams[idx])
        z = pts_cam[:, 2]
        valid = z > 1e-4
        if valid.any():
            uvw = Ks[idx] @ pts_cam[valid].T
            u = uvw[0] / uvw[2]
            v = uvw[1] / uvw[2]
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            cand_inframe[idx] = set(np.where(valid)[0][inside].tolist())
        else:
            cand_inframe[idx] = set()

    selected = [strong[0]]
    remaining = strong[1:]
    while len(selected) < k_max and remaining:
        best_idx, best_score = remaining[0], -1.0
        selected_sets = [cand_inframe[s] for s in selected]
        for idx in remaining:
            support = max(float(pixel_support[idx]), 1.0)
            covis = max(
                (len(cand_inframe[idx] & s) / support for s in selected_sets),
                default=0.0,
            )
            score = support * (1.0 - covis)
            if score > best_score:
                best_idx, best_score = idx, score
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def _target_instance_id(row: dict) -> int | None:
    objects = row.get("objects") if isinstance(row, dict) else None
    if not isinstance(objects, dict):
        return None
    inst_ids = objects.get("inst_ids")
    if not inst_ids:
        return None
    inst = inst_ids[0]
    if isinstance(inst, (list, tuple, np.ndarray)):
        inst = inst[0]
    try:
        return int(inst)
    except (TypeError, ValueError):
        return None


def _target_ids_in_mask(mask: np.ndarray, row: dict) -> list[int]:
    inst_id = _target_instance_id(row)
    if inst_id is None:
        return []
    ids = np.unique(mask)
    out = [int(i) for i in ids if int(i) > 0 and (int(i) == inst_id or int(i) % 1000 == inst_id)]
    return out


def _mask_sanity_for_view(
    points_world: np.ndarray,
    wrd2cam: np.ndarray,
    K: np.ndarray,
    mask,
    row: dict,
    pixel_support: float = 0.0,
    min_area_px: int = 256,
    min_hit_pts: int = 8,
    min_hit_frac: float = 0.05,
) -> dict:
    decoded = _decode_pan_arr(mask)
    target_ids = _target_ids_in_mask(decoded, row)
    out = {
        "target_ids": target_ids,
        "target_area_px": 0,
        "target_hit_count": 0,
        "target_hit_fraction": 0.0,
        "target_is_plurality": False,
        "support": float(pixel_support),
        "ok": True,
        "enforced": False,
    }
    if not target_ids:
        return out

    target_mask = np.isin(decoded, target_ids)
    out["target_area_px"] = int(target_mask.sum())
    out["enforced"] = True
    if len(points_world) == 0:
        out["ok"] = False
        return out

    u, v, in_frame = _project_world_points_to_pixels(
        points_world, wrd2cam, K, decoded.shape
    )
    if not in_frame.any():
        out["ok"] = False
        return out

    pix = decoded[v[in_frame].astype(np.int64), u[in_frame].astype(np.int64)]
    hits = int(np.isin(pix, target_ids).sum())
    uniq, counts = np.unique(pix[pix > 0], return_counts=True)
    plurality = int(uniq[np.argmax(counts)]) if len(uniq) else -1
    hit_frac = float(hits / max(int(in_frame.sum()), 1))
    out.update(
        target_hit_count=hits,
        target_hit_fraction=hit_frac,
        target_is_plurality=bool(plurality in set(target_ids)),
        ok=bool(
            out["target_area_px"] >= int(min_area_px)
            and hits >= int(min_hit_pts)
            and hit_frac >= float(min_hit_frac)
        ),
    )
    return out


class Trellis2MVDataset(Dataset):
    """PyTorch dataset over Trellis2 per-object mesh_dumps and mv_cond files."""

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
        self.feature_cache = Path(data_cfg.mv_feature_cache) if data_cfg.mv_feature_cache else None

        with (self.root / "metadata.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        self.instances = [row["sha256"] for row in rows]

        all_idx = list(range(len(self.instances)))
        rng = np.random.RandomState(42)
        rng.shuffle(all_idx)
        n_val = max(1, int(0.05 * len(all_idx)))
        keep_idx = sorted(all_idx[n_val:] if is_train else all_idx[:n_val])
        self.instances = [self.instances[i] for i in keep_idx]

        if getattr(data_cfg, "mv_filter_degenerate", False):
            keep_set = load_conditioning_filter(
                self.root,
                expected_frame_correction=getattr(data_cfg, "mv_frame_correction", False),
            )
            n_before = len(self.instances)
            self.instances = [s for s in self.instances if s in keep_set]
            print(
                f"[Trellis2MVDataset] mv_filter_degenerate: kept "
                f"{len(self.instances)}/{n_before} instances"
            )

        hf_path_abs = Path(hf_path).absolute().as_posix()
        try:
            hf_data = hf_datasets.load_from_disk(hf_path_abs)
        except Exception:
            hf_data = hf_datasets.load_dataset(hf_path_abs)
        if isinstance(hf_data, hf_datasets.Dataset):
            self._hf = hf_data
        else:
            split_key = next(
                (k for k in ("train", "validation", "val", "test") if k in hf_data),
                next(iter(hf_data)),
            )
            self._hf = hf_data[split_key]

        self._scene_id_to_idx = {sid: i for i, sid in enumerate(self._hf["scene_id"])}
        self._uid_to_idx = {}
        if "uid" in self._hf.column_names:
            for i, uid in enumerate(self._hf["uid"]):
                self._uid_to_idx[uid if isinstance(uid, str) else uid[0]] = i
        if getattr(data_cfg, "mv_frame_correction", False) and not self._uid_to_idx:
            raise ValueError("mv_frame_correction=True requires a uid column in the HF dataset.")

        self._pan_key = None
        if "panoptic_masks" in self._hf.features:
            self._pan_key = "panoptic_masks"
        elif "panoptic_mask" in self._hf.features:
            self._pan_key = "panoptic_mask"
        self._hf_has_panoptic = self._pan_key is not None
        self._cached_img_chw: tuple[int, int, int] | None = None

    def __len__(self) -> int:
        return len(self.instances)

    def _load_mesh(self, sha256: str):
        with (self.root / "mesh_dumps" / f"{sha256}.pickle").open("rb") as f:
            obj = pickle.load(f)["objects"][0]
        return np.asarray(obj["vertices"], dtype=np.float32), np.asarray(obj["faces"], dtype=np.int64)

    def _load_cond(self, sha256: str) -> dict:
        return torch.load(
            self.root / "mv_cond" / f"{sha256}.pt",
            map_location="cpu",
            weights_only=False,
        )["cond"]

    def _load_feature_cache(self, uid: str, n_avail: int) -> dict | None:
        if self.feature_cache is None:
            return None
        path = self.feature_cache / f"{uid}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"mv_feature_cache is set to {self.feature_cache}, but {path.name} is missing. "
                "Run scripts/data/precompute_mv_features.py for this split or unset mv_feature_cache."
            )
        with np.load(path) as z:
            missing = sorted(set(MV_FEATURE_CACHE_KEYS) - set(z.files))
            if missing:
                raise ValueError(
                    f"{path} is missing required cache keys {missing}; rebuild it with "
                    "scripts/data/precompute_mv_features.py."
                )
            cache_version = int(np.asarray(z["cache_version"]).item())
            if cache_version != MV_FEATURE_CACHE_VERSION:
                raise ValueError(
                    f"{path} has cache_version={cache_version}, expected "
                    f"{MV_FEATURE_CACHE_VERSION}; rebuild it with "
                    "scripts/data/precompute_mv_features.py."
                )
            view_indices = np.asarray(z["view_indices"], dtype=np.int64)
            view_mask = np.asarray(z["view_mask"], dtype=bool)
            if view_indices.ndim != 1 or view_mask.shape != view_indices.shape:
                raise ValueError(f"{path} has invalid view_indices/view_mask shapes.")
            if len(view_indices) == 0 or not bool(view_mask.any()):
                raise ValueError(f"{path} contains no valid cached views.")
            if int(view_indices.max(initial=0)) >= int(n_avail) or int(view_indices.min(initial=0)) < 0:
                raise ValueError(
                    f"{path} references HF view indices outside the available range [0, {n_avail})."
                )
            local_points = np.asarray(z["local_points"])
            conf = np.asarray(z["conf"])
            dino_feats = np.asarray(z["dino_feats"])
            if local_points.shape[:1] != view_indices.shape or conf.shape[:1] != view_indices.shape:
                raise ValueError(f"{path} feature count does not match view_indices.")
            if dino_feats.shape[:1] != view_indices.shape:
                raise ValueError(f"{path} dino feature count does not match view_indices.")
            ref_view = int(np.asarray(z["ref_view"]).item())
            if ref_view < 0 or ref_view >= len(view_indices):
                raise ValueError(f"{path} has invalid ref_view={ref_view}.")
            return {
                "cached_local_points": np.array(local_points, copy=True),
                "cached_conf": np.array(conf, copy=True),
                "cached_dino_feats": np.array(dino_feats, copy=True),
                "view_indices": np.array(view_indices, copy=True),
                "view_mask": np.array(view_mask, copy=True),
                "ref_view": ref_view,
            }

    def _reference_passes_mask_sanity(
        self,
        selected_views: list[int],
        obj_pts_world: np.ndarray,
        all_wrd2cams: np.ndarray,
        all_Ks_raw: np.ndarray,
        row: dict,
        pixel_support: np.ndarray,
    ) -> tuple[list[int], str | None]:
        if self._pan_key is None or not selected_views:
            return selected_views, None
        raw_masks = row[self._pan_key]
        if not isinstance(raw_masks, list):
            return selected_views, None

        min_area = int(getattr(self.data_cfg, "mv_mask_min_area_px", 256))
        min_hits = int(getattr(self.data_cfg, "mv_mask_min_hit_pts", 8))
        min_frac = float(getattr(self.data_cfg, "mv_mask_min_hit_frac", 0.05))
        scored = []
        any_enforced = False
        for view_idx in selected_views:
            score = _mask_sanity_for_view(
                obj_pts_world,
                all_wrd2cams[view_idx],
                all_Ks_raw[view_idx],
                raw_masks[view_idx],
                row,
                pixel_support=float(pixel_support[view_idx]),
                min_area_px=min_area,
                min_hit_pts=min_hits,
                min_hit_frac=min_frac,
            )
            any_enforced = any_enforced or bool(score["enforced"])
            scored.append((view_idx, score))

        if not any_enforced:
            return selected_views, None
        ok = [view_idx for view_idx, score in scored if score["ok"]]
        if not ok:
            return [], "no selected support view passed target-mask sanity"
        return ok + [view_idx for view_idx in selected_views if view_idx not in ok], None

    def __getitem__(self, i: int) -> dict:
        sha256 = self.instances[i]
        data_cfg = self.data_cfg
        load_images = data_cfg.load_images and self.image_preprocessor is not None
        has_pc = data_cfg.num_points > 0

        raw_vertices, raw_faces = self._load_mesh(sha256)
        vertices, faces = get_instance_mesh(raw_vertices, raw_faces)
        if vertices is not None and faces is not None:
            vertices = utils.normalize_vertices(vertices, bound=self.norm_bound)
            faces = np.asarray(faces)

        cond = self._load_cond(sha256)
        uid = cond["uid"]
        scene_id = cond["scene_id"]
        T_norm_from_output = np.asarray(cond["T_norm_from_output"], dtype=np.float32)
        T_output_from_norm = np.asarray(cond["T_output_from_norm"], dtype=np.float32)
        scene_pts_norm = np.asarray(cond["scene_point_clouds"], dtype=np.float32)
        bboxes_norm = np.asarray(cond["bboxes"], dtype=np.float32)
        T_obj_to_norm = np.asarray(cond["object_to_norm_transforms"], dtype=np.float32)
        obj_idx_in_scene = target_box_index(bboxes_norm, T_obj_to_norm)

        hf_idx = self._uid_to_idx.get(uid, self._scene_id_to_idx.get(scene_id))
        if hf_idx is None:
            return self._make_empty(uid, vertices, faces, "uid/scene_id not found in HF dataset")
        row = self._hf[hf_idx]
        n_avail = len(row["wrd2cam_rects"])
        if n_avail < 2:
            return self._make_empty(uid, vertices, faces, f"single-view HF row (n_views={n_avail})")
        cached_features = self._load_feature_cache(uid, n_avail)

        T_norm_to_world = None
        if getattr(data_cfg, "mv_frame_correction", False):
            hf_obj = row.get("objects") if isinstance(row, dict) else None
            if hf_idx == self._uid_to_idx.get(uid) and hf_obj and hf_obj.get("transforms"):
                T_norm_to_world = norm_to_world_transform(cond, hf_obj["transforms"][0])
            else:
                return self._make_empty(
                    uid,
                    vertices,
                    faces,
                    "frame correction unavailable (no uid-matched HF row)",
                )

        all_wrd2cams = np.stack(
            [np.asarray(row["wrd2cam_rects"][v], dtype=np.float32) for v in range(n_avail)]
        )
        all_Ks_raw = np.stack(
            [np.asarray(row["Ks"][v], dtype=np.float32) for v in range(n_avail)]
        )
        k0 = all_Ks_raw[0]
        raw_img_hw = (int(round(float(k0[1, 2]) * 2)), int(round(float(k0[0, 2]) * 2)))

        k_max = int(getattr(data_cfg, "mv_covis_k_max", 8) or n_avail)
        if cached_features is not None:
            view_indices = cached_features["view_indices"]
            view_mask = cached_features["view_mask"]
            ref_view = int(cached_features["ref_view"])
        else:
            obj_pts_world, pixel_support = covis_object_supports(
                cond,
                all_wrd2cams,
                all_Ks_raw,
                raw_img_hw,
                rng=np.random.RandomState(_stable_seed(sha256, "objpts")),
                T_norm_to_world=T_norm_to_world,
            )
            min_support = int(getattr(data_cfg, "mv_covis_min_support_pts", 50))
            selected_views = _select_diverse_views(
                obj_pts_world,
                all_wrd2cams,
                all_Ks_raw,
                raw_img_hw,
                pixel_support,
                k_max=min(k_max, n_avail),
                min_support_pts=min_support,
            )
            if not selected_views:
                return self._make_empty(
                    uid,
                    vertices,
                    faces,
                    f"object has no view with >= {min_support} projected support points",
                )
            selected_views, sanity_error = self._reference_passes_mask_sanity(
                selected_views, obj_pts_world, all_wrd2cams, all_Ks_raw, row, pixel_support
            )
            if not selected_views:
                return self._make_empty(uid, vertices, faces, sanity_error)

            pad_slots = int(getattr(data_cfg, "num_views", k_max) or k_max)
            if len(selected_views) < pad_slots:
                view_indices = np.asarray(
                    selected_views + [selected_views[-1]] * (pad_slots - len(selected_views)),
                    dtype=np.int64,
                )
            else:
                view_indices = np.asarray(selected_views[:pad_slots], dtype=np.int64)
            view_mask = np.asarray(
                [j < len(selected_views) and j < pad_slots for j in range(pad_slots)],
                dtype=bool,
            )
            ref_view = 0

        wrd2cam_rects_n = [all_wrd2cams[v] for v in view_indices]
        Ks_raw_n = [all_Ks_raw[v] for v in view_indices]
        T_world_to_norm = (
            np.linalg.inv(T_norm_to_world).astype(np.float32)
            if T_norm_to_world is not None
            else T_norm_from_output
        )

        scene_trans_raw_n = []
        K_adj_n = []
        pad_info_n = []
        pv_n_list = []
        cached_hw = None
        if cached_features is not None:
            cached_hw = cached_features["cached_local_points"].shape[1:3]
        for local_idx, view_idx in enumerate(view_indices):
            scene_trans_raw_n.append(
                (
                    T_world_to_norm
                    @ np.linalg.inv(wrd2cam_rects_n[local_idx])
                    @ _CAM_YUP_TO_OPENCV_4
                ).astype(np.float32)
            )
            K_n = Ks_raw_n[local_idx].copy()
            pad_left = pad_top = 0
            if load_images and cached_features is None:
                img_arr = np.asarray(row["images"][int(view_idx)])
                proc = self.image_preprocessor(images=img_arr, return_tensors="pt")
                pv_n = proc["pixel_values"]
                pv_n_list.append(pv_n)
                out_h, out_w = pv_n.shape[2], pv_n.shape[3]
                pad_left = (out_w - img_arr.shape[1]) // 2
                pad_top = (out_h - img_arr.shape[0]) // 2
            elif cached_hw is not None:
                out_h, out_w = int(cached_hw[0]), int(cached_hw[1])
                pad_left = max((out_w - raw_img_hw[1]) // 2, 0)
                pad_top = max((out_h - raw_img_hw[0]) // 2, 0)
            else:
                out_h, out_w = raw_img_hw
            K_n[0, 2] += pad_left
            K_n[1, 2] += pad_top
            K_adj_n.append(K_n.astype(np.float32))
            pad_info_n.append(
                {"pad_top": pad_top, "pad_left": pad_left, "out_h": out_h, "out_w": out_w}
            )

        if self.is_train and getattr(data_cfg, "random_scale", False):
            bound = float(
                np.random.uniform(getattr(data_cfg, "random_scale_min", 0.75), self.norm_bound)
            )
        else:
            bound = self.norm_bound
        bboxes_scene, scene_pts_scene, normalize_matrix = utils.normalize_bboxes_with_point_clouds(
            bboxes_norm, scene_pts_norm, bound=bound, return_matrix=True
        )
        M_shift = np.eye(4, dtype=np.float32)
        if self.is_train and getattr(data_cfg, "random_shift", False):
            bboxes_scene, scene_pts_scene, M_shift = utils.random_shift_bboxes_with_point_clouds(
                bboxes_scene,
                scene_pts_scene,
                max_shift=getattr(data_cfg, "random_shift_max", 0.2),
                bound=bound,
                return_matrix=True,
            )
        S = (M_shift @ normalize_matrix).astype(np.float32)
        S_inv = np.linalg.inv(S).astype(np.float32)
        scene_transforms_n = [(S @ st).astype(np.float32) for st in scene_trans_raw_n]
        obj_canon_transform = (np.linalg.inv(T_obj_to_norm) @ S_inv).astype(np.float32)
        T_scene_to_world = (
            (T_norm_to_world if T_norm_to_world is not None else T_output_from_norm) @ S_inv
        ).astype(np.float32)

        if has_pc:
            obj_bbox_scene = bboxes_scene[obj_idx_in_scene]
            bbox_min = obj_bbox_scene.min(axis=0) - 0.05
            bbox_max = obj_bbox_scene.max(axis=0) + 0.05
            in_bbox = ((scene_pts_scene >= bbox_min) & (scene_pts_scene <= bbox_max)).all(axis=1)
            obj_pts_scene = scene_pts_scene[in_bbox]
            if len(obj_pts_scene) == 0:
                return self._make_empty(uid, vertices, faces, "empty seed crop")
            dummy_2d = np.zeros((len(obj_pts_scene), 2), dtype=np.float32)
            sampled_pc, sampled_pc_2d, pc_valid, sample_inds = subsample_point_clouds(
                obj_pts_scene,
                dummy_2d,
                data_cfg.num_points,
                self.is_train,
                data_cfg,
                data_cfg.with_normals,
            )
            pts3d = obj_pts_scene[sample_inds] if sample_inds is not None else obj_pts_scene[: data_cfg.num_points]
            pts3d_h = np.concatenate([pts3d, np.ones((len(pts3d), 1), dtype=np.float32)], axis=1)
            pts3d_world = (T_scene_to_world @ pts3d_h.T).T[:, :3]
            sampled_pc_2d = _project_world_points_to_normalized_pixels(
                pts3d_world,
                wrd2cam_rects_n[ref_view],
                K_adj_n[ref_view],
                int(pad_info_n[ref_view]["out_h"]),
                int(pad_info_n[ref_view]["out_w"]),
            )

        pan_stack = None
        if self._pan_key is not None:
            raw_masks = row[self._pan_key]
            if isinstance(raw_masks, list):
                pan_stack = np.stack(
                    [
                        _pad_pan_arr(_decode_pan_arr(raw_masks[int(v)]), pad_info_n[j])
                        for j, v in enumerate(view_indices)
                    ],
                    axis=0,
                )
            else:
                raise ValueError(f"panoptic_masks must be a per-view list, got {type(raw_masks)}")

        ret = {
            "uid": uid,
            "bboxes": bboxes_scene,
            "obj_indices": obj_idx_in_scene,
            "vertices": vertices,
            "faces": faces,
        }
        if has_pc:
            ret.update(
                point_clouds=sampled_pc,
                point_clouds_2d=sampled_pc_2d,
                point_clouds_valid=pc_valid,
                scene_transforms=np.stack(scene_transforms_n),
                K_per_view=np.stack(K_adj_n),
                view_mask=view_mask,
                ref_view=ref_view,
                obj_canon_transform=obj_canon_transform,
                view_indices=view_indices,
            )
        if cached_features is not None:
            h, w = cached_features["cached_local_points"].shape[1:3]
            ret["pixel_values"] = torch.zeros((len(view_indices), 3, h, w), dtype=torch.float32)
            ret["cached_local_points"] = cached_features["cached_local_points"]
            ret["cached_conf"] = cached_features["cached_conf"]
            ret["cached_dino_feats"] = cached_features["cached_dino_feats"]
        elif load_images:
            pv_stack = torch.cat(pv_n_list, dim=0)
            ret["pixel_values"] = pv_stack
            if self._cached_img_chw is None:
                self._cached_img_chw = (pv_stack.shape[1], pv_stack.shape[2], pv_stack.shape[3])
        if pan_stack is not None:
            ret["panoptic_masks"] = pan_stack
        return ret

    def _make_empty(self, uid: str, vertices, faces, reason: str | None = None) -> dict:
        warnings.warn(
            f"[Trellis2MVDataset] {reason or 'unconditionable item'} for uid={uid!r}; "
            "item contributes no loss.",
            stacklevel=3,
        )
        n = int(getattr(self.data_cfg, "num_views", 8) or 8)
        ret = {
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
            "view_indices": np.zeros(n, dtype=np.int64),
        }
        if self.data_cfg.load_images and self.image_preprocessor is not None:
            if self._cached_img_chw is not None:
                c, h, w = self._cached_img_chw
            else:
                c, h, w = 3, 504, 672
            ret["pixel_values"] = torch.zeros((n, c, h, w), dtype=torch.float32)
            if self._hf_has_panoptic:
                ret["panoptic_masks"] = np.zeros((n, h, w), dtype=np.int32)
        return ret
