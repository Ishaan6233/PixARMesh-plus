"""Dataset loader for the trellis2-MV per-object file format.

Each instance is stored as two files:
  mesh_dumps/{sha256}.pickle  — {'objects': [{'vertices': (V,3), 'faces': (F,3)}]}
  mv_cond/{sha256}.pt         — {'cond': {T_norm_from_output, scene_point_clouds, bboxes, ...}}

Images and camera parameters are cross-referenced from a local HuggingFace
3d-front-multiview[-full] dataset by scene_id.  The pre-computed
T_norm_from_output (world→gravity-aligned norm frame) is combined with the HF
dataset's wrd2cam_rects to produce scene_transforms without needing the
original 21-view source renders.  HF wrd2cam_rects are y-up/z-forward camera
frames, so the loader adapts them to the OpenCV camera frame consumed by K,
Pi3X local_points, and grid_sample projection.

Reference view is selected by covisibility: the local camera whose visible
scene points overlap most with all other local cameras.
"""

import csv
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import datasets as hf_datasets
from src.data import utils
from src.data.mesh import get_instance_mesh, subsample_point_clouds
from src.utils.config import DataConfig

# ── Numpy 2.x compatibility shim ────────────────────────────────────────────
# mesh_dumps pickles were serialised with numpy 2.x (numpy._core); register
# the missing module alias so they load in numpy 1.x envs.  No-op on 2.x
# where numpy._core already exists natively.
if np.__version__ < "2":
    import numpy.core as _np_core

    _np_mod = types.ModuleType("numpy._core")
    _np_mod.numeric = _np_core.numeric
    sys.modules.setdefault("numpy._core", _np_mod)
    sys.modules.setdefault("numpy._core.numeric", _np_core.numeric)
# ─────────────────────────────────────────────────────────────────────────────

_COVIS_SAMPLE = 2048  # downsample scene pts before covisibility loop (speed)
_OBJ_PTS_SAMPLE = 512  # max object pts used for per-view covisibility scoring
_CAM_YUP_TO_OPENCV_4 = np.diag(np.array([-1, -1, 1, 1], dtype=np.float32))
_CAM_YUP_TO_OPENCV_3 = np.diag(np.array([-1, -1, 1], dtype=np.float32))


def _world_points_to_opencv_camera(
    points_world: np.ndarray,
    wrd2cam_yup: np.ndarray,
) -> np.ndarray:
    """Project HF world points into the OpenCV camera frame used by K/Pi3X."""
    pts = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    pts_cam_yup = (np.asarray(wrd2cam_yup, dtype=np.float32) @ pts_h.T).T[:, :3]
    return (pts_cam_yup @ _CAM_YUP_TO_OPENCV_3.T).astype(np.float32)


def _project_world_points_to_normalized_pixels(
    points_world: np.ndarray,
    wrd2cam_yup: np.ndarray,
    K: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Project HF world points with the y-up→OpenCV adapter and grid_sample coords."""
    pts_cam = _world_points_to_opencv_camera(points_world, wrd2cam_yup)
    K = np.asarray(K, dtype=np.float32)
    z = pts_cam[:, 2]
    px = np.where(z > 1e-4, pts_cam[:, 0] / z, 0.0)
    py = np.where(z > 1e-4, pts_cam[:, 1] / z, 0.0)
    u = K[0, 0] * px + K[0, 2]
    v = K[1, 1] * py + K[1, 2]
    u_norm = (u + 0.5) / float(out_w) * 2.0 - 1.0
    v_norm = (v + 0.5) / float(out_h) * 2.0 - 1.0
    return np.stack([u_norm, v_norm], axis=1).astype(np.float32)


def load_conditioning_filter(root, expected_frame_correction: bool | None = None) -> set:
    """Keep-set of instance sha256s from the degenerate-conditioning sidecar.

    Built by scripts/data/build_conditioning_filter.py: drops instances whose HF row
    has <2 views or whose object projects in-frame in NO view (covis support all 0).
    The keep-set is only valid for the frame mode it was scored in — a broken-frame
    sidecar marks 47.7% of objects blind (99.2% of them actually visible), so
    `expected_frame_correction` is checked against conditioning_filter.meta.json.
    """
    path = Path(root) / "conditioning_filter.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"mv_filter_degenerate=True but {path} does not exist; build it with:\n"
            "  PYTHONPATH=. python scripts/data/build_conditioning_filter.py "
            f"--mesh-dataset {root}"
        )
    meta_path = Path(root) / "conditioning_filter.meta.json"
    if expected_frame_correction is not None:
        if not meta_path.exists():
            # A CSV without meta is exactly what a stale pre-fix (frame-broken, 44%-keep)
            # sidecar looks like — refusing is the only safe default.
            raise ValueError(
                f"{path} has no conditioning_filter.meta.json, so its frame mode is "
                "unknown (a pre-frame-correction sidecar silently drops ~56% of "
                "instances); rebuild it with scripts/data/build_conditioning_filter.py."
            )
        import json

        built_fc = bool(json.loads(meta_path.read_text()).get("frame_correction"))
        if built_fc != bool(expected_frame_correction):
            raise ValueError(
                f"conditioning_filter.csv was built with frame_correction={built_fc} "
                f"but the loader runs with mv_frame_correction={expected_frame_correction}; "
                "rebuild the sidecar with scripts/data/build_conditioning_filter.py."
            )
    keep = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if str(row["keep"]).strip().lower() in ("true", "1"):
                keep.add(row["sha256"])
    return keep


def _decode_pan_arr(m) -> np.ndarray:
    """Decode an RGB-encoded (H,W,3) or direct (H,W) panoptic mask to int32."""
    arr = np.array(m, dtype=np.uint32)
    if arr.ndim == 3:
        return (arr[..., 0] * 65536 + arr[..., 1] * 256 + arr[..., 2]).astype(np.int32)
    return arr.astype(np.int32)


def _pad_pan_arr(dec: np.ndarray, pad_info: dict) -> np.ndarray:
    """Center-pad a panoptic mask from raw resolution to preprocessor output resolution."""
    oh, ow = pad_info["out_h"], pad_info["out_w"]
    pt, pl = pad_info["pad_top"], pad_info["pad_left"]
    h, w = dec.shape
    if (h, w) == (oh, ow):
        return dec
    out = np.zeros((oh, ow), dtype=dec.dtype)
    out[pt:pt + h, pl:pl + w] = dec
    return out


def _covisibility_scores(
    obj_pts_world: np.ndarray,
    wrd2cams: np.ndarray,
    Ks: np.ndarray,
    img_hw: tuple,
) -> np.ndarray:
    """Per-view object pixel support: count of obj_pts_world projecting in-frame.

    obj_pts_world : (M, 3)  object-region world-frame points
    wrd2cams      : (N, 4, 4)
    Ks            : (N, 3, 3)
    img_hw        : (H, W)
    returns       : (N,) float32
    """
    H, W = img_hw
    N = len(wrd2cams)
    scores = np.zeros(N, dtype=np.float32)
    if len(obj_pts_world) == 0:
        return scores
    pts_h = np.concatenate(
        [obj_pts_world, np.ones((len(obj_pts_world), 1), dtype=np.float32)], axis=1
    )
    for n in range(N):
        pts_cam = (wrd2cams[n] @ pts_h.T).T[:, :3]
        z = pts_cam[:, 2]
        valid = z > 1e-4
        if not valid.any():
            continue
        uvw = Ks[n] @ pts_cam[valid].T  # (3, k)
        u = uvw[0] / uvw[2]
        v = uvw[1] / uvw[2]
        scores[n] = float(((u >= 0) & (u < W) & (v >= 0) & (v < H)).sum())
    return scores


def _covisibility_ref_view(
    pts_world: np.ndarray,
    wrd2cams: np.ndarray,
    Ks: np.ndarray,
    img_hw: tuple,
) -> int:
    """Return index of view with highest scene-level cross-view covisibility."""
    H, W = img_hw
    N = len(wrd2cams)
    scores = np.zeros(N, dtype=np.int64)
    pts_h = np.concatenate([pts_world, np.ones((len(pts_world), 1), dtype=np.float32)], axis=1)
    for n in range(N):
        pts_cam_n = (wrd2cams[n] @ pts_h.T).T[:, :3]
        valid_n = pts_cam_n[:, 2] > 1e-4
        if not valid_n.any():
            continue
        sub = pts_cam_n[valid_n]
        cam2wrd_n = np.linalg.inv(wrd2cams[n])
        sub_h = np.concatenate([sub, np.ones((len(sub), 1), dtype=np.float32)], axis=1)
        sub_world_h = (cam2wrd_n @ sub_h.T).T
        for m in range(N):
            if m == n:
                continue
            pts_cam_m = (wrd2cams[m] @ sub_world_h.T).T[:, :3]
            z_m = pts_cam_m[:, 2]
            valid_m = z_m > 1e-4
            if not valid_m.any():
                continue
            uvw = Ks[m] @ pts_cam_m[valid_m].T
            u = uvw[0] / uvw[2]
            v = uvw[1] / uvw[2]
            scores[n] += int(((u >= 0) & (u < W) & (v >= 0) & (v < H)).sum())
    return int(np.argmax(scores))


def norm_to_world_transform(cond: dict, hf_obj_transform: np.ndarray) -> np.ndarray:
    """Exact per-scene norm→HF-world transform from matched object transforms.

    cond["T_output_from_norm"] is a PURE UNIFORM SCALE (verified 40/40 sampled
    instances, 2026-07-03) — it does NOT reach the HF camera world; the true frame
    offset is a per-scene rigid+scale transform (median 90° rotation). Both the HF row
    (`objects.transforms[0]`, object-canonical→HF-world) and mv_cond
    (`object_to_norm_transforms`, object-canonical→norm) describe the SAME object, so

        M = T_hf_obj @ inv(T_obj_to_norm)     (norm → HF-world)

    is exact and identical across all objects of a scene (max deviation 0.0 measured).
    Without it, scene_transforms register Pi3X geometry into a frame rotated ~90°
    from the seeds/bboxes/GT-layout frame, and covisibility scoring marks 47.7% of
    instances object-blind while their objects are plainly visible (panoptic masks).
    """
    T_on = np.array(cond["object_to_norm_transforms"], dtype=np.float32)
    return (np.array(hf_obj_transform, dtype=np.float32) @ np.linalg.inv(T_on)).astype(
        np.float32
    )


def target_box_index(bboxes_norm: np.ndarray, T_obj_to_norm: np.ndarray) -> int:
    """Index of this object's box among cond["bboxes"] (K,8,3), by nearest centroid.

    cond["bboxes"] holds ALL objects of the scene (only ~3% of cond files are
    single-box); object_to_norm_transforms is per-object-correct, so match each
    candidate box's centroid to the object's own transform translation. The uid
    "__objNNNN" suffix agrees on ~98% of a 500-object sample but picks a box several
    scene-units away on the other ~2%, so nearest-centroid is the robust choice.
    """
    centroids = np.asarray(bboxes_norm).mean(axis=1)  # (K, 3)
    return int(np.linalg.norm(centroids - np.asarray(T_obj_to_norm)[:3, 3], axis=1).argmin())


def covis_object_supports(
    cond: dict,
    wrd2cams: np.ndarray,
    Ks: np.ndarray,
    img_hw: tuple,
    rng=None,
    T_norm_to_world: np.ndarray | None = None,
) -> tuple:
    """Object-region covisibility supports, exactly as the training loader computes them.

    Single source of truth shared by __getitem__, scripts/eval/eval_pi3x_depth.py and
    scripts/data/build_conditioning_filter.py. `rng` (np.random.RandomState) makes the
    object-point subsample deterministic; None preserves the loader's global-RNG draw.

    T_norm_to_world: the norm→HF-world transform used to project points through the HF
    cameras. Pass `norm_to_world_transform(cond, hf_row_obj_transform)` for the
    frame-correct projection; None falls back to cond["T_output_from_norm"] (pure
    scale — the historical, frame-broken behavior kept for A/B measurement).

    Returns (obj_pts_for_scoring (M,3) world frame, pixel_support (N,) float32).
    """
    if T_norm_to_world is not None:
        T_out = np.array(T_norm_to_world, dtype=np.float32)
    else:
        T_out = np.array(cond["T_output_from_norm"], dtype=np.float32)
    sc_pts_norm = np.array(cond["scene_point_clouds"], dtype=np.float32)
    bboxes_norm = np.array(cond["bboxes"], dtype=np.float32)
    T_obj = np.array(cond["object_to_norm_transforms"], dtype=np.float32)
    obj_idx = target_box_index(bboxes_norm, T_obj)

    # Crop in the canonical/norm frame, where the stored object box is defined.
    # Cropping after the norm->world frame correction would turn a rotated box into a
    # looser world-axis AABB and can scoop nearby objects into the support count.
    corners_n = bboxes_norm[obj_idx]
    in_bbox = (
        (sc_pts_norm >= corners_n.min(0) - 0.1)
        & (sc_pts_norm <= corners_n.max(0) + 0.1)
    ).all(axis=1)
    # No whole-scene fallback: an empty crop means the object has no points in the
    # scene cloud at all — object-blind-equivalent — so let the supports come out
    # all-zero instead of scoring (and seeding) on the entire scene.
    obj_pts_norm = sc_pts_norm[in_bbox]
    obj_pts = (T_out[:3, :3] @ obj_pts_norm.T + T_out[:3, 3:]).T
    if len(obj_pts) > _OBJ_PTS_SAMPLE:
        chooser = rng if rng is not None else np.random
        obj_pts = obj_pts[chooser.choice(len(obj_pts), _OBJ_PTS_SAMPLE, replace=False)]
    return obj_pts, _covisibility_scores(obj_pts, wrd2cams, Ks, img_hw)


def _select_diverse_views(
    obj_pts_world: np.ndarray,
    wrd2cams: np.ndarray,
    Ks: np.ndarray,
    img_hw: tuple,
    pixel_support: np.ndarray,
    k_max: int,
    min_support_pts: int = 50,
) -> list:
    """Greedy diverse view selection maximising coverage and minimising overlap.

    Greedily builds a set of at most k_max views.  Each step picks the candidate
    that maximises  pixel_support[n] * (1 - max_covisibility_with_selected[n]).

    Returns selected world-view indices with the highest-support view first (used
    as the reference view by the caller).  Falls back gracefully when no view
    meets min_support_pts.
    """
    H, W = img_hw
    N = len(wrd2cams)

    if float(np.max(pixel_support)) <= 0:
        # Object-blind: the object projects in-frame in NO view. Previously this fell
        # through to the greedy loop, whose `sup < 1: continue` skipped every view and
        # silently returned one arbitrary view that cannot see the object (measured on
        # 741/18,334 train instances, 2026-07-03). Fail explicitly; callers mark the
        # instance invalid so it contributes no loss.
        return []

    candidates = [n for n in range(N) if pixel_support[n] >= min_support_pts]
    if not candidates:
        # No view meets threshold; take all sorted by support (best-effort)
        candidates = sorted(range(N), key=lambda n: -float(pixel_support[n]))
    elif len(candidates) < k_max:
        # Fewer than k_max views clear the support threshold: backfill with the
        # next-best sub-threshold views (real, distinct data) instead of letting
        # the caller pad the remaining slots by repeating an already-selected
        # view. Same downstream compute cost (the padded slot is processed by
        # Pi3X either way) but strictly more signal — a weak real view still
        # contributes some coverage, a duplicate contributes none (view_mask
        # gates it out entirely).
        backfill = sorted(
            (n for n in range(N) if n not in candidates),
            key=lambda n: -float(pixel_support[n]),
        )
        candidates = candidates + backfill[: k_max - len(candidates)]

    if len(candidates) <= k_max:
        return sorted(candidates, key=lambda n: -float(pixel_support[n]))

    # Precompute in-frame object-point indices per candidate view for fast covisibility
    pts_h = np.concatenate(
        [obj_pts_world, np.ones((len(obj_pts_world), 1), dtype=np.float32)], axis=1
    )
    cand_inframe: list = []  # list of np.ndarray of point indices
    for n in candidates:
        pts_cam = (wrd2cams[n] @ pts_h.T).T[:, :3]
        z = pts_cam[:, 2]
        valid = z > 1e-4
        if valid.any():
            uvw = Ks[n] @ pts_cam[valid].T
            u = uvw[0] / uvw[2]
            v = uvw[1] / uvw[2]
            inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            cand_inframe.append(np.where(valid)[0][inside])
        else:
            cand_inframe.append(np.array([], dtype=np.int64))

    cand_to_local = {c: i for i, c in enumerate(candidates)}

    # Initialise with highest-support view (= reference view)
    best_start = max(candidates, key=lambda n: float(pixel_support[n]))
    selected = [best_start]
    selected_sets = [set(cand_inframe[cand_to_local[best_start]].tolist())]
    remaining = [c for c in candidates if c != best_start]

    while len(selected) < k_max and remaining:
        best_n, best_score = None, -1.0
        for n in remaining:
            sup = float(pixel_support[n])
            if sup < 1:
                continue
            in_n = set(cand_inframe[cand_to_local[n]].tolist())
            max_covis = max(
                (len(in_n & s_set) / max(sup, 1.0) for s_set in selected_sets),
                default=0.0,
            )
            score = sup * (1.0 - max_covis)
            if score > best_score:
                best_score, best_n = score, n
        if best_n is None:
            break
        selected.append(best_n)
        selected_sets.append(set(cand_inframe[cand_to_local[best_n]].tolist()))
        remaining.remove(best_n)

    return selected


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

        # Degenerate-conditioning filter (applied AFTER the 95/5 split so each
        # instance's train/val membership is unchanged by filtering).
        if getattr(data_cfg, "mv_filter_degenerate", False):
            keep_set = load_conditioning_filter(
                self.root,
                expected_frame_correction=getattr(data_cfg, "mv_frame_correction", False),
            )
            n_before = len(self.instances)
            self.instances = [s for s in self.instances if s in keep_set]
            print(
                f"[Trellis2MVDataset] mv_filter_degenerate: kept "
                f"{len(self.instances)}/{n_before} instances "
                f"({n_before - len(self.instances)} degenerate dropped)"
            )

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
        # Per-object row lookup: HF rows are per-object (same scene rows share
        # images/cameras but objects.transforms is THIS row's object). The frame
        # correction needs the row matching the instance's uid, not the scene's
        # last row.
        self._uid_to_idx: dict[str, int] = {}
        if "uid" in hf_split.column_names:
            for i, u in enumerate(hf_split["uid"]):
                self._uid_to_idx[u if isinstance(u, str) else u[0]] = i
        if getattr(data_cfg, "mv_frame_correction", False) and not self._uid_to_idx:
            raise ValueError(
                f"mv_frame_correction=True requires a 'uid' column in the HF dataset "
                f"({hf_path}); without it every item would silently degrade to a "
                "no-loss empty (frame offset unrecoverable)."
            )

        # ── Cache column existence flags ─────────────────────────────────────
        self._hf_has_panoptic = (
            "panoptic_masks" in hf_split.features
            or "panoptic_mask" in hf_split.features
        )
        self._pan_key = (
            "panoptic_masks" if "panoptic_masks" in hf_split.features
            else ("panoptic_mask" if "panoptic_mask" in hf_split.features else None)
        )
        self._cached_img_chw: tuple | None = None  # (C, H, W) populated on first image load
        self._cached_pan_hw: tuple | None = None   # (H, W) populated on first panoptic load

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
        # (K,8,3) gravity_norm frame — ALL objects in the scene, not just this one; the
        # previous hardcoded [0] silently used a different object's box (see
        # target_box_index for the selection rationale).
        bboxes_norm = np.array(cond["bboxes"], dtype=np.float32)
        T_obj_to_norm = np.array(cond["object_to_norm_transforms"], dtype=np.float32)  # (4,4)
        uid = cond["uid"]
        scene_id = cond["scene_id"]
        obj_idx_in_scene = target_box_index(bboxes_norm, T_obj_to_norm)

        # ── 3. Local HF row ──────────────────────────────────────────────────
        # Prefer the per-object row (uid match): images/cameras are identical across
        # a scene's rows, but objects.transforms — needed for the frame correction —
        # belongs to this row's object.
        hf_idx = self._uid_to_idx.get(uid, self._scene_id_to_idx.get(scene_id))
        if hf_idx is None:
            return self._make_empty(uid, vertices, faces)

        row = self._hf[hf_idx]
        n_avail = len(row["wrd2cam_rects"])
        if n_avail < 2:
            # Multi-view conditioning is impossible on a single-view row; enforce the
            # sidecar's min-views rule at runtime so filter-off paths are safe too.
            return self._make_empty(
                uid, vertices, faces, reason=f"single-view HF row (n_views={n_avail})"
            )

        # ── 3b. Frame correction (norm → HF-world) ──────────────────────────
        # cond["T_output_from_norm"] is a pure scale and does NOT reach the HF camera
        # world (median 90° per-scene rotation missing); see norm_to_world_transform.
        # T_n2w is used for covisibility scoring, scene_transforms, and seed 2D
        # projection so conditioning geometry and supervision share ONE frame.
        T_n2w = None
        if getattr(data_cfg, "mv_frame_correction", False):
            hf_obj = row.get("objects") if isinstance(row, dict) else None
            if (
                hf_idx == self._uid_to_idx.get(uid)
                and hf_obj is not None
                and hf_obj.get("transforms")
            ):
                T_n2w = norm_to_world_transform(cond, hf_obj["transforms"][0])
            else:
                # No exact object match — the frame offset is unrecoverable; a
                # misregistered instance trains wrong, so contribute no loss.
                return self._make_empty(
                    uid, vertices, faces,
                    reason="frame correction unavailable (no uid-matched HF row)",
                )

        # Covisibility-based view selection ─────────────────────────────────
        # Load lightweight camera data for ALL available views first, score each
        # view by how many object-region points project in-frame, then greedily
        # select up to mv_covis_k_max diverse views.  This replaces the old
        # "take first num_views views" truncation that silently discarded most
        # views in 20+-view mesh_datasets scenes.

        k_max = getattr(data_cfg, "mv_covis_k_max", 8) or n_avail
        min_sup = getattr(data_cfg, "mv_covis_min_support_pts", 50)
        pad_slots = getattr(data_cfg, "num_views", k_max) or k_max  # tensor padding target

        all_wrd2cams = np.stack(
            [np.array(row["wrd2cam_rects"][v], dtype=np.float32) for v in range(n_avail)]
        )  # (n_avail, 4, 4)
        all_Ks_raw = np.stack(
            [np.array(row["Ks"][v], dtype=np.float32) for v in range(n_avail)]
        )  # (n_avail, 3, 3)

        # Approximate raw image HW from principal point (cx≈W/2, cy≈H/2) — avoids
        # loading any image before the selection decision.
        k0 = all_Ks_raw[0]
        raw_img_hw = (int(round(float(k0[1, 2]) * 2)), int(round(float(k0[0, 2]) * 2)))

        # Object-region points in world frame for scoring (shared helper — same code
        # path as build_conditioning_filter.py and eval_pi3x_depth.py)
        obj_pts_for_scoring, pixel_support = covis_object_supports(
            cond, all_wrd2cams, all_Ks_raw, raw_img_hw, T_norm_to_world=T_n2w
        )
        selected_views = _select_diverse_views(
            obj_pts_for_scoring, all_wrd2cams, all_Ks_raw, raw_img_hw,
            pixel_support, k_max=min(k_max, n_avail), min_support_pts=min_sup,
        )
        if not selected_views:
            # Object-blind instance (support 0 in every view): no view can condition
            # this object — contribute no loss instead of training on a blind view.
            return self._make_empty(
                uid, vertices, faces,
                reason="object projects in-frame in no view (covis support all 0)",
            )

        # Pad to pad_slots with the last selected view (maintains fixed tensor size)
        n_selected = len(selected_views)
        if n_selected < pad_slots:
            _vidx = selected_views + [selected_views[-1]] * (pad_slots - n_selected)
        else:
            _vidx = selected_views[:pad_slots]
            n_selected = pad_slots
        view_valid = np.array(
            [i < len(selected_views) and i < pad_slots for i in range(pad_slots)], dtype=bool
        )
        N_views = pad_slots
        # First selected view acts as the reference (highest pixel support)
        ref_view = 0

        wrd2cam_rects_n = [all_wrd2cams[v] for v in _vidx]
        Ks_raw_n = [all_Ks_raw[v] for v in _vidx]
        images_n = [row["images"][v] for v in _vidx]

        # ── 4. Per-view images, K_adj, and raw scene_transforms ─────────────
        # scene_trans_raw_n[n] maps OpenCV camera_n → gravity_norm frame (no
        # bbox-normalization yet). mv_frame_correction only recovers the exact
        # norm↔HF-world frame; the HF cameras themselves are y-up/z-forward
        # (x-left, y-up), so OpenCV/Pi3X camera points need the same
        # diag(-1,-1,1) adapter used by the SV path.
        T_world_to_norm = (
            np.linalg.inv(T_n2w).astype(np.float32) if T_n2w is not None
            else T_norm_from_output
        )
        scene_trans_raw_n = []
        K_adj_n = []
        pad_info_n = []
        pv_n_list = []
        for n in range(N_views):
            scene_trans_raw_n.append(
                (
                    T_world_to_norm
                    @ np.linalg.inv(wrd2cam_rects_n[n])
                    @ _CAM_YUP_TO_OPENCV_4
                ).astype(np.float32)
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
        T_scene_to_world = (
            (T_n2w if T_n2w is not None else T_output_from_norm) @ S_inv
        ).astype(np.float32)

        # ref_view = 0: the first selected view (highest object pixel support) is
        # the reference.  This was set during covisibility selection above.

        # ── 6. Object voxel seed (cond_pcs) ─────────────────────────────────
        if has_pc:
            obj_bbox_scene = bboxes_scene[obj_idx_in_scene]  # (8,3) corners in scene frame
            bbox_min = obj_bbox_scene.min(axis=0) - 0.05
            bbox_max = obj_bbox_scene.max(axis=0) + 0.05
            in_bbox = (
                (sc_pts_scene[:, 0] >= bbox_min[0]) & (sc_pts_scene[:, 0] <= bbox_max[0]) &
                (sc_pts_scene[:, 1] >= bbox_min[1]) & (sc_pts_scene[:, 1] <= bbox_max[1]) &
                (sc_pts_scene[:, 2] >= bbox_min[2]) & (sc_pts_scene[:, 2] <= bbox_max[2])
            )
            obj_pts_scene = sc_pts_scene[in_bbox]  # (K, 3) scene frame
            if len(obj_pts_scene) == 0:
                # No whole-scene fallback (same policy as covis_object_supports): seeding
                # on the entire scene gives every object identical, target-free
                # conditioning — worse than contributing no loss.
                return self._make_empty(
                    uid, vertices, faces,
                    reason="empty seed crop (no scene points in object bbox)",
                )

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
            K_ref = K_adj_n[ref_view]
            out_h_ref = pad_info_n[ref_view]["out_h"]
            out_w_ref = pad_info_n[ref_view]["out_w"]
            sampled_pc_2d = _project_world_points_to_normalized_pixels(
                pts3d_world, wrd2cam_rects_n[ref_view], K_ref, out_h_ref, out_w_ref
            )

        # ── 7. Panoptic masks ────────────────────────────────────────────────
        pan_stack = None
        if self._pan_key is not None:
            raw_masks = row[self._pan_key]
            if isinstance(raw_masks, list):
                raw_masks = [raw_masks[v] for v in _vidx]
                pan_stack = np.stack(
                    [_pad_pan_arr(_decode_pan_arr(m), pad_info_n[n])
                     for n, m in enumerate(raw_masks)],
                    axis=0,
                )
            else:
                raise ValueError(
                    f"panoptic_masks must be a list of per-view masks, got {type(raw_masks)}"
                )

        # ── 8. Assemble output ───────────────────────────────────────────────
        ret = {
            "uid": uid,
            "bboxes": bboxes_scene,   # (K, 8, 3) ALL scene objects, in [-bound, bound]
            # The collator indexes bboxes[obj_indices] for the layout supervision target
            # (and eval_layout_mv / infer do the same for gt-layout); it MUST point at
            # this instance's own box, not scene-object 0.
            "obj_indices": obj_idx_in_scene,
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
            if self._cached_img_chw is None:
                self._cached_img_chw = (pv_stack.shape[1], pv_stack.shape[2], pv_stack.shape[3])

        if pan_stack is not None:
            ret["panoptic_masks"] = pan_stack  # (N, H, W) int32

        return ret

    def _make_empty(self, uid: str, vertices, faces, reason: str | None = None) -> dict:
        """Fallback item for instances that cannot be conditioned.

        Used when scene_id is missing from the HF dataset, or when the object is
        object-blind (covisibility support 0 in every view). point_clouds_valid=False
        causes the collator to mask all loss for this item, so zero bboxes / identity
        transforms do not affect training. pixel_values and panoptic_masks are included
        so the collator's batch-level key-detection (based on examples[0]) doesn't
        KeyError when a fallback appears mid-batch.
        """
        import warnings
        warnings.warn(
            f"[Trellis2MVDataset] {reason or 'scene_id not found in HF data'} for "
            f"uid={uid!r}; item contributes no loss (pc_valid=False)",
            stacklevel=3,
        )
        n = getattr(self.data_cfg, "num_views", 8) or 8
        load_images = self.data_cfg.load_images and self.image_preprocessor is not None
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
        }
        if load_images:
            # Use cached shape from a successful item; fall back to DINOv2 default
            # for 484×648 3D-FRONT input (divisor pads to 504×672).
            if self._cached_img_chw is not None:
                C, H, W = self._cached_img_chw
            else:
                C, H, W = 3, 504, 672
            ret["pixel_values"] = torch.zeros((n, C, H, W), dtype=torch.float32)
            if self._hf_has_panoptic:
                ret["panoptic_masks"] = np.zeros((n, H, W), dtype=np.int32)
        return ret
