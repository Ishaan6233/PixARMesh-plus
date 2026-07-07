#!/usr/bin/env python3
"""Pi3X metric/scale-inconsistency diagnostic: per-view Pi3X depth vs GT depth.

Runs the frozen Pi3X encoder over MV scenes and compares its camera-frame depth
(local_points[..., 2]) against the GT `depths` stored in the HF dataset row,
per view. Camera-frame depth needs no extrinsics, so the comparison is immune
to any extrinsics issues upstream.

Reported per view: raw metric error (absRel/RMSE/silog), per-view scale
s_v = median(gt/pred), residual error after per-view scale and after RANSAC
scale+shift (align_depth — the same treatment SV gives Depth Pro), and error at
the per-scene scale s_scene. Per scene: the spread of s_v around s_scene
(r_std et al.) — the cross-view scale inconsistency that a scale-free eval
cannot erase.

Modes:
  all-views   — one joint Pi3X forward over all row views per scene (default).
  train-views — replicate the trellis2 loader's per-object covisibility view
                selection (<=8 views) and run Pi3X on exactly those subsets;
                requires the trellis2 mesh_dataset (--mesh-dataset) and runs
                per object instance. Train split only (no validation mv_cond).

Usage:
  # smoke test
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval/eval_pi3x_depth.py \
      --dataset datasets/3d-front-multiview --split validation \
      --out outputs/diagnostics/pi3x_depth/validation --num-samples 8

  # sharded full run (one process per GPU)
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python scripts/eval/eval_pi3x_depth.py \
      --dataset datasets/3d-front-multiview-full --split train \
      --out outputs/diagnostics/pi3x_depth/train --num-shards 4 --shard-idx 0

  # aggregation (CPU)
  python scripts/eval/eval_pi3x_depth.py --dataset ... --out ... --aggregate
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.utils import align_depth
from src.utils.config import ModelConfig

DEPTH_MAX_M = 10.0  # GT depth encoding range: (1 - raw/255) * 10 meters
_TRELLIS2_DEFAULT = (
    "datasets/mesh_datasets/datasets/"
    "3d-front-trellis2-slat-mv-da3-aug-srcperturb-r5-qfcat-obj015-light-bgtex-20260629"
)


def parse_args():
    p = argparse.ArgumentParser(description="Pi3X depth vs GT depth diagnostic")
    p.add_argument("--dataset", required=True, help="Path to HF 3d-front-multiview[-full] dataset")
    p.add_argument("--split", default=None, help="Split name (auto-detects val/validation/train)")
    p.add_argument("--mode", choices=["all-views", "train-views"], default="all-views")
    p.add_argument("--mesh-dataset", default=_TRELLIS2_DEFAULT,
                   help="trellis2 mesh_dataset root (metadata.csv + mv_cond/), train-views mode")
    p.add_argument("--pi3x-ckpt", default="checkpoints/pi3x")
    p.add_argument("--image-preprocessor", default="facebook/dpt-dinov2-small-nyu")
    p.add_argument("--image-size-divisor", type=int, default=28)
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=-1,
                   help="Cap on scenes/instances per shard (debug); -1 = all")
    p.add_argument("--max-views", type=int, default=21,
                   help="Cap on views per joint forward (strided subset above this)")
    p.add_argument("--min-valid-px", type=int, default=1000,
                   help="Views with fewer valid px are recorded as skipped")
    p.add_argument("--conf-thresh", type=float, default=0.5,
                   help="Post-sigmoid confidence threshold for the *_conf metric variants")
    p.add_argument("--align-subsample", type=int, default=20000,
                   help="Pixel subsample for the RANSAC scale+shift fit")
    p.add_argument("--include-far-plane", action="store_true",
                   help="Keep GT px saturated at the 10 m far plane (raw==0); default excludes")
    p.add_argument("--covis-k-max", type=int, default=8,
                   help="train-views: max covisibility-selected views (matches mv_covis_k_max)")
    p.add_argument("--covis-min-support", type=int, default=50,
                   help="train-views: min projected obj px for view eligibility")
    p.add_argument("--pad-to-slots", type=int, default=0,
                   help="train-views: pad the selected views to N slots by repeating the "
                        "last view, exactly like the wired trellis2 loader — the duplicates "
                        "enter the joint Pi3X forward (0 = off; wired ingestion uses 8)")
    p.add_argument("--autocast-bf16", action="store_true",
                   help="Run the Pi3X forward under bf16 autocast (OOM fallback; off = fp32)")
    p.add_argument("--aggregate", action="store_true",
                   help="Merge shard JSONLs into parquet + summary + plots (CPU only)")
    p.add_argument("--compare-to", default=None,
                   help="aggregate: another run's output dir; join per-(scene_id, view_idx)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers (unit-tested in tests/test_eval_pi3x_depth.py)
# ─────────────────────────────────────────────────────────────────────────────

def stable_seed(*parts) -> int:
    """Deterministic 32-bit seed from string parts (hash() is salted per process)."""
    key = "|".join(str(p) for p in parts)
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def decode_gt_depth(raw: np.ndarray) -> np.ndarray:
    """8-bit GT depth PNG -> meters; inverse-encoded (raw 0 = 10 m, raw 255 = 0 m).

    Matches src/data/mesh.py:605 exactly.
    """
    return (1.0 - raw.astype(np.float32) / 255.0) * DEPTH_MAX_M


def make_valid_mask(raw: np.ndarray, pred_z: np.ndarray, include_far_plane: bool = False):
    """Joint GT/pred validity mask.

    raw==255 decodes to depth 0 (the dataset's `depth > 1e-6` invalid convention);
    raw==0 is the saturated 10 m far plane — excluded by default because its true
    depth is only known to be >= 10 m.
    """
    if include_far_plane:
        gt_valid = raw <= 254
    else:
        gt_valid = (raw >= 1) & (raw <= 254)
    return gt_valid & np.isfinite(pred_z) & (pred_z > 1e-6)


def crop_padding(arr: np.ndarray, pad_top: int, pad_left: int, raw_hw: tuple) -> np.ndarray:
    """Crop (N, Hp, Wp) preprocessor-padded maps back to the raw (H, W) window."""
    h, w = raw_hw
    out = arr[:, pad_top:pad_top + h, pad_left:pad_left + w]
    assert out.shape[-2:] == (h, w), f"crop {out.shape} != raw {raw_hw}"
    return out


def fit_scale_shift(pred: np.ndarray, gt: np.ndarray):
    """RANSAC scale+shift via align_depth; returns (a, b, aligned = a*pred + b)."""
    aligned = align_depth(pred, gt)
    if float(np.ptp(pred)) <= 0:
        return float("nan"), float("nan"), aligned
    a, b = np.polyfit(pred, aligned, 1)  # exact recovery: aligned is affine in pred
    return float(a), float(b), aligned


def _err_stats(pred: np.ndarray, gt: np.ndarray) -> tuple:
    abs_rel = float(np.mean(np.abs(pred - gt) / gt))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    return abs_rel, rmse


def compute_view_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                         conf_sig: np.ndarray, seed: int,
                         align_subsample: int = 20000, conf_thresh: float = 0.5) -> dict:
    """All per-view depth metrics over the valid mask. pred/gt/conf_sig: (H, W)."""
    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)
    c = conf_sig[mask].astype(np.float64)

    abs_rel_raw, rmse_raw = _err_stats(p, g)
    e = np.log(p) - np.log(g)
    silog_raw = float(np.sqrt(max(np.mean(e**2) - np.mean(e) ** 2, 0.0)) * 100.0)

    s_v = float(np.median(g / p))
    s_v_log = float(np.exp(np.mean(np.log(g) - np.log(p))))
    abs_rel_pv, rmse_pv = _err_stats(s_v * p, g)

    # RANSAC scale+shift on a subsample (align_depth seeds off the global numpy RNG)
    rng = np.random.RandomState(seed)
    if len(p) > align_subsample:
        sub = rng.choice(len(p), align_subsample, replace=False)
    else:
        sub = np.arange(len(p))
    np.random.seed(seed)
    try:
        ss_a, ss_b, aligned = fit_scale_shift(p[sub], g[sub])
        abs_rel_ss, rmse_ss = _err_stats(aligned, g[sub])
    except Exception:
        ss_a = ss_b = abs_rel_ss = rmse_ss = float("nan")

    cm = c > conf_thresh
    n_valid_conf = int(cm.sum())
    if n_valid_conf > 0:
        abs_rel_raw_conf, _ = _err_stats(p[cm], g[cm])
        abs_rel_pv_conf, _ = _err_stats(s_v * p[cm], g[cm])
    else:
        abs_rel_raw_conf = abs_rel_pv_conf = float("nan")

    return {
        "n_valid": int(mask.sum()),
        "absRel_raw": abs_rel_raw,
        "rmse_raw": rmse_raw,
        "silog_raw": silog_raw,
        "s_v": s_v,
        "s_v_log": s_v_log,
        "absRel_pv_scale": abs_rel_pv,
        "rmse_pv_scale": rmse_pv,
        "ss_a": ss_a,
        "ss_b": ss_b,
        "absRel_ss": abs_rel_ss,
        "rmse_ss": rmse_ss,
        "mean_conf": float(c.mean()),
        "n_valid_conf": n_valid_conf,
        "absRel_raw_conf": abs_rel_raw_conf,
        "absRel_pv_scale_conf": abs_rel_pv_conf,
    }


def scene_scale_metrics(s_views: np.ndarray) -> dict:
    """Cross-view scale-inconsistency stats from the per-view scales of one scene."""
    s_views = np.asarray(s_views, dtype=np.float64)
    n = len(s_views)
    if n == 0:
        return {k: float("nan") for k in (
            "s_scene", "s_scene_geo", "r_std", "log_r_std", "r_mean_abs_dev", "r_spread")}
    s_scene = float(np.median(s_views))
    s_scene_geo = float(np.exp(np.mean(np.log(s_views))))
    out = {"s_scene": s_scene, "s_scene_geo": s_scene_geo}
    if n < 2:
        out.update({k: float("nan") for k in ("r_std", "log_r_std", "r_mean_abs_dev", "r_spread")})
        return out
    r = s_views / s_scene
    out["r_std"] = float(np.std(r))
    out["log_r_std"] = float(np.std(np.log(s_views)))
    out["r_mean_abs_dev"] = float(np.mean(np.abs(r - 1.0)))
    out["r_spread"] = float(s_views.max() / s_views.min() - 1.0)
    return out


def strided_view_subset(n_views: int, max_views: int) -> list:
    """Viewpoint-coverage-preserving subset; never chunk a joint forward instead."""
    if n_views <= max_views:
        return list(range(n_views))
    return sorted(set(np.round(np.linspace(0, n_views - 1, max_views)).astype(int).tolist()))


def _sanitize(rec: dict) -> dict:
    """JSON-safe record: numpy scalars -> python, NaN/inf -> None."""
    out = {}
    for k, v in rec.items():
        if isinstance(v, (np.floating, float)):
            v = float(v)
            out[k] = v if np.isfinite(v) else None
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        elif isinstance(v, np.bool_):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


def load_done_keys(scenes_path: Path) -> set:
    """Completion markers for resume: the `key` field of existing scene records."""
    done = set()
    if scenes_path.exists():
        with open(scenes_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / model setup
# ─────────────────────────────────────────────────────────────────────────────

def load_hf_split(dataset_path: str, split: str | None):
    import datasets as hf_datasets

    ds_path = str(Path(dataset_path).absolute())
    if (Path(dataset_path) / "dataset_dict.json").exists() or (
        Path(dataset_path) / "dataset_info.json"
    ).exists():
        raw = hf_datasets.load_from_disk(ds_path)
    else:
        raw = hf_datasets.load_dataset(ds_path)
    if isinstance(raw, hf_datasets.Dataset):
        return raw, "<flat>"
    if split is not None and split in raw:
        return raw[split], split
    for k in ("validation", "val", "test", "train"):
        if k in raw:
            return raw[k], k
    raise ValueError(f"No usable split in {dataset_path}; available: {list(raw.keys())}")


def build_pi3x(ckpt_path: str, device: str):
    from src.models.utils import get_pi3x_encoder

    model_cfg = ModelConfig(
        vocab_size=1, num_pos_tokens=512, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        pi3x_ckpt_path=ckpt_path, pi3x_disable_multimodal=True,
    )
    return get_pi3x_encoder(model_cfg).to(device).float().eval()


def preprocess_views(pil_images: list, processor):
    """DINOv2-preprocess N views -> ((1,N,3,Hp,Wp), pad_top, pad_left, (H,W))."""
    pvs, raw_hw = [], None
    for img in pil_images:
        arr = np.array(img)
        if raw_hw is None:
            raw_hw = arr.shape[:2]
        assert arr.shape[:2] == raw_hw, f"mixed view resolutions: {arr.shape[:2]} vs {raw_hw}"
        pvs.append(processor(images=arr, return_tensors="pt")["pixel_values"][0])
    pv = torch.stack(pvs).unsqueeze(0)  # (1, N, 3, Hp, Wp)
    out_h, out_w = pv.shape[-2:]
    pad_left = (out_w - raw_hw[1]) // 2
    pad_top = (out_h - raw_hw[0]) // 2
    return pv, pad_top, pad_left, raw_hw


def run_pi3x_depth(pi3x, pixel_values: torch.Tensor, device: str, autocast_bf16: bool):
    """One joint forward -> (pred_z (N,Hp,Wp), conf_sigmoid (N,Hp,Wp)) numpy float32."""
    with torch.inference_mode():
        pv = pixel_values.to(device).float()
        if autocast_bf16 and device.startswith("cuda"):
            with torch.autocast("cuda", torch.bfloat16):
                out = pi3x.forward_all_views_joint(pv)
        else:
            out = pi3x.forward_all_views_joint(pv)
        pred_z = out["local_points"][0, ..., 2].float().cpu().numpy()
        # Pi3X conf is a raw logit; every consumer sigmoids it — do the same.
        conf_sig = out["conf"][0, ..., 0].float().sigmoid().cpu().numpy()
    return pred_z, conf_sig


# ─────────────────────────────────────────────────────────────────────────────
# Per-scene processing (shared by both modes)
# ─────────────────────────────────────────────────────────────────────────────

def process_scene(pi3x, processor, args, *, uid, scene_id, images, depths, view_indices,
                  n_views_total, key, instance_uid=None, subset=None):
    """One joint Pi3X forward over `view_indices`; returns (scene_rec, view_recs)."""
    t0 = time.time()
    pv, pad_top, pad_left, raw_hw = preprocess_views([images[v] for v in view_indices], processor)
    pred_z, conf_sig = run_pi3x_depth(pi3x, pv, args.device, args.autocast_bf16)
    pred_z = crop_padding(pred_z, pad_top, pad_left, raw_hw)
    conf_sig = crop_padding(conf_sig, pad_top, pad_left, raw_hw)

    base = {"uid": uid, "scene_id": scene_id, "mode": args.mode,
            "instance_uid": instance_uid, "subset": subset}
    n_unique = len(set(view_indices))
    view_recs, pixels, s_used, seen_v = [], [], [], set()
    for k, v in enumerate(view_indices):
        # Pad-duplicated slots shape the joint forward above but are scored once.
        if v in seen_v:
            continue
        seen_v.add(v)
        raw = np.array(depths[v])
        if raw.ndim == 3:
            raw = raw[..., 0]
        assert raw.shape == raw_hw, f"GT depth {raw.shape} != image {raw_hw}"
        gt = decode_gt_depth(raw)
        mask = make_valid_mask(raw, pred_z[k], args.include_far_plane)
        n_valid = int(mask.sum())
        rec = dict(base, key=key, view_idx=int(v), n_views_scene=int(n_views_total),
                   n_views_used=n_unique, n_valid=n_valid)
        if n_valid < args.min_valid_px:
            rec["skipped"] = True
            pixels.append(None)
        else:
            m = compute_view_metrics(
                pred_z[k], gt, mask, conf_sig[k], seed=stable_seed(scene_id, v),
                align_subsample=args.align_subsample, conf_thresh=args.conf_thresh,
            )
            rec.update(m, skipped=False)
            pixels.append((pred_z[k][mask].astype(np.float32), gt[mask].astype(np.float32)))
            s_used.append(m["s_v"])
        view_recs.append(rec)

    stats = scene_scale_metrics(np.array(s_used))
    s_scene = stats["s_scene"]
    for rec, px in zip(view_recs, pixels):
        if px is None or not np.isfinite(s_scene):
            rec["absRel_scene_scale"] = rec["rmse_scene_scale"] = float("nan")
        else:
            p, g = px
            rec["absRel_scene_scale"], rec["rmse_scene_scale"] = _err_stats(
                s_scene * p.astype(np.float64), g.astype(np.float64))

    scored = [r for r in view_recs if not r["skipped"]]
    scene_rec = dict(
        base, key=key,
        n_views=int(n_views_total), n_views_used=n_unique, n_slots=len(view_indices),
        n_views_scored=len(scored),
        wall_time_s=round(time.time() - t0, 3), **stats,
    )
    for m in ("absRel_raw", "absRel_pv_scale", "absRel_scene_scale", "absRel_ss", "silog_raw"):
        vals = [r[m] for r in scored if np.isfinite(r.get(m, float("nan")))]
        scene_rec[f"mean_{m}"] = float(np.mean(vals)) if vals else float("nan")
    return scene_rec, view_recs


# ─────────────────────────────────────────────────────────────────────────────
# Work-item enumeration per mode
# ─────────────────────────────────────────────────────────────────────────────

def _norm_uid(u):
    return u[0] if isinstance(u, (list, tuple)) else u


def iter_mode_a(args, hf_split):
    """Yield one work item per unique scene of this shard.

    Rows are per-object (~4 rows share one scene) but images/depths are
    scene-level and identical across those rows, so only the first row per
    scene_id is processed and records are keyed by scene_id.
    """
    seen, first_rows = set(), []
    for i, sid in enumerate(hf_split["scene_id"]):
        if sid not in seen:
            seen.add(sid)
            first_rows.append(i)
    idxs = [i for k, i in enumerate(first_rows) if k % args.num_shards == args.shard_idx]
    if args.num_samples > 0:
        idxs = idxs[: args.num_samples]
    for i in idxs:
        row = hf_split[i]
        uid = _norm_uid(row["uid"])
        n = len(row["images"])
        yield {
            "key": row["scene_id"], "uid": uid, "scene_id": row["scene_id"],
            "images": row["images"], "depths": row["depths"],
            "view_indices": strided_view_subset(n, args.max_views),
            "n_views_total": n, "instance_uid": None, "subset": None,
        }


def load_trellis2_instances(mesh_root: Path) -> list:
    """metadata.csv rows + the trellis2 loader's deterministic 95/5 holdout flag."""
    insts = []
    with open(mesh_root / "metadata.csv", newline="") as f:
        for row in csv.DictReader(f):
            insts.append({"sha256": row["sha256"], "uid": row["uid"],
                          "scene_id": row["scene_id"]})
    # Same recipe as Trellis2MVDataset.__init__ (RandomState(42) shuffle, first 5% = holdout)
    all_idx = list(range(len(insts)))
    np.random.RandomState(42).shuffle(all_idx)
    holdout = set(all_idx[: max(1, int(0.05 * len(all_idx)))])
    for i, inst in enumerate(insts):
        inst["subset"] = "holdout" if i in holdout else "train"
    return insts


def select_covis_views(cond: dict, all_wrd2cams: np.ndarray, all_Ks: np.ndarray,
                       raw_img_hw: tuple, k_max: int, min_support: int, seed: int,
                       T_norm_to_world: np.ndarray | None = None) -> list:
    """The wired trellis2 view selection (shared helpers = single source of truth).

    May return [] for object-blind instances (covis support 0 in every view) —
    callers must handle the empty case.
    """
    from src.data.trellis2_mv import _select_diverse_views, covis_object_supports

    obj_pts, support = covis_object_supports(
        cond, all_wrd2cams, all_Ks, raw_img_hw, rng=np.random.RandomState(seed),
        T_norm_to_world=T_norm_to_world,
    )
    return _select_diverse_views(
        obj_pts, all_wrd2cams, all_Ks, raw_img_hw, support,
        k_max=min(k_max, len(all_wrd2cams)), min_support_pts=min_support,
    )


def iter_mode_b(args, hf_split):
    """Yield one work item per trellis2 object instance of this shard.

    Instances are grouped by scene and shards take whole scene groups, so the
    per-scene view-set cache in run_shard stays effective and one HF row decode
    serves all of a scene's objects.
    """
    mesh_root = Path(args.mesh_dataset)
    insts = load_trellis2_instances(mesh_root)

    scene_to_idx = {}
    for i, sid in enumerate(hf_split["scene_id"]):
        scene_to_idx.setdefault(sid, i)
    uid_to_idx = {_norm_uid(u): i for i, u in enumerate(hf_split["uid"])}

    groups: dict = {}
    for inst in insts:
        groups.setdefault(inst["scene_id"], []).append(inst)
    group_keys = sorted(groups)
    shard_keys = [k for i, k in enumerate(group_keys) if i % args.num_shards == args.shard_idx]

    n_emitted = 0
    for sid in shard_keys:
        hf_idx = scene_to_idx.get(sid)
        row = None
        for inst in sorted(groups[sid], key=lambda d: d["sha256"]):
            if args.num_samples > 0 and n_emitted >= args.num_samples:
                return
            n_emitted += 1
            if hf_idx is None:
                yield {"error": f"scene_id {sid} not in HF dataset", "key": inst["sha256"]}
                continue
            cond_path = mesh_root / "mv_cond" / f"{inst['sha256']}.pt"
            try:
                cond = torch.load(cond_path, map_location="cpu", weights_only=False)["cond"]
            except Exception as e:
                yield {"error": f"mv_cond load failed: {e}", "key": inst["sha256"]}
                continue
            uid_idx = uid_to_idx.get(cond["uid"])
            if uid_idx is None:
                yield {"error": "no uid-matched HF row (frame correction unavailable)",
                       "key": inst["sha256"]}
                continue
            obj_row = hf_split[uid_idx]
            transforms = (obj_row.get("objects") or {}).get("transforms")
            if not transforms:
                yield {"error": "uid-matched HF row has no objects.transforms",
                       "key": inst["sha256"]}
                continue
            if row is None:
                row = hf_split[hf_idx]  # decode images/depths once per scene
            all_w2c = np.stack([np.array(m, dtype=np.float32) for m in row["wrd2cam_rects"]])
            all_ks = np.stack([np.array(m, dtype=np.float32) for m in row["Ks"]])
            # Raw HW from the principal point (cx≈W/2, cy≈H/2) — trellis2_mv.py L397-400
            raw_hw = (int(round(float(all_ks[0][1, 2]) * 2)), int(round(float(all_ks[0][0, 2]) * 2)))
            from src.data.trellis2_mv import norm_to_world_transform

            selected = select_covis_views(
                cond, all_w2c, all_ks, raw_hw,
                k_max=args.covis_k_max, min_support=args.covis_min_support,
                seed=stable_seed(inst["sha256"], "objpts"),
                T_norm_to_world=norm_to_world_transform(cond, transforms[0]),
            )
            if not selected:
                yield {"error": "object-blind: covis support 0 in every view",
                       "key": inst["sha256"]}
                continue
            # Wired-loader parity: pad to fixed slots by repeating the last selected
            # view (trellis2_mv.py L424-425); duplicates enter the joint forward.
            if args.pad_to_slots > 0 and len(selected) < args.pad_to_slots:
                selected = list(selected) + [selected[-1]] * (args.pad_to_slots - len(selected))
            yield {
                "key": inst["sha256"], "uid": _norm_uid(row["uid"]), "scene_id": sid,
                "images": row["images"], "depths": row["depths"],
                "view_indices": [int(v) for v in selected],
                "n_views_total": len(row["images"]),
                "instance_uid": inst["uid"], "subset": inst["subset"],
            }


# ─────────────────────────────────────────────────────────────────────────────
# Shard runner
# ─────────────────────────────────────────────────────────────────────────────

def run_shard(args):
    from transformers import AutoImageProcessor

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"shard{args.shard_idx}"
    scenes_path = out_dir / f"scenes_{tag}.jsonl"
    views_path = out_dir / f"views_{tag}.jsonl"
    errors_path = out_dir / f"errors_{tag}.log"

    done = load_done_keys(scenes_path)
    if done:
        print(f"[{tag}] resuming: {len(done)} records already complete")

    hf_split, split_name = load_hf_split(args.dataset, args.split)
    keep_cols = ["uid", "scene_id", "images", "depths"]
    if args.mode == "train-views":
        keep_cols += ["wrd2cam_rects", "Ks", "objects"]
    hf_split = hf_split.select_columns([c for c in keep_cols if c in hf_split.column_names])
    print(f"[{tag}] dataset {args.dataset} split={split_name} rows={len(hf_split)} "
          f"mode={args.mode} shard {args.shard_idx}/{args.num_shards}")

    processor = AutoImageProcessor.from_pretrained(
        args.image_preprocessor, size_divisor=args.image_size_divisor)
    print(f"[{tag}] loading Pi3X from {args.pi3x_ckpt} ...")
    pi3x = build_pi3x(args.pi3x_ckpt, args.device)

    items = iter_mode_a(args, hf_split) if args.mode == "all-views" else iter_mode_b(args, hf_split)

    # train-views: identical (scene, view-set) across objects -> reuse computed records
    cache_scene, cache = None, {}
    n_done, running_r_std, running_abs_rel = 0, [], []
    with open(views_path, "a") as vf, open(scenes_path, "a") as sf, open(errors_path, "a") as ef:
        for item in tqdm(items, desc=tag, unit="scene"):
            key = item["key"]
            if key in done:
                continue
            if "error" in item:
                ef.write(f"{key}\t{item['error']}\n")
                ef.flush()
                continue
            try:
                if item["scene_id"] != cache_scene:
                    cache_scene, cache = item["scene_id"], {}
                vset = tuple(sorted(item["view_indices"]))
                if vset in cache:
                    scene_rec, view_recs = cache[vset]
                    scene_rec = dict(scene_rec, key=key, instance_uid=item["instance_uid"],
                                     subset=item["subset"])
                    view_recs = [dict(r, key=key, instance_uid=item["instance_uid"],
                                      subset=item["subset"]) for r in view_recs]
                else:
                    scene_rec, view_recs = process_scene(
                        pi3x, processor, args,
                        uid=item["uid"], scene_id=item["scene_id"], images=item["images"],
                        depths=item["depths"], view_indices=item["view_indices"],
                        n_views_total=item["n_views_total"], key=key,
                        instance_uid=item["instance_uid"], subset=item["subset"],
                    )
                    if args.mode == "train-views":
                        cache[vset] = (scene_rec, view_recs)
            except Exception:
                import traceback

                ef.write(f"{key}\n{traceback.format_exc()}\n")
                ef.flush()
                continue

            for r in view_recs:
                vf.write(json.dumps(_sanitize(r)) + "\n")
            vf.flush()
            # Scene line last + flush = durable completion marker for resume
            sf.write(json.dumps(_sanitize(scene_rec)) + "\n")
            sf.flush()

            n_done += 1
            if np.isfinite(scene_rec.get("r_std") or float("nan")):
                running_r_std.append(scene_rec["r_std"])
            if np.isfinite(scene_rec.get("mean_absRel_raw") or float("nan")):
                running_abs_rel.append(scene_rec["mean_absRel_raw"])
            if n_done % 50 == 0:
                print(f"\n[{tag}] {n_done} done | median r_std="
                      f"{np.median(running_r_std):.4f} | median absRel_raw="
                      f"{np.median(running_abs_rel):.4f}")

    print(f"[{tag}] finished: {n_done} new records -> {scenes_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _read_jsonl_glob(out_dir: Path, pattern: str):
    import pandas as pd

    frames = []
    for path in sorted(out_dir.glob(pattern)):
        try:
            frames.append(pd.read_json(path, lines=True))
        except ValueError:
            print(f"warning: unreadable {path}, skipping")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _pctls(series):
    s = series.dropna()
    if len(s) == 0:
        return {"n": 0}
    return {"n": int(len(s)), "mean": float(s.mean()), "p10": float(s.quantile(0.10)),
            "median": float(s.median()), "p90": float(s.quantile(0.90))}


def aggregate(args):
    import pandas as pd

    out_dir = Path(args.out)
    views = _read_jsonl_glob(out_dir, "views_shard*.jsonl")
    scenes = _read_jsonl_glob(out_dir, "scenes_shard*.jsonl")
    if scenes.empty or views.empty:
        print(f"No records under {out_dir} — nothing to aggregate.")
        return
    scenes = scenes.drop_duplicates(subset=["key"], keep="last")
    views = views.drop_duplicates(subset=["key", "view_idx"], keep="last")
    views = views[views["key"].isin(set(scenes["key"]))]  # drop orphans from crashed shards
    views.to_parquet(out_dir / "views.parquet", index=False)
    scenes.to_parquet(out_dir / "scenes.parquet", index=False)

    v = views[~views["skipped"].fillna(True).astype(bool)]
    sc = scenes[scenes["n_views_scored"] >= 2]

    summary = {
        "mode": scenes["mode"].iloc[0],
        "n_scenes": int(len(scenes)),
        "n_scenes_scored": int(len(sc)),
        "n_views": int(len(views)),
        "n_views_scored": int(len(v)),
        "raw_metric_error": {
            "absRel_raw": _pctls(v["absRel_raw"]),
            "rmse_raw_m": _pctls(v["rmse_raw"]),
            "silog_raw": _pctls(v["silog_raw"]),
            "s_v": _pctls(v["s_v"]),
        },
        "residual_after_alignment": {
            "absRel_pv_scale": _pctls(v["absRel_pv_scale"]),
            "absRel_ss": _pctls(v["absRel_ss"]),
        },
        "cross_view_inconsistency": {
            "r_std": _pctls(sc["r_std"]),
            "log_r_std": _pctls(sc["log_r_std"]),
            "r_mean_abs_dev": _pctls(sc["r_mean_abs_dev"]),
            "r_spread": _pctls(sc["r_spread"]),
        },
        "fusion_gap_absRel_scene_minus_pv": _pctls(
            sc["mean_absRel_scene_scale"] - sc["mean_absRel_pv_scale"]),
        "confidence_variant": {
            "absRel_raw_conf": _pctls(v["absRel_raw_conf"]),
            "absRel_pv_scale_conf": _pctls(v["absRel_pv_scale_conf"]),
        },
    }

    if args.compare_to:
        other = Path(args.compare_to) / "views.parquet"
        if other.exists():
            ov = pd.read_parquet(other)
            ov = ov[~ov["skipped"].fillna(True).astype(bool)]
            j = v.merge(ov, on=["scene_id", "view_idx"], suffixes=("", "_other"))
            if len(j):
                summary["compare_to"] = {
                    "dir": str(args.compare_to),
                    "n_joined_views": int(len(j)),
                    "delta_absRel_raw(this-other)": _pctls(j["absRel_raw"] - j["absRel_raw_other"]),
                    "delta_abs_log_scale": _pctls(np.abs(np.log(j["s_v"] / j["s_v_other"]))),
                }
        else:
            print(f"warning: --compare-to given but {other} missing")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    md = _summary_md(summary)
    (out_dir / "summary.md").write_text(md)
    print(md)
    _plots(out_dir, v, sc)
    print(f"\nWrote {out_dir}/views.parquet, scenes.parquet, summary.json, summary.md, plots/")


def _fmt(d: dict, pct: bool = False) -> str:
    if d.get("n", 0) == 0:
        return "n=0"
    f = 100.0 if pct else 1.0
    u = "%" if pct else ""
    return (f"median {d['median'] * f:.2f}{u}, mean {d['mean'] * f:.2f}{u}, "
            f"p90 {d['p90'] * f:.2f}{u} (n={d['n']})")


def _summary_md(s: dict) -> str:
    r, a, x = s["raw_metric_error"], s["residual_after_alignment"], s["cross_view_inconsistency"]
    lines = [
        f"# Pi3X depth vs GT — {s['mode']}",
        "",
        f"Scenes: {s['n_scenes']} ({s['n_scenes_scored']} with >=2 scored views) | "
        f"views scored: {s['n_views_scored']}/{s['n_views']}",
        "",
        "## 1. Raw metric error (Pi3X absolute scale vs GT)",
        f"- absRel_raw: {_fmt(r['absRel_raw'], pct=True)}",
        f"- RMSE: {_fmt(r['rmse_raw_m'])} m | silog: {_fmt(r['silog_raw'])}",
        f"- per-view scale s_v = median(gt/pred): {_fmt(r['s_v'])} "
        "(1.0 = perfectly metric)",
        "",
        "## 2. Residual after alignment",
        f"- after per-view scale: absRel {_fmt(a['absRel_pv_scale'], pct=True)}",
        f"- after RANSAC scale+shift (align_depth): absRel {_fmt(a['absRel_ss'], pct=True)}",
        "",
        "## 3. Cross-view scale inconsistency (headline; prior estimate 8-13%)",
        f"- r_std = std(s_v / s_scene): {_fmt(x['r_std'], pct=True)}",
        f"- log_r_std: {_fmt(x['log_r_std'], pct=True)} | "
        f"mean|r-1|: {_fmt(x['r_mean_abs_dev'], pct=True)}",
        f"- max/min spread - 1: {_fmt(x['r_spread'], pct=True)}",
        "",
        "## 4. Fusion-relevant gap (scale-free eval erases one scale/scene, not per-view)",
        f"- mean absRel@scene_scale - mean absRel@per-view scale: "
        f"{_fmt(s['fusion_gap_absRel_scene_minus_pv'], pct=True)}",
        "",
        "## 5. Confidence-filtered variant (sigmoid(conf) > thresh)",
        f"- absRel_raw: {_fmt(s['confidence_variant']['absRel_raw_conf'], pct=True)} | "
        f"absRel_pv: {_fmt(s['confidence_variant']['absRel_pv_scale_conf'], pct=True)}",
    ]
    if "compare_to" in s:
        c = s["compare_to"]
        lines += [
            "",
            f"## 6. Comparison vs {c['dir']} (joined on scene_id+view_idx, "
            f"n={c['n_joined_views']})",
            f"- delta absRel_raw (this - other): {_fmt(c['delta_absRel_raw(this-other)'], pct=True)}",
            f"- |log scale ratio|: {_fmt(c['delta_abs_log_scale'], pct=True)}",
        ]
    lines += [
        "",
        "_Caveat: GT depth is 8-bit quantized (bin 10/255 = 0.039 m -> ~1-2% absRel floor at "
        "2-4 m). Scale-ratio metrics are unaffected to first order._",
    ]
    return "\n".join(lines)


def _plots(out_dir: Path, v, sc):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = out_dir / "plots"
    plots.mkdir(exist_ok=True)

    def _save(name):
        plt.tight_layout()
        plt.savefig(plots / name, dpi=120)
        plt.close()

    if len(sc):
        r = sc["r_std"].dropna() * 100
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(r, bins=60)
        axes[0].set(xlabel="per-scene r_std (%)", ylabel="scenes",
                    title="Cross-view scale inconsistency")
        axes[1].plot(np.sort(r), np.linspace(0, 1, len(r)))
        axes[1].set(xlabel="per-scene r_std (%)", ylabel="CDF")
        axes[1].grid(alpha=0.3)
        _save("r_std_hist_cdf.png")

        plt.figure(figsize=(5, 4))
        plt.scatter(sc["n_views_scored"], sc["r_std"] * 100, s=4, alpha=0.3)
        plt.xlabel("views scored")
        plt.ylabel("r_std (%)")
        _save("r_std_vs_nviews.png")

    if len(v):
        plt.figure(figsize=(5, 4))
        plt.hist(v["s_v"].dropna(), bins=np.geomspace(0.2, 5, 60))
        plt.xscale("log")
        plt.axvline(1.0, color="k", ls="--", lw=1)
        plt.xlabel("per-view scale s_v = median(gt/pred)")
        plt.ylabel("views")
        _save("s_v_hist.png")

        plt.figure(figsize=(6, 4))
        for col, label in [("absRel_raw", "raw"), ("absRel_pv_scale", "per-view scale"),
                           ("absRel_ss", "scale+shift"), ("absRel_scene_scale", "scene scale")]:
            d = (v[col].dropna() * 100).clip(upper=100)
            plt.plot(np.sort(d), np.linspace(0, 1, len(d)), label=label)
        plt.xlabel("absRel (%)")
        plt.ylabel("CDF")
        plt.legend()
        plt.grid(alpha=0.3)
        _save("absrel_cdfs.png")

        vv = v.merge(sc[["key", "s_scene"]], on="key", how="inner")
        if len(vv):
            vv = vv[np.isfinite(vv["s_v"]) & np.isfinite(vv["s_scene"])]
            plt.figure(figsize=(5, 4))
            plt.scatter(vv["mean_conf"], np.abs(np.log(vv["s_v"] / vv["s_scene"])), s=3, alpha=0.2)
            plt.xlabel("mean sigmoid(conf)")
            plt.ylabel("|log r_v|")
            plt.title("Does confidence predict scale outliers?")
            _save("conf_vs_scale_outlier.png")
    print(f"plots -> {plots}")


def main():
    args = parse_args()
    if args.aggregate:
        aggregate(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
