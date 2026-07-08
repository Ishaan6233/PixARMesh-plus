#!/usr/bin/env python3
"""Empirical purity / size / effective-adaptive-size diagnostic for MV voxels.

Extends scripts/eval/eval_voxels.py (which only scores obj_voxels' purity vs.
a Chamfer proxy) with the three things needed to judge "how good are the
object AND scene voxels, and does adaptive_fps_voxelize's AnySplat-style
adaptive-size claim actually hold spatially":

  obj_purity          — fraction of obj_voxels projecting inside the target SAM mask
  ctx_target_overlap   — fraction of ctx_voxels ALSO inside the target mask (redundancy
                         with obj_voxels — ctx's job is to cover the REST of the scene)
  {obj,ctx}_mean_nn_dist    — mean nearest-neighbour spacing = physical voxel "size"
  {obj,ctx}_extent_diag     — AABB diagonal = physical footprint of the voxel cloud
  ctx_over_obj_extent_ratio — ctx_extent_diag / obj_extent_diag (>>1 expected: ctx should
                              spread past the object, not clump on top of it)
  {obj,ctx}_adaptive_corr    — Spearman corr(per-voxel Pi3X confidence, local voxel
                              density) across the voxel set. Positive & significant =
                              the adaptive-size property (finer voxels in high-confidence
                              regions) is actually manifesting; ~0 = it isn't.
  {obj,ctx}_adaptive_ratio   — mean NN-spacing in the low-confidence tercile / high-
                              confidence tercile of voxels. >>1 confirms finer voxels
                              where Pi3X is more confident.

Usage:
  python scripts/eval/eval_voxels_adaptive.py \\
      --dataset   datasets/3d-front-multiview \\
      --pi3x-ckpt checkpoints/pi3x \\
      --num-samples 100 \\
      --out       results/voxel_eval_adaptive
"""

import argparse
import json
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import scipy.spatial
import scipy.stats
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

import datasets as hf_datasets

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from metrics.chamfer import chamfer_distance
from src.data.mesh import get_mesh_dataset, transform_3d_front_multiview
from src.models.discovery import available_methods, get_discovery_fn
from src.models.frozen_geo_encoder import (
    _get_per_view_target_ids,
    _project_pts_to_views,
    _sample_mask_ids,
    build_geo_obj_pc,
)
from src.models.utils import get_pi3x_encoder
from src.utils.config import DataConfig, ModelConfig

# Reuse the proven collate/purity/ref-view helpers from the existing harness
# instead of duplicating them.
sys.path.insert(0, str(Path(__file__).parent))
from eval_voxels import _compute_purity, _eval_collate, _select_ref_view  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset",        required=True,
                   help="Path to the 3d-front-multiview dataset, OR (with "
                        "--dataset-type 3d-front-trellis2-mv) the trellis2 mesh_dataset dir")
    p.add_argument("--dataset-type",   default="3d-front-multiview",
                   choices=["3d-front-multiview", "3d-front-trellis2-mv"])
    p.add_argument("--trellis2-hf-path", default="datasets/3d-front-multiview-full",
                   help="Local HF dataset trellis2 cross-references for images/cameras/masks")
    p.add_argument("--pi3x-ckpt",      required=True)
    p.add_argument("--num-samples",    type=int, default=100)
    p.add_argument("--out",            default="results/voxel_eval_adaptive")
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pool-size",      type=int, default=8192)
    p.add_argument("--num-obj-voxels", type=int, default=512)
    p.add_argument("--num-ctx-voxels", type=int, default=1024)
    p.add_argument("--min-views",      type=int, default=3)
    p.add_argument("--depth-rtol",     type=float, default=100.0)
    p.add_argument("--conf-threshold", type=float, default=0.5)
    p.add_argument("--mask-seeded-pool", action="store_true", default=True)
    p.add_argument("--method",         default="consensus")
    p.add_argument("--voxel-sampling", default="fps",
                   choices=["fps", "grid", "adaptive", "adaptive_grid"],
                   help="Object-voxel downsampler after discovery.")
    p.add_argument("--knn-k",          type=int, default=4,
                   help="k for local-density estimate (mean dist to k nearest neighbours)")
    return p.parse_args()


def _local_nn_dist(pts: np.ndarray, k: int) -> np.ndarray:
    """Mean distance to the k nearest neighbours of each point — inverse local density,
    i.e. the *realized* voxel spacing at that location."""
    n = pts.shape[0]
    if n <= k:
        return np.full(n, np.nan)
    tree = scipy.spatial.cKDTree(pts)
    dists, _ = tree.query(pts, k=k + 1)   # includes self at column 0
    return dists[:, 1:].mean(axis=1)


def _sample_conf_at_points(
    pts: torch.Tensor,               # (V, 3) scene space — UNBATCHED (single item)
    scene_transforms: torch.Tensor,  # (N, 4, 4)
    K_per_view: torch.Tensor,        # (N, 3, 3)
    conf_map: torch.Tensor,          # (N, H, W) post-sigmoid Pi3X confidence
    view_mask: torch.Tensor,         # (N,) bool
    H: int, W: int,
) -> np.ndarray:
    """Best-view (max across visible views) Pi3X confidence at each point's projection.

    _project_pts_to_views takes single-item (unbatched) tensors — (V,3) / (N,4,4) /
    (N,3,3) — the same convention _compute_purity uses; no leading batch dim.
    """
    pix_coords, pts_cam = _project_pts_to_views(pts, scene_transforms, K_per_view, H, W)
    # pix_coords: (V, N, 2); pts_cam: (V, N, 3)
    V, N = pix_coords.shape[0], pix_coords.shape[1]
    c_flat = pix_coords.permute(1, 0, 2).reshape(N, V, 1, 2)   # (N, V, 1, 2)
    conf_flat = conf_map.float().reshape(N, 1, H, W)
    sampled = F.grid_sample(conf_flat, c_flat, mode="bilinear",
                             padding_mode="zeros", align_corners=False)
    sampled = sampled.squeeze(-1).squeeze(1)   # (N, V)
    in_front = pts_cam[..., 2] > 0             # (V, N)
    vm = view_mask[None, :].expand(V, N)       # (V, N)
    valid = (in_front & vm).permute(1, 0)      # (N, V) — matches sampled's layout
    sampled = sampled.masked_fill(~valid, float("-inf"))
    best, _ = sampled.max(dim=0)                # (V,)
    best[~valid.any(dim=0)] = float("nan")
    return best.cpu().numpy()


def _sample_canonical_surface(vertices: np.ndarray, faces: np.ndarray, n: int) -> np.ndarray:
    """Area-weighted uniform surface sampling of the GT canonical mesh. Duplicated from
    Front3DCollator._sample_canonical_surface (src/data/collator.py) rather than imported,
    so this eval-only diagnostic doesn't depend on the training collator."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    if v.ndim != 2 or len(v) == 0:
        return np.zeros((n, 3), dtype=np.float32)
    if f.ndim == 2 and f.shape[1] == 3 and len(f) > 0:
        tris = v[f.astype(np.int64)]
        cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        total = areas.sum()
        if total > 0:
            probs = areas / total
            ti = np.random.choice(len(f), size=n, p=probs)
            uu = np.random.rand(n, 1)
            ww = np.random.rand(n, 1)
            over = (uu + ww) > 1
            uu[over], ww[over] = 1 - uu[over], 1 - ww[over]
            a, b, c = tris[ti, 0], tris[ti, 1], tris[ti, 2]
            return (a + uu * (b - a) + ww * (c - a)).astype(np.float32)
    idx = np.random.randint(0, len(v), size=n)
    return v[idx].astype(np.float32)


def _canonicalize_like_model(obj_voxels: np.ndarray, obj_canon_transform: np.ndarray) -> np.ndarray:
    """Reproduce EdgeRunner.get_mv_inputs_with_cond's obj_geom_voxels computation exactly:
    rotate scene-frame obj_voxels by obj_canon_transform's ROTATION only, then re-center /
    re-scale by the voxels' OWN observed extent (normalize_vertices(0.95) convention).

    This — not a raw scene-frame comparison — is the right way to ground-truth-check
    discovery. obj_canon_transform's rotation is exact (built from data-loader-known GT
    transforms), but Pi3X is scale-invariant by design (CLAUDE.md) so obj_voxels' absolute
    scale/translation in scene frame is NOT metric-trustworthy — comparing them directly
    against the exact-metric GT mesh would conflate "discovery found the wrong object" with
    "Pi3X's known depth-scale drift," which are very different problems. Canonicalizing the
    same way the model itself does isolates shape/region correctness from that drift.
    """
    if obj_voxels.shape[0] == 0:
        return obj_voxels
    R = obj_canon_transform[:3, :3]
    v = obj_voxels @ R.T
    vmin, vmax = v.min(0), v.max(0)
    center = 0.5 * (vmin + vmax)
    scale = (2 * 0.95) / max(float((vmax - vmin).max()), 1e-6)
    return (v - center) * scale


def _adaptive_size_stats(conf: np.ndarray, nn_dist: np.ndarray, knn_k: int) -> dict:
    """Spearman corr(confidence, local density) + low/high-confidence-tercile spacing
    ratio. Returns NaNs when too few valid (non-degenerate) voxels are present."""
    valid = np.isfinite(conf) & np.isfinite(nn_dist) & (nn_dist > 0)
    if valid.sum() < max(3 * knn_k, 12):
        return {"adaptive_corr": float("nan"), "adaptive_ratio": float("nan"), "n_valid": int(valid.sum())}
    c = conf[valid]
    density = 1.0 / nn_dist[valid]
    corr, _pval = scipy.stats.spearmanr(c, density)
    order = np.argsort(c)
    n = len(c)
    lo_idx = order[: n // 3]
    hi_idx = order[-(n // 3):]
    lo_spacing = float(nn_dist[valid][lo_idx].mean())
    hi_spacing = float(nn_dist[valid][hi_idx].mean())
    ratio = lo_spacing / hi_spacing if hi_spacing > 1e-9 else float("nan")
    return {
        "adaptive_corr":  float(corr),
        "adaptive_ratio": ratio,
        "lo_conf_spacing": lo_spacing,
        "hi_conf_spacing": hi_spacing,
        "n_valid": int(valid.sum()),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    discover_fn = get_discovery_fn(args.method)
    print(f"Instance-discovery method: {args.method}  (available: {available_methods()})")

    per_sample_path = out_dir / "per_sample.jsonl"
    per_sample_path.write_text("")
    print(f"Streaming per-sample metrics to {per_sample_path}")

    is_trellis2 = args.dataset_type == "3d-front-trellis2-mv"
    data_cfg = DataConfig(
        type=args.dataset_type, path=args.dataset, num_views=(8 if is_trellis2 else 4),
        num_points=4096, norm_bound=0.95, load_images=True,
        image_preprocessor="facebook/dpt-dinov2-small-nyu", image_size_divisor=28,
        use_masked_obj_pc=True, random_scale=False, random_rotate=False,
        random_jitter_point_clouds=False, random_jitter_depth=False, random_shift=False,
        trellis2_hf_path=args.trellis2_hf_path, mv_covis_k_max=8, mv_covis_min_support_pts=50,
        # Match the live trellis2 training config: frame-correct scene_transforms/covis
        # (the diagnostic must measure the frame the model actually trains in).
        mv_frame_correction=is_trellis2,
    )
    print(f"Loading dataset (type={args.dataset_type}) ...")
    if is_trellis2:
        # Trellis2MVDataset emits the exact same per-item dict shape as
        # transform_3d_front_multiview, so the existing _eval_collate works unchanged.
        _, val_data, _ = get_mesh_dataset(data_cfg)
        if args.num_samples < len(val_data):
            val_data = torch.utils.data.Subset(val_data, range(args.num_samples))
    else:
        ds_path = str(Path(args.dataset).absolute())
        if (Path(args.dataset) / "dataset_dict.json").exists():
            raw = hf_datasets.load_from_disk(ds_path)
        else:
            raw = hf_datasets.load_dataset(ds_path)
        image_preprocessor = AutoImageProcessor.from_pretrained(
            data_cfg.image_preprocessor, size_divisor=data_cfg.image_size_divisor
        )
        val_key = next((k for k in ("val", "validation", "test") if k in raw), None)
        if val_key is None:
            raise ValueError(f"No val/validation/test split found. Available: {list(raw.keys())}")
        raw_val = raw[val_key]
        if args.num_samples < len(raw_val):
            raw_val = raw_val.select(range(args.num_samples))
        val_data = raw_val.with_transform(
            partial(transform_3d_front_multiview, is_train=False, data_cfg=data_cfg,
                    image_preprocessor=image_preprocessor)
        )
    loader = DataLoader(val_data, batch_size=1, collate_fn=_eval_collate, num_workers=0, shuffle=False)

    print(f"Loading Pi3X from {args.pi3x_ckpt} ...")
    model_cfg = ModelConfig(
        vocab_size=1, num_pos_tokens=512, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        pi3x_ckpt_path=args.pi3x_ckpt, pi3x_disable_multimodal=True,
    )
    pi3x = get_pi3x_encoder(model_cfg).to(device).float().eval()

    results = []
    for batch in loader:
        pixel_values     = batch["pixel_values"].to(device)
        scene_transforms = batch["scene_transforms"].to(device)
        K_per_view       = batch["K_per_view"].to(device)
        view_mask        = batch["view_mask"].to(device)
        panoptic_masks   = batch["panoptic_masks"].to(device)
        cond_pcs_2d      = batch["point_clouds_2d"].to(device)
        uid              = batch["uid"][0]
        B, N, C, H, W    = pixel_values.shape

        with torch.no_grad():
            pi3x_out = pi3x.forward_all_views_joint(pixel_values.float())
            lp   = pi3x_out["local_points"]
            conf = pi3x_out["conf"].sigmoid()   # (B, N, H, W, 1) — matches live consumption
            st, K_f = scene_transforms.float(), K_per_view.float()

            ref_idx = _select_ref_view(lp, view_mask)
            rv = int(ref_idx[0].item())
            seed_pcs = build_geo_obj_pc(lp[0:1, rv], cond_pcs_2d[0:1], st[0:1, rv]).float()

            lp_z = lp[..., 2]

            obj_voxels, ctx_voxels, diag, target_ids = discover_fn(
                local_points=lp, scene_transforms=st, panoptic_masks=panoptic_masks,
                K_per_view=K_f, view_mask=view_mask, seed_pcs=seed_pcs,
                num_obj_voxels=args.num_obj_voxels, num_ctx_voxels=args.num_ctx_voxels,
                conf=conf, pool_size=args.pool_size, min_views=args.min_views,
                conf_threshold=args.conf_threshold, depth_rtol=args.depth_rtol,
                adaptive_fallback=True, mask_seeded_pool=args.mask_seeded_pool,
                voxel_sampling=args.voxel_sampling,
                return_diagnostics=True, return_target_ids=True,
            )

        ov, cv = obj_voxels[0], ctx_voxels[0]
        tgt_ids_n = target_ids[0]
        conf_map_b = conf[0, ..., 0]   # (N, H, W)

        obj_purity = _compute_purity(ov, st[0], K_f[0], panoptic_masks[0], lp_z[0],
                                      view_mask[0], tgt_ids_n, depth_rtol=args.depth_rtol)
        ctx_overlap = _compute_purity(cv, st[0], K_f[0], panoptic_masks[0], lp_z[0],
                                       view_mask[0], tgt_ids_n, depth_rtol=args.depth_rtol)

        ov_np, cv_np = ov.float().cpu().numpy(), cv.float().cpu().numpy()
        obj_nn = _local_nn_dist(ov_np, args.knn_k)
        ctx_nn = _local_nn_dist(cv_np, args.knn_k)
        obj_extent_diag = float(np.linalg.norm(ov_np.max(0) - ov_np.min(0))) if len(ov_np) else 0.0
        ctx_extent_diag = float(np.linalg.norm(cv_np.max(0) - cv_np.min(0))) if len(cv_np) else 0.0

        obj_conf_at_v = _sample_conf_at_points(ov, st[0], K_f[0], conf_map_b, view_mask[0], H, W)
        ctx_conf_at_v = _sample_conf_at_points(cv, st[0], K_f[0], conf_map_b, view_mask[0], H, W)
        obj_adapt = _adaptive_size_stats(obj_conf_at_v, obj_nn, args.knn_k)
        ctx_adapt = _adaptive_size_stats(ctx_conf_at_v, ctx_nn, args.knn_k)

        # Ground-truth check: does obj_voxels actually land on the REQUESTED object's real
        # mesh/shape, not just on *some* self-consistent target? obj_purity/ctx_overlap can't
        # tell these apart (both are derived from the same seed-projected target ID). Compare
        # in the model's own canonical frame (see _canonicalize_like_model) so the check is
        # robust to Pi3X's known scale-invariance rather than conflated with it.
        gt_coverage_cd = gt_precision_cd = float("nan")
        if "vertices" in batch and ov_np.shape[0] > 0:
            verts, faces = batch["vertices"][0], batch["faces"][0]
            canon_transform = batch["obj_canon_transform"][0].cpu().numpy()
            if verts is not None and len(verts) > 0:
                gt_pts_canon = _sample_canonical_surface(verts, faces, n=2048)
                ov_canon = _canonicalize_like_model(ov_np, canon_transform)
                gt_t = torch.as_tensor(gt_pts_canon, device=device).unsqueeze(0).float()
                ov_t = torch.as_tensor(ov_canon, device=device).unsqueeze(0).float()
                # GT -> obj: how much of the true surface is covered (lower = better coverage)
                gt_coverage_cd = float(chamfer_distance(
                    gt_t, ov_t, squared=False, reduction="mean", single_directional=True
                ).item())
                # obj -> GT: how close discovered voxels are to the true surface (lower = better precision)
                gt_precision_cd = float(chamfer_distance(
                    ov_t, gt_t, squared=False, reduction="mean", single_directional=True
                ).item())

        rec = {
            "uid": uid,
            "obj_purity": obj_purity,
            "ctx_target_overlap": ctx_overlap,
            "n_obj_pts_raw": int(diag["n_obj_pts_raw"][0]),
            "obj_mean_nn_dist": float(np.nanmean(obj_nn)),
            "ctx_mean_nn_dist": float(np.nanmean(ctx_nn)),
            "obj_extent_diag": obj_extent_diag,
            "ctx_extent_diag": ctx_extent_diag,
            "ctx_over_obj_extent_ratio": (ctx_extent_diag / obj_extent_diag) if obj_extent_diag > 1e-9 else float("nan"),
            "obj_adaptive_corr": obj_adapt["adaptive_corr"],
            "obj_adaptive_ratio": obj_adapt["adaptive_ratio"],
            "ctx_adaptive_corr": ctx_adapt["adaptive_corr"],
            "ctx_adaptive_ratio": ctx_adapt["adaptive_ratio"],
            "gt_coverage_cd": gt_coverage_cd,
            "gt_precision_cd": gt_precision_cd,
        }
        results.append(rec)
        n = len(results)
        with per_sample_path.open("a") as f:
            f.write(json.dumps({"index": n, **rec}) + "\n")
        print(f"  [{n:4d}] {uid[:24]:<24}  obj_purity={obj_purity:.3f}  ctx_overlap={ctx_overlap:.3f}  "
              f"gt_cov={gt_coverage_cd:.4f}  gt_prec={gt_precision_cd:.4f}  "
              f"obj_adapt_corr={rec['obj_adaptive_corr']:.3f}  ctx_adapt_corr={rec['ctx_adaptive_corr']:.3f}")

    if not results:
        print("No samples evaluated.")
        return

    metric_keys = [
        "obj_purity", "ctx_target_overlap", "obj_mean_nn_dist", "ctx_mean_nn_dist",
        "obj_extent_diag", "ctx_extent_diag", "ctx_over_obj_extent_ratio",
        "obj_adaptive_corr", "obj_adaptive_ratio", "ctx_adaptive_corr", "ctx_adaptive_ratio",
        "gt_coverage_cd", "gt_precision_cd",
    ]
    agg = {}
    for k in metric_keys:
        vals = np.array([r[k] for r in results], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        agg[k] = {
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "std":  float(vals.std())  if len(vals) else float("nan"),
            "n":    int(len(vals)),
        }

    report = {"per_sample": results, "aggregate": agg}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print(f"\n=== Aggregate over {len(results)} objects ===")
    for k in metric_keys:
        a = agg[k]
        print(f"  {k:<28}  mean={a['mean']:.4f}  std={a['std']:.4f}  (n={a['n']})")

    # Cross-tab obj_purity (self-consistency) against gt_precision_cd (ground truth): the
    # two can disagree — purity only checks "found A self-consistent target," not "found
    # THE requested object." gt_found threshold is generous (well above typical obj voxel
    # spacing ~0.01-0.02) so it flags genuine misses, not sampling noise.
    gt_thresh = 0.05
    have_gt = [r for r in results if np.isfinite(r["gt_precision_cd"])]
    if have_gt:
        purity_hi = lambda r: r["obj_purity"] > 0.5
        gt_hi     = lambda r: r["gt_precision_cd"] < gt_thresh
        both      = sum(1 for r in have_gt if purity_hi(r) and gt_hi(r))
        wrong_obj = sum(1 for r in have_gt if purity_hi(r) and not gt_hi(r))
        missed    = sum(1 for r in have_gt if not purity_hi(r) and not gt_hi(r))
        other     = sum(1 for r in have_gt if not purity_hi(r) and gt_hi(r))
        n = len(have_gt)
        print(f"\n=== purity vs. ground truth (n={n}, gt_precision_cd threshold={gt_thresh}) ===")
        print(f"  high purity + found real object (genuinely good):        {both:4d} ({both/n:.1%})")
        print(f"  high purity + WRONG object (self-consistent but wrong):  {wrong_obj:4d} ({wrong_obj/n:.1%})")
        print(f"  low purity + missed (correctly flagged failure):         {missed:4d} ({missed/n:.1%})")
        print(f"  low purity + found real object anyway (odd):             {other:4d} ({other/n:.1%})")

    print(f"\nReport -> {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
