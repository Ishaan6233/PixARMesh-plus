#!/usr/bin/env python3
"""Standalone voxel quality evaluation for the PixARMesh+ multi-view pipeline.

Runs Pi3X + discover_instance_points_mv in isolation (no OPT decoder) and
reports four metrics per sample:

  purity         — fraction of obj_voxels that project inside the target SAM mask
  pool_hit_rate  — fraction of pool points matching the target mask (from diagnostics)
  seed_chamfer   — Chamfer(obj_voxels, cond_pcs), both in scene space
  fps_spread_cv  — std(nn_dists)/mean(nn_dists) for obj_voxels; near 0 = uniform

Usage:
  python scripts/eval_voxels.py \\
      --dataset   datasets/3d-front-multiview \\
      --pi3x-ckpt checkpoints/pi3x \\
      --num-samples 100 \\
      --out       results/voxel_eval
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import torch
import datasets as hf_datasets
import trimesh
from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.mesh import transform_3d_front_multiview
from src.models.frozen_geo_encoder import (
    _get_per_view_target_ids,
    _project_pts_to_views,
    _sample_mask_ids,
    build_geo_obj_pc,
    discover_instance_points_mv,
)
from src.models.utils import get_pi3x_encoder
from src.utils.config import DataConfig, ModelConfig
from metrics.chamfer import chamfer_distance


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate voxel quality from Pi3X + discover_instance_points_mv"
    )
    p.add_argument("--dataset",        required=True, help="Path to 3d-front-multiview dataset")
    p.add_argument("--pi3x-ckpt",      required=True, help="Path to Pi3X checkpoint directory")
    p.add_argument("--num-samples",    type=int, default=100)
    p.add_argument("--uid",            default=None, help="Evaluate a single scene UID")
    p.add_argument("--out",            default="results/voxel_eval")
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pool-size",      type=int, default=8192)
    p.add_argument("--num-obj-voxels", type=int, default=512)
    p.add_argument("--num-ctx-voxels", type=int, default=1024)
    p.add_argument("--min-views",      type=int, default=2,
                   help="Min views a pool point must match (1=any, 2=consensus)")
    p.add_argument("--depth-rtol",     type=float, default=0.10,
                   help="Depth relative tolerance for visibility test (default 0.10)")
    p.add_argument("--sweep-rtols",    default=None,
                   help="Comma-separated rtol values to sweep, e.g. '0.10,0.20,0.50,100.0'. "
                        "Pi3X runs once per batch; discover runs once per rtol value.")
    p.add_argument("--no-adaptive-fallback", action="store_true",
                   help="Disable adaptive depth_rtol=100 fallback in pool masking")
    p.add_argument("--mask-seeded-pool", action="store_true",
                   help="Build pool by back-projecting target SAM mask pixels from all N views "
                        "instead of seed-biased FPS (fixes pool-coverage bottleneck)")
    p.add_argument("--boundary-bias-alpha", type=float, default=0.0,
                   help="Over-weight mask boundary pixels in mask-seeded pool FPS (0=uniform)")
    p.add_argument("--batch-size",     type=int, default=1)
    p.add_argument("--no-ply",         action="store_true",
                   help="Skip per-sample PLY export (recommended for large runs)")
    return p.parse_args()


def _savefig(path, **kw):
    plt.tight_layout()
    plt.savefig(path, dpi=120, **kw)
    plt.close()
    print(f"  saved {Path(path).name}")


def _eval_collate(examples):
    """Minimal collator with padding for variable view counts (4–7+ views per scene)."""
    def _to_np(v):
        return np.array(v) if not isinstance(v, np.ndarray) else v

    batch = {
        "uid": [e["uid"] if isinstance(e["uid"], str) else e["uid"][0] for e in examples],
    }

    # Pad pixel_values to the max N_views in the batch
    if "pixel_values" in examples[0]:
        pvs = [torch.as_tensor(_to_np(e["pixel_values"])) for e in examples]
        max_n = max(p.shape[0] for p in pvs)
        padded = [torch.cat([p, p.new_zeros(max_n - p.shape[0], *p.shape[1:])]) for p in pvs]
        batch["pixel_values"] = torch.stack(padded, dim=0)

    # Per-view matrices: pad with identity to avoid singular matrix crashes
    for key, dtype in [("scene_transforms", torch.float32), ("K_per_view", torch.float32)]:
        if key not in examples[0]:
            continue
        arrs = [_to_np(e[key]) for e in examples]
        max_n = max(a.shape[0] for a in arrs)
        inner = arrs[0].shape[1:]      # (4, 4) or (3, 3)
        eye   = np.eye(inner[0], dtype=np.float32)
        def _pad_eye(a):
            n_pad = max_n - a.shape[0]
            return np.concatenate([a, np.stack([eye] * n_pad)]) if n_pad > 0 else a
        batch[key] = torch.tensor(np.stack([_pad_eye(a) for a in arrs]), dtype=dtype)

    # Scalars / dense per-object tensors — no view padding needed
    for key, dtype in [("point_clouds", torch.float32), ("point_clouds_2d", torch.float32)]:
        if key in examples[0]:
            batch[key] = torch.tensor(
                np.stack([_to_np(e[key]) for e in examples], axis=0), dtype=dtype
            )

    # view_mask: pad with False (padded views are ignored in all downstream code)
    if "view_mask" in examples[0]:
        vms = [_to_np(e["view_mask"]) for e in examples]
        max_n = max(v.shape[0] for v in vms)
        padded_vm = [np.concatenate([v, np.zeros(max_n - v.shape[0], dtype=bool)]) for v in vms]
        batch["view_mask"] = torch.tensor(np.stack(padded_vm), dtype=torch.bool)

    # panoptic_masks: pad with zeros (background, harmless)
    if "panoptic_masks" in examples[0]:
        pms = [_to_np(e["panoptic_masks"]) for e in examples]
        max_n = max(p.shape[0] for p in pms)
        H, W = pms[0].shape[1], pms[0].shape[2]
        padded_pm = [np.concatenate([p, np.zeros((max_n - p.shape[0], H, W), dtype=np.int64)]) for p in pms]
        batch["panoptic_masks"] = torch.tensor(np.stack(padded_pm), dtype=torch.long)

    return batch


def _select_ref_view(local_points: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
    """(B,) index of view with most valid-depth pixels."""
    B, N, H, W, _ = local_points.shape
    valid_count = (local_points[..., 2] > 0).reshape(B, N, -1).sum(-1).float()
    if view_mask is not None:
        valid_count = valid_count * view_mask.float()
    return valid_count.argmax(dim=1)


def _compute_purity(
    obj_voxels: torch.Tensor,        # (V, 3) scene space
    scene_transforms: torch.Tensor,  # (N, 4, 4)
    K_per_view: torch.Tensor,        # (N, 3, 3)
    panoptic_masks: torch.Tensor,    # (N, H, W)
    local_points_z: torch.Tensor,    # (N, H, W) depth
    view_mask: torch.Tensor,         # (N,) bool
    target_ids_n: torch.Tensor,      # (N,) long
    depth_rtol: float = 0.10,
) -> float:
    """Fraction of obj_voxels projecting inside the target SAM mask in >=1 view."""
    if obj_voxels.shape[0] == 0 or (target_ids_n > 0).sum() == 0:
        return 0.0
    H, W = panoptic_masks.shape[1:]
    pix_coords, pts_cam = _project_pts_to_views(
        obj_voxels, scene_transforms, K_per_view, H, W
    )
    voxel_ids = _sample_mask_ids(
        pix_coords, pts_cam, panoptic_masks, local_points_z, view_mask, depth_rtol=depth_rtol
    )
    V = obj_voxels.shape[0]
    N = scene_transforms.shape[0]
    is_target = torch.zeros(V, dtype=torch.bool, device=obj_voxels.device)
    for n in range(N):
        if not view_mask[n] or target_ids_n[n] <= 0:
            continue
        is_target |= (voxel_ids[:, n] == target_ids_n[n])
    return is_target.float().mean().item()


def _fps_spread_cv(pts: np.ndarray) -> float:
    """Coefficient of variation of nearest-neighbour distances. Near 0 = uniform."""
    if len(pts) < 2:
        return 0.0
    tree = scipy.spatial.cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    nn_dists = dists[:, 1]
    mean_d = float(nn_dists.mean())
    if mean_d < 1e-9:
        return 0.0
    return float(nn_dists.std() / mean_d)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    sweep_rtols = None
    if args.sweep_rtols:
        sweep_rtols = [float(r) for r in args.sweep_rtols.split(",")]
        print(f"Sweep mode: {len(sweep_rtols)} rtol values: {sweep_rtols}")
    adaptive_fallback = not args.no_adaptive_fallback

    if sweep_rtols:
        rtol_paths = {}
        rtol_results = {}
        for rtol in sweep_rtols:
            rtol_key = f"rtol_{rtol:g}"
            rd = out_dir / rtol_key
            rd.mkdir(parents=True, exist_ok=True)
            rtol_paths[rtol] = rd / "per_sample.jsonl"
            rtol_paths[rtol].write_text("")
            rtol_results[rtol] = []
        print(f"Sweep outputs → {out_dir}/rtol_*/per_sample.jsonl")
    else:
        per_sample_path = out_dir / "per_sample.jsonl"
        per_sample_path.write_text("")
        print(f"Streaming per-sample metrics to {per_sample_path}")

    # --- Dataset (deterministic: no augmentation) ---
    data_cfg = DataConfig(
        type="3d-front-multiview",
        path=args.dataset,
        num_views=4,
        num_points=4096,
        norm_bound=0.95,
        load_images=True,
        image_preprocessor="facebook/dpt-dinov2-small-nyu",
        image_size_divisor=28,
        use_masked_obj_pc=True,
        random_scale=False,
        random_rotate=False,
        random_jitter_point_clouds=False,
        random_jitter_depth=False,
        random_shift=False,
    )
    print("Loading dataset ...")
    ds_path = str(Path(args.dataset).absolute())
    # load_from_disk for save_to_disk format; fall back to load_dataset for parquet dirs
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
    raw_val = raw[val_key]  # untransformed — use for UID lookup

    if args.uid is not None:
        raw_uids = raw_val["uid"]
        indices = [
            i for i, u in enumerate(raw_uids)
            if (isinstance(u, list) and args.uid in u)
            or u == args.uid
            or (isinstance(u, str) and u.startswith(args.uid))
        ]
        if not indices:
            raise ValueError(f"UID {args.uid!r} not found in val split")
        raw_val = raw_val.select(indices)
    elif args.num_samples < len(raw_val):
        raw_val = raw_val.select(range(args.num_samples))

    val_data = raw_val.with_transform(
        partial(
            transform_3d_front_multiview,
            is_train=False,
            data_cfg=data_cfg,
            image_preprocessor=image_preprocessor,
        )
    )

    loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        collate_fn=_eval_collate,
        num_workers=0,
        shuffle=False,
    )

    # --- Pi3X encoder only (no OPT) ---
    model_cfg = ModelConfig(
        vocab_size=1,
        num_pos_tokens=512,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        pi3x_ckpt_path=args.pi3x_ckpt,
        pi3x_disable_multimodal=True,
    )
    print(f"Loading Pi3X from {args.pi3x_ckpt} ...")
    pi3x = get_pi3x_encoder(model_cfg).to(device).float().eval()

    # --- Evaluation loop ---
    results = []

    for batch in loader:
        pixel_values    = batch["pixel_values"].to(device)       # (B, N, C, H, W)
        scene_transforms = batch["scene_transforms"].to(device)  # (B, N, 4, 4)
        K_per_view      = batch["K_per_view"].to(device)         # (B, N, 3, 3)
        view_mask       = batch["view_mask"].to(device)          # (B, N) bool
        panoptic_masks  = batch["panoptic_masks"].to(device)     # (B, N, H, W)
        cond_pcs        = batch["point_clouds"].to(device)       # (B, P, 3)
        cond_pcs_2d     = batch["point_clouds_2d"].to(device)    # (B, P, 2)
        uids            = batch["uid"]
        B, N, C, H, W  = pixel_values.shape

        # Pi3X forward: runs ONCE per batch regardless of mode
        with torch.no_grad():
            pi3x_out = pi3x.forward_all_views_joint(pixel_values.float())
            lp   = pi3x_out["local_points"]   # (B, N, H, W, 3)
            conf = pi3x_out["conf"]            # (B, N, H, W, 1)

            st  = scene_transforms.float()
            K_f = K_per_view.float()

            ref_idx = _select_ref_view(lp, view_mask)
            seed_list = []
            for b in range(B):
                rv   = int(ref_idx[b].item())
                pc_b = build_geo_obj_pc(lp[b:b+1, rv], cond_pcs_2d[b:b+1], st[b:b+1, rv])
                seed_list.append(pc_b)
            seed_pcs = torch.cat(seed_list, dim=0).float()   # (B, P, 3)

        lp_z = lp[..., 2]   # (B, N, H, W)

        if sweep_rtols:
            # Pre-compute target IDs once per sample at fixed rtol=0.10 for purity
            target_ids_by_b = []
            for b in range(B):
                with torch.no_grad():
                    tids = _get_per_view_target_ids(
                        seed_pcs[b].float(), st[b], K_f[b],
                        panoptic_masks[b], lp_z[b], view_mask[b],
                        depth_rtol=0.10,
                    )
                target_ids_by_b.append(tids)

            for rtol in sweep_rtols:
                with torch.no_grad():
                    obj_voxels, _ctx, diag = discover_instance_points_mv(
                        local_points        = lp,
                        scene_transforms    = st,
                        panoptic_masks      = panoptic_masks,
                        K_per_view          = K_f,
                        view_mask           = view_mask,
                        seed_pcs            = seed_pcs,
                        num_obj_voxels      = args.num_obj_voxels,
                        num_ctx_voxels      = args.num_ctx_voxels,
                        conf                = conf,
                        pool_size           = args.pool_size,
                        min_views           = args.min_views,
                        depth_rtol          = rtol,
                        adaptive_fallback   = False,
                        mask_seeded_pool    = args.mask_seeded_pool,
                        boundary_bias_alpha = args.boundary_bias_alpha,
                        return_diagnostics  = True,
                    )

                for b in range(B):
                    uid_b = uids[b]
                    ov_b  = obj_voxels[b]
                    cp_b  = cond_pcs[b]

                    with torch.no_grad():
                        purity = _compute_purity(
                            ov_b, st[b], K_f[b], panoptic_masks[b], lp_z[b],
                            view_mask[b], target_ids_by_b[b], depth_rtol=0.10,
                        )

                    hit_rate = float(diag["pool_hit_rate"][b])

                    ov_cpu = ov_b.unsqueeze(0).float().cpu()
                    cp_cpu = cp_b.unsqueeze(0).float().cpu()
                    fwd_cd = float(chamfer_distance(
                        cp_cpu, ov_cpu, squared=False, reduction="mean", single_directional=True,
                    ).item())
                    bwd_cd = float(chamfer_distance(
                        ov_cpu, cp_cpu, squared=False, reduction="mean", single_directional=True,
                    ).item())
                    fps_cv = _fps_spread_cv(ov_b.float().cpu().numpy())

                    rec = {
                        "uid":           uid_b,
                        "purity":        purity,
                        "pool_hit_rate": hit_rate,
                        "chamfer_fwd":   fwd_cd,
                        "chamfer_bwd":   bwd_cd,
                        "fps_spread_cv": fps_cv,
                        "n_obj_pts_raw": int(diag["n_obj_pts_raw"][b]),
                    }
                    rtol_results[rtol].append(rec)
                    n_r = len(rtol_results[rtol])
                    with rtol_paths[rtol].open("a") as f:
                        f.write(json.dumps({"rtol": rtol, "index": n_r, **rec}) + "\n")
                        f.flush()
                        os.fsync(f.fileno())

                n_done = len(rtol_results[sweep_rtols[0]])
                recent = rtol_results[rtol][-B:]
                sample_str = "  ".join(
                    f"p={recent[b_]['purity']:.3f} hit={recent[b_]['pool_hit_rate']:.3f}"
                    for b_ in range(min(B, len(recent)))
                )
                print(f"  [{n_done:4d}] rtol={rtol:<8.3g}  {sample_str}")

        else:
            with torch.no_grad():
                obj_voxels, ctx_voxels, diag = discover_instance_points_mv(
                    local_points        = lp,
                    scene_transforms    = st,
                    panoptic_masks      = panoptic_masks,
                    K_per_view          = K_f,
                    view_mask           = view_mask,
                    seed_pcs            = seed_pcs,
                    num_obj_voxels      = args.num_obj_voxels,
                    num_ctx_voxels      = args.num_ctx_voxels,
                    conf                = conf,
                    pool_size           = args.pool_size,
                    min_views           = args.min_views,
                    depth_rtol          = args.depth_rtol,
                    adaptive_fallback   = adaptive_fallback,
                    mask_seeded_pool    = args.mask_seeded_pool,
                    boundary_bias_alpha = args.boundary_bias_alpha,
                    return_diagnostics  = True,
                )

            for b in range(B):
                uid_b  = uids[b]
                st_b   = st[b]
                K_b    = K_f[b]
                vm_b   = view_mask[b]
                pm_b   = panoptic_masks[b]
                lp_z_b = lp_z[b]
                ov_b   = obj_voxels[b]
                cv_b   = ctx_voxels[b]
                cp_b   = cond_pcs[b]

                with torch.no_grad():
                    target_ids_n = _get_per_view_target_ids(
                        seed_pcs[b].float(), st_b, K_b, pm_b, lp_z_b, vm_b,
                        depth_rtol=args.depth_rtol,
                    )
                    purity = _compute_purity(
                        ov_b, st_b, K_b, pm_b, lp_z_b, vm_b, target_ids_n,
                        depth_rtol=args.depth_rtol,
                    )

                hit_rate = float(diag["pool_hit_rate"][b])

                ov_cpu = ov_b.unsqueeze(0).float().cpu()
                cp_cpu = cp_b.unsqueeze(0).float().cpu()
                fwd_cd = float(chamfer_distance(
                    cp_cpu, ov_cpu, squared=False, reduction="mean", single_directional=True,
                ).item())
                bwd_cd = float(chamfer_distance(
                    ov_cpu, cp_cpu, squared=False, reduction="mean", single_directional=True,
                ).item())
                ov_np  = ov_b.float().cpu().numpy()
                fps_cv = _fps_spread_cv(ov_np)

                rec = {
                    "uid":           uid_b,
                    "purity":        purity,
                    "pool_hit_rate": hit_rate,
                    "chamfer_fwd":   fwd_cd,
                    "chamfer_bwd":   bwd_cd,
                    "fps_spread_cv": fps_cv,
                    "n_obj_pts_raw": int(diag["n_obj_pts_raw"][b]),
                }
                results.append(rec)
                n = len(results)
                with per_sample_path.open("a") as f:
                    f.write(json.dumps({"index": n, **rec}) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                print(
                    f"  [{n:4d}] {uid_b[:24]:<24}  "
                    f"purity={purity:.3f}  hit={hit_rate:.3f}  "
                    f"fwd={fwd_cd:.4f}  bwd={bwd_cd:.4f}  cv={fps_cv:.3f}"
                )

                if not args.no_ply:
                    cv_np = cv_b.float().cpu().numpy()
                    pts   = np.concatenate([ov_np, cv_np], axis=0)
                    colors = np.concatenate([
                        np.tile([0, 200, 0],     (len(ov_np), 1)),
                        np.tile([150, 150, 150], (len(cv_np), 1)),
                    ], axis=0)
                    trimesh.PointCloud(vertices=pts, colors=colors).export(
                        str(out_dir / f"{uid_b}_voxels.ply")
                    )

    # --- Sweep summary (early return) ---
    if sweep_rtols:
        print("\n=== Sweep Summary ===")
        print(f"  {'rtol':>10}  {'n':>6}  {'purity':>8}  {'hit_rate':>10}  {'fwd_cd':>8}  {'bwd_cd':>8}")
        agg_by_rtol = {}
        for rtol in sweep_rtols:
            recs = rtol_results[rtol]
            if not recs:
                print(f"  rtol={rtol:.3g}: no samples")
                continue
            p_arr = np.array([r["purity"]        for r in recs])
            h_arr = np.array([r["pool_hit_rate"]  for r in recs])
            f_arr = np.array([r["chamfer_fwd"]    for r in recs])
            b_arr = np.array([r["chamfer_bwd"]    for r in recs])
            agg = {
                "purity":        float(p_arr.mean()),
                "pool_hit_rate": float(h_arr.mean()),
                "chamfer_fwd":   float(f_arr.mean()),
                "chamfer_bwd":   float(b_arr.mean()),
                "n":             len(recs),
            }
            agg_by_rtol[rtol] = agg
            print(
                f"  {rtol:>10.3g}  {len(recs):>6}  "
                f"{agg['purity']:>8.4f}  {agg['pool_hit_rate']:>10.4f}  "
                f"{agg['chamfer_fwd']:>8.4f}  {agg['chamfer_bwd']:>8.4f}"
            )

        (out_dir / "sweep_summary.json").write_text(
            json.dumps({str(rtol): v for rtol, v in agg_by_rtol.items()}, indent=2)
        )

        rtols_with_data = [rtol for rtol in sweep_rtols if rtol in agg_by_rtol]
        purities  = [agg_by_rtol[r]["purity"]        for r in rtols_with_data]
        hit_rates = [agg_by_rtol[r]["pool_hit_rate"]  for r in rtols_with_data]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(hit_rates, purities, "o-", color="steelblue", lw=1.5, ms=7)
        for rtol, pur, hit in zip(rtols_with_data, purities, hit_rates):
            ax.annotate(f"rtol={rtol:g}", (hit, pur),
                        textcoords="offset points", xytext=(5, 3), fontsize=7)
        ax.set_xlabel("Pool hit rate (recall proxy)")
        ax.set_ylabel("Purity  (measured at depth_rtol=0.10)")
        ax.set_title("depth_rtol Sweep: Purity / Recall Tradeoff")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        _savefig(out_dir / "sweep_tradeoff.png")
        print(f"\nAll sweep outputs in {out_dir}/")
        return

    if not results:
        print("No samples evaluated.")
        return

    metric_keys = ["purity", "pool_hit_rate", "chamfer_fwd", "chamfer_bwd", "fps_spread_cv"]

    # --- Per-scene grouping ---
    def _scene_id(uid: str) -> str:
        return uid.split("__")[0]

    scene_groups: dict[str, list] = defaultdict(list)
    for r in results:
        scene_groups[_scene_id(r["uid"])].append(r)

    per_scene = {}
    for sid, recs in sorted(scene_groups.items()):
        per_scene[sid] = {"n_objects": len(recs)}
        for k in metric_keys:
            vals = np.array([r[k] for r in recs])
            per_scene[sid][k] = {
                "mean": float(vals.mean()),
                "std":  float(vals.std()),
                "min":  float(vals.min()),
                "max":  float(vals.max()),
            }

    # --- Global aggregate ---
    agg = {}
    for k in metric_keys:
        vals = np.array([r[k] for r in results])
        agg[k] = {
            "mean": float(vals.mean()),
            "std":  float(vals.std()),
            "min":  float(vals.min()),
            "max":  float(vals.max()),
            "p25":  float(np.percentile(vals, 25)),
            "p75":  float(np.percentile(vals, 75)),
        }

    report = {"per_sample": results, "per_scene": per_scene, "aggregate": agg}
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport → {report_path}")

    n_scenes = len(scene_groups)
    print(f"\nAggregate  ({len(results)} objects across {n_scenes} scenes):")
    for k in metric_keys:
        a = agg[k]
        print(f"  {k:<18}  mean={a['mean']:.4f}  std={a['std']:.4f}  "
              f"[{a['min']:.4f}, {a['max']:.4f}]")

    # --- Per-scene summary table to stdout ---
    if n_scenes > 1:
        col_w = 10
        header = f"  {'scene':<36}" + "".join(f"{'purity':>{col_w}}{'hit':>{col_w}}"
                                               f"{'fwd_cd':>{col_w}}{'bwd_cd':>{col_w}}"
                                               f"{'cv':>{col_w}}{'n_obj':>{col_w}}")
        print(f"\nPer-scene means:\n{header}")
        for sid in sorted(per_scene):
            ps = per_scene[sid]
            row = (f"  {sid[:36]:<36}"
                   f"{ps['purity']['mean']:>{col_w}.3f}"
                   f"{ps['pool_hit_rate']['mean']:>{col_w}.3f}"
                   f"{ps['chamfer_fwd']['mean']:>{col_w}.4f}"
                   f"{ps['chamfer_bwd']['mean']:>{col_w}.4f}"
                   f"{ps['fps_spread_cv']['mean']:>{col_w}.3f}"
                   f"{ps['n_objects']:>{col_w}}")
            print(row)

    # --- Plots ---
    purity_vals  = np.array([r["purity"]        for r in results])
    hit_vals     = np.array([r["pool_hit_rate"] for r in results])
    fwd_vals     = np.array([r["chamfer_fwd"]   for r in results])
    bwd_vals     = np.array([r["chamfer_bwd"]   for r in results])
    cv_vals      = np.array([r["fps_spread_cv"] for r in results])

    # 1. Purity histogram
    plt.figure()
    plt.hist(purity_vals, bins=20, range=(0, 1), color="steelblue", edgecolor="white")
    plt.axvline(float(purity_vals.mean()), color="navy", ls="--", lw=1.5,
                label=f"mean={purity_vals.mean():.3f}")
    plt.xlabel("Purity")
    plt.ylabel("Count")
    plt.title(f"Object Voxel Purity  (n={len(results)})")
    plt.legend()
    _savefig(out_dir / "purity_hist.png")

    # 2. Hit rate vs purity scatter
    plt.figure()
    plt.scatter(hit_vals, purity_vals, alpha=0.6, s=20, c="coral", edgecolors="none")
    plt.xlabel("Pool hit rate")
    plt.ylabel("Purity")
    plt.title("Hit Rate vs Purity")
    _savefig(out_dir / "hit_rate_scatter.png")

    # 3. Forward Chamfer histogram
    plt.figure()
    plt.hist(fwd_vals, bins=20, color="seagreen", edgecolor="white")
    plt.axvline(float(fwd_vals.mean()), color="darkgreen", ls="--", lw=1.5,
                label=f"mean={fwd_vals.mean():.4f}")
    plt.xlabel("Chamfer fwd: seed→voxels (scene units)")
    plt.ylabel("Count")
    plt.title("Forward Chamfer (Coverage)")
    plt.legend()
    _savefig(out_dir / "chamfer_fwd_hist.png")

    # 4. Backward Chamfer histogram
    plt.figure()
    plt.hist(bwd_vals, bins=20, color="mediumpurple", edgecolor="white")
    plt.axvline(float(bwd_vals.mean()), color="indigo", ls="--", lw=1.5,
                label=f"mean={bwd_vals.mean():.4f}")
    plt.xlabel("Chamfer bwd: voxels→seed (scene units)")
    plt.ylabel("Count")
    plt.title("Backward Chamfer (Precision)")
    plt.legend()
    _savefig(out_dir / "chamfer_bwd_hist.png")

    # 5. fwd vs bwd scatter coloured by purity
    fig, ax = plt.subplots()
    sc = ax.scatter(fwd_vals, bwd_vals, c=purity_vals, cmap="RdYlGn",
                    vmin=0, vmax=1, alpha=0.7, s=20, edgecolors="none")
    # diagonal reference: fwd == bwd
    lim = max(fwd_vals.max(), bwd_vals.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4, label="fwd = bwd")
    ax.set_xlabel("fwd Chamfer (coverage)")
    ax.set_ylabel("bwd Chamfer (precision)")
    ax.set_title("Coverage vs Precision  (colour = purity)")
    ax.legend(fontsize=8)
    fig.colorbar(sc, ax=ax, label="purity")
    _savefig(out_dir / "chamfer_fwd_vs_bwd.png")

    # 6. Violin of all metrics
    fig, axes = plt.subplots(1, len(metric_keys), figsize=(3 * len(metric_keys), 4))
    data_by_metric = [purity_vals, hit_vals, fwd_vals, bwd_vals, cv_vals]
    labels = ["purity", "hit_rate", "fwd_cd", "bwd_cd", "cv"]
    for ax, data, lbl in zip(axes, data_by_metric, labels):
        parts = ax.violinplot(data, showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)
        ax.set_title(lbl, fontsize=9)
        ax.set_xticks([])
        ax.text(0.5, -0.12, f"μ={data.mean():.3f}", ha="center",
                transform=ax.transAxes, fontsize=8)
    fig.suptitle(f"Metric distributions  (n={len(results)} obj, {n_scenes} scenes)", y=1.02)
    _savefig(out_dir / "metrics_violin.png", bbox_inches="tight")

    # 7. Per-scene purity bar (only if >1 scene)
    if n_scenes > 1:
        scene_ids = sorted(per_scene)
        scene_means = np.array([per_scene[s]["purity"]["mean"] for s in scene_ids])
        scene_stds  = np.array([per_scene[s]["purity"]["std"]  for s in scene_ids])
        # sort by mean purity descending
        order = np.argsort(scene_means)[::-1]
        fig, ax = plt.subplots(figsize=(max(6, n_scenes * 0.5), 4))
        x = np.arange(n_scenes)
        ax.bar(x, scene_means[order], yerr=scene_stds[order],
               color="steelblue", alpha=0.8, capsize=3, error_kw={"elinewidth": 0.8})
        ax.axhline(float(purity_vals.mean()), color="navy", ls="--", lw=1,
                   label=f"global mean={purity_vals.mean():.3f}")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Purity")
        ax.set_xlabel("Scene (sorted by purity)")
        ax.set_title("Per-scene Object Voxel Purity")
        ax.set_xticks(x)
        ax.set_xticklabels([scene_ids[i][:12] for i in order], rotation=45,
                           ha="right", fontsize=6)
        ax.legend(fontsize=8)
        _savefig(out_dir / "per_scene_purity.png", bbox_inches="tight")

    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
