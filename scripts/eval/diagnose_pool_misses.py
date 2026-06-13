#!/usr/bin/env python3
"""Diagnostic v2: classify why pool points miss the target mask.

GT surface proxy: multi-view mask point cloud (Pi3X reconstructed points that fall
inside the target SAM mask across all views). This avoids coordinate-transform issues
with CAD vertices and gives the internally-consistent reference for the question:
"Is this missed pool point consistent with Pi3X's own object reconstruction, just at a
view where Pi3X estimated a different depth?"

Metrics
-------
- Missed-point → GT-surface distance (object-scale-relative):
    near  (< 0.05 × obj_scale) → depth inconsistency, recoverable
    mid   (0.05–0.20 × obj_scale) → ambiguous
    far   (> 0.20 × obj_scale) → background / mask-bleed / Pi3X 3D error

- Per-miss occlusion classification (per view where 2D match but depth fail):
    occluded   → z_proj > pi3x_d × 2.0  (point clearly behind surface in that view)
    scale_err  → pi3x_d × (1+rtol) < z_proj ≤ pi3x_d × 2.0  (borderline / scale residue)
    other      → remaining failures (out of bounds, etc.)

- Per-(scene, view) depth scale fit — visibility-trimmed:
    ratio = z_proj / pi3x_d for visible seed points, trimmed to [0.5, 2.0]
    Report distribution of per-scene medians per view.
    Tight IQR around 1.0 → scalar fix viable; wide IQR → per-scene scatter dominates.

Usage
-----
  python scripts/eval/diagnose_pool_misses.py \\
      --dataset datasets/3d-front-multiview \\
      --pi3x-ckpt checkpoints/pi3x \\
      --num-samples 100 --batch-size 4 \\
      --out results/diag_v2
"""

import argparse
import json
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import torch
import torch.nn.functional as F
import datasets as hf_datasets
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.mesh import transform_3d_front_multiview
from src.models.frozen_geo_encoder import (
    _apply_scene_transform,
    _get_per_view_target_ids,
    _project_pts_to_views,
    _sample_mask_ids,
    build_geo_obj_pc,
    discover_instance_points_mv,
)
from src.models.utils import get_pi3x_encoder
from src.utils.config import DataConfig, ModelConfig
from scripts.eval.eval_voxels import _eval_collate, _select_ref_view


def parse_args():
    p = argparse.ArgumentParser(description="Diagnostic v2: pool miss attribution")
    p.add_argument("--dataset",     required=True)
    p.add_argument("--pi3x-ckpt",   required=True)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--out",         default="results/diag_v2")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pool-size",   type=int, default=8192)
    p.add_argument("--min-views",          type=int,   default=2)
    p.add_argument("--depth-rtol",         type=float, default=0.10)
    p.add_argument("--mask-seeded-pool",   action="store_true")
    p.add_argument("--boundary-bias-alpha",type=float, default=0.0)
    p.add_argument("--batch-size",         type=int,   default=1)
    return p.parse_args()


def _savefig(path, **kw):
    plt.tight_layout()
    plt.savefig(path, dpi=120, **kw)
    plt.close()
    print(f"  saved {Path(path).name}")


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # --- Dataset (identical config to eval_voxels.py) ---
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
    if (Path(args.dataset) / "dataset_dict.json").exists():
        raw = hf_datasets.load_from_disk(ds_path)
    else:
        raw = hf_datasets.load_dataset(ds_path)
    image_preprocessor = AutoImageProcessor.from_pretrained(
        data_cfg.image_preprocessor, size_divisor=data_cfg.image_size_divisor
    )
    val_key = next((k for k in ("val", "validation", "test") if k in raw), None)
    if val_key is None:
        raise ValueError(f"No val/test split found. Keys: {list(raw.keys())}")
    raw_val = raw[val_key]
    if args.num_samples < len(raw_val):
        raw_val = raw_val.select(range(args.num_samples))

    val_data = raw_val.with_transform(
        partial(transform_3d_front_multiview, is_train=False,
                data_cfg=data_cfg, image_preprocessor=image_preprocessor)
    )
    loader = DataLoader(val_data, batch_size=args.batch_size,
                        collate_fn=_eval_collate, num_workers=0, shuffle=False)

    # --- Pi3X ---
    model_cfg = ModelConfig(
        vocab_size=1, num_pos_tokens=512,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
        pi3x_ckpt_path=args.pi3x_ckpt, pi3x_disable_multimodal=True,
    )
    print(f"Loading Pi3X from {args.pi3x_ckpt} ...")
    pi3x = get_pi3x_encoder(model_cfg).to(device).float().eval()

    # --- Accumulators ---
    all_rel_dists: list[float] = []
    occ_counts = {"occluded": 0, "scale_err": 0, "other": 0}
    scale_medians: dict[int, list[float]] = defaultdict(list)
    per_sample_stats: list[dict] = []
    n_total = 0

    for batch in loader:
        pixel_values     = batch["pixel_values"].to(device)
        scene_transforms = batch["scene_transforms"].to(device)
        K_per_view       = batch["K_per_view"].to(device)
        view_mask        = batch["view_mask"].to(device)
        panoptic_masks   = batch["panoptic_masks"].to(device)
        cond_pcs_2d      = batch["point_clouds_2d"].to(device)
        uids             = batch["uid"]
        B, N, C, H, W   = pixel_values.shape

        with torch.no_grad():
            pi3x_out = pi3x.forward_all_views_joint(pixel_values.float())
            lp   = pi3x_out["local_points"]   # (B, N, H, W, 3)
            conf = pi3x_out["conf"]

            st  = scene_transforms.float()
            K_f = K_per_view.float()

            ref_idx = _select_ref_view(lp, view_mask)
            seed_list = []
            for b in range(B):
                rv = int(ref_idx[b].item())
                seed_list.append(build_geo_obj_pc(lp[b:b+1, rv], cond_pcs_2d[b:b+1], st[b:b+1, rv]))
            seed_pcs = torch.cat(seed_list, dim=0).float()

            _, _, diag = discover_instance_points_mv(
                local_points            = lp,
                scene_transforms        = st,
                panoptic_masks          = panoptic_masks,
                K_per_view              = K_f,
                view_mask               = view_mask,
                seed_pcs                = seed_pcs,
                num_obj_voxels          = 512,
                num_ctx_voxels          = 1024,
                conf                    = conf,
                pool_size               = args.pool_size,
                min_views               = args.min_views,
                depth_rtol              = args.depth_rtol,
                adaptive_fallback       = False,
                mask_seeded_pool        = args.mask_seeded_pool,
                boundary_bias_alpha     = args.boundary_bias_alpha,
                return_diagnostics      = True,
                return_pool_diagnostics = True,
            )

        lp_z = lp[..., 2]  # (B, N, H, W) Pi3X depths

        for b in range(B):
            uid_b      = uids[b]
            pool_pts_b = diag["pool_pts"][b].float()       # (P, 3) cpu
            hit_strict = diag["pool_hit_strict"][b]         # (P,) bool cpu
            hit_2d     = diag["pool_hit_2d"][b]             # (P,) bool cpu
            P = pool_pts_b.shape[0]
            if P == 0:
                continue

            missed    = hit_2d & ~hit_strict   # 2D-match, depth-fail
            n_missed  = int(missed.sum())
            n_clean   = int(hit_strict.sum())
            n_full_ms = int((~hit_2d).sum())
            n_total  += 1

            # ── Target IDs per view ──
            with torch.no_grad():
                target_ids_n = _get_per_view_target_ids(
                    seed_pcs[b].float(), st[b], K_f[b],
                    panoptic_masks[b], lp_z[b], view_mask[b],
                    depth_rtol=args.depth_rtol,
                )  # (N,) long, on device

            # ── GT surface proxy: Pi3X multi-view mask point cloud ──
            # panoptic_masks (H_pm, W_pm) and lp (H_lp, W_lp) may differ due to
            # the divisor-28 padding applied to pixel_values but not to masks.
            # Resize mask to lp resolution with nearest-neighbor for aligned indexing.
            mv_pts_list = []
            with torch.no_grad():
                for n in range(N):
                    if not view_mask[b, n] or target_ids_n[n] <= 0:
                        continue
                    lp_n     = lp[b, n]                    # (H_lp, W_lp, 3)
                    H_lp, W_lp = lp_n.shape[:2]
                    pm_n_rs  = F.interpolate(
                        panoptic_masks[b, n].float().unsqueeze(0).unsqueeze(0),
                        size=(H_lp, W_lp), mode="nearest",
                    ).squeeze().long()                     # (H_lp, W_lp)
                    mask_n   = pm_n_rs == target_ids_n[n]  # (H_lp, W_lp) bool
                    valid_n  = lp_n[..., 2] > 0
                    pts_c    = lp_n[mask_n & valid_n].float()  # (M_n, 3)
                    if pts_c.shape[0] == 0:
                        continue
                    pts_s = _apply_scene_transform(
                        pts_c.unsqueeze(0), st[b, n:n+1]
                    ).squeeze(0)
                    mv_pts_list.append(pts_s.cpu())

            if mv_pts_list:
                gt_surf   = torch.cat(mv_pts_list, dim=0).numpy()   # (M, 3)
                obj_scale = float(np.ptp(gt_surf, axis=0).max())
                obj_scale = max(obj_scale, 1e-3)
            else:
                gt_surf   = None
                obj_scale = 1.0

            # ── Missed pool pts → GT surface distance (object-scale-relative) ──
            missed_pts_np = pool_pts_b[missed].numpy()   # (n_missed, 3) cpu numpy
            rel_dists_b: list[float] = []
            if n_missed > 0 and gt_surf is not None and len(gt_surf) > 0:
                tree = scipy.spatial.cKDTree(gt_surf)
                dists, _ = tree.query(missed_pts_np, k=1)
                rel_dists_b = (dists / obj_scale).tolist()
                all_rel_dists.extend(rel_dists_b)

            rel_arr_b   = np.array(rel_dists_b) if rel_dists_b else np.array([])
            near_frac_b = float((rel_arr_b < 0.05).mean()) if len(rel_arr_b) > 0 else float("nan")
            far_frac_b  = float((rel_arr_b > 0.20).mean()) if len(rel_arr_b) > 0 else float("nan")

            # ── Occlusion classification: per (missed_pt × failing view) ──
            if n_missed > 0:
                missed_pool = pool_pts_b[missed].to(device)   # (n_missed, 3) → device
                with torch.no_grad():
                    pix_m, cam_m = _project_pts_to_views(missed_pool, st[b], K_f[b], H, W)
                    # ids at rtol=100: which views have 2D mask match (ignoring depth)
                    ids_2d = _sample_mask_ids(
                        pix_m, cam_m, panoptic_masks[b], lp_z[b], view_mask[b],
                        depth_rtol=100.0,
                    )  # (n_missed, N) long
                    # Pi3X depth at each projected pixel
                    d_flat   = lp_z[b].float().unsqueeze(1)           # (N, 1, H, W)
                    c_flat   = pix_m.permute(1, 0, 2).unsqueeze(2)    # (N, n_missed, 1, 2)
                    pi3x_d_m = F.grid_sample(
                        d_flat, c_flat, mode="bilinear",
                        padding_mode="zeros", align_corners=False,
                    ).squeeze(1).squeeze(-1).T.cpu()   # (n_missed, N)

                ids_cpu  = ids_2d.cpu()
                cam_cpu  = cam_m.cpu()
                tids_cpu = target_ids_n.cpu()

                for n in range(N):
                    if not bool(view_mask[b, n]) or int(tids_cpu[n]) <= 0:
                        continue
                    match_n = ids_cpu[:, n] == tids_cpu[n]    # (n_missed,) bool
                    z_n     = cam_cpu[:, n, 2]
                    pd_n    = pi3x_d_m[:, n]
                    sel     = match_n & (z_n > 0) & (pd_n > 1e-4)
                    zs, pds = z_n[sel], pd_n[sel]
                    occ_m   = zs > pds * 2.0
                    sca_m   = (~occ_m) & (zs > pds * (1.0 + args.depth_rtol))
                    occ_counts["occluded"]  += int(occ_m.sum())
                    occ_counts["scale_err"] += int(sca_m.sum())
                    occ_counts["other"]     += int(sel.sum()) - int(occ_m.sum()) - int(sca_m.sum())

            # ── Per-(scene, view) depth scale fit — visibility-trimmed ──
            seed_b = seed_pcs[b]   # (P_seed, 3) device
            if seed_b.shape[0] > 0:
                with torch.no_grad():
                    pix_s, cam_s = _project_pts_to_views(seed_b, st[b], K_f[b], H, W)
                    d_flat_s = lp_z[b].float().unsqueeze(1)
                    c_flat_s = pix_s.permute(1, 0, 2).unsqueeze(2)
                    pi3x_d_s = F.grid_sample(
                        d_flat_s, c_flat_s, mode="bilinear",
                        padding_mode="zeros", align_corners=False,
                    ).squeeze(1).squeeze(-1).T.cpu()   # (P_seed, N)

                cam_s_cpu = cam_s.cpu()
                for n in range(N):
                    if not bool(view_mask[b, n]):
                        continue
                    z_n   = cam_s_cpu[:, n, 2]
                    pd_n  = pi3x_d_s[:, n]
                    valid = (z_n > 0) & (pd_n > 1e-4)
                    ratio = z_n[valid] / pd_n[valid]
                    trim  = ratio[(ratio >= 0.5) & (ratio <= 2.0)]
                    if trim.shape[0] >= 5:
                        scale_medians[int(n)].append(float(trim.median().item()))

            per_sample_stats.append({
                "uid":         uid_b,
                "n_pool":      P,
                "n_clean_hit": n_clean,
                "n_missed":    n_missed,
                "n_full_miss": n_full_ms,
                "obj_scale":   obj_scale,
                "near_frac":   near_frac_b,
                "far_frac":    far_frac_b,
            })
            print(
                f"  [{n_total:4d}] {uid_b[:22]:<22}  "
                f"scale={obj_scale:.3f}  clean={n_clean}  missed={n_missed}  "
                f"near={near_frac_b:.2f}  far={far_frac_b:.2f}"
            )

    if not per_sample_stats:
        print("No samples evaluated.")
        return

    # ─── Global aggregation ───────────────────────────────────────────────────────
    all_rel = np.array(all_rel_dists) if all_rel_dists else np.array([])
    near_frac_g = float((all_rel < 0.05).mean()) if len(all_rel) > 0 else 0.0
    far_frac_g  = float((all_rel > 0.20).mean()) if len(all_rel) > 0 else 0.0
    mid_frac_g  = 1.0 - near_frac_g - far_frac_g

    if near_frac_g > 0.50:
        verdict = "LOGIC_FAILURE: majority near surface — depth check is the bottleneck"
    elif far_frac_g > 0.50:
        verdict = "GEOMETRY_ERROR: majority far from surface — Pi3X 3D inconsistency is the hard ceiling"
    elif near_frac_g + mid_frac_g > 0.65:
        verdict = "MIXED_RECOVERABLE: ~2/3 within object scale — logic + calibration fixes likely help"
    else:
        verdict = "MIXED_HARD: >35% far — Pi3X geometry error is a significant contributor"

    occ_total = sum(occ_counts.values())
    occ_fracs = {k: v / max(occ_total, 1) for k, v in occ_counts.items()}

    per_view_scale = {}
    for n, medians in sorted(scale_medians.items()):
        arr = np.array(medians)
        per_view_scale[str(n)] = {
            "median":   float(np.median(arr)),
            "p25":      float(np.percentile(arr, 25)),
            "p75":      float(np.percentile(arr, 75)),
            "n_scenes": len(arr),
        }

    summary = {
        "n_samples":        n_total,
        "n_missed_total":   len(all_rel),
        "near_frac":        near_frac_g,
        "mid_frac":         mid_frac_g,
        "far_frac":         far_frac_g,
        "verdict":          verdict,
        "occlusion_counts": occ_counts,
        "occlusion_fracs":  occ_fracs,
        "per_view_scale":   per_view_scale,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Gate-0 v2 Summary ===")
    print(f"  n_samples: {n_total},  n_missed_pts: {len(all_rel)}")
    print(f"  Near (<0.05 × scale): {near_frac_g:.1%}")
    print(f"  Mid  (0.05–0.20 × scale): {mid_frac_g:.1%}")
    print(f"  Far  (>0.20 × scale):  {far_frac_g:.1%}")
    print(f"  Verdict: {verdict}")
    if occ_total > 0:
        print(f"  Occlusion breakdown ({occ_total} miss-events):")
        for k, frac in occ_fracs.items():
            print(f"    {k}: {frac:.1%} ({occ_counts[k]})")
    if per_view_scale:
        print("  Per-view depth scale (trimmed medians):")
        for vk, vs in per_view_scale.items():
            print(f"    view {vk}: median={vs['median']:.3f}  "
                  f"IQR=[{vs['p25']:.3f}, {vs['p75']:.3f}]  (n={vs['n_scenes']})")

    # ─── Plot 1: Object-scale-relative distance histogram ────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(all_rel) > 0:
        cap  = min(float(all_rel.max()), 2.0)
        bins = np.linspace(0, cap, 40)
        ax.hist(all_rel, bins=bins, color="steelblue", edgecolor="white")
        ax.axvline(0.05, color="green",   ls="--", lw=1.5,
                   label=f"near < 0.05  ({near_frac_g:.1%})")
        ax.axvline(0.20, color="crimson", ls="--", lw=1.5,
                   label=f"far > 0.20   ({far_frac_g:.1%})")
        ax.set_xlabel("dist / obj_scale  (object-scale-relative)")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Missed pool pt → GT-surface distance  "
            f"(n={len(all_rel)} pts, {n_total} scenes)\n{verdict}"
        )
        ax.legend(fontsize=9)
    _savefig(out_dir / "miss_distance_hist.png")

    # ─── Plot 2: Occlusion classification bar ─────────────────────────────────────
    if occ_total > 0:
        fig, ax = plt.subplots(figsize=(5, 4))
        labels = ["occluded\n(z>2×d)", "scale_err\n(d×rtol<z≤2×d)", "other"]
        sizes  = [occ_counts["occluded"], occ_counts["scale_err"], occ_counts["other"]]
        colors = ["#4c72b0", "#dd8452", "#55a868"]
        bars   = ax.bar(labels, sizes, color=colors, edgecolor="white")
        for bar, sz in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + occ_total * 0.01,
                    f"{sz / occ_total:.1%}", ha="center", fontsize=9)
        ax.set_ylabel("Miss-events (point × view)")
        ax.set_title(f"Per-miss occlusion classification  (n={occ_total} events)")
        _savefig(out_dir / "occlusion_classification.png")

    # ─── Plot 3: Per-(scene, view) scale fit boxplot ──────────────────────────────
    if scale_medians:
        view_ids = sorted(scale_medians.keys())
        data     = [scale_medians[v] for v in view_ids]
        fig, ax  = plt.subplots(figsize=(max(4, len(view_ids) * 1.6), 4))
        bp = ax.boxplot(data, tick_labels=[f"view {v}" for v in view_ids],
                        patch_artist=True, showfliers=True,
                        medianprops={"color": "navy", "lw": 1.5})
        for patch in bp["boxes"]:
            patch.set_facecolor("lightsteelblue")
        ax.axhline(1.0, color="red", ls="--", lw=1.2, label="ideal = 1.0")
        ax.set_ylabel("per-scene median(z_proj / pi3x_d)  [trimmed to 0.5–2.0]")
        ax.set_title(
            "Per-(scene, view) depth scale ratio\n"
            "Tight IQR → scalar fix viable;  wide IQR → per-scene scatter dominates"
        )
        ax.legend(fontsize=9)
        _savefig(out_dir / "scale_fit_per_scene.png")

    print(f"\nAll outputs written to {out_dir}/")


if __name__ == "__main__":
    main()
