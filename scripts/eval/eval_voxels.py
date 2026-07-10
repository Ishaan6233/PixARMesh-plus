#!/usr/bin/env python3
"""Standalone voxel quality evaluation for the DA3 Trellis2-MV path.

Runs DA3 + `discover_instance_points_mv` in isolation, without the OPT decoder,
and reports object-voxel purity, pool hit rate, seed Chamfer, and FPS spread.
This is the Stage 2 gate for the DA3 port.
"""

from __future__ import annotations

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
from torch.utils.data import DataLoader, Subset
from transformers import AutoImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from metrics.chamfer import chamfer_distance
from src.data.trellis2_mv import Trellis2MVDataset
from src.models.discovery import available_methods, get_discovery_fn
from src.models.frozen_geo_encoder import (
    _get_per_view_target_ids,
    _project_pts_to_views,
    _sample_mask_ids,
    build_geo_obj_pc,
)
from src.models.utils import get_da3_encoder
from src.utils.config import DataConfig, ModelConfig

_TRELLIS2_DEFAULT = (
    "datasets/mesh_datasets/datasets/"
    "3d-front-trellis2-slat-mv-da3-aug-srcperturb-r5-qfcat-obj015-light-bgtex-20260629"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DA3 MV discovery/voxel quality")
    parser.add_argument("--mesh-dataset", default=_TRELLIS2_DEFAULT)
    parser.add_argument("--hf-dataset", default="datasets/3d-front-multiview-full")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--da3-ckpt", default="checkpoints/da3/DA3-GIANT")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--uid", default=None)
    parser.add_argument("--out", default="results/voxel_eval_da3")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pool-size", type=int, default=8192)
    parser.add_argument("--num-obj-voxels", type=int, default=512)
    parser.add_argument("--num-ctx-voxels", type=int, default=1024)
    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--depth-rtol", type=float, default=100.0)
    parser.add_argument("--conf-threshold", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--method",
        default="consensus",
        help=f"Discovery registry method. Available: {available_methods()}",
    )
    parser.add_argument(
        "--voxel-sampling",
        default="fps",
        choices=["fps", "grid", "adaptive", "adaptive_grid"],
    )
    parser.add_argument("--mask-seeded-pool", action="store_true", default=True)
    parser.add_argument("--no-mask-seeded-pool", action="store_false", dest="mask_seeded_pool")
    parser.add_argument("--boundary-bias-alpha", type=float, default=0.0)
    parser.add_argument("--no-adaptive-fallback", action="store_true")
    parser.add_argument("--no-ply", action="store_true")
    return parser.parse_args()


def _to_np(value):
    return np.asarray(value) if not isinstance(value, np.ndarray) else value


def _eval_collate(examples):
    batch = {
        "uid": [ex["uid"] if isinstance(ex["uid"], str) else ex["uid"][0] for ex in examples],
    }

    if "pixel_values" in examples[0]:
        pvs = [torch.as_tensor(_to_np(ex["pixel_values"])) for ex in examples]
        max_n = max(pv.shape[0] for pv in pvs)
        padded = [torch.cat([pv, pv.new_zeros(max_n - pv.shape[0], *pv.shape[1:])]) for pv in pvs]
        batch["pixel_values"] = torch.stack(padded, dim=0)

    for key, dtype in (("scene_transforms", torch.float32), ("K_per_view", torch.float32)):
        if key not in examples[0]:
            continue
        arrs = [_to_np(ex[key]) for ex in examples]
        max_n = max(arr.shape[0] for arr in arrs)
        eye = np.eye(arrs[0].shape[1], dtype=np.float32)

        def pad_eye(arr):
            n_pad = max_n - arr.shape[0]
            return np.concatenate([arr, np.stack([eye] * n_pad)]) if n_pad > 0 else arr

        batch[key] = torch.tensor(np.stack([pad_eye(arr) for arr in arrs]), dtype=dtype)

    for key, dtype in (("point_clouds", torch.float32), ("point_clouds_2d", torch.float32)):
        if key in examples[0]:
            batch[key] = torch.tensor(np.stack([_to_np(ex[key]) for ex in examples]), dtype=dtype)

    if "view_mask" in examples[0]:
        masks = [_to_np(ex["view_mask"]) for ex in examples]
        max_n = max(mask.shape[0] for mask in masks)
        padded = [np.concatenate([mask, np.zeros(max_n - mask.shape[0], dtype=bool)]) for mask in masks]
        batch["view_mask"] = torch.tensor(np.stack(padded), dtype=torch.bool)

    if "ref_view" in examples[0]:
        batch["ref_view"] = torch.tensor([int(ex["ref_view"]) for ex in examples], dtype=torch.long)

    if "panoptic_masks" in examples[0]:
        masks = [_to_np(ex["panoptic_masks"]) for ex in examples]
        max_n = max(mask.shape[0] for mask in masks)
        H, W = masks[0].shape[1:]
        padded = [
            np.concatenate([mask, np.zeros((max_n - mask.shape[0], H, W), dtype=np.int64)])
            for mask in masks
        ]
        batch["panoptic_masks"] = torch.tensor(np.stack(padded), dtype=torch.long)

    if "obj_canon_transform" in examples[0]:
        batch["obj_canon_transform"] = torch.tensor(
            np.stack([_to_np(ex["obj_canon_transform"]) for ex in examples]),
            dtype=torch.float32,
        )
    return batch


def _compute_purity(
    obj_voxels: torch.Tensor,
    scene_transforms: torch.Tensor,
    K_per_view: torch.Tensor,
    panoptic_masks: torch.Tensor,
    local_points_z: torch.Tensor,
    view_mask: torch.Tensor,
    target_ids_n: torch.Tensor,
    depth_rtol: float,
) -> float:
    if obj_voxels.shape[0] == 0 or (target_ids_n > 0).sum() == 0:
        return 0.0
    H, W = panoptic_masks.shape[1:]
    pix_coords, pts_cam = _project_pts_to_views(obj_voxels, scene_transforms, K_per_view, H, W)
    voxel_ids = _sample_mask_ids(
        pix_coords,
        pts_cam,
        panoptic_masks,
        local_points_z,
        view_mask,
        depth_rtol=depth_rtol,
    )
    is_target = torch.zeros(obj_voxels.shape[0], dtype=torch.bool, device=obj_voxels.device)
    for n in range(scene_transforms.shape[0]):
        if not view_mask[n] or target_ids_n[n] <= 0:
            continue
        is_target |= voxel_ids[:, n] == target_ids_n[n]
    return is_target.float().mean().item()


def _fps_spread_cv(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    tree = scipy.spatial.cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    nn_dists = dists[:, 1]
    mean_d = float(nn_dists.mean())
    return 0.0 if mean_d < 1e-9 else float(nn_dists.std() / mean_d)


def _cloud_fingerprint(pts: np.ndarray) -> tuple[list[float], list[float]]:
    if pts.shape[0] == 0:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    centroid = pts.mean(0)
    extent = pts.max(0) - pts.min(0)
    return [round(float(x), 5) for x in centroid], [round(float(x), 5) for x in extent]


def _savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def build_dataset(args):
    data_cfg = DataConfig(
        type="3d-front-trellis2-mv",
        path=args.mesh_dataset,
        trellis2_hf_path=args.hf_dataset,
        num_views=8,
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
        mv_frame_correction=True,
        mv_covis_k_max=8,
        mv_covis_min_support_pts=50,
        mv_filter_degenerate=True,
    )
    processor = AutoImageProcessor.from_pretrained(
        data_cfg.image_preprocessor,
        size_divisor=data_cfg.image_size_divisor,
    )
    ds = Trellis2MVDataset(
        str(Path(args.mesh_dataset).absolute()),
        str(Path(args.hf_dataset).absolute()),
        data_cfg,
        processor,
        is_train=args.split == "train",
    )
    if args.uid is not None:
        idxs = []
        for i, sha in enumerate(getattr(ds, "instances", [])):
            if args.uid in str(sha):
                idxs.append(i)
                continue
            try:
                if args.uid in str(ds._load_cond(sha).get("uid", "")):
                    idxs.append(i)
            except Exception:
                pass
        if not idxs:
            raise ValueError(f"UID/sha prefix {args.uid!r} not found")
        ds = Subset(ds, idxs)
    elif args.num_samples > 0 and args.num_samples < len(ds):
        ds = Subset(ds, list(range(args.num_samples)))
    return ds


def build_da3(args):
    cfg = ModelConfig(
        vocab_size=1,
        num_pos_tokens=512,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_da3=True,
        geo_encoder_type="da3",
        da3_ckpt_path=args.da3_ckpt,
    )
    return get_da3_encoder(cfg).to(args.device).float().eval()


def summarize(results: list[dict]) -> dict:
    metric_keys = ["purity", "pool_hit_rate", "chamfer_fwd", "chamfer_bwd", "fps_spread_cv"]
    agg = {}
    for key in metric_keys:
        vals = np.asarray([rec[key] for rec in results], dtype=np.float64)
        agg[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "p25": float(np.percentile(vals, 25)),
            "p75": float(np.percentile(vals, 75)),
        }

    by_scene = defaultdict(list)
    for rec in results:
        by_scene[rec["uid"].split("__")[0]].append(rec)
    multi = {scene: recs for scene, recs in by_scene.items() if len(recs) > 1}
    ratios = []
    per_metric = {}
    for key in ("purity", "pool_hit_rate", "fps_spread_cv"):
        within = [float(np.std([rec[key] for rec in recs])) for recs in multi.values()]
        mean_within = float(np.mean(within)) if within else 0.0
        global_std = agg[key]["std"]
        ratio = float(mean_within / global_std) if global_std > 1e-12 else 0.0
        ratios.append(ratio)
        per_metric[key] = {"mean_within_scene_std": mean_within, "discrimination_ratio": ratio}
    identical = None
    if multi and all("obj_centroid" in rec for recs in multi.values() for rec in recs):
        identical_count = 0
        for recs in multi.values():
            fingerprints = {(tuple(rec["obj_centroid"]), tuple(rec["obj_extent"])) for rec in recs}
            identical_count += int(len(fingerprints) == 1)
        identical = identical_count / len(multi)
    agg["object_discrimination"] = {
        "n_multi_object_scenes": len(multi),
        "per_metric": per_metric,
        "identical_cloud_scene_frac": identical,
        "mean_discrimination_ratio": float(np.mean(ratios)) if ratios else 0.0,
    }
    return agg


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    discover_fn = get_discovery_fn(args.method)
    print(f"Instance-discovery method: {args.method}")
    print(f"Available methods: {available_methods()}")
    print(
        f"Gate params: min_views={args.min_views}, depth_rtol={args.depth_rtol}, "
        f"conf_threshold={args.conf_threshold}, voxel_sampling={args.voxel_sampling}"
    )

    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=_eval_collate,
        num_workers=0,
        shuffle=False,
    )
    da3 = build_da3(args)
    per_sample_path = out_dir / "per_sample.jsonl"
    per_sample_path.write_text("")

    results = []
    adaptive_fallback = not args.no_adaptive_fallback
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        scene_transforms = batch["scene_transforms"].to(device)
        K_per_view = batch["K_per_view"].to(device)
        view_mask = batch["view_mask"].to(device)
        panoptic_masks = batch["panoptic_masks"].to(device)
        cond_pcs = batch["point_clouds"].to(device)
        cond_pcs_2d = batch["point_clouds_2d"].to(device)
        ref_view = batch["ref_view"].to(device)
        uids = batch["uid"]

        with torch.no_grad():
            da3_out = da3.forward_all_views_joint(pixel_values.float())
            local_points = da3_out["local_points"]
            conf = da3_out.get("conf")
            st = scene_transforms.float()
            K_f = K_per_view.float()
            ref_idx = ref_view
            seed_list = []
            for b in range(pixel_values.shape[0]):
                rv = int(ref_idx[b].item())
                seed_list.append(
                    build_geo_obj_pc(
                        local_points[b:b + 1, rv],
                        cond_pcs_2d[b:b + 1],
                        st[b:b + 1, rv],
                    )
                )
            seed_pcs = torch.cat(seed_list, dim=0).float()

            obj_voxels, ctx_voxels, diag = discover_fn(
                local_points=local_points,
                scene_transforms=st,
                panoptic_masks=panoptic_masks,
                K_per_view=K_f,
                view_mask=view_mask,
                seed_pcs=seed_pcs,
                num_obj_voxels=args.num_obj_voxels,
                num_ctx_voxels=args.num_ctx_voxels,
                conf=conf,
                pool_size=args.pool_size,
                min_views=args.min_views,
                conf_threshold=args.conf_threshold,
                depth_rtol=args.depth_rtol,
                adaptive_fallback=adaptive_fallback,
                mask_seeded_pool=args.mask_seeded_pool,
                boundary_bias_alpha=args.boundary_bias_alpha,
                voxel_sampling=args.voxel_sampling,
                return_diagnostics=True,
            )

        lp_z = local_points[..., 2]
        for b, uid in enumerate(uids):
            with torch.no_grad():
                target_ids = _get_per_view_target_ids(
                    seed_pcs[b].float(),
                    st[b],
                    K_f[b],
                    panoptic_masks[b],
                    lp_z[b],
                    view_mask[b],
                    depth_rtol=args.depth_rtol,
                )
                purity = _compute_purity(
                    obj_voxels[b],
                    st[b],
                    K_f[b],
                    panoptic_masks[b],
                    lp_z[b],
                    view_mask[b],
                    target_ids,
                    depth_rtol=args.depth_rtol,
                )

            ov_cpu = obj_voxels[b].unsqueeze(0).float().cpu()
            cp_cpu = cond_pcs[b].unsqueeze(0).float().cpu()
            fwd_cd = float(
                chamfer_distance(cp_cpu, ov_cpu, squared=False, reduction="mean", single_directional=True).item()
            )
            bwd_cd = float(
                chamfer_distance(ov_cpu, cp_cpu, squared=False, reduction="mean", single_directional=True).item()
            )
            ov_np = obj_voxels[b].float().cpu().numpy()
            centroid, extent = _cloud_fingerprint(ov_np)
            rec = {
                "uid": uid,
                "purity": purity,
                "pool_hit_rate": float(diag["pool_hit_rate"][b]),
                "chamfer_fwd": fwd_cd,
                "chamfer_bwd": bwd_cd,
                "fps_spread_cv": _fps_spread_cv(ov_np),
                "n_obj_pts_raw": int(diag["n_obj_pts_raw"][b]),
                "obj_centroid": centroid,
                "obj_extent": extent,
            }
            results.append(rec)
            with per_sample_path.open("a") as f:
                f.write(json.dumps({"index": len(results), **rec}) + "\n")
                f.flush()
                os.fsync(f.fileno())
            print(
                f"[{len(results):4d}] {uid[:24]:<24} "
                f"purity={purity:.3f} hit={rec['pool_hit_rate']:.3f} "
                f"fwd={fwd_cd:.4f} bwd={bwd_cd:.4f} cv={rec['fps_spread_cv']:.3f}"
            )

            if not args.no_ply:
                import trimesh

                cv_np = ctx_voxels[b].float().cpu().numpy()
                pts = np.concatenate([ov_np, cv_np], axis=0)
                colors = np.concatenate(
                    [
                        np.tile([0, 200, 0], (len(ov_np), 1)),
                        np.tile([150, 150, 150], (len(cv_np), 1)),
                    ],
                    axis=0,
                )
                trimesh.PointCloud(vertices=pts, colors=colors).export(str(out_dir / f"{uid}_voxels.ply"))

    if not results:
        print("No samples evaluated.")
        return

    aggregate = summarize(results)
    report = {"per_sample": results, "aggregate": aggregate}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport -> {out_dir / 'report.json'}")
    for key in ("purity", "pool_hit_rate", "chamfer_fwd", "chamfer_bwd", "fps_spread_cv"):
        stats = aggregate[key]
        print(
            f"{key:<18} mean={stats['mean']:.4f} std={stats['std']:.4f} "
            f"[{stats['min']:.4f}, {stats['max']:.4f}]"
        )

    purity_vals = np.asarray([rec["purity"] for rec in results])
    hit_vals = np.asarray([rec["pool_hit_rate"] for rec in results])
    plt.figure()
    plt.hist(purity_vals, bins=20, range=(0, 1), color="steelblue", edgecolor="white")
    plt.axvline(float(purity_vals.mean()), color="navy", ls="--", lw=1)
    plt.xlabel("Purity")
    plt.ylabel("Count")
    plt.title(f"DA3 object voxel purity (n={len(results)})")
    _savefig(out_dir / "purity_hist.png")

    plt.figure()
    plt.scatter(hit_vals, purity_vals, alpha=0.6, s=18, c="coral", edgecolors="none")
    plt.xlabel("Pool hit rate")
    plt.ylabel("Purity")
    plt.title("DA3 hit rate vs purity")
    _savefig(out_dir / "hit_rate_scatter.png")


if __name__ == "__main__":
    main()
