#!/usr/bin/env python3
"""Visual/numeric verification for the Trellis2-MV camera convention adapter."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.trellis2_mv import (  # noqa: E402
    _CAM_YUP_TO_OPENCV_4,
    _select_diverse_views,
    covis_object_supports,
    norm_to_world_transform,
    Trellis2MVDataset,
)
from src.models.frozen_geo_encoder import _project_pts_to_views  # noqa: E402
from src.utils.config import DataConfig  # noqa: E402


DEFAULT_TRELLIS2 = (
    "datasets/mesh_datasets/datasets/"
    "3d-front-trellis2-slat-mv-da3-aug-srcperturb-r5-qfcat-obj015-light-bgtex-20260629"
)
DEFAULT_HF = "datasets/3d-front-multiview-full"
DEFAULT_SHAS = [
    "train_008135_9c61f750-f0ac-47cc-852f-61e16db2a211_"
    "9c61f750-f0ac-47cc-852f-61e16db2a211__obj0000_obj0",
    "train_005485_7fd782e5-117f-43cd-b891-15c4f6ed86e8_"
    "7fd782e5-117f-43cd-b891-15c4f6ed86e8__obj0000_obj0",
    "train_016429_f1a605ec-155d-48fd-ac89-d54c3ebdbfb6_"
    "f1a605ec-155d-48fd-ac89-d54c3ebdbfb6__obj0000_obj0",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trellis2", default=DEFAULT_TRELLIS2)
    p.add_argument("--hf", default=DEFAULT_HF)
    p.add_argument("--out-dir", default="outputs/vis")
    p.add_argument("--sha", action="append", dest="shas",
                   help="Trellis2 sha256 key to verify; repeatable")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num-points", type=int, default=4096)
    return p.parse_args()


def make_cfg(args: argparse.Namespace) -> DataConfig:
    return DataConfig(
        type="3d-front-trellis2-mv",
        path=args.trellis2,
        trellis2_hf_path=args.hf,
        num_views=8,
        num_points=args.num_points,
        norm_bound=0.95,
        load_images=False,
        use_masked_obj_pc=False,
        random_scale=False,
        random_rotate=False,
        random_jitter_point_clouds=False,
        random_jitter_depth=False,
        random_shift=False,
        mv_covis_k_max=8,
        mv_covis_min_support_pts=50,
        mv_frame_correction=True,
        mv_filter_degenerate=False,
    )


def norm_to_pixel(xy_norm: np.ndarray, h: int, w: int) -> np.ndarray:
    xy = np.asarray(xy_norm, dtype=np.float64)
    px = np.empty_like(xy, dtype=np.float64)
    px[:, 0] = (xy[:, 0] + 1.0) * float(w) / 2.0 - 0.5
    px[:, 1] = (xy[:, 1] + 1.0) * float(h) / 2.0 - 0.5
    return px


def pixel_to_norm(uv: np.ndarray, h: int, w: int) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64)
    xy = np.empty_like(uv, dtype=np.float64)
    xy[:, 0] = (uv[:, 0] + 0.5) / float(w) * 2.0 - 1.0
    xy[:, 1] = (uv[:, 1] + 0.5) / float(h) * 2.0 - 1.0
    return xy


def project_scene_points(points_scene: np.ndarray, scene_transform: np.ndarray,
                         K: np.ndarray, h: int, w: int) -> np.ndarray:
    pts = np.asarray(points_scene, dtype=np.float32)
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    cam_h = (np.linalg.inv(scene_transform).astype(np.float32) @ pts_h.T).T
    cam = cam_h[:, :3]
    z = cam[:, 2]
    u = np.where(z > 1e-4, K[0, 0] * cam[:, 0] / z + K[0, 2], np.nan)
    v = np.where(z > 1e-4, K[1, 1] * cam[:, 1] / z + K[1, 2], np.nan)
    return pixel_to_norm(np.stack([u, v], axis=1), h, w).astype(np.float32)


def select_reference_world_view(ds: Trellis2MVDataset, sha: str,
                                cfg: DataConfig, seed: int) -> tuple[int, dict]:
    cond = ds._load_cond(sha)
    uid = cond["uid"]
    scene_id = cond["scene_id"]
    hf_idx = ds._uid_to_idx.get(uid, ds._scene_id_to_idx.get(scene_id))
    if hf_idx is None:
        raise RuntimeError(f"{sha}: no matching HF row for uid={uid!r}")
    row = ds._hf[hf_idx]
    all_w2c = np.stack([np.array(m, dtype=np.float32) for m in row["wrd2cam_rects"]])
    all_K = np.stack([np.array(m, dtype=np.float32) for m in row["Ks"]])
    raw_hw = (int(round(float(all_K[0][1, 2]) * 2.0)),
              int(round(float(all_K[0][0, 2]) * 2.0)))
    T_n2w = norm_to_world_transform(cond, row["objects"]["transforms"][0])
    np.random.seed(seed)
    obj_pts_world, support = covis_object_supports(
        cond, all_w2c, all_K, raw_hw, T_norm_to_world=T_n2w
    )
    selected = _select_diverse_views(
        obj_pts_world,
        all_w2c,
        all_K,
        raw_hw,
        support,
        k_max=min(cfg.mv_covis_k_max, len(all_w2c)),
        min_support_pts=cfg.mv_covis_min_support_pts,
    )
    if not selected:
        raise RuntimeError(f"{sha}: object-blind, no selected reference view")
    return int(selected[0]), row


def project_world_with_sign(points_world: np.ndarray, wrd2cam: np.ndarray, K: np.ndarray,
                            h: int, w: int, signs: tuple[int, int, int]) -> np.ndarray:
    D = np.diag(np.array(signs, dtype=np.float32))
    pts = np.asarray(points_world, dtype=np.float32)
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    cam_yup = (np.asarray(wrd2cam, dtype=np.float32) @ pts_h.T).T[:, :3]
    cam = cam_yup @ D.T
    z = cam[:, 2]
    u = np.where(z > 1e-4, K[0, 0] * cam[:, 0] / z + K[0, 2], np.nan)
    v = np.where(z > 1e-4, K[1, 1] * cam[:, 1] / z + K[1, 2], np.nan)
    return pixel_to_norm(np.stack([u, v], axis=1), h, w).astype(np.float32)


def finite_median_or_penalty(values: np.ndarray, penalty: float = 1.0e6) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return penalty
    return float(np.median(values[finite]))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shas = args.shas or DEFAULT_SHAS
    cfg = make_cfg(args)
    ds = Trellis2MVDataset(args.trellis2, args.hf, cfg, image_preprocessor=None,
                           is_train=False)

    samples = []
    all_roundtrip_err = []
    all_raw_err = []
    sign_errs: dict[str, list[float]] = {
        f"{sx},{sy},{sz}": [] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    }

    for offset, sha in enumerate(shas):
        ref_world_view, row = select_reference_world_view(ds, sha, cfg, args.seed + offset)
        np.random.seed(args.seed + offset)
        ds.instances = [sha]
        item = ds[0]
        if not item.get("point_clouds_valid", False):
            raise RuntimeError(f"{sha}: loader returned point_clouds_valid=False")

        ref_slot = int(item["ref_view"])
        K = np.asarray(item["K_per_view"][ref_slot], dtype=np.float32)
        h = int(round(float(K[1, 2]) * 2.0))
        w = int(round(float(K[0, 2]) * 2.0))
        pts_scene = np.asarray(item["point_clouds"], dtype=np.float32)
        st = np.asarray(item["scene_transforms"][ref_slot], dtype=np.float32)

        pix_model, _ = _project_pts_to_views(
            torch.as_tensor(pts_scene, dtype=torch.float32),
            torch.as_tensor(item["scene_transforms"], dtype=torch.float32),
            torch.as_tensor(item["K_per_view"], dtype=torch.float32),
            h,
            w,
        )
        patched_norm = pix_model[:, ref_slot].cpu().numpy()
        emitted_norm = np.asarray(item["point_clouds_2d"], dtype=np.float32)
        raw_norm = project_scene_points(pts_scene, st @ _CAM_YUP_TO_OPENCV_4, K, h, w)

        patched_px = norm_to_pixel(patched_norm, h, w)
        emitted_px = norm_to_pixel(emitted_norm, h, w)
        raw_px = norm_to_pixel(raw_norm, h, w)
        roundtrip_err = np.linalg.norm(patched_px - emitted_px, axis=1)
        raw_err = np.linalg.norm(raw_px - emitted_px, axis=1)
        all_roundtrip_err.extend(roundtrip_err[np.isfinite(roundtrip_err)].tolist())
        all_raw_err.extend(raw_err[np.isfinite(raw_err)].tolist())

        pts_world_h = (
            np.linalg.inv(np.asarray(row["wrd2cam_rects"][ref_world_view], dtype=np.float32))
            @ (_CAM_YUP_TO_OPENCV_4 @ np.linalg.inv(st) @ np.concatenate(
                [pts_scene, np.ones((len(pts_scene), 1), dtype=np.float32)], axis=1
            ).T)
        ).T
        pts_world = pts_world_h[:, :3]
        wrd2cam = np.asarray(row["wrd2cam_rects"][ref_world_view], dtype=np.float32)
        K_raw = np.asarray(row["Ks"][ref_world_view], dtype=np.float32)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    key = f"{sx},{sy},{sz}"
                    cand = project_world_with_sign(pts_world, wrd2cam, K_raw, h, w,
                                                   (sx, sy, sz))
                    cand_px = norm_to_pixel(cand, h, w)
                    err = np.linalg.norm(cand_px - emitted_px, axis=1)
                    sign_errs[key].append(finite_median_or_penalty(err))

        image = np.asarray(row["images"][ref_world_view])
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)

        samples.append({
            "sha": sha,
            "short": sha.split("_")[2][:8],
            "ref_world_view": ref_world_view,
            "image": image,
            "raw_px": raw_px,
            "patched_px": patched_px,
            "roundtrip_max_px": float(np.nanmax(roundtrip_err)),
            "roundtrip_median_px": float(np.nanmedian(roundtrip_err)),
            "raw_median_px": float(np.nanmedian(raw_err)),
        })

    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 6.2), squeeze=False)
    for col, sample in enumerate(samples):
        for row_idx, (label, color, key) in enumerate([
            ("RAW/no adapter", "red", "raw_px"),
            ("PATCHED adapter", "lime", "patched_px"),
        ]):
            ax = axes[row_idx, col]
            ax.imshow(sample["image"])
            pts = sample[key]
            finite = np.isfinite(pts).all(axis=1)
            in_bounds = (
                finite
                & (pts[:, 0] >= 0) & (pts[:, 0] < sample["image"].shape[1])
                & (pts[:, 1] >= 0) & (pts[:, 1] < sample["image"].shape[0])
            )
            ax.scatter(pts[in_bounds, 0], pts[in_bounds, 1], s=2, c=color, alpha=0.9,
                       linewidths=0)
            ax.set_title(f"{sample['short']} view {sample['ref_world_view']} {label}")
            ax.axis("off")
    fig.suptitle("Trellis2-MV seed reprojection: raw convention vs patched adapter")
    fig.tight_layout()
    live_path = out_dir / "live_seed_flip_ab_patched.png"
    fig.savefig(live_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bins = np.linspace(0.0, max(1.0, np.nanpercentile(all_raw_err, 99)), 80)
    ax.hist(all_raw_err, bins=bins, alpha=0.55, label="raw/no adapter", color="red")
    ax.hist(all_roundtrip_err, bins=bins, alpha=0.75, label="patched round-trip",
            color="green")
    ax.set_xlabel("pixel error vs emitted patched seed coords")
    ax.set_ylabel("seed points")
    ax.set_title("Seed projection round-trip error")
    ax.legend()
    hist_path = out_dir / "seed_projection_roundtrip_hist.png"
    fig.tight_layout()
    fig.savefig(hist_path, dpi=160)
    plt.close(fig)

    sign_summary = {k: finite_median_or_penalty(np.asarray(v)) for k, v in sign_errs.items()}
    keys = sorted(sign_summary)
    vals = [sign_summary[k] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["green" if k == "-1,-1,1" else "gray" for k in keys]
    ax.bar(range(len(keys)), [max(v, 1.0e-4) for v in vals], color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(len(keys)), keys, rotation=45, ha="right")
    ax.set_ylabel("median pixel error")
    ax.set_xlabel("diag(sx,sy,sz) camera adapter")
    ax.set_title("Camera sign sweep against patched loader seed pixels")
    best_idx = keys.index("-1,-1,1")
    ax.text(best_idx, max(sign_summary["-1,-1,1"], 1.0e-4) * 3.0, "best",
            ha="center", va="bottom", color="green")
    sweep_path = out_dir / "camera_sign_sweep.png"
    fig.tight_layout()
    fig.savefig(sweep_path, dpi=160)
    plt.close(fig)

    summary = {
        "samples": [
            {
                "sha": s["sha"],
                "ref_world_view": s["ref_world_view"],
                "roundtrip_max_px": s["roundtrip_max_px"],
                "roundtrip_median_px": s["roundtrip_median_px"],
                "raw_median_px": s["raw_median_px"],
            }
            for s in samples
        ],
        "overall": {
            "roundtrip_max_px": float(np.nanmax(all_roundtrip_err)),
            "roundtrip_median_px": float(np.nanmedian(all_roundtrip_err)),
            "raw_median_px": float(np.nanmedian(all_raw_err)),
            "best_sign": min(sign_summary, key=sign_summary.get),
        },
        "sign_sweep_median_px": sign_summary,
        "artifacts": {
            "live_seed_flip_ab_patched": str(live_path),
            "seed_projection_roundtrip_hist": str(hist_path),
            "camera_sign_sweep": str(sweep_path),
        },
    }
    summary_path = out_dir / "camera_fix_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["overall"], indent=2))
    print(f"wrote {live_path}")
    print(f"wrote {hist_path}")
    print(f"wrote {sweep_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
