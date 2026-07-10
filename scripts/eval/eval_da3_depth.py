#!/usr/bin/env python3
"""DA3 metric/scale diagnostic for Trellis2 multi-view training views.

The script runs the frozen DA3 encoder over either all HF scene views or the
same covisibility-selected subsets used by `Trellis2MVDataset`, then compares
predicted camera-frame depth (`local_points[..., 2]`) against GT depth maps.
The headline metric for this port is per-scene cross-view scale spread:
`r_std = std(s_v / median(s_v))`, where `s_v = median(gt_depth / pred_depth)`.
"""

from __future__ import annotations

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
from src.models.utils import get_da3_encoder
from src.utils.config import ModelConfig

DEPTH_MAX_M = 10.0
_TRELLIS2_DEFAULT = (
    "datasets/mesh_datasets/datasets/"
    "3d-front-trellis2-slat-mv-da3-aug-srcperturb-r5-qfcat-obj015-light-bgtex-20260629"
)


def parse_args():
    parser = argparse.ArgumentParser(description="DA3 depth vs GT depth diagnostic")
    parser.add_argument("--dataset", default="datasets/3d-front-multiview-full")
    parser.add_argument("--split", default=None)
    parser.add_argument("--mode", choices=["all-views", "train-views"], default="train-views")
    parser.add_argument("--mesh-dataset", default=_TRELLIS2_DEFAULT)
    parser.add_argument("--da3-ckpt", default="checkpoints/da3/DA3-GIANT")
    parser.add_argument("--image-preprocessor", default="facebook/dpt-dinov2-small-nyu")
    parser.add_argument("--image-size-divisor", type=int, default=28)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-views", type=int, default=21)
    parser.add_argument("--min-valid-px", type=int, default=1000)
    parser.add_argument("--conf-thresh", type=float, default=0.0)
    parser.add_argument("--align-subsample", type=int, default=20000)
    parser.add_argument("--include-far-plane", action="store_true")
    parser.add_argument("--covis-k-max", type=int, default=8)
    parser.add_argument("--covis-min-support", type=int, default=50)
    parser.add_argument("--pad-to-slots", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def stable_seed(*parts) -> int:
    key = "|".join(str(p) for p in parts)
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def decode_gt_depth(raw: np.ndarray) -> np.ndarray:
    return (1.0 - raw.astype(np.float32) / 255.0) * DEPTH_MAX_M


def make_valid_mask(raw: np.ndarray, pred_z: np.ndarray, include_far_plane: bool = False):
    if include_far_plane:
        gt_valid = raw <= 254
    else:
        gt_valid = (raw >= 1) & (raw <= 254)
    return gt_valid & np.isfinite(pred_z) & (pred_z > 1e-6)


def crop_padding(arr: np.ndarray, pad_top: int, pad_left: int, raw_hw: tuple[int, int]):
    h, w = raw_hw
    out = arr[:, pad_top:pad_top + h, pad_left:pad_left + w]
    assert out.shape[-2:] == (h, w), f"crop {out.shape} != raw {raw_hw}"
    return out


def fit_scale_shift(pred: np.ndarray, gt: np.ndarray):
    aligned = align_depth(pred, gt)
    if float(np.ptp(pred)) <= 0:
        return float("nan"), float("nan"), aligned
    a, b = np.polyfit(pred, aligned, 1)
    return float(a), float(b), aligned


def _err_stats(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    return (
        float(np.mean(np.abs(pred - gt) / gt)),
        float(np.sqrt(np.mean((pred - gt) ** 2))),
    )


def compute_view_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    conf: np.ndarray,
    seed: int,
    align_subsample: int,
    conf_thresh: float,
) -> dict:
    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)
    c = conf[mask].astype(np.float64)

    abs_rel_raw, rmse_raw = _err_stats(p, g)
    log_err = np.log(p) - np.log(g)
    silog_raw = float(np.sqrt(max(np.mean(log_err**2) - np.mean(log_err) ** 2, 0.0)) * 100.0)
    s_v = float(np.median(g / p))
    s_v_log = float(np.exp(np.mean(np.log(g) - np.log(p))))
    abs_rel_pv, rmse_pv = _err_stats(s_v * p, g)

    rng = np.random.RandomState(seed)
    sub = rng.choice(len(p), align_subsample, replace=False) if len(p) > align_subsample else np.arange(len(p))
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
        "mean_conf": float(c.mean()) if len(c) else float("nan"),
        "n_valid_conf": n_valid_conf,
        "absRel_raw_conf": abs_rel_raw_conf,
        "absRel_pv_scale_conf": abs_rel_pv_conf,
    }


def scene_scale_metrics(s_views: np.ndarray) -> dict:
    s_views = np.asarray(s_views, dtype=np.float64)
    if len(s_views) == 0:
        return {k: float("nan") for k in ("s_scene", "s_scene_geo", "r_std", "log_r_std", "r_mean_abs_dev", "r_spread")}
    s_scene = float(np.median(s_views))
    s_scene_geo = float(np.exp(np.mean(np.log(s_views))))
    out = {"s_scene": s_scene, "s_scene_geo": s_scene_geo}
    if len(s_views) < 2:
        out.update({k: float("nan") for k in ("r_std", "log_r_std", "r_mean_abs_dev", "r_spread")})
        return out
    r = s_views / s_scene
    out["r_std"] = float(np.std(r))
    out["log_r_std"] = float(np.std(np.log(s_views)))
    out["r_mean_abs_dev"] = float(np.mean(np.abs(r - 1.0)))
    out["r_spread"] = float(s_views.max() / s_views.min() - 1.0)
    return out


def strided_view_subset(n_views: int, max_views: int) -> list[int]:
    if n_views <= max_views:
        return list(range(n_views))
    return sorted(set(np.round(np.linspace(0, n_views - 1, max_views)).astype(int).tolist()))


def _sanitize(rec: dict) -> dict:
    out = {}
    for key, value in rec.items():
        if isinstance(value, (np.floating, float)):
            value = float(value)
            out[key] = value if np.isfinite(value) else None
        elif isinstance(value, (np.integer, int)):
            out[key] = int(value)
        elif isinstance(value, np.bool_):
            out[key] = bool(value)
        else:
            out[key] = value
    return out


def _norm_uid(uid):
    return uid[0] if isinstance(uid, (list, tuple)) else uid


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
    for key in ("validation", "val", "test", "train"):
        if key in raw:
            return raw[key], key
    raise ValueError(f"No usable split in {dataset_path}; available: {list(raw.keys())}")


def build_da3(ckpt_path: str, device: str):
    cfg = ModelConfig(
        vocab_size=1,
        num_pos_tokens=512,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_da3=True,
        geo_encoder_type="da3",
        da3_ckpt_path=ckpt_path,
    )
    return get_da3_encoder(cfg).to(device).float().eval()


def preprocess_views(pil_images: list, processor):
    pvs = []
    raw_hw = None
    for img in pil_images:
        arr = np.asarray(img)
        if raw_hw is None:
            raw_hw = arr.shape[:2]
        assert arr.shape[:2] == raw_hw, f"mixed view resolutions: {arr.shape[:2]} vs {raw_hw}"
        pvs.append(processor(images=arr, return_tensors="pt")["pixel_values"][0])
    pv = torch.stack(pvs).unsqueeze(0)
    out_h, out_w = pv.shape[-2:]
    pad_left = (out_w - raw_hw[1]) // 2
    pad_top = (out_h - raw_hw[0]) // 2
    return pv, pad_top, pad_left, raw_hw


def run_da3_depth(da3, pixel_values: torch.Tensor, device: str):
    with torch.inference_mode():
        out = da3.forward_all_views_joint(pixel_values.to(device).float())
        pred_z = out["local_points"][0, ..., 2].float().cpu().numpy()
        conf = out.get("conf")
        if conf is None:
            conf_np = np.ones_like(pred_z, dtype=np.float32)
        else:
            conf_np = conf[0, ..., 0].float().cpu().numpy()
    return pred_z, conf_np


def process_scene(da3, processor, args, *, uid, scene_id, images, depths, view_indices, n_views_total, key, instance_uid=None, subset=None):
    t0 = time.time()
    pv, pad_top, pad_left, raw_hw = preprocess_views([images[v] for v in view_indices], processor)
    pred_z, conf = run_da3_depth(da3, pv, args.device)
    pred_z = crop_padding(pred_z, pad_top, pad_left, raw_hw)
    conf = crop_padding(conf, pad_top, pad_left, raw_hw)

    base = {"uid": uid, "scene_id": scene_id, "mode": args.mode, "instance_uid": instance_uid, "subset": subset}
    n_unique = len(set(view_indices))
    view_recs, pixels, s_used, seen = [], [], [], set()
    for k, view_idx in enumerate(view_indices):
        if view_idx in seen:
            continue
        seen.add(view_idx)
        raw = np.asarray(depths[view_idx])
        if raw.ndim == 3:
            raw = raw[..., 0]
        gt = decode_gt_depth(raw)
        mask = make_valid_mask(raw, pred_z[k], args.include_far_plane)
        rec = dict(base, key=key, view_idx=int(view_idx), n_views_scene=int(n_views_total), n_views_used=n_unique, n_valid=int(mask.sum()))
        if rec["n_valid"] < args.min_valid_px:
            rec["skipped"] = True
            pixels.append(None)
        else:
            metrics = compute_view_metrics(
                pred_z[k],
                gt,
                mask,
                conf[k],
                seed=stable_seed(scene_id, view_idx),
                align_subsample=args.align_subsample,
                conf_thresh=args.conf_thresh,
            )
            rec.update(metrics, skipped=False)
            pixels.append((pred_z[k][mask].astype(np.float32), gt[mask].astype(np.float32)))
            s_used.append(metrics["s_v"])
        view_recs.append(rec)

    stats = scene_scale_metrics(np.asarray(s_used))
    s_scene = stats["s_scene"]
    for rec, px in zip(view_recs, pixels):
        if px is None or not np.isfinite(s_scene):
            rec["absRel_scene_scale"] = rec["rmse_scene_scale"] = float("nan")
        else:
            pred, gt = px
            rec["absRel_scene_scale"], rec["rmse_scene_scale"] = _err_stats(
                s_scene * pred.astype(np.float64),
                gt.astype(np.float64),
            )

    scored = [rec for rec in view_recs if not rec["skipped"]]
    scene_rec = dict(
        base,
        key=key,
        n_views=int(n_views_total),
        n_views_used=n_unique,
        n_slots=len(view_indices),
        n_views_scored=len(scored),
        wall_time_s=round(time.time() - t0, 3),
        **stats,
    )
    for metric in ("absRel_raw", "absRel_pv_scale", "absRel_scene_scale", "absRel_ss", "silog_raw"):
        vals = [rec[metric] for rec in scored if np.isfinite(rec.get(metric, float("nan")))]
        scene_rec[f"mean_{metric}"] = float(np.mean(vals)) if vals else float("nan")
    return scene_rec, view_recs


def load_trellis2_instances(mesh_root: Path) -> list[dict]:
    instances = []
    with open(mesh_root / "metadata.csv", newline="") as f:
        for row in csv.DictReader(f):
            instances.append({"sha256": row["sha256"], "uid": row["uid"], "scene_id": row["scene_id"]})
    all_idx = list(range(len(instances)))
    np.random.RandomState(42).shuffle(all_idx)
    holdout = set(all_idx[: max(1, int(0.05 * len(all_idx)))])
    for i, inst in enumerate(instances):
        inst["subset"] = "holdout" if i in holdout else "train"
    return instances


def select_covis_views(cond, all_wrd2cams, all_Ks, raw_img_hw, args, seed, T_norm_to_world):
    from src.data.trellis2_mv import _select_diverse_views, covis_object_supports

    obj_pts, support = covis_object_supports(
        cond,
        all_wrd2cams,
        all_Ks,
        raw_img_hw,
        rng=np.random.RandomState(seed),
        T_norm_to_world=T_norm_to_world,
    )
    return _select_diverse_views(
        obj_pts,
        all_wrd2cams,
        all_Ks,
        raw_img_hw,
        support,
        k_max=min(args.covis_k_max, len(all_wrd2cams)),
        min_support_pts=args.covis_min_support,
    )


def iter_all_views(args, hf_split):
    seen, first_rows = set(), []
    for i, scene_id in enumerate(hf_split["scene_id"]):
        if scene_id not in seen:
            seen.add(scene_id)
            first_rows.append(i)
    idxs = [i for k, i in enumerate(first_rows) if k % args.num_shards == args.shard_idx]
    if args.num_samples > 0:
        idxs = idxs[: args.num_samples]
    for i in idxs:
        row = hf_split[i]
        n_views = len(row["images"])
        yield {
            "key": row["scene_id"],
            "uid": _norm_uid(row["uid"]),
            "scene_id": row["scene_id"],
            "images": row["images"],
            "depths": row["depths"],
            "view_indices": strided_view_subset(n_views, args.max_views),
            "n_views_total": n_views,
            "instance_uid": None,
            "subset": None,
        }


def iter_train_views(args, hf_split):
    mesh_root = Path(args.mesh_dataset)
    instances = load_trellis2_instances(mesh_root)
    scene_to_idx = {}
    for i, scene_id in enumerate(hf_split["scene_id"]):
        scene_to_idx.setdefault(scene_id, i)
    uid_to_idx = {_norm_uid(uid): i for i, uid in enumerate(hf_split["uid"])}

    groups: dict[str, list[dict]] = {}
    for inst in instances:
        if inst["subset"] != "train":
            continue
        groups.setdefault(inst["scene_id"], []).append(inst)
    shard_scenes = [sid for i, sid in enumerate(sorted(groups)) if i % args.num_shards == args.shard_idx]

    emitted = 0
    for scene_id in shard_scenes:
        row_idx = scene_to_idx.get(scene_id)
        row = hf_split[row_idx] if row_idx is not None else None
        for inst in sorted(groups[scene_id], key=lambda item: item["sha256"]):
            if args.num_samples > 0 and emitted >= args.num_samples:
                return
            emitted += 1
            if row is None:
                yield {"key": inst["sha256"], "error": f"scene_id {scene_id} not in HF dataset"}
                continue
            cond_path = mesh_root / "mv_cond" / f"{inst['sha256']}.pt"
            try:
                cond = torch.load(cond_path, map_location="cpu", weights_only=False)["cond"]
            except Exception as exc:
                yield {"key": inst["sha256"], "error": f"mv_cond load failed: {exc}"}
                continue
            uid_idx = uid_to_idx.get(cond["uid"])
            if uid_idx is None:
                yield {"key": inst["sha256"], "error": "no uid-matched HF row for frame correction"}
                continue
            obj_row = hf_split[uid_idx]
            transforms = (obj_row.get("objects") or {}).get("transforms")
            if not transforms:
                yield {"key": inst["sha256"], "error": "uid-matched HF row has no objects.transforms"}
                continue

            all_w2c = np.stack([np.asarray(m, dtype=np.float32) for m in row["wrd2cam_rects"]])
            all_ks = np.stack([np.asarray(m, dtype=np.float32) for m in row["Ks"]])
            raw_hw = (int(round(float(all_ks[0][1, 2]) * 2)), int(round(float(all_ks[0][0, 2]) * 2)))
            from src.data.trellis2_mv import norm_to_world_transform

            selected = select_covis_views(
                cond,
                all_w2c,
                all_ks,
                raw_hw,
                args,
                seed=stable_seed(inst["sha256"], "objpts"),
                T_norm_to_world=norm_to_world_transform(cond, transforms[0]),
            )
            if not selected:
                yield {"key": inst["sha256"], "error": "object-blind: covis support 0 in every view"}
                continue
            if args.pad_to_slots > 0 and len(selected) < args.pad_to_slots:
                selected = list(selected) + [selected[-1]] * (args.pad_to_slots - len(selected))
            yield {
                "key": inst["sha256"],
                "uid": _norm_uid(row["uid"]),
                "scene_id": scene_id,
                "images": row["images"],
                "depths": row["depths"],
                "view_indices": [int(v) for v in selected],
                "n_views_total": len(row["images"]),
                "instance_uid": inst["uid"],
                "subset": inst["subset"],
            }


def _summary(records: list[dict]) -> dict:
    scored = [rec for rec in records if rec.get("n_views_scored", 0) >= 2]

    def pctls(key):
        vals = np.asarray([rec[key] for rec in scored if rec.get(key) is not None], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return {"n": 0}
        return {
            "n": int(len(vals)),
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90)),
        }

    return {
        "n_scenes": len(records),
        "n_scenes_scored": len(scored),
        "cross_view_inconsistency": {
            "r_std": pctls("r_std"),
            "log_r_std": pctls("log_r_std"),
            "r_mean_abs_dev": pctls("r_mean_abs_dev"),
            "r_spread": pctls("r_spread"),
        },
        "raw_metric_error": {
            "mean_absRel_raw": pctls("mean_absRel_raw"),
            "mean_absRel_scene_scale": pctls("mean_absRel_scene_scale"),
        },
    }


def run(args):
    from transformers import AutoImageProcessor

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes_path = out_dir / f"scenes_shard{args.shard_idx}.jsonl"
    views_path = out_dir / f"views_shard{args.shard_idx}.jsonl"
    errors_path = out_dir / f"errors_shard{args.shard_idx}.log"

    hf_split, split_name = load_hf_split(args.dataset, args.split)
    keep_cols = ["uid", "scene_id", "images", "depths"]
    if args.mode == "train-views":
        keep_cols += ["wrd2cam_rects", "Ks", "objects"]
    hf_split = hf_split.select_columns([col for col in keep_cols if col in hf_split.column_names])
    print(
        f"dataset={args.dataset} split={split_name} rows={len(hf_split)} "
        f"mode={args.mode} shard={args.shard_idx}/{args.num_shards}"
    )

    processor = AutoImageProcessor.from_pretrained(
        args.image_preprocessor,
        size_divisor=args.image_size_divisor,
    )
    print(f"loading DA3 from {args.da3_ckpt} on {args.device}")
    da3 = build_da3(args.da3_ckpt, args.device)

    items = iter_all_views(args, hf_split) if args.mode == "all-views" else iter_train_views(args, hf_split)
    scene_records = []
    cache_scene, cache = None, {}
    with open(views_path, "w") as vf, open(scenes_path, "w") as sf, open(errors_path, "w") as ef:
        for item in tqdm(items, desc="da3-depth", unit="item"):
            key = item["key"]
            if "error" in item:
                ef.write(f"{key}\t{item['error']}\n")
                continue
            try:
                if item["scene_id"] != cache_scene:
                    cache_scene, cache = item["scene_id"], {}
                vset = tuple(item["view_indices"])
                if args.mode == "train-views" and vset in cache:
                    scene_rec, view_recs = cache[vset]
                    scene_rec = dict(scene_rec, key=key, instance_uid=item["instance_uid"], subset=item["subset"])
                    view_recs = [dict(rec, key=key, instance_uid=item["instance_uid"], subset=item["subset"]) for rec in view_recs]
                else:
                    scene_rec, view_recs = process_scene(
                        da3,
                        processor,
                        args,
                        uid=item["uid"],
                        scene_id=item["scene_id"],
                        images=item["images"],
                        depths=item["depths"],
                        view_indices=item["view_indices"],
                        n_views_total=item["n_views_total"],
                        key=key,
                        instance_uid=item["instance_uid"],
                        subset=item["subset"],
                    )
                    if args.mode == "train-views":
                        cache[vset] = (scene_rec, view_recs)
            except Exception:
                import traceback

                ef.write(f"{key}\n{traceback.format_exc()}\n")
                continue

            for rec in view_recs:
                vf.write(json.dumps(_sanitize(rec)) + "\n")
            sf.write(json.dumps(_sanitize(scene_rec)) + "\n")
            scene_records.append(_sanitize(scene_rec))

    summary = _summary(scene_records)
    (out_dir / f"summary_shard{args.shard_idx}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
