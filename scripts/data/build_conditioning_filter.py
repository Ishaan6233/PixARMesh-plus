#!/usr/bin/env python3
"""Build the degenerate-conditioning sidecar for the Trellis2-MV dataset.

For every instance in metadata.csv, computes (a) how many views its HF scene row
has and (b) the max per-view covisibility support of its object points — via the
same `covis_object_supports` + `norm_to_world_transform` frame correction the
training loader uses (mv_frame_correction). Instances with <2 views or support 0
in every view (object-blind) get keep=False; the loader drops them when
`mv_filter_degenerate: true`. Scoring WITHOUT the frame correction over-drops
massively (47.7% marked blind, 99.2% of them actually visible — 2026-07-03);
conditioning_filter.meta.json records the frame mode so the loader can refuse a
mismatched sidecar.

Usage:
  PYTHONPATH=. python scripts/data/build_conditioning_filter.py \
      [--mesh-dataset <trellis2 root>] [--hf-dataset datasets/3d-front-multiview-full] \
      [--workers 16]

Writes <mesh-dataset>/conditioning_filter.csv
(columns: sha256, scene_id, hf_n_views, support_max, n_views_with_support, keep).
"""

import argparse
import csv
import hashlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import datasets as hf_datasets
from src.data.trellis2_mv import covis_object_supports, norm_to_world_transform

_TRELLIS2_DEFAULT = (
    "datasets/mesh_datasets/datasets/"
    "3d-front-trellis2-slat-mv-da3-aug-srcperturb-r5-qfcat-obj015-light-bgtex-20260629"
)

# Worker globals (populated once per fork)
_SCENE_CAMS: dict = {}
_UID_OBJ_T: dict = {}  # uid -> HF objects.transforms[0] (object-canonical -> HF-world)
_MESH_ROOT: Path | None = None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mesh-dataset", default=_TRELLIS2_DEFAULT)
    p.add_argument("--hf-dataset", default="datasets/3d-front-multiview-full")
    p.add_argument("--min-views", type=int, default=2,
                   help="HF rows with fewer views than this are dropped")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--num-samples", type=int, default=-1, help="debug cap")
    return p.parse_args()


def _stable_seed(*parts) -> int:
    key = "|".join(str(p) for p in parts)
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def _score_instance(inst: dict) -> dict:
    sha, scene_id = inst["sha256"], inst["scene_id"]
    rec = {"sha256": sha, "scene_id": scene_id, "hf_n_views": 0,
           "support_max": 0.0, "n_views_with_support": 0, "keep": False}
    cams = _SCENE_CAMS.get(scene_id)
    if cams is None:
        rec["error"] = "scene_id not in HF dataset"
        return rec
    w2c, ks = cams
    rec["hf_n_views"] = len(w2c)
    try:
        cond = torch.load(_MESH_ROOT / "mv_cond" / f"{sha}.pt",
                          map_location="cpu", weights_only=False)["cond"]
    except Exception as e:
        rec["error"] = f"mv_cond load failed: {e}"
        return rec
    t_hf = _UID_OBJ_T.get(cond["uid"])
    if t_hf is None:
        # Mirrors the loader: without a uid-matched row the frame offset is
        # unrecoverable, so the instance cannot be conditioned correctly.
        rec["error"] = "no uid-matched HF row (objects.transforms missing)"
        return rec
    raw_hw = (int(round(float(ks[0][1, 2]) * 2)), int(round(float(ks[0][0, 2]) * 2)))
    # Same seeding convention as eval_pi3x_depth.select_covis_views
    _, support = covis_object_supports(
        cond, w2c, ks, raw_hw,
        rng=np.random.RandomState(_stable_seed(sha, "objpts")),
        T_norm_to_world=norm_to_world_transform(cond, t_hf),
    )
    rec["support_max"] = float(support.max())
    rec["n_views_with_support"] = int((support > 0).sum())
    return rec


def main():
    args = parse_args()
    mesh_root = Path(args.mesh_dataset)

    insts = []
    with open(mesh_root / "metadata.csv", newline="") as f:
        for row in csv.DictReader(f):
            insts.append({"sha256": row["sha256"], "scene_id": row["scene_id"]})
    if args.num_samples > 0:
        insts = insts[: args.num_samples]
    print(f"{len(insts)} instances from {mesh_root / 'metadata.csv'}")

    print(f"Loading cameras from {args.hf_dataset} ...")
    ds = hf_datasets.load_from_disk(str(Path(args.hf_dataset).absolute()))["train"]
    ds = ds.select_columns(["scene_id", "wrd2cam_rects", "Ks", "uid", "objects"])
    global _SCENE_CAMS, _MESH_ROOT
    _MESH_ROOT = mesh_root
    seen = set()
    for row in tqdm(ds, desc="cameras", unit="row"):
        sid = row["scene_id"]
        u = row["uid"] if isinstance(row["uid"], str) else row["uid"][0]
        tr = (row.get("objects") or {}).get("transforms")
        if tr:
            _UID_OBJ_T[u] = np.array(tr[0], dtype=np.float32)
        if sid in seen:
            continue
        seen.add(sid)
        _SCENE_CAMS[sid] = (
            np.stack([np.array(m, dtype=np.float32) for m in row["wrd2cam_rects"]]),
            np.stack([np.array(m, dtype=np.float32) for m in row["Ks"]]),
        )
    print(f"{len(_SCENE_CAMS)} unique scenes, {len(_UID_OBJ_T)} uid transforms")

    with Pool(args.workers) as pool:
        recs = list(tqdm(pool.imap(_score_instance, insts, chunksize=16),
                         total=len(insts), desc="scoring", unit="inst"))

    n_err = n_few_views = n_blind = 0
    for r in recs:
        if "error" in r:
            n_err += 1
            continue
        if r["hf_n_views"] < args.min_views:
            n_few_views += 1
        elif r["support_max"] <= 0:
            n_blind += 1
        else:
            r["keep"] = True

    out_path = mesh_root / "conditioning_filter.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sha256", "scene_id", "hf_n_views", "support_max",
            "n_views_with_support", "keep"])
        w.writeheader()
        for r in recs:
            r.pop("error", None)
            w.writerow(r)

    meta = {
        "frame_correction": True,
        "min_views": args.min_views,
        "n_total": len(recs),
        "n_keep": int(sum(r["keep"] for r in recs)),
    }
    with open(mesh_root / "conditioning_filter.meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    n_drop = len(recs) - sum(r["keep"] for r in recs)
    print(f"\nWrote {out_path} (+ conditioning_filter.meta.json)")
    print(f"keep=False: {n_drop}/{len(recs)} ({100 * n_drop / len(recs):.1f}%) — "
          f"{n_few_views} with <{args.min_views} views, {n_blind} object-blind, "
          f"{n_err} load-errors")
    print("Expected (frame-corrected): ~1,522 single-view + ~0.3% truly blind ≈ 9%")


if __name__ == "__main__":
    main()
