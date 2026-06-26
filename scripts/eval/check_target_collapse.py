"""Confirm (or refute) MV target-ID collapse for one scene.

Loads every object of a single scene, runs the real eval pipeline
(Pi3X -> build_geo_obj_pc seed -> _get_per_view_target_ids), and prints, per
object: the discovered per-view target instance IDs, the GT inst_id, and the
seed footprint. If different objects discover the SAME target_ids_n, the
per-object seed is not localising the object (target-ID collapse) — the failure
is upstream of the panoptic masks.

Usage:
    PYTHONPATH=. micromamba run -n pixarmesh124 python scripts/eval/check_target_collapse.py \
        --dataset datasets/3d-front-multiview --pi3x-ckpt checkpoints/pi3x \
        --scene 00110bde
"""
import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

import datasets as hf_datasets
from scripts.eval.eval_voxels import _eval_collate, _select_ref_view
from src.data.mesh import transform_3d_front_multiview
from src.models.frozen_geo_encoder import _get_per_view_target_ids, build_geo_obj_pc
from src.models.utils import get_pi3x_encoder
from src.utils.config import DataConfig, ModelConfig


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--pi3x-ckpt", required=True)
    p.add_argument("--scene", required=True, help="Scene UID prefix (matches uid.startswith)")
    p.add_argument("--depth-rtol", type=float, default=0.10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    data_cfg = DataConfig(
        type="3d-front-multiview", path=args.dataset, num_views=4, num_points=4096,
        norm_bound=0.95, load_images=True,
        image_preprocessor="facebook/dpt-dinov2-small-nyu", image_size_divisor=28,
        use_masked_obj_pc=True, random_scale=False, random_rotate=False,
        random_jitter_point_clouds=False, random_jitter_depth=False, random_shift=False,
    )
    raw = hf_datasets.load_from_disk(str(Path(args.dataset).absolute()))
    val = raw[next(k for k in ("val", "validation", "test") if k in raw)]
    idx = [i for i, u in enumerate(val["uid"]) if str(u).startswith(args.scene)]
    if not idx:
        raise SystemExit(f"no objects for scene prefix {args.scene!r}")
    print(f"scene {args.scene}: {len(idx)} objects -> {[val['uid'][i] for i in idx]}")
    gt_inst = [val[i]["objects"].get("inst_ids") for i in idx]

    ip = AutoImageProcessor.from_pretrained(
        data_cfg.image_preprocessor, size_divisor=data_cfg.image_size_divisor
    )
    val_sel = val.select(idx).with_transform(
        partial(transform_3d_front_multiview, is_train=False, data_cfg=data_cfg,
                image_preprocessor=ip)
    )
    loader = DataLoader(val_sel, batch_size=len(idx), collate_fn=_eval_collate,
                        num_workers=0, shuffle=False)

    model_cfg = ModelConfig(vocab_size=1, num_pos_tokens=512, bos_token_id=1,
                            eos_token_id=2, pad_token_id=0, pi3x_ckpt_path=args.pi3x_ckpt,
                            pi3x_disable_multimodal=True)
    pi3x = get_pi3x_encoder(model_cfg).to(device).float().eval()

    batch = next(iter(loader))
    pixel_values = batch["pixel_values"].to(device)
    st = batch["scene_transforms"].to(device).float()
    K_f = batch["K_per_view"].to(device).float()
    view_mask = batch["view_mask"].to(device)
    panoptic_masks = batch["panoptic_masks"].to(device)
    cond_pcs_2d = batch["point_clouds_2d"].to(device)
    uids = batch["uid"]
    B = pixel_values.shape[0]

    with torch.no_grad():
        out = pi3x.forward_all_views_joint(pixel_values.float())
        lp = out["local_points"]
        lp_z = lp[..., 2]
        ref_idx = _select_ref_view(lp, view_mask)
        target_rows = []
        for b in range(B):
            rv = int(ref_idx[b].item())
            seed = build_geo_obj_pc(lp[b:b+1, rv], cond_pcs_2d[b:b+1], st[b:b+1, rv])[0]
            tids = _get_per_view_target_ids(
                seed.float(), st[b], K_f[b], panoptic_masks[b], lp_z[b], view_mask[b],
                depth_rtol=args.depth_rtol,
            )
            s = seed.reshape(-1, 3).cpu().numpy()
            target_rows.append(tids.cpu().tolist())
            print(f"\n{uids[b]}  (GT inst_id={gt_inst[b]})")
            print(f"  discovered target_ids_n (per view) = {tids.cpu().tolist()}")
            print(f"  seed centroid={np.round(s.mean(0), 3)}  extent={np.round(s.ptp(0), 3)}")

    # verdict
    uniq = {tuple(t) for t in target_rows}
    print("\n" + "=" * 60)
    if len(uniq) == 1 and B > 1:
        print(f"COLLAPSE CONFIRMED: all {B} distinct objects share target_ids_n = "
              f"{target_rows[0]}  -> seed does not localise the object.")
    else:
        print(f"NO COLLAPSE: {len(uniq)} distinct target_ids_n across {B} objects.")


if __name__ == "__main__":
    main()
