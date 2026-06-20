"""Precompute frozen Pi3X + DINOv2 features for the multi-view dataset.

Pi3X (local_points, conf) and DINOv2 (spatial features) are frozen and depend ONLY on
the un-augmented per-view images, so their outputs are identical across all training
epochs and can be cached once. Loading them at train time skips the two heaviest forwards
(a 36-layer Pi3X ViT × N views + DINOv2 × N views) → ~2-4x faster training.

Writes one <uid>.npz per object to --out (fp16): local_points (N,H,W,3), conf (N,H,W,1),
dino_feats (N,C_d,H',W'). Point the training DataConfig.mv_feature_cache at --out.

Usage (run once per split, multi-GPU shards by process):
  accelerate launch --num_processes 8 --module scripts.data.precompute_mv_features \
      --data-path datasets/3d-front-multiview-full --split train \
      --out datasets/mv-feature-cache/train
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("RUN_TS", "precompute")

import numpy as np
import torch
from accelerate import PartialState
from tqdm import tqdm

from src.utils.inference import prepare_mv_model_for_inference, _filter_dataclass_kwargs
from src.utils.config import DataConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="datasets/3d-front-multiview-full")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config-name", default="edgerunner_3d_front_multiview")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    state = PartialState()
    device = state.device
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the model purely for its frozen Pi3X + DINOv2 encoders.
    model, model_cfg, data_cfg = prepare_mv_model_for_inference(config_name=args.config_name)
    model.to(device).eval()

    # Load the requested split with the MV transform (deterministic, is_train=False).
    from functools import partial
    import datasets
    from transformers import AutoImageProcessor
    from src.data.mesh import transform_3d_front_multiview

    data = datasets.load_from_disk(Path(args.data_path).absolute().as_posix())
    split = args.split if args.split in data else next(iter(data))
    img_proc = AutoImageProcessor.from_pretrained(
        data_cfg.image_preprocessor, size_divisor=data_cfg.image_size_divisor
    )
    ds = data[split].with_transform(
        partial(transform_3d_front_multiview, is_train=False, data_cfg=data_cfg,
                image_preprocessor=img_proc)
    )

    n_total = len(ds) if args.limit is None else min(len(ds), args.limit)
    shard = list(range(state.process_index, n_total, state.num_processes))

    pi3x = model.pi3x_encoder
    dino = model.cond_encoder_img

    done = skipped = 0
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for i in tqdm(shard, position=state.process_index, dynamic_ncols=True):
                ex = ds[i]
                uid = ex["uid"] if not isinstance(ex["uid"], list) else ex["uid"][0]
                fpath = out_dir / f"{uid}.npz"
                if fpath.exists():
                    skipped += 1
                    continue
                pv = torch.as_tensor(np.asarray(ex["pixel_values"]), dtype=torch.float32)
                pv = pv.unsqueeze(0).to(device)          # (1, N, C, H, W)
                _, N, C, H, W = pv.shape

                out = pi3x.forward_all_views_joint(pv)
                lp = out["local_points"][0].float().cpu().numpy().astype(np.float16)
                cf = (out["conf"][0].float().cpu().numpy().astype(np.float16)
                      if out.get("conf") is not None else None)
                df = dino(pixel_values=pv.reshape(N, C, H, W))
                df = df.reshape(N, *df.shape[1:]).float().cpu().numpy().astype(np.float16)

                np.savez_compressed(
                    fpath, local_points=lp,
                    conf=cf if cf is not None else np.zeros((N, H, W, 1), np.float16),
                    dino_feats=df,
                )
                done += 1
    print(f"[rank {state.process_index}] wrote {done}, skipped {skipped} -> {out_dir}")


if __name__ == "__main__":
    main()
