#!/usr/bin/env python3
"""Precompute frozen DA3 + DINOv2 features for Trellis2-MV.

Writes one `<uid>.npz` per Trellis2 object. Each file contains:
  - cache_version: cache contract version checked by the loader
  - local_points: (N,H,W,3) DA3 camera-frame points
  - conf: (N,H,W,1) DA3 confidence
  - dino_feats: (N,C,Hf,Wf) frozen DINOv2 feature maps
  - view_indices: (N,) selected HF view indices used by the dataset item
  - view_mask: (N,) valid selected-view slots; padded slots are False
  - ref_view: scalar local slot used to seed the object

Point `DataConfig.mv_feature_cache` at the output directory to skip the frozen
DA3/DINO forwards during training.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("RUN_TS", "precompute")

import numpy as np
import torch
from accelerate import PartialState
from tqdm import tqdm
from transformers import AutoImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.trellis2_mv import (
    MV_FEATURE_CACHE_EMPTY_MARKER_KEY,
    MV_FEATURE_CACHE_KEYS,
    MV_FEATURE_CACHE_POLICY_KEY,
    MV_FEATURE_CACHE_VERSION,
    Trellis2MVDataset,
    view_selection_policy_fingerprint,
)
from src.models.utils import get_da3_encoder, get_image_condition_encoder
from src.utils.config import DataConfig, ModelConfig


_REQUIRED_CACHE_KEYS = set(MV_FEATURE_CACHE_KEYS)


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute DA3+DINO features for Trellis2-MV")
    parser.add_argument("--config-name", default="edgerunner_3d_front_trellis2_mv")
    parser.add_argument("--mesh-dataset", default=None)
    parser.add_argument("--hf-dataset", default=None)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--da3-ckpt", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _filter_dataclass_kwargs(dataclass_type, values: dict):
    allowed = {field.name for field in fields(dataclass_type)}
    return {key: value for key, value in values.items() if key in allowed}


def _compose_configs(config_name: str) -> tuple[DataConfig, ModelConfig]:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    config_dir = Path(__file__).resolve().parents[2] / "configs"
    try:
        OmegaConf.register_new_resolver("sub", lambda x, y: x - y)
    except ValueError:
        pass
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)

    if OmegaConf.select(cfg, "dataset.src_data") is not None:
        data_values = OmegaConf.to_container(cfg.dataset.src_data, resolve=True)
    elif OmegaConf.select(cfg, "data") is not None:
        data_values = OmegaConf.to_container(cfg.data, resolve=True)
    else:
        data_values = OmegaConf.to_container(cfg.dataset, resolve=True)
    data_cfg = DataConfig(**_filter_dataclass_kwargs(DataConfig, data_values))

    model_values = OmegaConf.to_container(cfg.model, resolve=True)
    ds_model = OmegaConf.select(cfg, "dataset.model")
    if ds_model is not None:
        model_values.update(OmegaConf.to_container(ds_model, resolve=True))
    model_cfg = ModelConfig(**_filter_dataclass_kwargs(ModelConfig, model_values))
    return data_cfg, model_cfg


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)


def _cache_is_current(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as z:
            if not _REQUIRED_CACHE_KEYS.issubset(set(z.files)):
                return False
            return int(np.asarray(z["cache_version"]).item()) == MV_FEATURE_CACHE_VERSION
    except Exception:
        return False


def _write_empty_cache_marker(path: Path, ex: dict, data_cfg: DataConfig, reason: str) -> None:
    """Write a cache marker for items the runtime loader already deems empty."""
    view_mask_raw = np.asarray(ex.get("view_mask", []), dtype=bool).reshape(-1)
    view_indices_raw = np.asarray(ex.get("view_indices", []), dtype=np.int64).reshape(-1)
    n_slots = len(view_mask_raw) or len(view_indices_raw) or int(getattr(data_cfg, "num_views", 8) or 8)
    if len(view_indices_raw) == n_slots:
        view_indices = view_indices_raw
    else:
        view_indices = np.zeros(n_slots, dtype=np.int64)
    view_mask = np.zeros(n_slots, dtype=bool)
    ref_view = np.asarray(0, dtype=np.int64)
    np.savez_compressed(
        path,
        cache_version=np.asarray(MV_FEATURE_CACHE_VERSION, dtype=np.int64),
        local_points=np.zeros((n_slots, 1, 1, 3), dtype=np.float16),
        conf=np.zeros((n_slots, 1, 1, 1), dtype=np.float16),
        dino_feats=np.zeros((n_slots, 1, 1, 1), dtype=np.float16),
        view_indices=view_indices,
        view_mask=view_mask,
        ref_view=ref_view,
        **{
            MV_FEATURE_CACHE_EMPTY_MARKER_KEY: np.asarray(True),
            "empty_reason": np.asarray(reason),
            MV_FEATURE_CACHE_POLICY_KEY: np.asarray(view_selection_policy_fingerprint(data_cfg)),
        },
    )


def main():
    args = parse_args()
    state = PartialState()
    device = state.device
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg, model_cfg = _compose_configs(args.config_name)
    if args.mesh_dataset is not None:
        data_cfg.path = args.mesh_dataset
    if args.hf_dataset is not None:
        data_cfg.trellis2_hf_path = args.hf_dataset
    if args.da3_ckpt is not None:
        model_cfg.da3_ckpt_path = args.da3_ckpt
    data_cfg.load_images = True
    data_cfg.mv_feature_cache = ""

    image_processor = AutoImageProcessor.from_pretrained(
        data_cfg.image_preprocessor,
        size_divisor=data_cfg.image_size_divisor,
    )
    dataset = Trellis2MVDataset(
        str(Path(data_cfg.path).absolute()),
        str(Path(data_cfg.trellis2_hf_path).absolute()),
        data_cfg,
        image_processor,
        is_train=args.split == "train",
    )

    n_total = len(dataset) if args.limit is None else min(len(dataset), args.limit)
    shard = list(range(state.process_index, n_total, state.num_processes))

    da3 = get_da3_encoder(model_cfg).to(device).eval()
    dino = get_image_condition_encoder(model_cfg).to(device).eval()

    wrote = markers = skipped = invalid = 0
    with torch.no_grad():
        for idx in tqdm(shard, position=state.process_index, dynamic_ncols=True):
            ex = dataset[idx]
            uid = ex["uid"] if isinstance(ex["uid"], str) else ex["uid"][0]
            out_path = out_dir / f"{uid}.npz"
            if not args.overwrite and _cache_is_current(out_path):
                skipped += 1
                continue
            if "pixel_values" not in ex or not bool(np.asarray(ex.get("view_mask", [])).any()):
                _write_empty_cache_marker(
                    out_path,
                    ex,
                    data_cfg,
                    "runtime loader returned an empty MV conditioning example during precompute",
                )
                markers += 1
                invalid += 1
                continue

            pixel_values = torch.as_tensor(
                np.asarray(ex["pixel_values"]),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            _, N, C, H, W = pixel_values.shape
            with _autocast(device):
                geo_out = da3.forward_all_views_joint(pixel_values)
                dino_flat = dino(pixel_values=pixel_values.reshape(N, C, H, W))
            local_points = geo_out["local_points"][0].float().cpu().numpy().astype(np.float16)
            conf = geo_out.get("conf")
            if conf is None:
                conf_np = np.ones((*local_points.shape[:3], 1), dtype=np.float16)
            else:
                conf_np = conf[0].float().cpu().numpy().astype(np.float16)
            dino_np = dino_flat.reshape(N, *dino_flat.shape[1:]).float().cpu().numpy().astype(np.float16)
            view_indices = np.asarray(ex["view_indices"], dtype=np.int64)
            view_mask = np.asarray(ex["view_mask"], dtype=bool)
            ref_view = np.asarray(int(ex["ref_view"]), dtype=np.int64)

            np.savez_compressed(
                out_path,
                cache_version=np.asarray(MV_FEATURE_CACHE_VERSION, dtype=np.int64),
                local_points=local_points,
                conf=conf_np,
                dino_feats=dino_np,
                view_indices=view_indices,
                view_mask=view_mask,
                ref_view=ref_view,
                **{
                    MV_FEATURE_CACHE_POLICY_KEY: np.asarray(
                        view_selection_policy_fingerprint(data_cfg)
                    )
                },
            )
            wrote += 1

    print(
        f"[rank {state.process_index}] wrote={wrote} markers={markers} skipped={skipped} "
        f"invalid={invalid} out={out_dir}"
    )


if __name__ == "__main__":
    main()
