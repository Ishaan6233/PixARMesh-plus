#!/usr/bin/env python3
"""Evaluate predicted stage-1 layout boxes for DA3 Trellis2-MV checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.collator import get_mesh_data_collator
from src.data.mesh import get_mesh_dataset
from src.data.utils import dequantize_points
from src.models.mv_voxel_encoder import _project_to_views
from src.models.utils import (
    get_condition_encoder,
    get_da3_encoder,
    get_image_condition_encoder,
    get_model,
)
from src.utils.config import DataConfig, ModelConfig, mv_prefix_len

LAYOUT_COORD_TOKENS = 24
LAYOUT_SENTINEL_BIN = -1
LAYOUT_TERMINAL_TOKEN_ID = 2


@dataclass(frozen=True)
class LayoutDecodeResult:
    boxes: np.ndarray
    raw_bins: np.ndarray
    token_valid: np.ndarray
    structural_valid: np.ndarray
    sample_valid: np.ndarray
    invalid_reasons: list[str | None]
    token_counts: np.ndarray
    tokens: list[np.ndarray]
    scored_tokens: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Stage-1 checkpoint or final dir.")
    parser.add_argument("--config-name", default="edgerunner_3d_front_trellis2_mv_stage1")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--out", default="outputs/da3/eval/layout_mv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mv-feature-cache", default="")
    parser.add_argument("--view-limit", type=int, default=0, help="Keep only the first N valid views.")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--shuffle-views", action="store_true")
    parser.add_argument("--max-visuals", type=int, default=24)
    parser.add_argument(
        "--visual-selection",
        choices=["first", "all", "best", "worst", "failures"],
        default="first",
        help="Which validation cases to save as visual evidence.",
    )
    parser.add_argument(
        "--visual-uids",
        default="",
        help="Optional newline or JSON list of fixed UIDs to visualize.",
    )
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override, repeatable. Use for architecture-changing controls.",
    )
    return parser.parse_args()


def _filter_dataclass_kwargs(dataclass_type, values: dict[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(dataclass_type)}
    return {key: value for key, value in values.items() if key in allowed}


def _to_container(node):
    return OmegaConf.to_container(node, resolve=True)


def compose_configs(args: argparse.Namespace) -> tuple[DataConfig, ModelConfig]:
    os.environ.setdefault("RUN_TS", "eval-layout-mv")
    OmegaConf.register_new_resolver("sub", lambda x, y: x - y, replace=True)
    repo_root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(
        config_dir=str(repo_root / "configs"),
        version_base=None,
    ):
        cfg = compose(config_name=args.config_name, overrides=args.override)
    OmegaConf.resolve(cfg)
    data_values = _to_container(cfg.dataset.src_data)
    if args.mv_feature_cache:
        data_values["mv_feature_cache"] = args.mv_feature_cache
    model_values = _to_container(cfg.model)
    ds_model = OmegaConf.select(cfg, "dataset.model")
    if ds_model is not None:
        model_values.update(_to_container(ds_model))
    model_values["local_path"] = args.checkpoint
    data_cfg = DataConfig(**_filter_dataclass_kwargs(DataConfig, data_values))
    model_cfg = ModelConfig(**_filter_dataclass_kwargs(ModelConfig, model_values))
    if getattr(model_cfg, "mv_voxel_encoder", False) or model_cfg.mv_obj_pc_cond:
        model_cfg.prefix_len = mv_prefix_len(model_cfg)
    return data_cfg, model_cfg


def build_model(model_cfg: ModelConfig, data_cfg: DataConfig, device: torch.device):
    cond_encoder_img = get_image_condition_encoder(model_cfg) if model_cfg.img_cond else None
    cond_encoder = (
        get_condition_encoder(
            model_cfg.local_cond_path,
            model_cfg,
            cond_encoder_img=cond_encoder_img,
        )
        if model_cfg.cond
        else None
    )
    geo_encoder = None
    if not getattr(data_cfg, "mv_feature_cache", "") and (
        getattr(model_cfg, "use_da3", False) or getattr(model_cfg, "geo_encoder_type", "") == "da3"
    ):
        geo_encoder = get_da3_encoder(model_cfg)
    model = get_model(
        model_cfg.local_path,
        model_cfg,
        cond_encoder=cond_encoder,
        cond_encoder_img=cond_encoder_img,
        geo_encoder=geo_encoder,
    )
    return model.to(device).eval()


def _layout_token_rows(tokens: np.ndarray) -> list[np.ndarray]:
    try:
        arr = np.asarray(tokens)
    except ValueError:
        arr = np.asarray(tokens, dtype=object)
    if arr.dtype == object:
        values = arr.tolist()
        if arr.ndim == 1 and not any(isinstance(value, (list, tuple, np.ndarray)) for value in values):
            return [np.asarray(values, dtype=np.int64).reshape(-1)]
        return [np.asarray(value, dtype=np.int64).reshape(-1) for value in values]
    if arr.ndim == 1:
        return [arr.astype(np.int64, copy=False).reshape(-1)]
    if arr.ndim == 2:
        return [arr[row].astype(np.int64, copy=False).reshape(-1) for row in range(arr.shape[0])]
    raise ValueError(f"layout tokens must be a 1D or 2D array, got shape {arr.shape}")


def _structural_status(
    row: np.ndarray, terminal_token_id: int | None
) -> tuple[bool, str | None]:
    token_count = int(row.size)
    if token_count == LAYOUT_COORD_TOKENS:
        return True, None
    if (
        token_count == LAYOUT_COORD_TOKENS + 1
        and terminal_token_id is not None
        and int(row[-1]) == int(terminal_token_id)
    ):
        return True, None
    if token_count == 0:
        return False, "empty"
    if token_count == 1:
        return False, "short"
    if token_count < LAYOUT_COORD_TOKENS:
        return False, "truncated" if token_count % 3 else "short"
    return False, "overlong"


def decode_layout_tokens(
    tokens: np.ndarray,
    pos_token_offset: int,
    num_pos_tokens: int,
    terminal_token_id: int | None = LAYOUT_TERMINAL_TOKEN_ID,
) -> LayoutDecodeResult:
    rows = _layout_token_rows(tokens)
    n = len(rows)
    scored_tokens = np.full(
        (n, LAYOUT_COORD_TOKENS),
        int(pos_token_offset) + LAYOUT_SENTINEL_BIN,
        dtype=np.int64,
    )
    token_counts = np.asarray([row.size for row in rows], dtype=np.int64)
    structural_valid = np.zeros(n, dtype=bool)
    invalid_reasons: list[str | None] = []
    for idx, row in enumerate(rows):
        token_count = min(row.size, LAYOUT_COORD_TOKENS)
        if token_count:
            scored_tokens[idx, :token_count] = row[:token_count]
        structural_valid[idx], reason = _structural_status(row, terminal_token_id)
        invalid_reasons.append(reason)

    raw_bins = scored_tokens - int(pos_token_offset)
    token_valid = (raw_bins >= 0) & (raw_bins < int(num_pos_tokens))
    sample_valid = structural_valid & token_valid.all(axis=1)
    for idx, reason in enumerate(invalid_reasons):
        if reason is None and not bool(token_valid[idx].all()):
            invalid_reasons[idx] = "invalid_position_token"

    clipped_bins = np.clip(raw_bins, 0, int(num_pos_tokens) - 1)
    boxes = dequantize_points(clipped_bins.reshape(-1, 8, 3), int(num_pos_tokens))
    return LayoutDecodeResult(
        boxes=boxes.astype(np.float32),
        raw_bins=raw_bins.astype(np.int64),
        token_valid=token_valid,
        structural_valid=structural_valid,
        sample_valid=sample_valid,
        invalid_reasons=invalid_reasons,
        token_counts=token_counts,
        tokens=rows,
        scored_tokens=scored_tokens,
    )


def bbox_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    pred_valid: np.ndarray,
    gt_bins: np.ndarray,
    pred_bins: np.ndarray,
    gt_tokens: np.ndarray,
    pred_tokens: np.ndarray,
    structural_valid: bool = True,
) -> dict:
    pred_min, pred_max = pred.min(axis=0), pred.max(axis=0)
    gt_min, gt_max = gt.min(axis=0), gt.max(axis=0)
    inter_min = np.maximum(pred_min, gt_min)
    inter_max = np.minimum(pred_max, gt_max)
    inter = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(np.prod(inter))
    pred_vol = float(np.prod(np.maximum(pred_max - pred_min, 0.0)))
    gt_vol = float(np.prod(np.maximum(gt_max - gt_min, 0.0)))
    union = pred_vol + gt_vol - inter_vol
    center_error = float(np.linalg.norm(pred.mean(axis=0) - gt.mean(axis=0)))
    pred_size = np.maximum(pred_max - pred_min, 1e-6)
    gt_size = np.maximum(gt_max - gt_min, 1e-6)
    return {
        "token_accuracy": float(
            (pred_tokens[:LAYOUT_COORD_TOKENS] == gt_tokens[:LAYOUT_COORD_TOKENS]).mean()
        ),
        "valid_token_frac": float(pred_valid[:LAYOUT_COORD_TOKENS].mean())
        if structural_valid
        else 0.0,
        "bin_mae": float(
            np.abs(pred_bins[:LAYOUT_COORD_TOKENS] - gt_bins[:LAYOUT_COORD_TOKENS]).mean()
        ),
        "corner_l1": float(np.abs(pred - gt).mean()),
        "corner_l2": float(np.linalg.norm(pred - gt, axis=-1).mean()),
        "center_error": center_error,
        "size_rel_error": float(np.abs(pred_size - gt_size).mean() / np.maximum(gt_size.mean(), 1e-6)),
        "aabb_iou": float(inter_vol / union) if union > 1e-9 else 0.0,
    }


def summarize(records: list[dict]) -> dict:
    metric_keys = [
        "token_accuracy",
        "valid_token_frac",
        "bin_mae",
        "corner_l1",
        "corner_l2",
        "center_error",
        "size_rel_error",
        "aabb_iou",
    ]
    summary = {}
    for key in metric_keys:
        vals = np.asarray([r[key] for r in records], dtype=np.float64)
        summary[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "p05": float(np.percentile(vals, 5)),
            "p50": float(np.percentile(vals, 50)),
            "p95": float(np.percentile(vals, 95)),
        }
    return summary


def summarize_validity(records: list[dict]) -> dict[str, Any]:
    valid_count = sum(1 for record in records if bool(record.get("valid")))
    invalid_reasons = Counter(
        str(record.get("invalid_reason") or "unknown")
        for record in records
        if not bool(record.get("valid"))
    )
    total = len(records)
    return {
        "num_valid": int(valid_count),
        "num_invalid": int(total - valid_count),
        "valid_sample_frac": float(valid_count / total) if total else None,
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
    }


def _view_counts(mask: torch.Tensor) -> list[int]:
    return [int(value) for value in mask.detach().to(dtype=torch.int64).sum(dim=1).cpu().tolist()]


def _view_fraction(enabled_count: int | None, num_slots: int | None) -> float | None:
    if enabled_count is None or not num_slots:
        return None
    return float(enabled_count) / float(num_slots)


def row_view_usage(ablation_meta: dict[str, Any], row: int) -> dict[str, Any]:
    """Extract per-object view usage after view-limit/reference/shuffle controls."""

    def row_value(key: str) -> Any:
        values = ablation_meta.get(key)
        if isinstance(values, list) and row < len(values):
            return values[row]
        return None

    enabled = row_value("enabled_view_counts")
    num_slots = ablation_meta.get("num_view_slots")
    return {
        "input_valid_view_count": row_value("input_valid_view_counts"),
        "enabled_view_count": enabled,
        "num_view_slots": num_slots,
        "enabled_view_fraction": _view_fraction(enabled, num_slots),
        "ref_view": row_value("ref_views"),
    }


def summarize_view_usage(records: list[dict]) -> dict[str, Any]:
    usages = [record.get("view_usage") or {} for record in records]
    enabled_counts = [
        int(usage["enabled_view_count"])
        for usage in usages
        if usage.get("enabled_view_count") is not None
    ]
    input_counts = [
        int(usage["input_valid_view_count"])
        for usage in usages
        if usage.get("input_valid_view_count") is not None
    ]
    enabled_fracs = [
        float(usage["enabled_view_fraction"])
        for usage in usages
        if usage.get("enabled_view_fraction") is not None
    ]
    if not enabled_counts:
        return {
            "num_records_with_view_usage": 0,
            "mean_input_valid_view_count": None,
            "mean_enabled_view_count": None,
            "min_enabled_view_count": None,
            "max_enabled_view_count": None,
            "multi_view_record_frac": None,
            "single_view_record_frac": None,
            "zero_view_record_frac": None,
            "mean_enabled_view_fraction": None,
        }
    enabled_arr = np.asarray(enabled_counts, dtype=np.float64)
    input_arr = np.asarray(input_counts, dtype=np.float64) if input_counts else enabled_arr
    return {
        "num_records_with_view_usage": int(len(enabled_counts)),
        "mean_input_valid_view_count": float(input_arr.mean()),
        "mean_enabled_view_count": float(enabled_arr.mean()),
        "min_enabled_view_count": int(enabled_arr.min()),
        "max_enabled_view_count": int(enabled_arr.max()),
        "multi_view_record_frac": float((enabled_arr > 1).mean()),
        "single_view_record_frac": float((enabled_arr == 1).mean()),
        "zero_view_record_frac": float((enabled_arr == 0).mean()),
        "mean_enabled_view_fraction": float(np.asarray(enabled_fracs, dtype=np.float64).mean())
        if enabled_fracs
        else None,
    }


def apply_view_ablation(batch: dict, args: argparse.Namespace) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "view_limit": int(getattr(args, "view_limit", 0) or 0),
        "reference_only": bool(getattr(args, "reference_only", False)),
        "shuffle_views": bool(getattr(args, "shuffle_views", False)),
        "shuffle_perm": None,
        "shuffle_semantics": None,
        "num_view_slots": None,
        "input_valid_view_counts": None,
        "enabled_view_counts": None,
        "ref_views": None,
    }
    if "view_mask" not in batch:
        return meta
    view_mask = batch["view_mask"]
    meta["num_view_slots"] = int(view_mask.shape[1])
    meta["input_valid_view_counts"] = _view_counts(view_mask)
    if "ref_view" in batch:
        meta["ref_views"] = [int(value) for value in batch["ref_view"].detach().cpu().tolist()]
    if args.reference_only:
        new_mask = torch.zeros_like(view_mask)
        ref = batch.get("ref_view", torch.zeros(view_mask.shape[0], dtype=torch.long))
        for b in range(view_mask.shape[0]):
            r = int(ref[b].item())
            if r < 0 or r >= view_mask.shape[1] or not bool(view_mask[b, r]):
                raise ValueError(
                    f"reference-only control requires ref_view to be valid; row {b} has ref_view={r}."
                )
        new_mask.scatter_(1, ref.view(-1, 1).to(new_mask.device), True)
        batch["view_mask"] = new_mask
    elif args.view_limit > 0:
        for b in range(view_mask.shape[0]):
            valid = torch.nonzero(view_mask[b], as_tuple=False).flatten()
            if valid.numel() > args.view_limit:
                view_mask[b, valid[args.view_limit :]] = False
    if args.shuffle_views:
        perm = torch.arange(batch["view_mask"].shape[1] - 1, -1, -1)
        meta["shuffle_perm"] = perm.detach().cpu().tolist()
        meta["shuffle_semantics"] = (
            "per-view observations/features/masks are permuted while cameras, "
            "view_mask, ref_view, and view_indices remain fixed"
        )
        # Deliberately break view-to-camera alignment: shuffle per-view observations
        # while leaving cameras, validity masks, and ref_view indices fixed.
        for key in (
            "pixel_values",
            "panoptic_masks",
            "cached_local_points",
            "cached_conf",
            "cached_dino_feats",
        ):
            if key in batch:
                batch[key] = batch[key][:, perm]
    meta["enabled_view_counts"] = _view_counts(batch["view_mask"])
    return meta


def filter_batch_rows(batch: dict[str, Any], keep: torch.Tensor) -> dict[str, Any]:
    """Keep only valid examples in a collated batch without disturbing per-view dims."""
    keep = keep.to(dtype=torch.bool)
    n = int(keep.numel())
    keep_cpu = keep.detach().cpu()
    filtered = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == (n,):
            filtered[key] = value[keep.to(value.device)]
        elif isinstance(value, list) and len(value) == n:
            filtered[key] = [item for item, ok in zip(value, keep_cpu.tolist()) if ok]
        else:
            filtered[key] = value
    return filtered


def _as_uid(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value:
        return _as_uid(value[0])
    return str(value)


def _load_visual_uids(path: str) -> set[str]:
    if not path:
        return set()
    text = Path(path).read_text().strip()
    if not text:
        return set()
    if text.startswith("["):
        return {str(uid) for uid in json.loads(text)}
    return {line.strip() for line in text.splitlines() if line.strip()}


def make_eval_collate(collator):
    def _collate(examples: list[dict]) -> dict:
        batch = collator(examples)
        batch["uid"] = [_as_uid(ex.get("uid", i)) for i, ex in enumerate(examples)]
        if "view_indices" in examples[0]:
            batch["view_indices"] = torch.as_tensor(
                np.stack([np.asarray(ex["view_indices"]) for ex in examples]),
                dtype=torch.long,
            )
        if "point_clouds_valid" in examples[0]:
            batch["point_clouds_valid"] = torch.as_tensor(
                [bool(ex["point_clouds_valid"]) for ex in examples],
                dtype=torch.bool,
            )
        return batch

    return _collate


def edge_pairs() -> list[tuple[int, int]]:
    return [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def _tensor_row_np(diagnostics: dict[str, Any], key: str, row: int) -> np.ndarray | None:
    value = diagnostics.get(key)
    if value is None:
        return None
    return value[row].detach().float().cpu().numpy()


def _tensor_row_list(diagnostics: dict[str, Any], key: str, row: int) -> list | None:
    value = diagnostics.get(key)
    if value is None:
        return None
    return value[row].detach().cpu().tolist()


def _slice_tensor_row(value: torch.Tensor, row: int) -> torch.Tensor:
    return value[row : row + 1].detach().cpu()


def capture_visual_payload(
    uid: str,
    pred: np.ndarray,
    gt: np.ndarray,
    batch: dict,
    row: int,
    diagnostics: dict[str, Any],
    record: dict[str, Any],
    ablation_meta: dict[str, Any],
) -> dict[str, Any]:
    batch_keys = (
        "cond_pcs",
        "cond_pcs_2d",
        "scene_transforms",
        "K_per_view",
        "view_mask",
        "panoptic_masks",
        "view_indices",
    )
    diag_keys = (
        "obj_voxels",
        "ctx_voxels",
        "obj_geom_voxels",
        "seed_pcs",
        "seed_pcs_2d",
        "view_mask",
        "obj_view_mask",
        "ref_view",
        "obj_aabb",
        "mv_target_ids",
        "view_conf_mean",
    )
    return {
        "uid": uid,
        "pred": pred.copy(),
        "gt": gt.copy(),
        "record": dict(record),
        "ablation_meta": dict(ablation_meta),
        "batch": {
            key: _slice_tensor_row(value, row)
            for key, value in batch.items()
            if key in batch_keys and torch.is_tensor(value)
        },
        "diagnostics": {
            key: _slice_tensor_row(value, row)
            for key, value in diagnostics.items()
            if key in diag_keys and torch.is_tensor(value)
        },
    }


def visual_rank(record: dict[str, Any], mode: str) -> tuple:
    if mode == "best":
        return (-float(record["aabb_iou"]), float(record["bin_mae"]), float(record["center_error"]))
    if mode == "failures":
        return (
            float(record["valid_token_frac"]) >= 1.0,
            float(record["aabb_iou"]),
            -float(record["bin_mae"]),
        )
    return (float(record["aabb_iou"]), -float(record["bin_mae"]), -float(record["center_error"]))


def _visual_limit(max_visuals: int, n: int) -> int:
    return n if max_visuals <= 0 else min(max_visuals, n)


def select_visual_payloads(
    payloads: list[dict[str, Any]],
    *,
    mode: str,
    max_visuals: int,
    fixed_uids: set[str],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if fixed_uids:
        selected = [payload for payload in payloads if payload["uid"] in fixed_uids]
        mode = "fixed_uids"
    elif mode == "all":
        selected = payloads
    elif mode == "first":
        selected = payloads[: _visual_limit(max_visuals, len(payloads))]
    else:
        selected = sorted(payloads, key=lambda payload: visual_rank(payload["record"], mode))
        selected = selected[: _visual_limit(max_visuals, len(selected))]

    out = []
    selected_uids = [payload["uid"] for payload in selected]
    for rank, payload in enumerate(selected):
        out.append(
            (
                payload,
                {
                    "mode": mode,
                    "rank": rank,
                    "max_visuals": int(max_visuals),
                    "fixed_uids": sorted(fixed_uids),
                    "selected_uids": selected_uids,
                    "total_candidates": len(payloads),
                },
            )
        )
    return out


def save_visuals(
    out_dir: Path,
    uid: str,
    pred: np.ndarray,
    gt: np.ndarray,
    batch: dict,
    row: int,
    diagnostics: dict[str, Any],
    *,
    record: dict[str, Any],
    selection_meta: dict[str, Any],
    ablation_meta: dict[str, Any],
) -> None:
    case_dir = out_dir / "visuals" / uid.replace("/", "_")
    case_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    for box, color, label in ((gt, "tab:green", "gt"), (pred, "tab:red", "pred")):
        xy = box[:, [0, 2]]
        for i, j in edge_pairs():
            ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color=color, linewidth=1.5)
        ax.scatter(xy[:, 0], xy[:, 1], color=color, s=8, label=label)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.set_title(uid)
    fig.tight_layout()
    fig.savefig(case_dir / "topdown_bbox.png", dpi=140)
    plt.close(fig)

    obj_voxels = _tensor_row_np(diagnostics, "obj_voxels", row)
    ctx_voxels = _tensor_row_np(diagnostics, "ctx_voxels", row)
    obj_geom_voxels = _tensor_row_np(diagnostics, "obj_geom_voxels", row)
    seed_pcs = _tensor_row_np(diagnostics, "seed_pcs", row)
    seed_pcs_2d = _tensor_row_np(diagnostics, "seed_pcs_2d", row)
    cond_pcs = batch.get("cond_pcs")
    cond_np = cond_pcs[row].detach().float().cpu().numpy() if torch.is_tensor(cond_pcs) else None
    if obj_voxels is not None or ctx_voxels is not None or cond_np is not None:
        fig, ax = plt.subplots(figsize=(7, 6))
        if ctx_voxels is not None:
            ax.scatter(ctx_voxels[:, 0], ctx_voxels[:, 2], s=2, alpha=0.18, label="ctx voxels")
        if obj_voxels is not None:
            ax.scatter(obj_voxels[:, 0], obj_voxels[:, 2], s=4, alpha=0.6, label="obj voxels")
        if cond_np is not None:
            ax.scatter(cond_np[:, 0], cond_np[:, 2], s=3, alpha=0.3, label="seed obj pc")
        for box, color, label in ((gt, "tab:green", "gt bbox"), (pred, "tab:red", "pred bbox")):
            xy = box[:, [0, 2]]
            for i, j in edge_pairs():
                ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color=color, linewidth=1.2)
            ax.scatter(xy[:, 0], xy[:, 1], color=color, s=8, label=label)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(case_dir / "topdown_voxels_bbox.png", dpi=140)
        plt.close(fig)

    if obj_geom_voxels is not None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(obj_geom_voxels[:, 0], obj_geom_voxels[:, 2], s=4, alpha=0.7)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{uid} canonical obj voxels")
        fig.tight_layout()
        fig.savefig(case_dir / "canonical_obj_voxels.png", dpi=140)
        plt.close(fig)

    view_mask = _tensor_row_np(diagnostics, "view_mask", row)
    obj_view_mask = _tensor_row_np(diagnostics, "obj_view_mask", row)
    ref_view = _tensor_row_np(diagnostics, "ref_view", row)
    obj_aabb = _tensor_row_np(diagnostics, "obj_aabb", row)
    view_indices = batch.get("view_indices")
    meta = {
        "uid": uid,
        "metrics": {
            key: record[key]
            for key in (
                "token_accuracy",
                "valid_token_frac",
                "bin_mae",
                "corner_l1",
                "corner_l2",
                "center_error",
                "size_rel_error",
                "aabb_iou",
            )
            if key in record
        },
        "selection": selection_meta,
        "ablation": ablation_meta,
        "view_mask": view_mask.astype(bool).tolist() if view_mask is not None else None,
        "obj_view_mask": obj_view_mask.astype(bool).tolist() if obj_view_mask is not None else None,
        "ref_view": int(ref_view.item()) if ref_view is not None else None,
        "view_indices": view_indices[row].detach().cpu().tolist() if torch.is_tensor(view_indices) else None,
        "obj_aabb": obj_aabb.tolist() if obj_aabb is not None else None,
        "mv_target_ids": _tensor_row_list(diagnostics, "mv_target_ids", row),
        "view_conf_mean": _tensor_row_list(diagnostics, "view_conf_mean", row),
        "projection": None,
    }

    if obj_voxels is not None or ctx_voxels is not None or obj_geom_voxels is not None or seed_pcs is not None:
        np.savez_compressed(
            case_dir / "conditioning_points.npz",
            obj_voxels=obj_voxels if obj_voxels is not None else np.empty((0, 3), dtype=np.float32),
            ctx_voxels=ctx_voxels if ctx_voxels is not None else np.empty((0, 3), dtype=np.float32),
            obj_geom_voxels=obj_geom_voxels if obj_geom_voxels is not None else np.empty((0, 3), dtype=np.float32),
            seed_pcs=seed_pcs if seed_pcs is not None else np.empty((0, 3), dtype=np.float32),
            seed_pcs_2d=seed_pcs_2d if seed_pcs_2d is not None else np.empty((0, 2), dtype=np.float32),
            cond_pcs=cond_np if cond_np is not None else np.empty((0, 3), dtype=np.float32),
        )

    if "panoptic_masks" not in batch:
        (case_dir / "conditioning.json").write_text(json.dumps(meta, indent=2))
        return
    st = batch["scene_transforms"][row : row + 1].float()
    K = batch["K_per_view"][row : row + 1].float()
    masks = batch["panoptic_masks"][row].detach().cpu().numpy()
    view_mask = batch["view_mask"][row].detach().cpu().numpy().astype(bool)
    H, W = masks.shape[-2:]
    boxes = torch.tensor(np.stack([gt, pred], axis=0), dtype=torch.float32, device=st.device)
    pix, voxels_cam = _project_to_views(boxes, st.repeat(2, 1, 1, 1), K.repeat(2, 1, 1, 1), H, W)
    pix = pix.detach().cpu().numpy()
    voxels_cam_np = voxels_cam.detach().cpu().numpy()
    projection_summary = []
    for view_idx, ok in enumerate(view_mask.tolist()):
        gt_uv = pix[0, :, view_idx]
        pred_uv = pix[1, :, view_idx]
        gt_in_bounds = np.isfinite(gt_uv).all(axis=-1) & (np.abs(gt_uv) <= 1.0).all(axis=-1)
        pred_in_bounds = np.isfinite(pred_uv).all(axis=-1) & (np.abs(pred_uv) <= 1.0).all(axis=-1)
        projection_summary.append(
            {
                "view_idx": int(view_idx),
                "view_enabled": bool(ok),
                "gt_in_bounds_corners": int(gt_in_bounds.sum()),
                "pred_in_bounds_corners": int(pred_in_bounds.sum()),
                "gt_positive_depth_corners": int((voxels_cam_np[0, :, view_idx, 2] > 0).sum()),
                "pred_positive_depth_corners": int((voxels_cam_np[1, :, view_idx, 2] > 0).sum()),
            }
        )
        if not ok:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.imshow(masks[view_idx], cmap="gray")
        for box_idx, color in ((0, "lime"), (1, "red")):
            uv = pix[box_idx, :, view_idx]
            u = (uv[:, 0] + 1.0) * 0.5 * W - 0.5
            v = (uv[:, 1] + 1.0) * 0.5 * H - 0.5
            for i, j in edge_pairs():
                ax.plot([u[i], u[j]], [v[i], v[j]], color=color, linewidth=1.2)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(case_dir / f"view{view_idx:02d}_projection.png", dpi=140)
        plt.close(fig)
    meta["projection"] = {
        "height": int(H),
        "width": int(W),
        "edge_pairs": edge_pairs(),
        "per_view": projection_summary,
    }
    (case_dir / "conditioning.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data_cfg, model_cfg = compose_configs(args)
    model = build_model(model_cfg, data_cfg, device)
    train_set, val_set, _ = get_mesh_dataset(data_cfg)
    dataset = train_set if args.split == "train" else val_set
    if args.num_samples > 0 and args.num_samples < len(dataset):
        dataset = Subset(dataset, list(range(args.num_samples)))
    collator = get_mesh_data_collator(data_cfg, model_cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=make_eval_collate(collator),
    )

    per_sample = out_dir / "per_sample.jsonl"
    per_sample.write_text("")
    records = []
    visual_payloads: list[dict[str, Any]] = []
    fixed_visual_uids = _load_visual_uids(args.visual_uids)
    skipped_invalid = 0
    for batch in loader:
        if "point_clouds_valid" in batch:
            valid_rows = batch["point_clouds_valid"].to(dtype=torch.bool)
            skipped_invalid += int((~valid_rows).sum().item())
            if not bool(valid_rows.any()):
                continue
            if not bool(valid_rows.all()):
                batch = filter_batch_rows(batch, valid_rows)
        ablation_meta = apply_view_ablation(batch, args)
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        input_ids = batch["input_ids"]
        prefix_len = int(model_cfg.prefix_len)
        prefix_ids = input_ids[:, : prefix_len + 1]
        gt_tokens = input_ids[:, prefix_len + 1 : prefix_len + 25].detach().cpu().numpy()
        with torch.no_grad():
            embeds, diagnostics = model.get_mv_inputs_with_cond(
                input_ids=prefix_ids,
                pixel_values=batch["pixel_values"],
                scene_transforms=batch["scene_transforms"],
                K_per_view=batch["K_per_view"],
                view_mask=batch["view_mask"],
                panoptic_masks=batch.get("panoptic_masks"),
                cond_pcs=batch["cond_pcs"],
                cond_pcs_2d=batch["cond_pcs_2d"],
                cond_num_faces=batch.get("cond_num_faces"),
                cached_local_points=batch.get("cached_local_points"),
                cached_conf=batch.get("cached_conf"),
                cached_dino_feats=batch.get("cached_dino_feats"),
                obj_canon_transform=batch.get("obj_canon_transform"),
                gt_obj_vertices=batch.get("gt_obj_vertices"),
                ref_view=batch.get("ref_view"),
                return_diagnostics=True,
            )
            generated = model.generate(
                inputs_embeds=embeds,
                max_new_tokens=LAYOUT_COORD_TOKENS + 1,
                use_cache=True,
                do_sample=False,
            ).detach().cpu().numpy()
        pred_decode = decode_layout_tokens(
            generated,
            model_cfg.pos_token_offset,
            model_cfg.num_pos_tokens,
            model_cfg.eos_token_id,
        )
        gt_decode = decode_layout_tokens(
            gt_tokens,
            model_cfg.pos_token_offset,
            model_cfg.num_pos_tokens,
            model_cfg.eos_token_id,
        )
        for i in range(len(pred_decode.tokens)):
            uid = batch["uid"][i] if "uid" in batch else f"{len(records):06d}"
            view_usage = row_view_usage(ablation_meta, i)
            rec = {
                "index": len(records),
                "uid": uid,
                "valid": bool(pred_decode.sample_valid[i]),
                "invalid_reason": pred_decode.invalid_reasons[i],
                "layout_token_count": int(pred_decode.token_counts[i]),
                "expected_layout_tokens": LAYOUT_COORD_TOKENS,
                "view_usage": view_usage,
                **view_usage,
                **bbox_metrics(
                    pred_decode.boxes[i],
                    gt_decode.boxes[i],
                    pred_decode.token_valid[i],
                    gt_decode.raw_bins[i],
                    pred_decode.raw_bins[i],
                    gt_decode.scored_tokens[i],
                    pred_decode.scored_tokens[i],
                    bool(pred_decode.structural_valid[i]),
                ),
                "pred_tokens": pred_decode.tokens[i].astype(int).tolist(),
                "pred_tokens_scored": pred_decode.scored_tokens[i].astype(int).tolist(),
                "gt_tokens": gt_tokens[i].astype(int).tolist(),
            }
            records.append(rec)
            with per_sample.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            if not args.no_visuals:
                visual_payloads.append(
                    capture_visual_payload(
                        uid,
                        pred_decode.boxes[i],
                        gt_decode.boxes[i],
                        batch,
                        i,
                        diagnostics,
                        rec,
                        ablation_meta,
                    )
                )

    saved_visuals = 0
    if not args.no_visuals:
        for payload, selection_meta in select_visual_payloads(
            visual_payloads,
            mode=args.visual_selection,
            max_visuals=args.max_visuals,
            fixed_uids=fixed_visual_uids,
        ):
            save_visuals(
                out_dir,
                payload["uid"],
                payload["pred"],
                payload["gt"],
                payload["batch"],
                0,
                payload["diagnostics"],
                record=payload["record"],
                selection_meta=selection_meta,
                ablation_meta=payload["ablation_meta"],
            )
            saved_visuals += 1

    report = {
        "checkpoint": args.checkpoint,
        "config_name": args.config_name,
        "split": args.split,
        "requested_num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "mv_feature_cache": args.mv_feature_cache,
        "device": str(device),
        "num_records": len(records),
        "view_limit": args.view_limit,
        "reference_only": args.reference_only,
        "shuffle_views": args.shuffle_views,
        "visual_selection": args.visual_selection,
        "visual_uids": sorted(fixed_visual_uids),
        "saved_visuals": saved_visuals,
        "overrides": args.override,
        "skipped_invalid": skipped_invalid,
        "validity": summarize_validity(records),
        "view_usage": summarize_view_usage(records),
        "summary": summarize(records) if records else {},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
