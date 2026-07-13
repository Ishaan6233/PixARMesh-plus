"""Shared contracts for MV layout-loss evidence scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

DEFAULT_MESH_DATASET = (
    "datasets/mesh_datasets/datasets/"
    "3d-front-trellis2-slat-mv-da3-aug-srcperturb-r5-qfcat-obj015-light-bgtex-20260629"
)
DEFAULT_HF_DATASET = "datasets/3d-front-multiview-full"
DEFAULT_MV_FEATURE_CACHE = (
    "${MV_FEATURE_CACHE:-datasets/mv-feature-cache/da3/trellis2-mv}"
)

STAGE1_TRAIN_ROOT = "outputs/da3/train/mv_layout_loss"
STAGE2_TRAIN_ROOT = "outputs/da3/train/mv_layout_loss_stage2"
STAGE2_INFER_ROOT = "outputs/da3/infer/mv_layout_loss_stage2"
STAGE2_EVAL_ROOT = "outputs/da3/eval/stage2_best"
LAYOUT_EVAL_ROOT = "outputs/da3/eval/layout_mv"

ABLATIONS = {
    "A_ce": "mv_layout_loss_ce",
    "B_ordinal": "mv_layout_loss_ordinal",
    "C_coord": "mv_layout_loss_coord",
}
DEFAULT_LAYOUT_RUNS = tuple(ABLATIONS)

NEGATIVE_CONTROLS = {
    "no_aabb": ["dataset.model.mv_obj_aabb_token=false"],
    "no_voxel_encoder": ["dataset.model.mv_use_voxel_encoder=false"],
    "no_obj_pc_cond": ["dataset.model.mv_obj_pc_cond=false"],
    "no_obj_pc_appearance": ["dataset.model.mv_obj_pc_appearance=false"],
    "one_view_eval": ["--view-limit", "1"],
    "two_view_eval": ["--view-limit", "2"],
    "four_view_eval": ["--view-limit", "4"],
    "eight_view_eval": ["--view-limit", "8"],
    "reference_only_eval": ["--reference-only"],
    "shuffled_views_eval": ["--shuffle-views"],
}

UID_KEYS = ("uid", "image_id", "scene_id", "sha256", "model_id")
CATEGORY_KEYS = (
    "category",
    "object_category",
    "model_category",
    "semantic_category",
    "class",
    "label",
    "synset",
    "category_id",
)

REQUIRED_VERIFIER_TERMS = {
    "loss_verifier.md": (
        "src/models/loss.py",
        "src/models/edgerunner.py",
        "gradient",
        "token",
    ),
    "data_verifier.md": (
        "src/data/trellis2_mv.py",
        "src/data/collator.py",
        "view mask",
        "leakage",
    ),
    "experiment_verifier.md": (
        "summary.json",
        "per_sample.jsonl",
        "eval_obj_results.jsonl",
        "paired",
    ),
    "visual_verifier.md": (
        "gallery_manifest.json",
        "conditioning.json",
        "projection",
        "failure",
    ),
}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name, Path(path)


def layout_seed_dir(layout_root: Path, run: str, seed: int) -> Path:
    return layout_root / run / f"seed{seed}"


def is_downstream_summary(record: dict[str, Any]) -> bool:
    return "avg_cd" in record or "avg_f_score" in record or "num_evaluated" in record


def downstream_object_key(record: dict[str, Any]) -> str | None:
    if "uid" not in record:
        return None
    if "obj_id" in record:
        return f"{record['uid']}::{record['obj_id']}"
    return str(record["uid"])


def downstream_seed_label(path: Path) -> str:
    for part in reversed(path.parts):
        if part.startswith("seed"):
            return part
    return path.parent.name


def stage1_output_dir(run_name: str, *, root: str | Path = STAGE1_TRAIN_ROOT) -> str:
    return (Path(root) / run_name).as_posix()


def stage2_output_dir(run_name: str, *, root: str | Path = STAGE2_TRAIN_ROOT) -> str:
    return (Path(root) / run_name).as_posix()


def checkpoint_dir(root: str | Path, run_name: str, seed: int | str) -> Path:
    return Path(root) / run_name / f"seed{seed}" / "checkpoints"


def stage1_checkpoint_dir(
    run_name: str,
    seed: int | str,
    *,
    root: str | Path = STAGE1_TRAIN_ROOT,
) -> Path:
    return checkpoint_dir(root, run_name, seed)


def stage1_checkpoint(
    run_name: str,
    seed: int | str,
    *,
    root: str | Path = STAGE1_TRAIN_ROOT,
) -> str:
    return (stage1_checkpoint_dir(run_name, seed, root=root) / "final").as_posix()


def stage2_run_name(seed: int | str) -> str:
    return f"E_stage2_best_seed{seed}"


def stage2_checkpoint_dir(
    run_name: str,
    seed: int | str,
    *,
    root: str | Path = STAGE2_TRAIN_ROOT,
) -> Path:
    return checkpoint_dir(root, run_name, seed)


def stage2_checkpoint(
    run_name: str,
    seed: int | str,
    *,
    root: str | Path = STAGE2_TRAIN_ROOT,
) -> str:
    return (stage2_checkpoint_dir(run_name, seed, root=root) / "final").as_posix()


def stage2_infer_root(
    run_name: str, seed: int | str, *, root: str | Path = STAGE2_INFER_ROOT
) -> str:
    return (Path(root) / run_name / f"seed{seed}").as_posix()


def stage2_pred_dir(run_name: str, seed: int | str) -> str:
    return f"{stage2_infer_root(run_name, seed)}/obj/edgerunner/gt_layout_gt_mask_pred_depth"


def stage2_eval_dir(seed: int | str, *, root: str | Path = STAGE2_EVAL_ROOT) -> str:
    return (Path(root) / f"seed{seed}").as_posix()


def stage2_eval_results(seed: int | str, *, root: str | Path = STAGE2_EVAL_ROOT) -> str:
    return f"{stage2_eval_dir(seed, root=root)}/eval_obj_results.jsonl"
