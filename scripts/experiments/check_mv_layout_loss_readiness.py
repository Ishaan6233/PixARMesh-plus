#!/usr/bin/env python3
"""Check prerequisites before launching MV layout-loss evidence runs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval.mv_layout_evidence_common import (
    CATEGORY_KEYS,
    checkpoint_dir,
    DEFAULT_HF_DATASET,
    DEFAULT_LAYOUT_RUNS,
    DEFAULT_MESH_DATASET,
    DEFAULT_MV_FEATURE_CACHE,
    is_downstream_summary,
    read_csv,
    read_jsonl,
    STAGE1_TRAIN_ROOT,
    stage2_run_name,
    STAGE2_TRAIN_ROOT,
    UID_KEYS,
)
from src.data.trellis2_mv import MV_FEATURE_CACHE_KEYS, MV_FEATURE_CACHE_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dataset", default=DEFAULT_MESH_DATASET)
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    parser.add_argument("--mv-feature-cache", default=DEFAULT_MV_FEATURE_CACHE)
    parser.add_argument(
        "--sv-downstream", default="outputs/sv/eval/baseline/eval_obj_results.jsonl"
    )
    parser.add_argument(
        "--uid-metadata", default="metadata/layout_uid_categories.jsonl"
    )
    parser.add_argument(
        "--require-uid-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require UID/category metadata for object-category stability gates.",
    )
    parser.add_argument(
        "--cache-precompute-script",
        default="",
        help="Script to run when the MV feature cache is missing or stale.",
    )
    parser.add_argument(
        "--run", action="append", default=[], help="Stage-1 run name. Repeatable."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--stage1-train-root", default=STAGE1_TRAIN_ROOT)
    parser.add_argument("--stage2-train-root", default=STAGE2_TRAIN_ROOT)
    parser.add_argument(
        "--require-gpu", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--allow-existing-checkpoints", action="store_true")
    parser.add_argument("--sample-cache-files", type=int, default=3)
    parser.add_argument(
        "--out",
        default="outputs/da3/experiments/mv_layout_loss_ablation/readiness.json",
    )
    return parser.parse_args()


def _literal_path(value: str) -> Path:
    """Return a usable local path for literal/default CLI values with shell syntax."""
    if value.startswith("${") and ":-" in value and value.endswith("}"):
        return Path(value.split(":-", 1)[1][:-1])
    return Path(value)


def _record_issue(report: dict[str, Any], message: str) -> None:
    report["issues"].append(message)


def _record_warning(report: dict[str, Any], message: str) -> None:
    report["warnings"].append(message)


def _record_remediation(
    report: dict[str, Any], check: str, action: str, command: str | None = None
) -> None:
    entry = {"check": check, "action": action}
    if command:
        entry["command"] = command
    if entry not in report["remediations"]:
        report["remediations"].append(entry)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _loader_split_indices(n_rows: int) -> dict[str, list[int]]:
    all_idx = list(range(n_rows))
    rng = np.random.RandomState(42)
    rng.shuffle(all_idx)
    n_val = max(1, int(0.05 * len(all_idx))) if all_idx else 0
    return {
        "train": sorted(all_idx[n_val:]),
        "val": sorted(all_idx[:n_val]),
    }


def _expected_cache_uids_by_split(
    mesh_root: Path,
) -> tuple[dict[str, list[str]] | None, list[str]]:
    metadata = mesh_root / "metadata.csv"
    cond_filter = mesh_root / "conditioning_filter.csv"
    if not metadata.exists() or not cond_filter.exists():
        return None, []
    with metadata.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, [f"metadata.csv is empty: {metadata}"]
    if "sha256" not in (rows[0].keys() if rows else []):
        return None, [f"metadata.csv has no sha256 column: {metadata}"]
    with cond_filter.open(newline="") as f:
        filter_rows = list(csv.DictReader(f))
    if not filter_rows:
        return None, [f"conditioning_filter.csv is empty: {cond_filter}"]
    if "sha256" not in (filter_rows[0].keys() if filter_rows else []) or "keep" not in filter_rows[0]:
        return None, [f"conditioning_filter.csv must contain sha256 and keep columns: {cond_filter}"]
    keep_set = {row["sha256"] for row in filter_rows if _truthy(row.get("keep"))}

    issues = []
    by_split = {"train": [], "val": []}
    for split, indices in _loader_split_indices(len(rows)).items():
        for idx in indices:
            row = rows[idx]
            sha256 = row.get("sha256", "")
            if sha256 not in keep_set:
                continue
            uid = str(row.get("uid") or "").strip()
            if not uid:
                issues.append(
                    f"metadata row for kept sha256={sha256} has no uid; cannot verify cache filename coverage"
                )
                continue
            by_split[split].append(uid)
    return by_split, issues


def check_gpu(report: dict[str, Any], require_gpu: bool) -> None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        result = subprocess.CompletedProcess(
            ["nvidia-smi"], returncode=127, stdout="", stderr=str(exc)
        )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    devices = lines if result.returncode == 0 else []
    report["checks"]["gpu"] = {
        "required": require_gpu,
        "returncode": result.returncode,
        "devices": devices,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    if result.returncode != 0 or not devices:
        message = "nvidia-smi did not report usable GPUs"
        if require_gpu:
            _record_issue(report, message)
            _record_remediation(
                report,
                "gpu",
                "Run the ablations on a machine where nvidia-smi can see the training GPUs. "
                "Use --no-require-gpu only for local dry-run validation.",
            )
        else:
            _record_warning(report, message)


def check_mesh_dataset(report: dict[str, Any], root: Path) -> None:
    metadata = root / "metadata.csv"
    cond_filter = root / "conditioning_filter.csv"
    cond_meta = root / "conditioning_filter.meta.json"
    mesh_dir = root / "mesh_dumps"
    mv_cond_dir = root / "mv_cond"
    entry: dict[str, Any] = {
        "path": str(root),
        "exists": root.exists(),
        "metadata_csv": str(metadata),
        "conditioning_filter": str(cond_filter),
        "conditioning_filter_meta": str(cond_meta),
        "mesh_dumps": str(mesh_dir),
        "mv_cond": str(mv_cond_dir),
    }
    report["checks"]["mesh_dataset"] = entry
    for path, label in (
        (root, "mesh dataset root"),
        (metadata, "metadata.csv"),
        (cond_filter, "conditioning_filter.csv"),
        (cond_meta, "conditioning_filter.meta.json"),
        (mesh_dir, "mesh_dumps"),
        (mv_cond_dir, "mv_cond"),
    ):
        if not path.exists():
            _record_issue(report, f"missing {label}: {path}")
            _record_remediation(
                report,
                "mesh_dataset",
                "Point --mesh-dataset at the Trellis2-MV mesh dataset root containing "
                "metadata.csv, conditioning_filter.csv, conditioning_filter.meta.json, "
                "mesh_dumps/, and mv_cond/.",
            )
    if metadata.exists():
        with metadata.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = [next(reader, None) for _ in range(3)]
        entry["metadata_columns"] = reader.fieldnames or []
        entry["metadata_sample_count"] = sum(row is not None for row in rows)
    if cond_meta.exists():
        try:
            entry["conditioning_filter_meta_payload"] = json.loads(
                cond_meta.read_text()
            )
        except json.JSONDecodeError as exc:
            _record_issue(
                report, f"invalid conditioning_filter.meta.json: {cond_meta}: {exc}"
            )
            _record_remediation(
                report,
                "mesh_dataset",
                "Regenerate or repair conditioning_filter.meta.json before treating the split/filtering as fixed.",
            )


def check_hf_dataset(report: dict[str, Any], root: Path) -> None:
    entry = {
        "path": str(root),
        "exists": root.exists(),
        "dataset_dict_json": str(root / "dataset_dict.json"),
        "train_state_json": str(root / "train" / "state.json"),
    }
    report["checks"]["hf_dataset"] = entry
    for path, label in (
        (root, "HF dataset root"),
        (root / "dataset_dict.json", "dataset_dict.json"),
        (root / "train" / "state.json", "train/state.json"),
    ):
        if not path.exists():
            _record_issue(report, f"missing {label}: {path}")
            _record_remediation(
                report,
                "hf_dataset",
                "Point --hf-dataset at the local multiview HF dataset root with dataset_dict.json and train/state.json.",
            )


def check_feature_cache(
    report: dict[str, Any],
    *,
    mesh_root: Path,
    root: Path,
    sample_count: int,
    precompute_script: str,
) -> None:
    entry: dict[str, Any] = {
        "path": str(root),
        "exists": root.exists(),
        "expected_cache_version": MV_FEATURE_CACHE_VERSION,
        "expected_keys": list(MV_FEATURE_CACHE_KEYS),
        "sampled": [],
        "coverage": {},
    }
    report["checks"]["mv_feature_cache"] = entry

    def record_cache_remediation() -> None:
        _record_remediation(
            report,
            "mv_feature_cache",
            "Build or rebuild the frozen DA3+DINO Trellis2-MV feature cache before launching A-D training.",
            f"bash {precompute_script}",
        )

    if not root.exists():
        _record_issue(report, f"missing MV feature cache root: {root}")
        record_cache_remediation()
        return
    files = sorted(root.glob("*.npz"))
    entry["npz_count"] = len(files)
    if not files:
        _record_issue(report, f"MV feature cache has no .npz files: {root}")
        record_cache_remediation()
        return
    required = set(MV_FEATURE_CACHE_KEYS)
    expected_by_split, coverage_input_issues = _expected_cache_uids_by_split(mesh_root)
    expected_uids: set[str] | None = None
    if expected_by_split is not None:
        expected_flat = expected_by_split["train"] + expected_by_split["val"]
        expected_uids = set(expected_flat)
        present_uids = {path.stem for path in files}
        duplicates = sorted(uid for uid, count in Counter(expected_flat).items() if count > 1)
        missing = sorted(expected_uids - present_uids)
        extra = sorted(present_uids - expected_uids)
        entry["coverage"] = {
            "expected_total": len(expected_flat),
            "expected_unique": len(expected_uids),
            "expected_by_split": {
                split: len(uids) for split, uids in expected_by_split.items()
            },
            "present_total": len(files),
            "missing_count": len(missing),
            "missing_examples": missing[:20],
            "extra_count": len(extra),
            "extra_examples": extra[:20],
            "duplicate_expected_uids": duplicates[:20],
        }
        if duplicates:
            _record_issue(
                report,
                f"conditioning-filter keep set maps to {len(duplicates)} duplicate cache UIDs; "
                "cache filenames are not 1:1",
            )
        if missing:
            _record_issue(
                report,
                f"MV feature cache is missing {len(missing)} files required by the conditioning-filter keep set; "
                f"examples: {', '.join(missing[:5])}",
            )
        if extra:
            _record_issue(
                report,
                f"MV feature cache has {len(extra)} unexpected .npz files outside the conditioning-filter keep set; "
                f"examples: {', '.join(extra[:5])}",
            )
    for message in coverage_input_issues:
        _record_issue(report, message)

    cache_has_issue = False
    if expected_uids is not None:
        files_to_check = [root / f"{uid}.npz" for uid in sorted(expected_uids) if (root / f"{uid}.npz").exists()]
    else:
        files_to_check = files[: max(sample_count, 0)]
    empty_marker_count = 0
    for idx, path in enumerate(files_to_check):
        sample: dict[str, Any] = {"path": str(path)}
        try:
            with np.load(path) as z:
                keys = set(z.files)
                missing = sorted(required - keys)
                sample["missing_keys"] = missing
                sample["cache_version"] = (
                    int(np.asarray(z["cache_version"]).item())
                    if "cache_version" in z
                    else None
                )
                if missing:
                    cache_has_issue = True
                    _record_issue(report, f"{path} missing cache keys {missing}")
                if sample["cache_version"] != MV_FEATURE_CACHE_VERSION:
                    cache_has_issue = True
                    _record_issue(
                        report,
                        f"{path} has cache_version={sample['cache_version']}, expected {MV_FEATURE_CACHE_VERSION}",
                    )
                if not missing and "view_mask" in z.files:
                    sample["empty_marker"] = not bool(np.asarray(z["view_mask"], dtype=bool).any())
                    if sample["empty_marker"]:
                        empty_marker_count += 1
        except Exception as exc:  # noqa: BLE001 - report corrupt cache files as readiness failures.
            sample["error"] = repr(exc)
            cache_has_issue = True
            _record_issue(report, f"could not read MV feature cache file {path}: {exc}")
        if idx < max(sample_count, 0):
            entry["sampled"].append(sample)
    if expected_uids is not None:
        entry["coverage"]["empty_marker_count"] = empty_marker_count
        entry["coverage"]["feature_file_count"] = len(files_to_check) - empty_marker_count
        entry["coverage"]["checked_expected_files"] = len(files_to_check)
    if cache_has_issue or coverage_input_issues or entry["coverage"].get("missing_count") or entry["coverage"].get("extra_count"):
        record_cache_remediation()


def _metadata_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    if path.suffix == ".csv":
        return read_csv(path)
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("records"), list):
            return [row for row in payload["records"] if isinstance(row, dict)]
        if all(isinstance(value, dict) for value in payload.values()):
            return [{"uid": key, **value} for key, value in payload.items()]
        return [payload]
    return []


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return None


def check_sv_downstream(report: dict[str, Any], path: Path) -> None:
    rows = read_jsonl(path)
    object_rows = [row for row in rows if not is_downstream_summary(row)]
    summary = rows[-1] if rows else None
    report["checks"]["sv_downstream"] = {
        "path": str(path),
        "exists": path.exists(),
        "rows": len(rows),
        "object_rows": len(object_rows),
        "summary": summary if summary and is_downstream_summary(summary) else None,
    }
    if not path.exists():
        _record_issue(report, f"missing SV downstream file: {path}")
        _record_remediation(
            report,
            "sv_downstream",
            "Run or point --sv-downstream at the fixed SV baseline eval_obj_results.jsonl with object rows and a summary row.",
        )
        return
    if not object_rows:
        _record_issue(report, f"SV downstream has no object-level rows: {path}")
        _record_remediation(
            report,
            "sv_downstream",
            "Regenerate the SV baseline so it contains per-object rows needed for paired downstream comparison.",
        )
    if not summary or not is_downstream_summary(summary):
        _record_issue(report, f"SV downstream has no aggregate summary row: {path}")
        _record_remediation(
            report,
            "sv_downstream",
            "Regenerate the SV baseline so the final row reports aggregate CD/F metrics.",
        )


def check_uid_metadata(report: dict[str, Any], path_value: str, required: bool) -> None:
    if not path_value:
        message = (
            "no --uid-metadata provided; category-stability gates cannot be enabled"
        )
        if required:
            _record_issue(report, message)
            _record_remediation(
                report,
                "uid_metadata",
                "Pass --uid-metadata with JSONL/CSV/JSON metadata, or use --no-require-uid-metadata "
                "only for dry-run checks that will not support category-stability claims.",
            )
        else:
            _record_warning(report, message)
        report["checks"]["uid_metadata"] = {"required": required, "path": ""}
        return
    path = Path(path_value)
    report["checks"]["uid_metadata"] = {
        "required": required,
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        message = f"missing UID category metadata: {path}"
        if required:
            _record_issue(report, message)
        else:
            _record_warning(report, message)
        _record_remediation(
            report,
            "uid_metadata",
            "Create JSONL/CSV/JSON metadata with a UID key "
            f"{list(UID_KEYS)} and a category key {list(CATEGORY_KEYS)}; "
            "do not use raw model_id as semantic category evidence unless the claim is scoped to per-model stability.",
        )
        return
    try:
        rows = _metadata_rows(path)
    except Exception as exc:  # noqa: BLE001 - readiness should surface malformed sidecars.
        message = f"could not read UID category metadata {path}: {exc}"
        if required:
            _record_issue(report, message)
        else:
            _record_warning(report, message)
        _record_remediation(
            report,
            "uid_metadata",
            "Repair the UID category metadata file so it parses as JSONL, CSV, JSON list, or JSON records mapping.",
        )
        return
    category_values = []
    missing_uid = 0
    missing_category = 0
    for row in rows:
        uid = _first_present(row, UID_KEYS)
        category = _first_present(row, CATEGORY_KEYS)
        if uid is None:
            missing_uid += 1
        if category is None:
            missing_category += 1
        else:
            category_values.append(category)
    report["checks"]["uid_metadata"].update(
        {
            "rows": len(rows),
            "category_count": len(set(category_values)),
            "missing_uid_rows": missing_uid,
            "missing_category_rows": missing_category,
            "category_keys": list(CATEGORY_KEYS),
            "uid_keys": list(UID_KEYS),
        }
    )
    metadata_issues = []
    if not rows:
        metadata_issues.append(f"UID category metadata is empty: {path}")
    if missing_uid:
        metadata_issues.append(
            f"UID category metadata has {missing_uid} rows without a UID key"
        )
    if missing_category:
        metadata_issues.append(
            f"UID category metadata has {missing_category} rows without a category key"
        )
    if len(set(category_values)) < 2:
        metadata_issues.append(
            f"UID category metadata has fewer than two categories: {path}"
        )
    for message in metadata_issues:
        if required:
            _record_issue(report, message)
        else:
            _record_warning(report, message)
    if not rows or missing_uid or missing_category or len(set(category_values)) < 2:
        _record_remediation(
            report,
            "uid_metadata",
            "Fix the metadata so every row has one accepted UID key and one accepted category key, "
            "with at least two semantic categories represented.",
        )


def check_existing_outputs(
    report: dict[str, Any],
    *,
    runs: list[str],
    seeds: list[int],
    stage1_root: Path,
    stage2_root: Path,
    allow_existing: bool,
) -> None:
    existing = []
    for run in runs:
        for seed in seeds:
            path = checkpoint_dir(stage1_root, run, seed)
            if path.exists():
                existing.append(str(path))
    for seed in seeds:
        path = checkpoint_dir(stage2_root, stage2_run_name(seed), seed)
        if path.exists():
            existing.append(str(path))
    report["checks"]["existing_checkpoints"] = {
        "allow_existing": allow_existing,
        "paths": existing,
    }
    if existing and not allow_existing:
        _record_issue(
            report,
            "existing checkpoint directories would trip generated stale-output guards: "
            + ", ".join(existing[:20]),
        )
        _record_remediation(
            report,
            "existing_checkpoints",
            "Use a fresh output root/run name or archive the existing checkpoint directories after confirming "
            "they are stale. Pass --allow-existing-checkpoints only when auditing already-completed runs.",
        )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    runs = args.run or list(DEFAULT_LAYOUT_RUNS)
    precompute_script = args.cache_precompute_script or str(
        Path(args.out).parent / "precompute_cache.sh"
    )
    report: dict[str, Any] = {
        "ok": False,
        "issues": [],
        "warnings": [],
        "remediations": [],
        "runs": runs,
        "seeds": args.seeds,
        "checks": {},
    }
    check_gpu(report, args.require_gpu)
    check_mesh_dataset(report, Path(args.mesh_dataset))
    check_hf_dataset(report, Path(args.hf_dataset))
    check_feature_cache(
        report,
        mesh_root=Path(args.mesh_dataset),
        root=_literal_path(args.mv_feature_cache),
        sample_count=args.sample_cache_files,
        precompute_script=precompute_script,
    )
    check_sv_downstream(report, Path(args.sv_downstream))
    check_uid_metadata(report, args.uid_metadata, args.require_uid_metadata)
    check_existing_outputs(
        report,
        runs=runs,
        seeds=args.seeds,
        stage1_root=Path(args.stage1_train_root),
        stage2_root=Path(args.stage2_train_root),
        allow_existing=args.allow_existing_checkpoints,
    )
    report["ok"] = not report["issues"]
    return report


def main() -> int:
    args = parse_args()
    report = build_report(args)
    text = json.dumps(report, indent=2) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
