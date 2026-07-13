#!/usr/bin/env python3
"""Check that an MV layout-loss evidence bundle is complete enough to review."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval.mv_layout_evidence_common import (
    downstream_object_key,
    downstream_seed_label,
    is_downstream_summary,
    parse_named_path,
    read_json,
    read_jsonl,
    REQUIRED_VERIFIER_TERMS,
)
from scripts.eval.write_mv_layout_council_review import category_stability_issues

REQUIRED_LAYOUT_METRICS = [
    "token_accuracy",
    "valid_token_frac",
    "bin_mae",
    "corner_l1",
    "corner_l2",
    "center_error",
    "size_rel_error",
    "aabb_iou",
]

REQUIRED_VERIFIER_FILES = [
    "loss_verifier.md",
    "data_verifier.md",
    "experiment_verifier.md",
    "visual_verifier.md",
    "council_review.md",
]

REQUIRED_DOWNSTREAM_KEYS = ["avg_cd", "avg_f_score", "num_evaluated", "coverage"]
REQUIRED_DOWNSTREAM_PROTOCOL_KEYS = (
    "num_sample_points",
    "align_sample_points",
    "alignment_protocol",
    "no_align",
    "mask_area_thresh",
    "evaluator_seed",
    "rng_policy",
    "eval_protocol_fingerprint",
)
DOWNSTREAM_PROTOCOL_COMPARE_KEYS = REQUIRED_DOWNSTREAM_PROTOCOL_KEYS
MIN_DOWNSTREAM_COVERAGE = 0.95
REQUIRED_LAYOUT_REPORT_KEYS = (
    "config_name",
    "split",
    "requested_num_samples",
    "batch_size",
    "mv_feature_cache",
    "num_records",
    "validity",
    "view_usage",
    "view_limit",
    "reference_only",
    "shuffle_views",
)
REQUIRED_LAYOUT_RECORD_KEYS = (
    "valid",
    "invalid_reason",
    "layout_token_count",
    "expected_layout_tokens",
)
LAYOUT_IDENTITY_KEYS = (
    "config_name",
    "split",
    "requested_num_samples",
    "batch_size",
    "mv_feature_cache",
)
REQUIRED_VISUAL_METADATA_KEYS = (
    "selection",
    "ablation",
    "projection",
    "view_mask",
    "ref_view",
    "obj_aabb",
)
REQUIRED_VIEW_USAGE_KEYS = (
    "num_records_with_view_usage",
    "mean_input_valid_view_count",
    "mean_enabled_view_count",
    "min_enabled_view_count",
    "max_enabled_view_count",
    "multi_view_record_frac",
    "single_view_record_frac",
    "zero_view_record_frac",
    "mean_enabled_view_fraction",
)
PASS_MARKERS = ("status: pass", "verdict: pass")
RECOMMENDATION_MARKERS = (
    "recommendation: merge",
    "recommendation: keep-experimental",
    "recommendation: reject",
)
FAIL_MARKERS = (
    "status: fail",
    "verdict: fail",
    "status: blocked",
    "verdict: blocked",
    "blocker",
    "todo",
    "not run",
    "missing evidence",
    "unresolved",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-root", default="outputs/da3/eval/layout_mv")
    parser.add_argument(
        "--run", action="append", default=[], help="Layout run name, repeatable."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--ce-run", default="A_ce")
    parser.add_argument(
        "--sv-downstream", required=True, help="SV eval_obj_results.jsonl/JSON summary."
    )
    parser.add_argument(
        "--downstream",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="MV downstream eval_obj_results.jsonl/JSON summary. Repeatable.",
    )
    parser.add_argument(
        "--verifier-dir",
        default="outputs/da3/experiments/mv_layout_loss_ablation/verifiers",
        help="Directory containing verifier/council markdown findings.",
    )
    parser.add_argument(
        "--figure-dir",
        default="outputs/da3/experiments/mv_layout_loss_ablation/evidence_figures",
        help="Directory containing per-seed plots and fixed/improved/regressed galleries.",
    )
    parser.add_argument(
        "--summary-json",
        default="outputs/da3/experiments/mv_layout_loss_ablation/evidence_summary/summary.json",
        help="Evidence summary JSON used for category-stability gates.",
    )
    parser.add_argument(
        "--paper-target",
        default="configs/eval/pixarmesh_paper_target.json",
        help="Tracked PixARMesh paper target spec that MV downstream seeds must beat.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10000,
        help="Scene-clustered paired-bootstrap resamples for three-seed downstream certification.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260713,
        help="Deterministic RNG seed for the scene-clustered paired bootstrap.",
    )
    parser.add_argument(
        "--best-layout-run",
        default="",
        help="Best layout run name for category-stability gates.",
    )
    parser.add_argument("--require-category-stability", action="store_true")
    parser.add_argument("--require-visuals", action="store_true")
    parser.add_argument("--require-figures", action="store_true")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def read_uid_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_summary(path: Path) -> dict[str, Any] | None:
    if path.suffix == ".jsonl":
        records = read_jsonl(path)
        return records[-1] if records else None
    return read_json(path)


def read_downstream_objects(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix != ".jsonl":
        return {}
    out = {}
    for record in read_jsonl(path):
        if is_downstream_summary(record):
            continue
        key = downstream_object_key(record)
        if key is None:
            continue
        out[key] = record
    return out


def record_issue(report: dict[str, Any], message: str) -> None:
    report["issues"].append(message)


def metric_keys_present(record: dict[str, Any]) -> list[str]:
    return [metric for metric in REQUIRED_LAYOUT_METRICS if metric in record]


def check_layout_seed(
    report: dict[str, Any],
    *,
    layout_root: Path,
    run: str,
    seed: int,
    require_visuals: bool = False,
) -> dict[str, Any]:
    seed_dir = layout_root / run / f"seed{seed}"
    report_path = seed_dir / "report.json"
    per_sample_path = seed_dir / "per_sample.jsonl"
    layout_report = read_json(report_path)
    records = read_jsonl(per_sample_path)
    entry = {
        "run": run,
        "seed": seed,
        "report": str(report_path),
        "per_sample": str(per_sample_path),
        "num_records": len(records),
        "visual_cases": 0,
    }
    if layout_report is None:
        record_issue(report, f"missing layout report: {report_path}")
        return entry
    if not records:
        record_issue(
            report, f"missing or empty per-sample layout records: {per_sample_path}"
        )
        return entry
    entry["identity"] = {key: layout_report.get(key) for key in LAYOUT_IDENTITY_KEYS}
    entry["report_num_records"] = layout_report.get("num_records")
    missing_report_keys = [
        key for key in REQUIRED_LAYOUT_REPORT_KEYS if key not in layout_report
    ]
    if missing_report_keys:
        record_issue(
            report,
            f"{run}/seed{seed} layout report missing identity keys: {missing_report_keys}",
        )
    if layout_report.get("num_records") != len(records):
        record_issue(
            report,
            f"{run}/seed{seed} report num_records={layout_report.get('num_records')} "
            f"but per_sample rows={len(records)}",
        )
    validity = layout_report.get("validity")
    if not isinstance(validity, dict):
        record_issue(report, f"{run}/seed{seed} layout report missing validity summary")
    else:
        invalid_records = [record for record in records if record.get("valid") is False]
        valid_records = [record for record in records if record.get("valid") is True]
        if validity.get("num_valid") != len(valid_records):
            record_issue(
                report,
                f"{run}/seed{seed} validity num_valid={validity.get('num_valid')} "
                f"but per_sample valid rows={len(valid_records)}",
            )
        if validity.get("num_invalid") != len(invalid_records):
            record_issue(
                report,
                f"{run}/seed{seed} validity num_invalid={validity.get('num_invalid')} "
                f"but per_sample invalid rows={len(invalid_records)}",
            )
        expected_frac = len(valid_records) / len(records)
        if validity.get("valid_sample_frac") != expected_frac:
            record_issue(
                report,
                f"{run}/seed{seed} validity valid_sample_frac={validity.get('valid_sample_frac')} "
                f"but per_sample fraction={expected_frac}",
            )
    view_usage = layout_report.get("view_usage")
    entry["view_usage"] = view_usage
    if not isinstance(view_usage, dict):
        record_issue(
            report, f"{run}/seed{seed} layout report missing view_usage summary"
        )
    else:
        missing_view_usage = [
            key for key in REQUIRED_VIEW_USAGE_KEYS if key not in view_usage
        ]
        if missing_view_usage:
            record_issue(
                report,
                f"{run}/seed{seed} view_usage missing keys: {missing_view_usage}",
            )
        if view_usage.get("num_records_with_view_usage") != len(records):
            record_issue(
                report,
                f"{run}/seed{seed} view_usage num_records_with_view_usage="
                f"{view_usage.get('num_records_with_view_usage')} but per_sample rows={len(records)}",
            )
        if float(view_usage.get("multi_view_record_frac") or 0.0) <= 0.0:
            record_issue(
                report,
                f"{run}/seed{seed} has no evidence that any evaluated object used multiple views",
            )
        if float(view_usage.get("mean_enabled_view_count") or 0.0) <= 1.0:
            record_issue(
                report, f"{run}/seed{seed} mean enabled view count is not multi-view"
            )
    missing_view_records = [
        idx
        for idx, record in enumerate(records)
        if record.get("enabled_view_count") is None
        or record.get("input_valid_view_count") is None
        or not isinstance(record.get("view_usage"), dict)
    ]
    if missing_view_records:
        record_issue(
            report,
            f"{run}/seed{seed} has per-sample rows without view-usage fields: {missing_view_records[:20]}",
        )
    if layout_report.get("view_limit") not in (0, None):
        record_issue(
            report,
            f"{run}/seed{seed} main layout eval unexpectedly used view_limit={layout_report.get('view_limit')}",
        )
    if bool(layout_report.get("reference_only")):
        record_issue(
            report,
            f"{run}/seed{seed} main layout eval unexpectedly used reference_only=true",
        )
    if bool(layout_report.get("shuffle_views")):
        record_issue(
            report,
            f"{run}/seed{seed} main layout eval unexpectedly used shuffle_views=true",
        )
    missing_metrics = sorted(
        {
            metric
            for record in records
            for metric in REQUIRED_LAYOUT_METRICS
            if metric not in record
        }
    )
    if missing_metrics:
        record_issue(
            report, f"{run}/seed{seed} missing layout metrics: {missing_metrics}"
        )
    missing_record_keys = sorted(
        {
            key
            for record in records
            for key in REQUIRED_LAYOUT_RECORD_KEYS
            if key not in record
        }
    )
    if missing_record_keys:
        record_issue(
            report,
            f"{run}/seed{seed} missing layout validity fields: {missing_record_keys}",
        )
    for record in records:
        if "uid" not in record:
            record_issue(report, f"{run}/seed{seed} has a per-sample row without uid")
            break
        if len(metric_keys_present(record)) != len(REQUIRED_LAYOUT_METRICS):
            break
    visual_dir = seed_dir / "visuals"
    visual_jsons = (
        list(visual_dir.glob("*/conditioning.json")) if visual_dir.exists() else []
    )
    entry["visual_cases"] = len(visual_jsons)
    if require_visuals and not visual_jsons:
        record_issue(report, f"{run}/seed{seed} has no visual conditioning metadata")
    for path in visual_jsons:
        meta = read_json(path)
        if not meta:
            record_issue(report, f"empty visual metadata: {path}")
            continue
        for key in REQUIRED_VISUAL_METADATA_KEYS:
            if key not in meta:
                record_issue(report, f"{path} missing visual metadata key {key!r}")
            elif key != "ref_view" and not meta[key]:
                record_issue(report, f"{path} has empty visual metadata key {key!r}")
            elif key == "ref_view" and meta[key] is None:
                record_issue(report, f"{path} has null visual metadata key {key!r}")
        case_dir = path.parent
        if not (case_dir / "topdown_bbox.png").exists():
            record_issue(report, f"{case_dir} missing topdown_bbox.png")
        if not (case_dir / "conditioning_points.npz").exists():
            record_issue(report, f"{case_dir} missing conditioning_points.npz")
        projection = meta.get("projection") or {}
        if (
            not isinstance(projection.get("per_view"), list)
            or not projection["per_view"]
        ):
            record_issue(
                report, f"{path} missing non-empty projection per_view summary"
            )
            continue
        enabled_views = [
            item for item in projection["per_view"] if item.get("view_enabled")
        ]
        if not enabled_views:
            record_issue(report, f"{path} projection metadata has no enabled views")
        for item in enabled_views:
            view_idx = item.get("view_idx")
            if not isinstance(view_idx, int):
                record_issue(
                    report,
                    f"{path} projection metadata has non-integer view_idx: {view_idx!r}",
                )
                continue
            if not (case_dir / f"view{view_idx:02d}_projection.png").exists():
                record_issue(
                    report, f"{case_dir} missing view{view_idx:02d}_projection.png"
                )
    return entry


def check_layout_eval_identity(report: dict[str, Any]) -> None:
    expected: dict[str, Any] | None = None
    expected_label = ""
    for entry in report["layout"]:
        identity = entry.get("identity")
        if not identity:
            continue
        label = f"{entry['run']}/seed{entry['seed']}"
        if expected is None:
            expected = identity
            expected_label = label
            continue
        for key in LAYOUT_IDENTITY_KEYS:
            if identity.get(key) != expected.get(key):
                record_issue(
                    report,
                    f"{label} layout eval identity {key}={identity.get(key)!r} "
                    f"differs from {expected_label} {key}={expected.get(key)!r}",
                )


def check_pairing(
    report: dict[str, Any],
    *,
    layout_root: Path,
    runs: list[str],
    seeds: list[int],
    ce_run: str,
) -> None:
    for seed in seeds:
        ce_records = read_jsonl(
            layout_root / ce_run / f"seed{seed}" / "per_sample.jsonl"
        )
        ce_uids = {str(record.get("uid")) for record in ce_records if "uid" in record}
        if not ce_uids:
            record_issue(
                report, f"{ce_run}/seed{seed} has no UIDs for paired comparison"
            )
            continue
        for run in runs:
            if run == ce_run:
                continue
            records = read_jsonl(layout_root / run / f"seed{seed}" / "per_sample.jsonl")
            run_uids = {str(record.get("uid")) for record in records if "uid" in record}
            shared = ce_uids & run_uids
            if not shared:
                record_issue(
                    report,
                    f"{run}/seed{seed} has no paired UIDs with {ce_run}/seed{seed}",
                )
            if run_uids != ce_uids:
                record_issue(
                    report,
                    f"{run}/seed{seed} UID set differs from {ce_run}/seed{seed}: "
                    f"missing={len(ce_uids - run_uids)} extra={len(run_uids - ce_uids)}",
                )


def check_downstream(
    report: dict[str, Any], *, name: str, path: Path
) -> dict[str, dict[str, Any]]:
    summary = read_summary(path)
    object_records = read_downstream_objects(path)
    report["downstream"][name] = {
        "path": str(path),
        "present": summary is not None,
        "object_records": len(object_records),
    }
    if summary is None:
        record_issue(report, f"missing downstream summary for {name}: {path}")
        return object_records
    missing = [key for key in REQUIRED_DOWNSTREAM_KEYS if key not in summary]
    if missing:
        record_issue(report, f"downstream summary for {name} missing keys: {missing}")
    missing_protocol = [key for key in REQUIRED_DOWNSTREAM_PROTOCOL_KEYS if key not in summary]
    if missing_protocol:
        record_issue(
            report,
            f"downstream summary for {name} missing protocol/provenance keys: {missing_protocol}",
        )
    if not object_records:
        record_issue(
            report,
            f"downstream summary for {name} lacks object-level JSONL records: {path}",
        )
    bad_records = [
        key
        for key, record in object_records.items()
        if record.get("cd") is None or record.get("f_score") is None
    ]
    if bad_records:
        record_issue(
            report,
            f"downstream summary for {name} has object rows without cd/f_score: {bad_records[:20]}",
        )
    bad_protocol_records = [
        key
        for key, record in object_records.items()
        if any(proto_key not in record for proto_key in REQUIRED_DOWNSTREAM_PROTOCOL_KEYS)
        or "object_eval_seed" not in record
    ]
    if bad_protocol_records:
        record_issue(
            report,
            f"downstream summary for {name} has object rows without eval protocol provenance: "
            f"{bad_protocol_records[:20]}",
        )
    coverage = summary.get("coverage")
    if coverage is not None and float(coverage) < MIN_DOWNSTREAM_COVERAGE:
        record_issue(
            report,
            f"downstream summary for {name} coverage={float(coverage):.1%} below "
            f"{MIN_DOWNSTREAM_COVERAGE:.0%}",
        )
    report["downstream"][name].update(
        {key: summary.get(key) for key in REQUIRED_DOWNSTREAM_KEYS}
    )
    report["downstream"][name]["protocol"] = {
        key: summary.get(key) for key in REQUIRED_DOWNSTREAM_PROTOCOL_KEYS
    }
    return object_records


def check_downstream_protocol_match(report: dict[str, Any], label: str) -> None:
    sv_protocol = report["downstream"].get("SV", {}).get("protocol") or {}
    mv_protocol = report["downstream"].get(label, {}).get("protocol") or {}
    mismatches = []
    for key in DOWNSTREAM_PROTOCOL_COMPARE_KEYS:
        if sv_protocol.get(key) != mv_protocol.get(key):
            mismatches.append(
                {"key": key, "sv": sv_protocol.get(key), "mv": mv_protocol.get(key)}
            )
    if mismatches:
        record_issue(
            report,
            f"downstream protocol mismatch for {label} vs SV: {mismatches}",
        )


def check_paper_target(report: dict[str, Any], label: str, paper_target: dict[str, Any] | None) -> None:
    if paper_target is None:
        record_issue(report, "missing PixARMesh paper target specification")
        return
    target = paper_target.get("object_level") or {}
    target_cd = target.get("mean_cd")
    target_f = target.get("mean_f_score")
    min_cov = float(paper_target.get("coverage_min", MIN_DOWNSTREAM_COVERAGE))
    entry = report["downstream"].get(label, {})
    if target_cd is None or target_f is None:
        record_issue(report, "paper target spec lacks object_level.mean_cd/mean_f_score")
        return
    cd = entry.get("avg_cd")
    f_score = entry.get("avg_f_score")
    coverage = entry.get("coverage")
    paper_check = {
        "target_mean_cd": target_cd,
        "target_mean_f_score": target_f,
        "coverage_min": min_cov,
        "cd_beats_paper": cd is not None and float(cd) < float(target_cd),
        "f_score_beats_paper": f_score is not None and float(f_score) > float(target_f),
        "coverage_passes": coverage is not None and float(coverage) >= min_cov,
    }
    entry["paper_target"] = paper_check
    if not paper_check["cd_beats_paper"]:
        record_issue(report, f"downstream {label} does not beat PixARMesh paper CD target")
    if not paper_check["f_score_beats_paper"]:
        record_issue(report, f"downstream {label} does not beat PixARMesh paper F-score target")
    if not paper_check["coverage_passes"]:
        record_issue(report, f"downstream {label} does not meet paper-target coverage gate")


def paired_downstream_deltas(
    mv_records: dict[str, dict[str, Any]],
    sv_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    shared = sorted(set(mv_records) & set(sv_records))
    cd_deltas = [
        float(mv_records[key]["cd"]) - float(sv_records[key]["cd"])
        for key in shared
        if mv_records[key].get("cd") is not None
        and sv_records[key].get("cd") is not None
    ]
    f_deltas = [
        float(mv_records[key]["f_score"]) - float(sv_records[key]["f_score"])
        for key in shared
        if mv_records[key].get("f_score") is not None
        and sv_records[key].get("f_score") is not None
    ]
    avg_cd_delta = sum(cd_deltas) / len(cd_deltas) if cd_deltas else None
    avg_f_delta = sum(f_deltas) / len(f_deltas) if f_deltas else None
    return {
        "paired_count": len(shared),
        "avg_cd_delta_vs_sv": avg_cd_delta,
        "avg_f_score_delta_vs_sv": avg_f_delta,
        "cd_beats_sv": avg_cd_delta is not None and avg_cd_delta < 0,
        "f_score_beats_sv": avg_f_delta is not None and avg_f_delta > 0,
    }


def _downstream_cluster_key(
    key: str,
    mv_record: dict[str, Any],
    sv_record: dict[str, Any],
) -> str:
    # eval_obj rows use uid as the scene id and obj_id as the object index.
    # Prefer explicit scene_id if future manifests add it; otherwise uid is the
    # correct scene cluster for the current object-level eval rows.
    return str(
        mv_record.get("scene_id")
        or sv_record.get("scene_id")
        or mv_record.get("uid")
        or sv_record.get("uid")
        or key
    )


def scene_clustered_paired_bootstrap(
    mv_record_sets: list[dict[str, dict[str, Any]]],
    sv_records: dict[str, dict[str, Any]],
    *,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    clusters: dict[str, list[tuple[float, float]]] = {}
    for mv_records in mv_record_sets:
        for key in sorted(set(mv_records) & set(sv_records)):
            mv_record = mv_records[key]
            sv_record = sv_records[key]
            if (
                mv_record.get("cd") is None
                or sv_record.get("cd") is None
                or mv_record.get("f_score") is None
                or sv_record.get("f_score") is None
            ):
                continue
            cluster = _downstream_cluster_key(key, mv_record, sv_record)
            clusters.setdefault(cluster, []).append(
                (
                    float(mv_record["cd"]) - float(sv_record["cd"]),
                    float(mv_record["f_score"]) - float(sv_record["f_score"]),
                )
            )

    resamples = int(resamples)
    if resamples <= 0:
        return {
            "resamples": resamples,
            "seed": int(seed),
            "confidence": float(confidence),
            "cluster_count": 0,
            "paired_observation_count": 0,
            "error": "bootstrap resamples must be positive",
        }

    cluster_values = [np.asarray(values, dtype=np.float64) for values in clusters.values()]
    if not cluster_values:
        return {
            "resamples": resamples,
            "seed": int(seed),
            "confidence": float(confidence),
            "cluster_count": 0,
            "paired_observation_count": 0,
            "error": "no paired downstream observations",
        }

    all_values = np.concatenate(cluster_values, axis=0)
    point = all_values.mean(axis=0)
    rng = np.random.default_rng(int(seed))
    cd_means = np.empty(resamples, dtype=np.float64)
    f_means = np.empty(resamples, dtype=np.float64)
    n_clusters = len(cluster_values)
    for i in range(resamples):
        sample_ids = rng.integers(0, n_clusters, size=n_clusters)
        sample = np.concatenate([cluster_values[idx] for idx in sample_ids], axis=0)
        means = sample.mean(axis=0)
        cd_means[i] = means[0]
        f_means[i] = means[1]

    alpha = (1.0 - float(confidence)) / 2.0
    cd_low, cd_high = np.quantile(cd_means, [alpha, 1.0 - alpha])
    f_low, f_high = np.quantile(f_means, [alpha, 1.0 - alpha])
    return {
        "resamples": resamples,
        "seed": int(seed),
        "confidence": float(confidence),
        "cluster_count": n_clusters,
        "paired_observation_count": int(len(all_values)),
        "cd_delta_mean": float(point[0]),
        "cd_delta_ci_low": float(cd_low),
        "cd_delta_ci_high": float(cd_high),
        "f_score_delta_mean": float(point[1]),
        "f_score_delta_ci_low": float(f_low),
        "f_score_delta_ci_high": float(f_high),
        "cd_upper_bound_below_zero": bool(cd_high < 0.0),
        "f_score_lower_bound_above_zero": bool(f_low > 0.0),
    }


def check_verifier_findings(report: dict[str, Any], verifier_dir: Path) -> None:
    report["verifiers"] = {}
    for name in REQUIRED_VERIFIER_FILES:
        path = verifier_dir / name
        text = path.read_text().strip() if path.exists() else ""
        report["verifiers"][name] = {"path": str(path), "bytes": len(text.encode())}
        if not text:
            record_issue(report, f"missing or empty verifier finding: {path}")
            continue
        lowered = text.lower()
        if "checked" not in lowered and "command" not in lowered:
            record_issue(
                report,
                f"verifier finding lacks checked commands/artifacts section: {path}",
            )
        if not any(marker in lowered for marker in PASS_MARKERS):
            record_issue(
                report, f"verifier finding lacks explicit status: pass marker: {path}"
            )
        missing_terms = [
            term
            for term in REQUIRED_VERIFIER_TERMS.get(name, ())
            if term.lower() not in lowered
        ]
        if missing_terms:
            record_issue(
                report,
                f"verifier finding lacks role-specific evidence terms {missing_terms}: {path}",
            )
        if name == "council_review.md" and not any(
            marker in lowered for marker in RECOMMENDATION_MARKERS
        ):
            record_issue(
                report,
                f"council review lacks explicit merge/keep-experimental/reject recommendation: {path}",
            )
        if name == "council_review.md" and any(
            marker in lowered for marker in PASS_MARKERS
        ):
            if "recommendation: merge" not in lowered:
                record_issue(
                    report,
                    f"council review has pass status without recommendation: merge: {path}",
                )
        fail_markers = [marker for marker in FAIL_MARKERS if marker in lowered]
        if fail_markers:
            record_issue(
                report,
                f"verifier finding contains unresolved failure markers {fail_markers}: {path}",
            )


def check_figure_artifacts(
    report: dict[str, Any], figure_dir: Path, require_figures: bool
) -> None:
    report["figures"] = {"path": str(figure_dir), "required": require_figures}
    if not require_figures:
        return
    required_files = [
        "per_seed_metrics.png",
        "paired_delta_vs_ce.png",
        "fixed_uids.txt",
        "improved_uids.txt",
        "regressed_uids.txt",
        "failure_uids.txt",
        "gallery_uids.txt",
        "ranked_uids.json",
        "ranked_failures.json",
        "gallery_manifest.json",
    ]
    for name in required_files:
        path = figure_dir / name
        report["figures"][name] = {"path": str(path), "exists": path.exists()}
        if not path.exists():
            record_issue(report, f"missing figure artifact: {path}")
    uid_lists = {
        "fixed": read_uid_list(figure_dir / "fixed_uids.txt"),
        "improved": read_uid_list(figure_dir / "improved_uids.txt"),
        "regressed": read_uid_list(figure_dir / "regressed_uids.txt"),
        "failures": read_uid_list(figure_dir / "failure_uids.txt"),
        "gallery": read_uid_list(figure_dir / "gallery_uids.txt"),
    }
    report["figures"]["uid_list_counts"] = {
        name: len(uids) for name, uids in uid_lists.items()
    }
    expected_gallery_uids = sorted(
        set(
            uid_lists["fixed"]
            + uid_lists["improved"]
            + uid_lists["regressed"]
            + uid_lists["failures"]
        )
    )
    if (figure_dir / "gallery_uids.txt").exists() and sorted(
        uid_lists["gallery"]
    ) != expected_gallery_uids:
        record_issue(
            report,
            f"{figure_dir / 'gallery_uids.txt'} does not match the union of ranked UID lists",
        )
    manifest = read_json(figure_dir / "gallery_manifest.json")
    if manifest is None:
        return
    created = int(manifest.get("created_composites") or 0)
    report["figures"]["created_composites"] = created
    if created <= 0:
        record_issue(
            report,
            f"{figure_dir / 'gallery_manifest.json'} has no created comparison composites",
        )
    empty_justifications = manifest.get("empty_uid_list_justifications") or {}
    if not isinstance(empty_justifications, dict):
        record_issue(
            report, "gallery manifest empty_uid_list_justifications must be an object"
        )
        empty_justifications = {}
    for name in ("fixed", "improved", "regressed", "failures"):
        gallery = next(
            (
                item
                for item in manifest.get("galleries", [])
                if item.get("name") == name
            ),
            None,
        )
        if gallery is None:
            record_issue(report, f"gallery manifest missing {name!r} gallery")
            continue
        if name == "failures" and not uid_lists[name]:
            record_issue(
                report,
                "gallery 'failures' has an empty UID list; failure exemplars are required",
            )
        if name == "failures" and not gallery.get("created"):
            record_issue(report, "gallery 'failures' has no created comparison images")
        elif uid_lists[name] and not gallery.get("created"):
            record_issue(
                report,
                f"gallery {name!r} has no created comparison images for nonempty UID list",
            )
        elif name in {"fixed", "improved"} and not uid_lists[name]:
            justification = str(empty_justifications.get(name) or "").strip()
            if not justification:
                record_issue(
                    report,
                    f"gallery {name!r} has an empty UID list without an explicit recorded justification",
                )


def check_category_stability(
    report: dict[str, Any],
    *,
    summary_json: Path,
    best_layout_run: str,
    require_category_stability: bool,
) -> None:
    report["category_stability"] = {
        "required": require_category_stability,
        "summary_json": str(summary_json),
        "best_layout_run": best_layout_run,
    }
    if not require_category_stability:
        return
    if not best_layout_run:
        record_issue(report, "--require-category-stability needs --best-layout-run")
        return
    summary = read_json(summary_json)
    if summary is None:
        record_issue(report, f"missing category-stability summary JSON: {summary_json}")
        return
    issues = category_stability_issues(summary, best_layout_run)
    report["category_stability"]["issues"] = issues
    for issue in issues:
        record_issue(report, f"category stability: {issue}")


def build_evidence_report(
    *,
    layout_root: Path,
    runs: list[str],
    seeds: list[int],
    ce_run: str,
    sv_downstream: Path,
    downstream: list[str],
    verifier_dir: Path,
    figure_dir: Path = Path(
        "outputs/da3/experiments/mv_layout_loss_ablation/evidence_figures"
    ),
    summary_json: Path = Path(
        "outputs/da3/experiments/mv_layout_loss_ablation/evidence_summary/summary.json"
    ),
    paper_target: Path = Path("configs/eval/pixarmesh_paper_target.json"),
    bootstrap_resamples: int = 10000,
    bootstrap_seed: int = 20260713,
    best_layout_run: str = "",
    require_visuals: bool = False,
    require_figures: bool = False,
    require_category_stability: bool = False,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "ok": False,
        "issues": [],
        "layout": [],
        "downstream": {},
        "downstream_groups": {},
        "verifiers": {},
        "figures": {},
        "category_stability": {},
    }
    for run in runs:
        for seed in seeds:
            report["layout"].append(
                check_layout_seed(
                    report,
                    layout_root=layout_root,
                    run=run,
                    seed=seed,
                    require_visuals=require_visuals,
                )
            )
    check_layout_eval_identity(report)
    check_pairing(
        report, layout_root=layout_root, runs=runs, seeds=seeds, ce_run=ce_run
    )
    paper_target_payload = read_json(paper_target)
    report["paper_target"] = {
        "path": str(paper_target),
        "present": paper_target_payload is not None,
        "source": (paper_target_payload or {}).get("source"),
        "object_level": (paper_target_payload or {}).get("object_level"),
        "coverage_min": (paper_target_payload or {}).get("coverage_min"),
    }
    sv_records = check_downstream(report, name="SV", path=sv_downstream)
    downstream_groups: dict[str, list[Path]] = {}
    downstream_records_by_group: dict[str, list[dict[str, dict[str, Any]]]] = {}
    for item in downstream:
        name, path = parse_named_path(item)
        paths = downstream_groups.setdefault(name, [])
        paths.append(path)
        label = name if len(paths) == 1 else f"{name}#{len(paths)}"
        mv_records = check_downstream(report, name=label, path=path)
        downstream_records_by_group.setdefault(name, []).append(mv_records)
        check_downstream_protocol_match(report, label)
        check_paper_target(report, label, paper_target_payload)
        if sv_records and mv_records and set(sv_records) != set(mv_records):
            record_issue(
                report,
                f"downstream object set for {label} differs from SV: "
                f"missing={len(set(sv_records) - set(mv_records))} "
                f"extra={len(set(mv_records) - set(sv_records))}",
            )
        if sv_records and mv_records:
            deltas = paired_downstream_deltas(mv_records, sv_records)
            report["downstream"][label]["paired_vs_sv"] = deltas
            if not deltas["cd_beats_sv"]:
                record_issue(
                    report, f"downstream {label} does not beat SV on paired CD"
                )
            if not deltas["f_score_beats_sv"]:
                record_issue(
                    report, f"downstream {label} does not beat SV on paired F-score"
                )
    for name, paths in sorted(downstream_groups.items()):
        seed_labels = [downstream_seed_label(path) for path in paths]
        seed_counts = Counter(seed_labels)
        duplicates = sorted(seed for seed, count in seed_counts.items() if count > 1)
        expected = {f"seed{seed}" for seed in seeds}
        observed = set(seed_labels)
        report["downstream_groups"][name] = {
            "paths": [str(path) for path in paths],
            "count": len(paths),
            "expected_seed_count": len(seeds),
            "seed_labels": seed_labels,
            "expected_seed_labels": sorted(expected),
            "duplicate_seed_labels": duplicates,
            "missing_expected_seed_labels": sorted(expected - observed),
            "unexpected_seed_labels": sorted(observed - expected),
        }
        if len(seeds) > 1 and len(paths) != len(seeds):
            record_issue(
                report,
                f"downstream group {name!r} has {len(paths)} files but "
                f"{len(seeds)} seeds were requested; pass one --downstream {name}=... per seed",
            )
        if duplicates:
            record_issue(
                report,
                f"downstream group {name!r} has duplicate seed labels {duplicates}",
            )
        if len(seeds) > 1 and expected - observed:
            record_issue(
                report,
                f"downstream group {name!r} is missing expected seed labels {sorted(expected - observed)}",
            )
        if len(seeds) > 1 and observed - expected:
            record_issue(
                report,
                f"downstream group {name!r} has unexpected seed labels {sorted(observed - expected)}",
            )
        if len(seeds) >= 3 and sv_records and paths:
            bootstrap = scene_clustered_paired_bootstrap(
                downstream_records_by_group.get(name, []),
                sv_records,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            )
            report["downstream_groups"][name]["scene_clustered_paired_bootstrap"] = bootstrap
            if bootstrap.get("error"):
                record_issue(report, f"downstream group {name!r} bootstrap failed: {bootstrap['error']}")
            elif not bootstrap.get("cd_upper_bound_below_zero"):
                record_issue(
                    report,
                    f"downstream group {name!r} CD bootstrap upper 95% bound is not below zero "
                    f"({bootstrap.get('cd_delta_ci_high')})",
                )
            if not bootstrap.get("error") and not bootstrap.get("f_score_lower_bound_above_zero"):
                record_issue(
                    report,
                    f"downstream group {name!r} F-score bootstrap lower 95% bound is not above zero "
                    f"({bootstrap.get('f_score_delta_ci_low')})",
                )
    check_verifier_findings(report, verifier_dir)
    check_figure_artifacts(report, figure_dir, require_figures)
    check_category_stability(
        report,
        summary_json=summary_json,
        best_layout_run=best_layout_run,
        require_category_stability=require_category_stability,
    )
    report["ok"] = not report["issues"]
    return report


def main() -> int:
    args = parse_args()
    runs = args.run or ["A_ce", "B_ordinal", "C_coord"]
    report = build_evidence_report(
        layout_root=Path(args.layout_root),
        runs=runs,
        seeds=args.seeds,
        ce_run=args.ce_run,
        sv_downstream=Path(args.sv_downstream),
        downstream=args.downstream,
        verifier_dir=Path(args.verifier_dir),
        figure_dir=Path(args.figure_dir),
        summary_json=Path(args.summary_json),
        paper_target=Path(args.paper_target),
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        best_layout_run=args.best_layout_run,
        require_visuals=args.require_visuals,
        require_figures=args.require_figures,
        require_category_stability=args.require_category_stability,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(text, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
