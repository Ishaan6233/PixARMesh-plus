#!/usr/bin/env python3
"""Write an adversarial council review from MV layout evidence artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

LAYOUT_IMPROVEMENT_METRICS = [
    "bin_mae",
    "corner_l1",
    "corner_l2",
    "center_error",
    "size_rel_error",
    "aabb_iou",
]

LOWER_IS_BETTER = {
    "bin_mae",
    "corner_l1",
    "corner_l2",
    "center_error",
    "size_rel_error",
}

REQUIRED_CONTROLS = [
    "one_view_eval",
    "two_view_eval",
    "four_view_eval",
    "eight_view_eval",
    "reference_only_eval",
    "shuffled_views_eval",
    "no_aabb",
    "no_voxel_encoder",
    "no_obj_pc_cond",
    "no_obj_pc_appearance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        default="outputs/da3/experiments/mv_layout_loss_ablation/evidence_summary/summary.json",
    )
    parser.add_argument("--best-layout-run", required=True)
    parser.add_argument("--downstream-run", default="E_stage2_best")
    parser.add_argument(
        "--best-layout-report",
        action="append",
        required=True,
        help=(
            "report.json for a best stage-1 run/seed used as the negative-control "
            "baseline. Repeat once per seed/control report."
        ),
    )
    parser.add_argument(
        "--negative-control",
        action="append",
        default=[],
        metavar="NAME=REPORT_JSON",
        help="Negative-control layout report. Repeatable.",
    )
    parser.add_argument(
        "--required-control",
        action="append",
        default=[],
        help="Required control name. Defaults to the planned core controls.",
    )
    parser.add_argument(
        "--require-category-stability",
        action="store_true",
        help="Fail unless the best layout run improves CE within every metadata category.",
    )
    parser.add_argument(
        "--out",
        default="outputs/da3/experiments/mv_layout_loss_ablation/verifiers/council_review.md",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name, Path(path)


def metric_mean(report: dict[str, Any], metric: str) -> float | None:
    value = report.get("summary", {}).get(metric, {}).get("mean")
    return float(value) if value is not None else None


def control_degrades(control_report: dict[str, Any], baseline_report: dict[str, Any]) -> bool:
    base_iou = metric_mean(baseline_report, "aabb_iou")
    ctrl_iou = metric_mean(control_report, "aabb_iou")
    base_bin = metric_mean(baseline_report, "bin_mae")
    ctrl_bin = metric_mean(control_report, "bin_mae")
    degraded_iou = base_iou is not None and ctrl_iou is not None and ctrl_iou < base_iou
    degraded_bin = base_bin is not None and ctrl_bin is not None and ctrl_bin > base_bin
    return degraded_iou or degraded_bin


def _as_report_list(value: Any) -> list[dict[str, Any] | None]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def category_stability_issues(summary: dict[str, Any], best_layout_run: str) -> list[str]:
    """Return blocking issues for category-stratified CE deltas."""
    issues: list[str] = []
    layout_runs = summary.get("layout", {}).get("runs", {})
    best_layout = layout_runs.get(best_layout_run) or {}
    categories = best_layout.get("category_paired_delta_vs_ce") or {}
    if not categories:
        return [f"{best_layout_run} has no category-paired layout evidence"]
    if len(categories) < 2:
        issues.append(f"{best_layout_run} has category evidence for fewer than two categories")
    missing_category_count = int(best_layout.get("missing_category_count") or 0)
    if missing_category_count:
        issues.append(f"{best_layout_run} has {missing_category_count} paired records without category metadata")
    for category, metrics in sorted(categories.items()):
        for metric in LAYOUT_IMPROVEMENT_METRICS:
            delta = metrics.get(metric, {})
            n = int(delta.get("n") or 0)
            improved_seed_count = int(delta.get("improved_seed_count") or 0)
            mean = delta.get("mean")
            direction_ok = mean is not None and (
                mean < 0 if metric in LOWER_IS_BETTER else mean > 0
            )
            all_cells_ok = n > 0 and improved_seed_count == n
            if not (direction_ok and all_cells_ok):
                issues.append(
                    f"{best_layout_run} is not stable for category {category!r} metric {metric}"
                )
    return issues


def evaluate_council(
    *,
    summary: dict[str, Any] | None,
    best_layout_run: str,
    downstream_run: str,
    best_layout_report: dict[str, Any] | list[dict[str, Any] | None] | None,
    negative_controls: dict[str, dict[str, Any] | list[dict[str, Any] | None] | None],
    required_controls: list[str],
    require_category_stability: bool = False,
) -> dict[str, Any]:
    issues: list[str] = []
    checks: dict[str, Any] = {}
    if summary is None:
        return {"status": "fail", "issues": ["summary.json is missing or unreadable"], "checks": checks}

    layout_runs = summary.get("layout", {}).get("runs", {})
    best_layout = layout_runs.get(best_layout_run)
    if best_layout is None:
        issues.append(f"best layout run {best_layout_run!r} is absent from summary")
    else:
        layout_checks = {}
        for metric in LAYOUT_IMPROVEMENT_METRICS:
            delta = best_layout.get("paired_delta_vs_ce", {}).get(metric, {})
            n = int(delta.get("n") or 0)
            improved_seed_count = int(delta.get("improved_seed_count") or 0)
            mean = delta.get("mean")
            direction_ok = mean is not None and (
                mean < 0 if metric in LOWER_IS_BETTER else mean > 0
            )
            all_seeds_ok = n > 0 and improved_seed_count == n
            layout_checks[metric] = {
                "paired_delta_mean": mean,
                "improved_seed_count": improved_seed_count,
                "n": n,
                "passes": bool(direction_ok and all_seeds_ok),
            }
            if not layout_checks[metric]["passes"]:
                issues.append(
                    f"{best_layout_run} does not improve {metric} against CE across all paired seeds"
                )
        visual_counts = [
            int(seed_info.get("visual_count") or 0)
            for seed_info in best_layout.get("seeds", {}).values()
        ]
        layout_checks["visual_artifacts"] = {
            "visual_counts": visual_counts,
            "passes": bool(visual_counts and all(count > 0 for count in visual_counts)),
        }
        if not layout_checks["visual_artifacts"]["passes"]:
            issues.append(f"{best_layout_run} lacks visual artifacts for one or more seeds")
        if require_category_stability:
            cat_issues = category_stability_issues(summary, best_layout_run)
            category_checks = {
                "required": True,
                "category_count": len(best_layout.get("category_paired_delta_vs_ce") or {}),
                "missing_category_count": int(best_layout.get("missing_category_count") or 0),
                "passes": not cat_issues,
            }
            layout_checks["category_stability"] = category_checks
            issues.extend(cat_issues)
        checks["layout_vs_ce"] = layout_checks

    downstream_cmp = summary.get("downstream", {}).get("comparison_to_sv", {}).get(downstream_run, {})
    downstream_checks = {
        "cd_beats_sv": bool(downstream_cmp.get("cd_beats_sv")),
        "f_score_beats_sv": bool(downstream_cmp.get("f_score_beats_sv")),
        "object_set_equal": bool(downstream_cmp.get("object_set_equal")),
        "paired_count": int(downstream_cmp.get("paired_count") or 0),
        "avg_cd_delta_vs_sv": downstream_cmp.get("avg_cd_delta_vs_sv"),
        "avg_f_score_delta_vs_sv": downstream_cmp.get("avg_f_score_delta_vs_sv"),
    }
    checks["downstream_vs_sv"] = downstream_checks
    if not downstream_checks["object_set_equal"] or downstream_checks["paired_count"] <= 0:
        issues.append(f"{downstream_run} is not UID/object-paired with SV downstream evidence")
    if not downstream_checks["cd_beats_sv"]:
        issues.append(f"{downstream_run} does not beat SV on downstream CD")
    if not downstream_checks["f_score_beats_sv"]:
        issues.append(f"{downstream_run} does not beat SV on downstream F-score")

    control_checks = {}
    baseline_reports = _as_report_list(best_layout_report)
    if not baseline_reports:
        issues.append("best layout report is missing or unreadable")
    for name in required_controls:
        controls = _as_report_list(negative_controls.get(name))
        if not controls:
            control_checks[name] = {"present": False, "passes": False, "n": 0}
            issues.append(f"required negative control {name!r} is missing")
            continue
        if baseline_reports and len(controls) != len(baseline_reports):
            issues.append(
                f"negative control {name!r} has {len(controls)} reports but "
                f"{len(baseline_reports)} best-layout reports"
            )
        per_report = []
        for idx, control in enumerate(controls):
            baseline = baseline_reports[idx] if idx < len(baseline_reports) else None
            passes = baseline is not None and control is not None and control_degrades(control, baseline)
            per_report.append({"index": idx, "passes": bool(passes)})
        control_passes = bool(per_report) and all(item["passes"] for item in per_report)
        control_checks[name] = {
            "present": True,
            "passes": control_passes,
            "n": len(controls),
            "per_report": per_report,
        }
        if not control_passes:
            issues.append(f"negative control {name!r} does not degrade relative to the best layout report")
    checks["negative_controls"] = control_checks

    status = "pass" if not issues else "fail"
    return {
        "status": status,
        "recommendation": recommendation_for(status, issues),
        "issues": issues,
        "checks": checks,
    }


def recommendation_for(status: str, issues: list[str]) -> str:
    """Return the deterministic merge/hold/reject recommendation for the bundle."""
    if status == "pass":
        return "merge"
    missing_or_incomplete_markers = (
        "missing",
        "unreadable",
        "absent",
        "lacks",
        "not uid/object-paired",
        "has no",
        "required negative control",
        "reports but",
    )
    issue_text = "\n".join(issues).lower()
    if any(marker in issue_text for marker in missing_or_incomplete_markers):
        return "keep-experimental"
    return "reject"


def write_markdown(result: dict[str, Any], out_path: Path) -> None:
    status = result["status"]
    recommendation = result.get("recommendation", recommendation_for(status, result.get("issues", [])))
    lines = [
        f"status: {status}",
        f"recommendation: {recommendation}",
        "",
        "# MV Layout Council Review",
        "",
        "Checked commands/artifacts:",
        "- evidence summary JSON",
        "- best stage-1 layout report",
        "- downstream SV comparison in summary JSON",
        "- required negative-control layout reports",
        "- optional category-stability evidence when required",
        "",
        "## Verdict",
    ]
    if status == "pass":
        lines.append("The evidence satisfies the deterministic council gate; merge is supportable after independent verifier files also pass.")
    else:
        lines.append(f"The evidence does not satisfy the deterministic council gate; recommendation is {recommendation}.")
    lines.extend(["", "## Issues"])
    if result["issues"]:
        lines.extend(f"- {issue}" for issue in result["issues"])
    else:
        lines.append("- none")
    lines.extend(["", "## Checks", "```json", json.dumps(result["checks"], indent=2), "```", ""])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    required_controls = args.required_control or REQUIRED_CONTROLS
    controls: dict[str, list[dict[str, Any] | None]] = {}
    for name, path in (parse_named_path(item) for item in args.negative_control):
        controls.setdefault(name, []).append(read_json(path))
    result = evaluate_council(
        summary=read_json(Path(args.summary_json)),
        best_layout_run=args.best_layout_run,
        downstream_run=args.downstream_run,
        best_layout_report=[read_json(Path(path)) for path in args.best_layout_report],
        negative_controls=controls,
        required_controls=required_controls,
        require_category_stability=args.require_category_stability,
    )
    write_markdown(result, Path(args.out))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
