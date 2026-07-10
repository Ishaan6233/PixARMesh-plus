#!/usr/bin/env python3
"""Check that an MV layout-loss evidence bundle is complete enough to review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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

REQUIRED_DOWNSTREAM_KEYS = ["avg_cd", "avg_f_score", "num_evaluated"]
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
    parser.add_argument("--run", action="append", default=[], help="Layout run name, repeatable.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--ce-run", default="A_ce")
    parser.add_argument("--sv-downstream", required=True, help="SV eval_obj_results.jsonl/JSON summary.")
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
    parser.add_argument("--best-layout-run", default="", help="Best layout run name for category-stability gates.")
    parser.add_argument("--require-category-stability", action="store_true")
    parser.add_argument("--require-visuals", action="store_true")
    parser.add_argument("--require-figures", action="store_true")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name, Path(path)


def read_summary(path: Path) -> dict[str, Any] | None:
    if path.suffix == ".jsonl":
        records = read_jsonl(path)
        return records[-1] if records else None
    return read_json(path)


def is_downstream_summary(record: dict[str, Any]) -> bool:
    return "avg_cd" in record or "avg_f_score" in record or "num_evaluated" in record


def downstream_object_key(record: dict[str, Any]) -> str | None:
    if "uid" not in record:
        return None
    if "obj_id" in record:
        return f"{record['uid']}::{record['obj_id']}"
    return str(record["uid"])


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
        record_issue(report, f"missing or empty per-sample layout records: {per_sample_path}")
        return entry
    missing_metrics = sorted(
        {
            metric
            for record in records
            for metric in REQUIRED_LAYOUT_METRICS
            if metric not in record
        }
    )
    if missing_metrics:
        record_issue(report, f"{run}/seed{seed} missing layout metrics: {missing_metrics}")
    for record in records:
        if "uid" not in record:
            record_issue(report, f"{run}/seed{seed} has a per-sample row without uid")
            break
        if len(metric_keys_present(record)) != len(REQUIRED_LAYOUT_METRICS):
            break
    visual_dir = seed_dir / "visuals"
    visual_jsons = list(visual_dir.glob("*/conditioning.json")) if visual_dir.exists() else []
    entry["visual_cases"] = len(visual_jsons)
    if require_visuals and not visual_jsons:
        record_issue(report, f"{run}/seed{seed} has no visual conditioning metadata")
    for path in visual_jsons:
        meta = read_json(path)
        if not meta:
            record_issue(report, f"empty visual metadata: {path}")
            continue
        for key in ("selection", "ablation", "projection"):
            if key not in meta:
                record_issue(report, f"{path} missing visual metadata key {key!r}")
            elif not meta[key]:
                record_issue(report, f"{path} has empty visual metadata key {key!r}")
        case_dir = path.parent
        if not (case_dir / "topdown_bbox.png").exists():
            record_issue(report, f"{case_dir} missing topdown_bbox.png")
        if not (case_dir / "conditioning_points.npz").exists():
            record_issue(report, f"{case_dir} missing conditioning_points.npz")
        projection = meta.get("projection") or {}
        if not isinstance(projection.get("per_view"), list) or not projection["per_view"]:
            record_issue(report, f"{path} missing non-empty projection per_view summary")
    return entry


def check_pairing(
    report: dict[str, Any],
    *,
    layout_root: Path,
    runs: list[str],
    seeds: list[int],
    ce_run: str,
) -> None:
    for seed in seeds:
        ce_records = read_jsonl(layout_root / ce_run / f"seed{seed}" / "per_sample.jsonl")
        ce_uids = {str(record.get("uid")) for record in ce_records if "uid" in record}
        if not ce_uids:
            record_issue(report, f"{ce_run}/seed{seed} has no UIDs for paired comparison")
            continue
        for run in runs:
            if run == ce_run:
                continue
            records = read_jsonl(layout_root / run / f"seed{seed}" / "per_sample.jsonl")
            run_uids = {str(record.get("uid")) for record in records if "uid" in record}
            shared = ce_uids & run_uids
            if not shared:
                record_issue(report, f"{run}/seed{seed} has no paired UIDs with {ce_run}/seed{seed}")
            if run_uids != ce_uids:
                record_issue(
                    report,
                    f"{run}/seed{seed} UID set differs from {ce_run}/seed{seed}: "
                    f"missing={len(ce_uids - run_uids)} extra={len(run_uids - ce_uids)}",
                )


def check_downstream(report: dict[str, Any], *, name: str, path: Path) -> dict[str, dict[str, Any]]:
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
    if not object_records:
        record_issue(report, f"downstream summary for {name} lacks object-level JSONL records: {path}")
    bad_records = [
        key for key, record in object_records.items() if record.get("cd") is None or record.get("f_score") is None
    ]
    if bad_records:
        record_issue(
            report,
            f"downstream summary for {name} has object rows without cd/f_score: {bad_records[:20]}",
        )
    report["downstream"][name].update({key: summary.get(key) for key in REQUIRED_DOWNSTREAM_KEYS})
    return object_records


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
            record_issue(report, f"verifier finding lacks checked commands/artifacts section: {path}")
        if not any(marker in lowered for marker in PASS_MARKERS):
            record_issue(report, f"verifier finding lacks explicit status: pass marker: {path}")
        if name == "council_review.md" and not any(marker in lowered for marker in RECOMMENDATION_MARKERS):
            record_issue(report, f"council review lacks explicit merge/keep-experimental/reject recommendation: {path}")
        fail_markers = [marker for marker in FAIL_MARKERS if marker in lowered]
        if fail_markers:
            record_issue(report, f"verifier finding contains unresolved failure markers {fail_markers}: {path}")


def check_figure_artifacts(report: dict[str, Any], figure_dir: Path, require_figures: bool) -> None:
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
        "ranked_uids.json",
        "ranked_failures.json",
        "gallery_manifest.json",
    ]
    for name in required_files:
        path = figure_dir / name
        report["figures"][name] = {"path": str(path), "exists": path.exists()}
        if not path.exists():
            record_issue(report, f"missing figure artifact: {path}")
    manifest = read_json(figure_dir / "gallery_manifest.json")
    if manifest is None:
        return
    created = int(manifest.get("created_composites") or 0)
    report["figures"]["created_composites"] = created
    if created <= 0:
        record_issue(report, f"{figure_dir / 'gallery_manifest.json'} has no created comparison composites")
    for name in ("fixed", "improved", "regressed", "failures"):
        gallery = next((item for item in manifest.get("galleries", []) if item.get("name") == name), None)
        if gallery is None:
            record_issue(report, f"gallery manifest missing {name!r} gallery")
            continue
        if name in {"fixed", "improved", "failures"} and not gallery.get("created"):
            record_issue(report, f"gallery {name!r} has no created comparison images")


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
    figure_dir: Path = Path("outputs/da3/experiments/mv_layout_loss_ablation/evidence_figures"),
    summary_json: Path = Path("outputs/da3/experiments/mv_layout_loss_ablation/evidence_summary/summary.json"),
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
    check_pairing(report, layout_root=layout_root, runs=runs, seeds=seeds, ce_run=ce_run)
    sv_records = check_downstream(report, name="SV", path=sv_downstream)
    for item in downstream:
        name, path = parse_named_path(item)
        mv_records = check_downstream(report, name=name, path=path)
        if sv_records and mv_records and set(sv_records) != set(mv_records):
            record_issue(
                report,
                f"downstream object set for {name} differs from SV: "
                f"missing={len(set(sv_records) - set(mv_records))} "
                f"extra={len(set(mv_records) - set(sv_records))}",
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
    runs = args.run or ["A_ce", "B_ordinal", "C_coord", "D_geometry"]
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
