#!/usr/bin/env python3
"""Summarize MV layout-loss ablations and downstream SV/MV evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

LAYOUT_METRICS = [
    "token_accuracy",
    "valid_token_frac",
    "bin_mae",
    "corner_l1",
    "corner_l2",
    "center_error",
    "size_rel_error",
    "aabb_iou",
]

HIGHER_IS_BETTER = {
    "token_accuracy",
    "valid_token_frac",
    "aabb_iou",
    "avg_f_score",
}

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

UID_KEYS = ("uid", "image_id", "scene_id", "sha256", "model_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-root", default="outputs/da3/eval/layout_mv")
    parser.add_argument("--run", action="append", default=[], help="Layout run name, repeatable.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--ce-run", default="A_ce")
    parser.add_argument(
        "--downstream",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Downstream eval_obj_results.jsonl or JSON summary for a run. Repeatable.",
    )
    parser.add_argument("--sv-downstream", default="", help="SV baseline eval_obj_results.jsonl/JSON.")
    parser.add_argument(
        "--uid-metadata",
        default="",
        help=(
            "Optional JSON/JSONL/CSV mapping from uid/image_id to category metadata. "
            "When provided, summaries include category-paired CE deltas."
        ),
    )
    parser.add_argument("--out", default="outputs/da3/experiments/mv_layout_loss_ablation/evidence_summary")
    return parser.parse_args()


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


def _stringify_metadata(record: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in record.items() if value is not None}


def _metadata_uid(record: dict[str, Any]) -> str | None:
    for key in UID_KEYS:
        value = record.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return None


def load_uid_metadata(path: str | Path) -> dict[str, dict[str, str]]:
    """Load optional UID metadata for category-stratified evidence summaries."""
    path = Path(path)
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        rows = read_jsonl(path)
    elif path.suffix == ".csv":
        rows = read_csv(path)
    else:
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            if "records" in payload and isinstance(payload["records"], list):
                rows = payload["records"]
            elif all(isinstance(value, dict) for value in payload.values()):
                rows = [{"uid": key, **value} for key, value in payload.items()]
            else:
                rows = [payload]
        else:
            rows = []

    out: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        meta = _stringify_metadata(row)
        uid = _metadata_uid(meta)
        if uid is not None:
            out[uid] = meta
    return out


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def ci95(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    m = mean(values)
    assert m is not None
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return float(1.96 * math.sqrt(var) / math.sqrt(len(values)))


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": mean(values),
        "ci95": ci95(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def layout_seed_dir(layout_root: Path, run: str, seed: int) -> Path:
    return layout_root / run / f"seed{seed}"


def per_uid(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["uid"]): record for record in records if "uid" in record}


def _category_from_mapping(uid: str, uid_metadata: dict[str, dict[str, str]]) -> str | None:
    candidates = [uid]
    if "::" in uid:
        candidates.append(uid.split("::", 1)[0])
    if "/" in uid:
        candidates.append(uid.rsplit("/", 1)[-1])
    for candidate in candidates:
        meta = uid_metadata.get(candidate)
        if not meta:
            continue
        for key in CATEGORY_KEYS:
            value = meta.get(key)
            if value is not None and str(value) != "":
                return str(value)
    return None


def record_category(record: dict[str, Any], uid_metadata: dict[str, dict[str, str]]) -> str | None:
    for key in CATEGORY_KEYS:
        value = record.get(key)
        if value is not None and str(value) != "":
            return str(value)
    uid = record.get("uid")
    if uid is None:
        return None
    return _category_from_mapping(str(uid), uid_metadata)


def short_list(values: set[str], limit: int = 20) -> list[str]:
    return sorted(values)[:limit]


def _category_bucket() -> dict[str, list[Any]]:
    return {metric: [] for metric in LAYOUT_METRICS}


def layout_summary(
    layout_root: Path,
    runs: list[str],
    seeds: list[int],
    ce_run: str,
    uid_metadata: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    uid_metadata = uid_metadata or {}
    out: dict[str, Any] = {
        "runs": {},
        "missing": [],
        "uid_metadata_count": len(uid_metadata),
    }
    ce_records_by_seed = {
        seed: per_uid(read_jsonl(layout_seed_dir(layout_root, ce_run, seed) / "per_sample.jsonl"))
        for seed in seeds
    }

    for run in runs:
        run_summary: dict[str, Any] = {"seeds": {}, "metrics": {}, "paired_delta_vs_ce": {}}
        per_seed_metric_values: dict[str, list[float]] = {metric: [] for metric in LAYOUT_METRICS}
        per_seed_paired_delta: dict[str, list[float]] = {metric: [] for metric in LAYOUT_METRICS}
        per_seed_improved: dict[str, list[bool]] = {metric: [] for metric in LAYOUT_METRICS}
        per_category_delta: dict[str, dict[str, list[float]]] = {}
        per_category_improved: dict[str, dict[str, list[bool]]] = {}
        per_category_record_count: dict[str, int] = {}
        missing_category_count = 0

        for seed in seeds:
            seed_dir = layout_seed_dir(layout_root, run, seed)
            report = read_json(seed_dir / "report.json")
            records = read_jsonl(seed_dir / "per_sample.jsonl")
            if report is None or not records:
                out["missing"].append(str(seed_dir))
                continue
            run_summary["seeds"][str(seed)] = {
                "report": str(seed_dir / "report.json"),
                "per_sample": str(seed_dir / "per_sample.jsonl"),
                "num_records": len(records),
                "visual_count": len(list((seed_dir / "visuals").glob("*"))) if (seed_dir / "visuals").exists() else 0,
            }
            ce_records = ce_records_by_seed.get(seed, {})
            run_records = per_uid(records)
            ce_uids = set(ce_records)
            run_uids = set(run_records)
            shared = sorted(ce_uids & run_uids)
            run_summary["seeds"][str(seed)].update(
                {
                    "ce_uid_count": len(ce_uids),
                    "run_uid_count": len(run_uids),
                    "paired_uid_count": len(shared),
                    "exact_uid_match": run_uids == ce_uids,
                    "missing_vs_ce": short_list(ce_uids - run_uids),
                    "extra_vs_ce": short_list(run_uids - ce_uids),
                }
            )
            if run != ce_run and run_uids != ce_uids:
                out["missing"].append(
                    f"{run}/seed{seed} UID mismatch vs {ce_run}: "
                    f"missing={len(ce_uids - run_uids)} extra={len(run_uids - ce_uids)}"
                )
            for metric in LAYOUT_METRICS:
                vals = [float(record[metric]) for record in records if metric in record]
                seed_mean = mean(vals)
                if seed_mean is not None:
                    per_seed_metric_values[metric].append(seed_mean)

            if shared:
                category_seed_values: dict[str, dict[str, list[float]]] = {}
                for metric in LAYOUT_METRICS:
                    deltas = [
                        float(run_records[uid][metric]) - float(ce_records[uid][metric])
                        for uid in shared
                        if metric in run_records[uid] and metric in ce_records[uid]
                    ]
                    delta_mean = mean(deltas)
                    if delta_mean is None:
                        continue
                    per_seed_paired_delta[metric].append(delta_mean)
                    improved = delta_mean > 0 if metric in HIGHER_IS_BETTER else delta_mean < 0
                    per_seed_improved[metric].append(improved)
                if uid_metadata:
                    for uid in shared:
                        category = record_category(run_records[uid], uid_metadata) or record_category(
                            ce_records[uid], uid_metadata
                        )
                        if category is None:
                            missing_category_count += 1
                            continue
                        per_category_record_count[category] = per_category_record_count.get(category, 0) + 1
                        category_seed_values.setdefault(category, _category_bucket())
                        for metric in LAYOUT_METRICS:
                            if metric in run_records[uid] and metric in ce_records[uid]:
                                category_seed_values[category][metric].append(
                                    float(run_records[uid][metric]) - float(ce_records[uid][metric])
                                )
                    for category, metric_values in category_seed_values.items():
                        per_category_delta.setdefault(category, _category_bucket())
                        per_category_improved.setdefault(category, {metric: [] for metric in LAYOUT_METRICS})
                        for metric, values in metric_values.items():
                            delta_mean = mean(values)
                            if delta_mean is None:
                                continue
                            per_category_delta[category][metric].append(delta_mean)
                            improved = delta_mean > 0 if metric in HIGHER_IS_BETTER else delta_mean < 0
                            per_category_improved[category][metric].append(improved)

        for metric, values in per_seed_metric_values.items():
            run_summary["metrics"][metric] = summarize_values(values)
        for metric, values in per_seed_paired_delta.items():
            run_summary["paired_delta_vs_ce"][metric] = {
                **summarize_values(values),
                "improved_seed_count": int(sum(per_seed_improved[metric])),
            }
        if uid_metadata:
            run_summary["category_record_count"] = per_category_record_count
            run_summary["missing_category_count"] = missing_category_count
            run_summary["category_paired_delta_vs_ce"] = {}
            for category, metric_values in sorted(per_category_delta.items()):
                run_summary["category_paired_delta_vs_ce"][category] = {}
                for metric, values in metric_values.items():
                    run_summary["category_paired_delta_vs_ce"][category][metric] = {
                        **summarize_values(values),
                        "improved_seed_count": int(sum(per_category_improved[category][metric])),
                    }
        out["runs"][run] = run_summary
    return out


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name, Path(path)


def read_downstream_summary(path: Path) -> dict[str, Any] | None:
    if not path:
        return None
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


def downstream_object_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path or not path.exists() or path.suffix != ".jsonl":
        return {}
    out = {}
    for record in read_jsonl(path):
        if is_downstream_summary(record):
            continue
        key = downstream_object_key(record)
        if key is None or record.get("cd") is None or record.get("f_score") is None:
            continue
        out[key] = record
    return out


def paired_downstream_comparison(
    run_records: dict[str, dict[str, Any]],
    sv_records: dict[str, dict[str, Any]],
    run_summary: dict[str, Any],
    sv_summary: dict[str, Any],
) -> dict[str, Any]:
    run_ids = set(run_records)
    sv_ids = set(sv_records)
    shared = sorted(run_ids & sv_ids)
    cd_deltas = [
        float(run_records[key]["cd"]) - float(sv_records[key]["cd"])
        for key in shared
    ]
    f_deltas = [
        float(run_records[key]["f_score"]) - float(sv_records[key]["f_score"])
        for key in shared
    ]
    cmp: dict[str, Any] = {
        "object_pairing_available": bool(run_records and sv_records),
        "object_set_equal": run_ids == sv_ids and bool(run_ids),
        "paired_count": len(shared),
        "run_object_count": len(run_ids),
        "sv_object_count": len(sv_ids),
        "missing_vs_sv": short_list(sv_ids - run_ids),
        "extra_vs_sv": short_list(run_ids - sv_ids),
        "avg_cd_delta_vs_sv": mean(cd_deltas),
        "avg_f_score_delta_vs_sv": mean(f_deltas),
        "cd_improved_object_count": int(sum(delta < 0 for delta in cd_deltas)),
        "f_score_improved_object_count": int(sum(delta > 0 for delta in f_deltas)),
    }
    if cmp["avg_cd_delta_vs_sv"] is not None:
        cmp["cd_beats_sv"] = bool(cmp["avg_cd_delta_vs_sv"] < 0)
    if cmp["avg_f_score_delta_vs_sv"] is not None:
        cmp["f_score_beats_sv"] = bool(cmp["avg_f_score_delta_vs_sv"] > 0)
    for metric in ("avg_cd", "avg_f_score", "num_evaluated"):
        if metric in run_summary and metric in sv_summary:
            cmp[f"aggregate_{metric}_delta_vs_sv"] = run_summary[metric] - sv_summary[metric]
    return cmp


def downstream_summary(items: list[str], sv_path: str) -> dict[str, Any]:
    out: dict[str, Any] = {"sv": None, "runs": {}, "comparison_to_sv": {}, "missing": []}
    sv_records = downstream_object_records(Path(sv_path)) if sv_path else {}
    if sv_path:
        sv_path_obj = Path(sv_path)
        sv_summary = read_downstream_summary(sv_path_obj)
        out["sv"] = sv_summary
        if sv_summary is None:
            out["missing"].append(sv_path)
        if not sv_records:
            out["missing"].append(f"{sv_path}: no object-level downstream records for pairing")
    else:
        sv_summary = None

    for item in items:
        name, path = parse_named_path(item)
        summary = read_downstream_summary(path)
        records = downstream_object_records(path)
        out["runs"][name] = summary
        if summary is None:
            out["missing"].append(str(path))
            continue
        if not records:
            out["missing"].append(f"{path}: no object-level downstream records for pairing")
        if sv_summary is None:
            continue
        cmp = paired_downstream_comparison(records, sv_records, summary, sv_summary)
        if not cmp["object_set_equal"]:
            out["missing"].append(
                f"{path}: downstream object set mismatch vs {sv_path} "
                f"missing={len(set(sv_records) - set(records))} "
                f"extra={len(set(records) - set(sv_records))}"
            )
        out["comparison_to_sv"][name] = cmp
    return out


def fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def markdown_report(summary: dict[str, Any]) -> str:
    lines = ["# MV Layout Loss Evidence Summary", ""]
    lines.append("## Layout Metrics")
    lines.append("")
    header = "| Run | Metric | Mean | 95% CI | Paired Δ vs CE | Improved Seeds |"
    lines.extend([header, "|---|---:|---:|---:|---:|---:|"])
    for run, run_summary in summary["layout"]["runs"].items():
        for metric in LAYOUT_METRICS:
            vals = run_summary["metrics"][metric]
            delta = run_summary["paired_delta_vs_ce"][metric]
            lines.append(
                f"| {run} | {metric} | {fmt(vals['mean'])} | {fmt(vals['ci95'])} "
                f"| {fmt(delta['mean'])} | {fmt(delta['improved_seed_count'])}/{fmt(delta['n'])} |"
            )
    category_rows = []
    for run, run_summary in summary["layout"]["runs"].items():
        for category, metrics in run_summary.get("category_paired_delta_vs_ce", {}).items():
            for metric in ("bin_mae", "corner_l1", "corner_l2", "center_error", "size_rel_error", "aabb_iou"):
                delta = metrics.get(metric, {})
                if delta.get("n"):
                    category_rows.append(
                        (
                            run,
                            category,
                            metric,
                            delta.get("mean"),
                            delta.get("ci95"),
                            delta.get("improved_seed_count"),
                            delta.get("n"),
                        )
                    )
    if category_rows:
        lines.append("")
        lines.append("## Category Stability")
        lines.append("")
        lines.extend(
            [
                "| Run | Category | Metric | Paired Δ vs CE | 95% CI | Improved Seed-Category Cells |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for run, category, metric, delta, ci, improved, n in category_rows:
            lines.append(
                f"| {run} | {category} | {metric} | {fmt(delta)} | {fmt(ci)} | {fmt(improved)}/{fmt(n)} |"
            )
    lines.append("")
    lines.append("## Downstream CD/F")
    lines.append("")
    lines.extend(["| Run | avg_cd | avg_f_score | N | ΔCD vs SV | ΔF vs SV |", "|---|---:|---:|---:|---:|---:|"])
    downstream = summary["downstream"]
    for run, vals in downstream["runs"].items():
        cmp = downstream["comparison_to_sv"].get(run, {})
        vals = vals or {}
        lines.append(
            f"| {run} | {fmt(vals.get('avg_cd'))} | {fmt(vals.get('avg_f_score'))} "
            f"| {fmt(vals.get('num_evaluated'))} | {fmt(cmp.get('avg_cd_delta_vs_sv'))} "
            f"| {fmt(cmp.get('avg_f_score_delta_vs_sv'))} |"
        )
    if summary["layout"]["missing"] or downstream["missing"]:
        lines.append("")
        lines.append("## Missing Evidence")
        for path in summary["layout"]["missing"]:
            lines.append(f"- layout: `{path}`")
        for path in downstream["missing"]:
            lines.append(f"- downstream: `{path}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    runs = args.run or ["A_ce", "B_ordinal", "C_coord", "D_geometry"]
    uid_metadata = load_uid_metadata(args.uid_metadata) if args.uid_metadata else {}
    summary = {
        "layout": layout_summary(Path(args.layout_root), runs, args.seeds, args.ce_run, uid_metadata),
        "downstream": downstream_summary(args.downstream, args.sv_downstream),
        "seeds": args.seeds,
        "ce_run": args.ce_run,
        "uid_metadata": args.uid_metadata,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "summary.md").write_text(markdown_report(summary))
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
