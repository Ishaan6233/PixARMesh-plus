#!/usr/bin/env python3
"""Build plots and fixed comparison galleries for MV layout-loss evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from scripts.eval.summarize_mv_layout_evidence import (
    HIGHER_IS_BETTER,
    LAYOUT_METRICS,
    layout_seed_dir,
    read_json,
    read_jsonl,
)


LOWER_IS_BETTER = set(LAYOUT_METRICS) - set(HIGHER_IS_BETTER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        default="outputs/da3/experiments/mv_layout_loss_ablation/evidence_summary/summary.json",
    )
    parser.add_argument("--layout-root", default="outputs/da3/eval/layout_mv")
    parser.add_argument("--run", action="append", default=[], help="Layout run name, repeatable.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--ce-run", default="A_ce")
    parser.add_argument("--baseline-run", default="A_ce")
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--rank-metric", default="aabb_iou", choices=LAYOUT_METRICS)
    parser.add_argument("--max-gallery-cases", type=int, default=12)
    parser.add_argument(
        "--image-name",
        action="append",
        default=["topdown_bbox.png", "topdown_voxels_bbox.png"],
        help="Visual image filename to combine from each case directory. Repeatable.",
    )
    parser.add_argument(
        "--out",
        default="outputs/da3/experiments/mv_layout_loss_ablation/evidence_figures",
    )
    return parser.parse_args()


def per_uid(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["uid"]): record for record in records if "uid" in record}


def metric_improvement(candidate: float, baseline: float, metric: str) -> float:
    if metric in HIGHER_IS_BETTER:
        return candidate - baseline
    return baseline - candidate


def seed_records(layout_root: Path, run: str, seed: int) -> dict[str, dict[str, Any]]:
    return per_uid(read_jsonl(layout_seed_dir(layout_root, run, seed) / "per_sample.jsonl"))


def ranked_uid_scores(
    *,
    layout_root: Path,
    baseline_run: str,
    candidate_run: str,
    seeds: list[int],
    metric: str,
) -> list[dict[str, Any]]:
    by_uid: dict[str, list[dict[str, Any]]] = {}
    for seed in seeds:
        baseline = seed_records(layout_root, baseline_run, seed)
        candidate = seed_records(layout_root, candidate_run, seed)
        for uid in sorted(set(baseline) & set(candidate)):
            if metric not in baseline[uid] or metric not in candidate[uid]:
                continue
            score = metric_improvement(float(candidate[uid][metric]), float(baseline[uid][metric]), metric)
            by_uid.setdefault(uid, []).append(
                {
                    "seed": seed,
                    "baseline": float(baseline[uid][metric]),
                    "candidate": float(candidate[uid][metric]),
                    "improvement": score,
                }
            )
    ranked = []
    for uid, values in by_uid.items():
        ranked.append(
            {
                "uid": uid,
                "mean_improvement": sum(item["improvement"] for item in values) / len(values),
                "seeds": values,
            }
        )
    return sorted(ranked, key=lambda item: item["mean_improvement"], reverse=True)


def write_uid_list(path: Path, items: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(str(item["uid"]) for item in items) + ("\n" if items else ""))


def plot_per_seed_metrics(summary: dict[str, Any], out_path: Path) -> None:
    runs = summary.get("layout", {}).get("runs", {})
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), squeeze=False)
    for ax, metric in zip(axes.flatten(), LAYOUT_METRICS):
        for run, run_info in runs.items():
            xs = []
            ys = []
            for seed, seed_info in sorted(run_info.get("seeds", {}).items(), key=lambda item: int(item[0])):
                records = read_jsonl(Path(seed_info["per_sample"]))
                values = [float(record[metric]) for record in records if metric in record]
                if values:
                    xs.append(int(seed))
                    ys.append(sum(values) / len(values))
            if xs:
                ax.plot(xs, ys, marker="o", label=run)
        ax.set_title(metric)
        ax.set_xlabel("seed")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_paired_deltas(summary: dict[str, Any], out_path: Path) -> None:
    runs = summary.get("layout", {}).get("runs", {})
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), squeeze=False)
    for ax, metric in zip(axes.flatten(), LAYOUT_METRICS):
        labels = []
        means = []
        cis = []
        for run, run_info in runs.items():
            delta = run_info.get("paired_delta_vs_ce", {}).get(metric, {})
            if delta.get("mean") is None:
                continue
            labels.append(run)
            means.append(float(delta["mean"]))
            cis.append(float(delta["ci95"]) if delta.get("ci95") is not None else 0.0)
        if labels:
            ax.bar(labels, means, yerr=cis, capsize=3)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.tick_params(axis="x", rotation=35)
        ax.set_title(f"{metric} paired delta vs CE")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def visual_case_dir(layout_root: Path, run: str, seed: int, uid: str) -> Path:
    return layout_seed_dir(layout_root, run, seed) / "visuals" / uid.replace("/", "_")


def combine_case_image(
    *,
    layout_root: Path,
    baseline_run: str,
    candidate_run: str,
    seed: int,
    uid: str,
    image_name: str,
    out_path: Path,
) -> bool:
    baseline_path = visual_case_dir(layout_root, baseline_run, seed, uid) / image_name
    candidate_path = visual_case_dir(layout_root, candidate_run, seed, uid) / image_name
    if not baseline_path.exists() or not candidate_path.exists():
        return False
    baseline_img = mpimg.imread(baseline_path)
    candidate_img = mpimg.imread(candidate_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, image, title in (
        (axes[0], baseline_img, f"{baseline_run} seed{seed}"),
        (axes[1], candidate_img, f"{candidate_run} seed{seed}"),
    ):
        ax.imshow(image)
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle(f"{uid} - {image_name}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def write_gallery(
    *,
    layout_root: Path,
    baseline_run: str,
    candidate_run: str,
    selected: list[dict[str, Any]],
    image_names: list[str],
    out_dir: Path,
    gallery_name: str,
) -> dict[str, Any]:
    gallery_dir = out_dir / gallery_name
    created = []
    missing = []
    for item in selected:
        uid = str(item["uid"])
        seeds = [int(seed_info["seed"]) for seed_info in item.get("seeds", [])]
        if not seeds:
            missing.append({"uid": uid, "reason": "no paired seed records"})
            continue
        seed = seeds[0]
        for image_name in image_names:
            out_name = f"seed{seed}_{uid.replace('/', '_')}_{image_name}"
            ok = combine_case_image(
                layout_root=layout_root,
                baseline_run=baseline_run,
                candidate_run=candidate_run,
                seed=seed,
                uid=uid,
                image_name=image_name,
                out_path=gallery_dir / out_name,
            )
            if ok:
                created.append({"uid": uid, "seed": seed, "image_name": image_name, "path": str(gallery_dir / out_name)})
            else:
                missing.append({"uid": uid, "seed": seed, "image_name": image_name})
    return {"name": gallery_name, "created": created, "missing": missing}


def build_figures(
    *,
    summary_json: Path,
    layout_root: Path,
    runs: list[str],
    seeds: list[int],
    ce_run: str,
    baseline_run: str,
    candidate_run: str,
    rank_metric: str,
    max_gallery_cases: int,
    image_names: list[str],
    out_dir: Path,
) -> dict[str, Any]:
    summary = read_json(summary_json)
    if summary is None:
        summary = {
            "layout": {
                "runs": {},
                "missing": [str(summary_json)],
            },
            "seeds": seeds,
            "ce_run": ce_run,
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_per_seed_metrics(summary, out_dir / "per_seed_metrics.png")
    plot_paired_deltas(summary, out_dir / "paired_delta_vs_ce.png")

    ranked = ranked_uid_scores(
        layout_root=layout_root,
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        seeds=seeds,
        metric=rank_metric,
    )
    fixed = ranked[:max_gallery_cases]
    improved = [item for item in ranked if item["mean_improvement"] > 0][:max_gallery_cases]
    regressed = list(reversed([item for item in ranked if item["mean_improvement"] < 0]))[:max_gallery_cases]
    write_uid_list(out_dir / "fixed_uids.txt", fixed)
    write_uid_list(out_dir / "improved_uids.txt", improved)
    write_uid_list(out_dir / "regressed_uids.txt", regressed)
    (out_dir / "ranked_uids.json").write_text(json.dumps(ranked, indent=2) + "\n")

    galleries = [
        write_gallery(
            layout_root=layout_root,
            baseline_run=baseline_run,
            candidate_run=candidate_run,
            selected=items,
            image_names=image_names,
            out_dir=out_dir,
            gallery_name=name,
        )
        for name, items in (
            ("fixed", fixed),
            ("improved", improved),
            ("regressed", regressed),
        )
    ]
    manifest = {
        "summary_json": str(summary_json),
        "layout_root": str(layout_root),
        "runs": runs,
        "seeds": seeds,
        "ce_run": ce_run,
        "baseline_run": baseline_run,
        "candidate_run": candidate_run,
        "rank_metric": rank_metric,
        "plots": {
            "per_seed_metrics": str(out_dir / "per_seed_metrics.png"),
            "paired_delta_vs_ce": str(out_dir / "paired_delta_vs_ce.png"),
        },
        "uid_lists": {
            "fixed": str(out_dir / "fixed_uids.txt"),
            "improved": str(out_dir / "improved_uids.txt"),
            "regressed": str(out_dir / "regressed_uids.txt"),
        },
        "galleries": galleries,
        "created_composites": sum(len(gallery["created"]) for gallery in galleries),
    }
    (out_dir / "gallery_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    args = parse_args()
    runs = args.run or ["A_ce", "B_ordinal", "C_coord", "D_geometry"]
    manifest = build_figures(
        summary_json=Path(args.summary_json),
        layout_root=Path(args.layout_root),
        runs=runs,
        seeds=args.seeds,
        ce_run=args.ce_run,
        baseline_run=args.baseline_run,
        candidate_run=args.candidate_run,
        rank_metric=args.rank_metric,
        max_gallery_cases=args.max_gallery_cases,
        image_names=args.image_name,
        out_dir=Path(args.out),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
