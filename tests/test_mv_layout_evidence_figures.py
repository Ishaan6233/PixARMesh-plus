import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.eval.build_mv_layout_evidence_figures import build_figures, ranked_uid_scores
from scripts.eval.summarize_mv_layout_evidence import layout_summary


METRICS = {
    "token_accuracy": 1.0,
    "valid_token_frac": 1.0,
    "corner_l1": 0.0,
    "corner_l2": 0.0,
    "center_error": 0.0,
    "size_rel_error": 0.0,
}


def _record(uid: str, *, bin_mae: float, aabb_iou: float) -> dict:
    return {"uid": uid, "bin_mae": bin_mae, "aabb_iou": aabb_iou, **METRICS}


def _write_seed(root, run, seed, records):
    seed_dir = root / run / f"seed{seed}"
    seed_dir.mkdir(parents=True)
    (seed_dir / "report.json").write_text(json.dumps({"num_records": len(records)}))
    with (seed_dir / "per_sample.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    for record in records:
        visual_dir = seed_dir / "visuals" / record["uid"]
        visual_dir.mkdir(parents=True)
        for image_name in ("topdown_bbox.png", "topdown_voxels_bbox.png"):
            plt.imsave(visual_dir / image_name, np.zeros((4, 4, 3), dtype=np.float32))


def test_ranked_uid_scores_orders_improved_before_regressed(tmp_path):
    _write_seed(
        tmp_path,
        "A_ce",
        11,
        [_record("better", bin_mae=5.0, aabb_iou=0.2), _record("worse", bin_mae=2.0, aabb_iou=0.8)],
    )
    _write_seed(
        tmp_path,
        "D_geometry",
        11,
        [_record("better", bin_mae=2.0, aabb_iou=0.6), _record("worse", bin_mae=4.0, aabb_iou=0.4)],
    )

    ranked = ranked_uid_scores(
        layout_root=tmp_path,
        baseline_run="A_ce",
        candidate_run="D_geometry",
        seeds=[11],
        metric="aabb_iou",
    )

    assert [item["uid"] for item in ranked] == ["better", "worse"]
    assert ranked[0]["mean_improvement"] > 0
    assert ranked[1]["mean_improvement"] < 0


def test_build_figures_writes_plots_uid_lists_and_composites(tmp_path):
    layout_root = tmp_path / "layout"
    _write_seed(
        layout_root,
        "A_ce",
        11,
        [_record("better", bin_mae=5.0, aabb_iou=0.2), _record("worse", bin_mae=2.0, aabb_iou=0.8)],
    )
    _write_seed(
        layout_root,
        "D_geometry",
        11,
        [_record("better", bin_mae=2.0, aabb_iou=0.6), _record("worse", bin_mae=4.0, aabb_iou=0.4)],
    )
    summary = {
        "layout": layout_summary(layout_root, ["A_ce", "D_geometry"], [11], "A_ce"),
        "downstream": {"runs": {}, "comparison_to_sv": {}, "missing": []},
        "seeds": [11],
        "ce_run": "A_ce",
    }
    summary_json = tmp_path / "summary.json"
    summary_json.write_text(json.dumps(summary))

    manifest = build_figures(
        summary_json=summary_json,
        layout_root=layout_root,
        runs=["A_ce", "D_geometry"],
        seeds=[11],
        ce_run="A_ce",
        baseline_run="A_ce",
        candidate_run="D_geometry",
        rank_metric="aabb_iou",
        max_gallery_cases=2,
        image_names=["topdown_bbox.png"],
        out_dir=tmp_path / "figures",
    )

    assert (tmp_path / "figures" / "per_seed_metrics.png").exists()
    assert (tmp_path / "figures" / "paired_delta_vs_ce.png").exists()
    assert (tmp_path / "figures" / "fixed_uids.txt").read_text().splitlines() == ["better", "worse"]
    assert (tmp_path / "figures" / "improved_uids.txt").read_text().splitlines() == ["better"]
    assert (tmp_path / "figures" / "regressed_uids.txt").read_text().splitlines() == ["worse"]
    assert manifest["created_composites"] >= 3
    assert (tmp_path / "figures" / "gallery_manifest.json").exists()
