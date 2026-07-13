import json

from scripts.eval.check_mv_layout_evidence_bundle import build_evidence_report


METRICS = {
    "token_accuracy": 1.0,
    "valid_token_frac": 1.0,
    "bin_mae": 0.0,
    "corner_l1": 0.0,
    "corner_l2": 0.0,
    "center_error": 0.0,
    "size_rel_error": 0.0,
    "aabb_iou": 1.0,
}


def _write_layout(root, run, seed, uid="uid-a"):
    seed_dir = root / run / f"seed{seed}"
    visual_dir = seed_dir / "visuals" / uid
    visual_dir.mkdir(parents=True)
    view_usage = {
        "num_records_with_view_usage": 1,
        "mean_input_valid_view_count": 3.0,
        "mean_enabled_view_count": 3.0,
        "min_enabled_view_count": 3,
        "max_enabled_view_count": 3,
        "multi_view_record_frac": 1.0,
        "single_view_record_frac": 0.0,
        "zero_view_record_frac": 0.0,
        "mean_enabled_view_fraction": 0.375,
    }
    (seed_dir / "report.json").write_text(
        json.dumps(
            {
                "checkpoint": f"checkpoints/{run}/seed{seed}",
                "config_name": "edgerunner_3d_front_trellis2_mv_stage1",
                "split": "val",
                "requested_num_samples": 200,
                "batch_size": 1,
                "mv_feature_cache": "datasets/mv-feature-cache/da3/trellis2-mv",
                "device": "cuda",
                "num_records": 1,
                "validity": {
                    "num_valid": 1,
                    "num_invalid": 0,
                    "valid_sample_frac": 1.0,
                    "invalid_reasons": {},
                },
                "view_usage": view_usage,
                "view_limit": 0,
                "reference_only": False,
                "shuffle_views": False,
                "overrides": [],
            }
        )
    )
    with (seed_dir / "per_sample.jsonl").open("w") as f:
        f.write(
            json.dumps(
                {
                    "uid": uid,
                    "valid": True,
                    "invalid_reason": None,
                    "layout_token_count": 24,
                    "expected_layout_tokens": 24,
                    "view_usage": {
                        "input_valid_view_count": 3,
                        "enabled_view_count": 3,
                        "num_view_slots": 8,
                        "enabled_view_fraction": 0.375,
                        "ref_view": 0,
                    },
                    "input_valid_view_count": 3,
                    "enabled_view_count": 3,
                    "num_view_slots": 8,
                    "enabled_view_fraction": 0.375,
                    "ref_view": 0,
                    **METRICS,
                }
            )
            + "\n"
        )
    (visual_dir / "conditioning.json").write_text(
        json.dumps(
            {
                "selection": {"mode": "fixed_uids"},
                "ablation": {"view_limit": 0},
                "projection": {"per_view": [{"view_idx": 0, "view_enabled": True}]},
                "view_mask": [True],
                "ref_view": 0,
                "obj_aabb": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            }
        )
    )
    (visual_dir / "topdown_bbox.png").write_bytes(b"png")
    (visual_dir / "view00_projection.png").write_bytes(b"png")
    (visual_dir / "conditioning_points.npz").write_bytes(b"npz")


def _write_downstream(path, *, cd=None, f_score=None):
    is_sv = "sv" in {part.lower() for part in path.parts}
    if cd is None:
        cd = 0.2 if is_sv else 0.1
    if f_score is None:
        f_score = 0.1 if is_sv else 0.2
    path.parent.mkdir(parents=True)
    with path.open("w") as f:
        f.write(
            json.dumps({"uid": "x", "obj_id": 0, "cd": cd, "f_score": f_score}) + "\n"
        )
        f.write(
            json.dumps({"avg_cd": cd, "avg_f_score": f_score, "num_evaluated": 1})
            + "\n"
        )


def _write_verifiers(root):
    root.mkdir(parents=True)
    texts = {
        "loss_verifier.md": (
            "status: pass\n\nChecked commands and artifacts.\n"
            "Reviewed src/models/loss.py and src/models/edgerunner.py for gradient flow and token offsets.\n"
        ),
        "data_verifier.md": (
            "status: pass\n\nChecked commands and artifacts.\n"
            "Reviewed src/data/trellis2_mv.py and src/data/collator.py for view mask handling and GT leakage.\n"
        ),
        "experiment_verifier.md": (
            "status: pass\n\nChecked commands and artifacts.\n"
            "Reviewed summary.json, per_sample.jsonl, eval_obj_results.jsonl, and paired UID/object metrics.\n"
        ),
        "visual_verifier.md": (
            "status: pass\n\nChecked commands and artifacts.\n"
            "Reviewed gallery_manifest.json, conditioning.json, projection images, and failure galleries.\n"
        ),
    }
    for name, text in texts.items():
        (root / name).write_text(text)
    (root / "council_review.md").write_text(
        "status: pass\nrecommendation: merge\n\nFindings\n\nChecked commands and artifacts.\n"
    )


def _write_figures(root):
    root.mkdir(parents=True)
    for name in ("per_seed_metrics.png", "paired_delta_vs_ce.png"):
        (root / name).write_bytes(b"png")
    for name in (
        "fixed_uids.txt",
        "improved_uids.txt",
        "regressed_uids.txt",
        "failure_uids.txt",
    ):
        (root / name).write_text("uid-a\n")
    (root / "gallery_uids.txt").write_text("uid-a\n")
    (root / "ranked_uids.json").write_text(json.dumps([{"uid": "uid-a"}]))
    (root / "ranked_failures.json").write_text(json.dumps([{"uid": "uid-a"}]))
    (root / "gallery_manifest.json").write_text(
        json.dumps(
            {
                "created_composites": 3,
                "galleries": [
                    {
                        "name": "fixed",
                        "created": [{"path": "fixed.png"}],
                        "missing": [],
                    },
                    {
                        "name": "improved",
                        "created": [{"path": "improved.png"}],
                        "missing": [],
                    },
                    {
                        "name": "regressed",
                        "created": [{"path": "regressed.png"}],
                        "missing": [],
                    },
                    {
                        "name": "failures",
                        "created": [{"path": "failures.png"}],
                        "missing": [],
                    },
                ],
            }
        )
    )


def _write_category_summary(path, *, stable=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    good = {}
    bad = {}
    for metric in (
        "bin_mae",
        "corner_l1",
        "corner_l2",
        "center_error",
        "size_rel_error",
    ):
        good[metric] = {"mean": -0.1, "n": 1, "improved_seed_count": 1}
        bad[metric] = {
            "mean": 0.1 if not stable else -0.1,
            "n": 1,
            "improved_seed_count": 0 if not stable else 1,
        }
    good["aabb_iou"] = {"mean": 0.1, "n": 1, "improved_seed_count": 1}
    bad["aabb_iou"] = {
        "mean": -0.1 if not stable else 0.1,
        "n": 1,
        "improved_seed_count": 0 if not stable else 1,
    }
    path.write_text(
        json.dumps(
            {
                "layout": {
                    "runs": {
                        "B_ordinal": {
                            "category_paired_delta_vs_ce": {
                                "chair": good,
                                "table": bad,
                            },
                            "missing_category_count": 0,
                        }
                    }
                }
            }
        )
    )


def test_evidence_bundle_checker_accepts_complete_bundle(tmp_path):
    layout_root = tmp_path / "layout"
    for run in ("A_ce", "B_ordinal"):
        _write_layout(layout_root, run, 11)
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv = tmp_path / "mv" / "eval_obj_results.jsonl"
    _write_downstream(sv)
    _write_downstream(mv)
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)

    report = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        require_visuals=True,
    )

    assert report["ok"]
    assert report["issues"] == []


def test_evidence_bundle_checker_requires_downstream_file_per_requested_seed(tmp_path):
    layout_root = tmp_path / "layout"
    for run in ("A_ce", "B_ordinal"):
        for seed in (11, 23):
            _write_layout(layout_root, run, seed)
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv11 = tmp_path / "mv" / "seed11" / "eval_obj_results.jsonl"
    mv23 = tmp_path / "mv" / "seed23" / "eval_obj_results.jsonl"
    _write_downstream(sv)
    _write_downstream(mv11)
    _write_downstream(mv23)
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)

    incomplete = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11, 23],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv11}"],
        verifier_dir=verifier_dir,
        require_visuals=True,
    )
    assert not incomplete["ok"]
    assert any(
        "has 1 files but 2 seeds were requested" in issue
        for issue in incomplete["issues"]
    )

    complete = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11, 23],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv11}", f"MV={mv23}"],
        verifier_dir=verifier_dir,
        require_visuals=True,
    )
    assert complete["ok"]
    assert complete["downstream_groups"]["MV"]["count"] == 2


def test_evidence_bundle_checker_rejects_duplicate_downstream_seed_labels(tmp_path):
    layout_root = tmp_path / "layout"
    for run in ("A_ce", "B_ordinal"):
        for seed in (11, 23):
            _write_layout(layout_root, run, seed)
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv11 = tmp_path / "mv" / "seed11" / "eval_obj_results.jsonl"
    _write_downstream(sv)
    _write_downstream(mv11)
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)

    report = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11, 23],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv11}", f"MV={mv11}"],
        verifier_dir=verifier_dir,
        require_visuals=True,
    )

    assert not report["ok"]
    assert report["downstream_groups"]["MV"]["duplicate_seed_labels"] == ["seed11"]
    assert any("duplicate seed labels" in issue for issue in report["issues"])
    assert any(
        "missing expected seed labels ['seed23']" in issue for issue in report["issues"]
    )


def test_evidence_bundle_checker_rejects_downstream_seed_that_does_not_beat_sv(
    tmp_path,
):
    layout_root = tmp_path / "layout"
    for run in ("A_ce", "B_ordinal"):
        _write_layout(layout_root, run, 11)
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv = tmp_path / "mv" / "seed11" / "eval_obj_results.jsonl"
    _write_downstream(sv, cd=0.2, f_score=0.2)
    _write_downstream(mv, cd=0.25, f_score=0.1)
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)

    report = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        require_visuals=True,
    )

    assert not report["ok"]
    assert any("does not beat SV on paired CD" in issue for issue in report["issues"])
    assert any(
        "does not beat SV on paired F-score" in issue for issue in report["issues"]
    )


def test_evidence_bundle_checker_reports_missing_artifacts(tmp_path):
    report = build_evidence_report(
        layout_root=tmp_path / "layout",
        runs=["A_ce"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=tmp_path / "missing_sv.jsonl",
        downstream=[],
        verifier_dir=tmp_path / "missing_verifiers",
        require_visuals=True,
    )

    assert not report["ok"]
    assert any("missing layout report" in issue for issue in report["issues"])
    assert any("missing downstream summary" in issue for issue in report["issues"])
    assert any(
        "missing or empty verifier finding" in issue for issue in report["issues"]
    )


def test_evidence_bundle_checker_rejects_partial_layout_uid_overlap(tmp_path):
    layout_root = tmp_path / "layout"
    _write_layout(layout_root, "A_ce", 11, uid="uid-a")
    _write_layout(layout_root, "B_ordinal", 11, uid="uid-b")
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv = tmp_path / "mv" / "eval_obj_results.jsonl"
    _write_downstream(sv)
    _write_downstream(mv)
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)

    report = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        require_visuals=True,
    )

    assert not report["ok"]
    assert any("UID set differs" in issue for issue in report["issues"])


def test_evidence_bundle_checker_rejects_layout_eval_identity_mismatch(tmp_path):
    layout_root = tmp_path / "layout"
    for run in ("A_ce", "B_ordinal"):
        _write_layout(layout_root, run, 11)
    report_path = layout_root / "B_ordinal" / "seed11" / "report.json"
    report_json = json.loads(report_path.read_text())
    report_json["mv_feature_cache"] = "datasets/other-cache"
    report_path.write_text(json.dumps(report_json))
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv = tmp_path / "mv" / "eval_obj_results.jsonl"
    _write_downstream(sv)
    _write_downstream(mv)
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)

    report = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        require_visuals=True,
    )

    assert not report["ok"]
    assert any(
        "layout eval identity mv_feature_cache" in issue for issue in report["issues"]
    )


def test_evidence_bundle_checker_rejects_main_layout_view_control(tmp_path):
    layout_root = tmp_path / "layout"
    for run in ("A_ce", "B_ordinal"):
        _write_layout(layout_root, run, 11)
    report_path = layout_root / "B_ordinal" / "seed11" / "report.json"
    report_json = json.loads(report_path.read_text())
    report_json["view_limit"] = 1
    report_path.write_text(json.dumps(report_json))
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv = tmp_path / "mv" / "eval_obj_results.jsonl"
    _write_downstream(sv)
    _write_downstream(mv)
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)

    report = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        require_visuals=True,
    )

    assert not report["ok"]
    assert any("unexpectedly used view_limit=1" in issue for issue in report["issues"])


def test_evidence_bundle_checker_requires_view_usage_summary(tmp_path):
    layout_root = tmp_path / "layout"
    for run in ("A_ce", "B_ordinal"):
        _write_layout(layout_root, run, 11)
    report_path = layout_root / "B_ordinal" / "seed11" / "report.json"
    report_json = json.loads(report_path.read_text())
    report_json.pop("view_usage")
    report_path.write_text(json.dumps(report_json))
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv = tmp_path / "mv" / "eval_obj_results.jsonl"
    _write_downstream(sv)
    _write_downstream(mv)
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)

    report = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        require_visuals=True,
    )

    assert not report["ok"]
    assert any(
        "layout report missing view_usage summary" in issue
        for issue in report["issues"]
    )


def test_evidence_bundle_checker_requires_layout_validity_fields(tmp_path):
    layout_root = tmp_path / "layout"
    for run in ("A_ce", "B_ordinal"):
        _write_layout(layout_root, run, 11)
    report_path = layout_root / "B_ordinal" / "seed11" / "report.json"
    report_json = json.loads(report_path.read_text())
    report_json.pop("validity")
    report_path.write_text(json.dumps(report_json))
    per_sample = layout_root / "B_ordinal" / "seed11" / "per_sample.jsonl"
    record = json.loads(per_sample.read_text())
    record.pop("valid")
    per_sample.write_text(json.dumps(record) + "\n")
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv = tmp_path / "mv" / "eval_obj_results.jsonl"
    _write_downstream(sv)
    _write_downstream(mv)
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)

    report = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        require_visuals=True,
    )

    assert not report["ok"]
    assert any("layout report missing validity summary" in issue for issue in report["issues"])
    assert any("missing layout validity fields" in issue for issue in report["issues"])


def test_evidence_bundle_checker_rejects_effectively_single_view_main_eval(tmp_path):
    layout_root = tmp_path / "layout"
    for run in ("A_ce", "B_ordinal"):
        _write_layout(layout_root, run, 11)
    report_path = layout_root / "B_ordinal" / "seed11" / "report.json"
    report_json = json.loads(report_path.read_text())
    report_json["view_usage"]["mean_enabled_view_count"] = 1.0
    report_json["view_usage"]["min_enabled_view_count"] = 1
    report_json["view_usage"]["max_enabled_view_count"] = 1
    report_json["view_usage"]["multi_view_record_frac"] = 0.0
    report_json["view_usage"]["single_view_record_frac"] = 1.0
    report_path.write_text(json.dumps(report_json))
    per_sample = layout_root / "B_ordinal" / "seed11" / "per_sample.jsonl"
    record = json.loads(per_sample.read_text())
    record["enabled_view_count"] = 1
    record["view_usage"]["enabled_view_count"] = 1
    per_sample.write_text(json.dumps(record) + "\n")
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv = tmp_path / "mv" / "eval_obj_results.jsonl"
    _write_downstream(sv)
    _write_downstream(mv)
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)

    report = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        require_visuals=True,
    )

    assert not report["ok"]
    assert any(
        "has no evidence that any evaluated object used multiple views" in issue
        for issue in report["issues"]
    )
    assert any(
        "mean enabled view count is not multi-view" in issue
        for issue in report["issues"]
    )


def test_evidence_bundle_checker_rejects_unpaired_downstream_objects(tmp_path):
    layout_root = tmp_path / "layout"
    for run in ("A_ce", "B_ordinal"):
        _write_layout(layout_root, run, 11)
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv = tmp_path / "mv" / "eval_obj_results.jsonl"
    _write_downstream(sv)
    mv.parent.mkdir(parents=True)
    with mv.open("w") as f:
        f.write(json.dumps({"uid": "y", "obj_id": 0, "cd": 0.1, "f_score": 0.2}) + "\n")
        f.write(
            json.dumps({"avg_cd": 0.1, "avg_f_score": 0.2, "num_evaluated": 1}) + "\n"
        )
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)

    report = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        require_visuals=True,
    )

    assert not report["ok"]
    assert any("downstream object set" in issue for issue in report["issues"])


def test_evidence_bundle_checker_rejects_placeholder_visual_metadata(tmp_path):
    layout_root = tmp_path / "layout"
    for run in ("A_ce", "B_ordinal"):
        _write_layout(layout_root, run, 11)
    visual_dir = layout_root / "B_ordinal" / "seed11" / "visuals" / "uid-a"
    (visual_dir / "conditioning.json").write_text(
        json.dumps({"selection": {}, "ablation": {}, "projection": {}})
    )
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv = tmp_path / "mv" / "eval_obj_results.jsonl"
    _write_downstream(sv)
    _write_downstream(mv)
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)

    report = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        require_visuals=True,
    )

    assert not report["ok"]
    assert any("empty visual metadata" in issue for issue in report["issues"])


def test_evidence_bundle_checker_rejects_missing_view_projection_image(tmp_path):
    layout_root = tmp_path / "layout"
    for run in ("A_ce", "B_ordinal"):
        _write_layout(layout_root, run, 11)
    missing = (
        layout_root
        / "B_ordinal"
        / "seed11"
        / "visuals"
        / "uid-a"
        / "view00_projection.png"
    )
    missing.unlink()
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv = tmp_path / "mv" / "eval_obj_results.jsonl"
    _write_downstream(sv)
    _write_downstream(mv)
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)

    report = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        require_visuals=True,
    )

    assert not report["ok"]
    assert any("missing view00_projection.png" in issue for issue in report["issues"])


def test_evidence_bundle_checker_rejects_unresolved_verifier_markers(tmp_path):
    layout_root = tmp_path / "layout"
    for run in ("A_ce", "B_ordinal"):
        _write_layout(layout_root, run, 11)
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv = tmp_path / "mv" / "eval_obj_results.jsonl"
    _write_downstream(sv)
    _write_downstream(mv)
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)
    (verifier_dir / "visual_verifier.md").write_text(
        "status: pass\n\nBlocker: projection metadata has not been checked.\n\nChecked commands.\n"
    )

    report = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        require_visuals=True,
    )

    assert not report["ok"]
    assert any("unresolved failure markers" in issue for issue in report["issues"])


def test_evidence_bundle_checker_rejects_thin_verifier_finding(tmp_path):
    layout_root = tmp_path / "layout"
    for run in ("A_ce", "B_ordinal"):
        _write_layout(layout_root, run, 11)
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv = tmp_path / "mv" / "eval_obj_results.jsonl"
    _write_downstream(sv)
    _write_downstream(mv)
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)
    (verifier_dir / "loss_verifier.md").write_text(
        "status: pass\n\nChecked commands and artifacts.\n"
    )

    report = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        require_visuals=True,
    )

    assert not report["ok"]
    assert any("role-specific evidence terms" in issue for issue in report["issues"])


def test_evidence_bundle_checker_rejects_council_without_recommendation(tmp_path):
    layout_root = tmp_path / "layout"
    for run in ("A_ce", "B_ordinal"):
        _write_layout(layout_root, run, 11)
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv = tmp_path / "mv" / "eval_obj_results.jsonl"
    _write_downstream(sv)
    _write_downstream(mv)
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)
    (verifier_dir / "council_review.md").write_text(
        "status: pass\n\nFindings\n\nChecked commands and artifacts.\n"
    )

    report = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        require_visuals=True,
    )

    assert not report["ok"]
    assert any(
        "lacks explicit merge/keep-experimental/reject recommendation" in issue
        for issue in report["issues"]
    )


def test_evidence_bundle_checker_rejects_passing_council_without_merge_recommendation(
    tmp_path,
):
    layout_root = tmp_path / "layout"
    for run in ("A_ce", "B_ordinal"):
        _write_layout(layout_root, run, 11)
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv = tmp_path / "mv" / "eval_obj_results.jsonl"
    _write_downstream(sv)
    _write_downstream(mv)
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)
    (verifier_dir / "council_review.md").write_text(
        "status: pass\nrecommendation: reject\n\nFindings\n\nChecked commands and artifacts.\n"
    )

    report = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        require_visuals=True,
    )

    assert not report["ok"]
    assert any(
        "pass status without recommendation: merge" in issue
        for issue in report["issues"]
    )


def test_evidence_bundle_checker_requires_figure_artifacts_when_enabled(tmp_path):
    layout_root = tmp_path / "layout"
    for run in ("A_ce", "B_ordinal"):
        _write_layout(layout_root, run, 11)
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv = tmp_path / "mv" / "eval_obj_results.jsonl"
    _write_downstream(sv)
    _write_downstream(mv)
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)
    figure_dir = tmp_path / "figures"

    missing = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        figure_dir=figure_dir,
        require_visuals=True,
        require_figures=True,
    )
    assert not missing["ok"]
    assert any("missing figure artifact" in issue for issue in missing["issues"])

    _write_figures(figure_dir)
    present = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        figure_dir=figure_dir,
        require_visuals=True,
        require_figures=True,
    )
    assert present["ok"]


def test_evidence_bundle_checker_requires_empty_fixed_improved_justification(tmp_path):
    layout_root = tmp_path / "layout"
    for run in ("A_ce", "B_ordinal"):
        _write_layout(layout_root, run, 11)
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv = tmp_path / "mv" / "eval_obj_results.jsonl"
    _write_downstream(sv)
    _write_downstream(mv)
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)
    figure_dir = tmp_path / "figures"
    _write_figures(figure_dir)
    (figure_dir / "fixed_uids.txt").write_text("")
    (figure_dir / "improved_uids.txt").write_text("")
    manifest_path = figure_dir / "gallery_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("empty_uid_list_justifications", None)
    manifest_path.write_text(json.dumps(manifest))

    missing_justification = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        figure_dir=figure_dir,
        require_visuals=True,
        require_figures=True,
    )

    assert not missing_justification["ok"]
    assert any(
        "gallery 'fixed' has an empty UID list" in issue
        for issue in missing_justification["issues"]
    )
    assert any(
        "gallery 'improved' has an empty UID list" in issue
        for issue in missing_justification["issues"]
    )

    manifest["empty_uid_list_justifications"] = {
        "fixed": "No paired UID had the requested rank metric.",
        "improved": "No ranked UID improved over CE.",
    }
    manifest_path.write_text(json.dumps(manifest))
    justified = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        figure_dir=figure_dir,
        require_visuals=True,
        require_figures=True,
    )

    assert justified["ok"]


def test_evidence_bundle_checker_requires_failure_gallery_even_when_uid_list_empty(
    tmp_path,
):
    layout_root = tmp_path / "layout"
    for run in ("A_ce", "B_ordinal"):
        _write_layout(layout_root, run, 11)
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv = tmp_path / "mv" / "eval_obj_results.jsonl"
    _write_downstream(sv)
    _write_downstream(mv)
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)
    figure_dir = tmp_path / "figures"
    _write_figures(figure_dir)
    (figure_dir / "failure_uids.txt").write_text("")
    (figure_dir / "gallery_uids.txt").write_text("uid-a\n")
    manifest = json.loads((figure_dir / "gallery_manifest.json").read_text())
    for gallery in manifest["galleries"]:
        if gallery["name"] == "failures":
            gallery["created"] = []
    (figure_dir / "gallery_manifest.json").write_text(json.dumps(manifest))

    report = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        figure_dir=figure_dir,
        require_visuals=True,
        require_figures=True,
    )

    assert not report["ok"]
    assert any("failure exemplars are required" in issue for issue in report["issues"])
    assert any(
        "gallery 'failures' has no created comparison images" in issue
        for issue in report["issues"]
    )


def test_evidence_bundle_checker_rejects_missing_gallery_uid_union(tmp_path):
    layout_root = tmp_path / "layout"
    for run in ("A_ce", "B_ordinal"):
        _write_layout(layout_root, run, 11)
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv = tmp_path / "mv" / "eval_obj_results.jsonl"
    _write_downstream(sv)
    _write_downstream(mv)
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)
    figure_dir = tmp_path / "figures"
    _write_figures(figure_dir)
    (figure_dir / "gallery_uids.txt").unlink()

    report = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        figure_dir=figure_dir,
        require_visuals=True,
        require_figures=True,
    )

    assert not report["ok"]
    assert any(
        "missing figure artifact" in issue and "gallery_uids.txt" in issue
        for issue in report["issues"]
    )


def test_evidence_bundle_checker_rejects_empty_regressed_gallery_for_regressed_uids(
    tmp_path,
):
    layout_root = tmp_path / "layout"
    for run in ("A_ce", "B_ordinal"):
        _write_layout(layout_root, run, 11)
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv = tmp_path / "mv" / "eval_obj_results.jsonl"
    _write_downstream(sv)
    _write_downstream(mv)
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)
    figure_dir = tmp_path / "figures"
    _write_figures(figure_dir)
    manifest = json.loads((figure_dir / "gallery_manifest.json").read_text())
    for gallery in manifest["galleries"]:
        if gallery["name"] == "regressed":
            gallery["created"] = []
    (figure_dir / "gallery_manifest.json").write_text(json.dumps(manifest))

    report = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        figure_dir=figure_dir,
        require_visuals=True,
        require_figures=True,
    )

    assert not report["ok"]
    assert any(
        "gallery 'regressed' has no created comparison images" in issue
        for issue in report["issues"]
    )


def test_evidence_bundle_checker_requires_category_stability(tmp_path):
    layout_root = tmp_path / "layout"
    for run in ("A_ce", "B_ordinal"):
        _write_layout(layout_root, run, 11)
    sv = tmp_path / "sv" / "eval_obj_results.jsonl"
    mv = tmp_path / "mv" / "eval_obj_results.jsonl"
    _write_downstream(sv)
    _write_downstream(mv)
    verifier_dir = tmp_path / "verifiers"
    _write_verifiers(verifier_dir)
    summary_json = tmp_path / "summary" / "summary.json"

    _write_category_summary(summary_json, stable=False)
    failing = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        summary_json=summary_json,
        best_layout_run="B_ordinal",
        require_visuals=True,
        require_category_stability=True,
    )
    assert not failing["ok"]
    assert any("category stability" in issue for issue in failing["issues"])

    _write_category_summary(summary_json, stable=True)
    passing = build_evidence_report(
        layout_root=layout_root,
        runs=["A_ce", "B_ordinal"],
        seeds=[11],
        ce_run="A_ce",
        sv_downstream=sv,
        downstream=[f"MV={mv}"],
        verifier_dir=verifier_dir,
        summary_json=summary_json,
        best_layout_run="B_ordinal",
        require_visuals=True,
        require_category_stability=True,
    )
    assert passing["ok"]
