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
    (seed_dir / "report.json").write_text(json.dumps({"num_records": 1}))
    with (seed_dir / "per_sample.jsonl").open("w") as f:
        f.write(json.dumps({"uid": uid, **METRICS}) + "\n")
    (visual_dir / "conditioning.json").write_text(
        json.dumps(
            {
                "selection": {"mode": "fixed_uids"},
                "ablation": {"view_limit": 0},
                "projection": {"per_view": [{"view_idx": 0, "view_enabled": True}]},
            }
        )
    )
    (visual_dir / "topdown_bbox.png").write_bytes(b"png")
    (visual_dir / "conditioning_points.npz").write_bytes(b"npz")


def _write_downstream(path):
    path.parent.mkdir(parents=True)
    with path.open("w") as f:
        f.write(json.dumps({"uid": "x", "obj_id": 0, "cd": 0.1, "f_score": 0.2}) + "\n")
        f.write(json.dumps({"avg_cd": 0.1, "avg_f_score": 0.2, "num_evaluated": 1}) + "\n")


def _write_verifiers(root):
    root.mkdir(parents=True)
    for name in (
        "loss_verifier.md",
        "data_verifier.md",
        "experiment_verifier.md",
        "visual_verifier.md",
        "council_review.md",
    ):
        (root / name).write_text("status: pass\n\nFindings\n\nChecked commands and artifacts.\n")


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
    assert any("missing or empty verifier finding" in issue for issue in report["issues"])


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
        f.write(json.dumps({"avg_cd": 0.1, "avg_f_score": 0.2, "num_evaluated": 1}) + "\n")
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
