import json

from scripts.eval.write_mv_layout_council_review import (
    evaluate_council,
    read_report_bundle,
    recommendation_for,
    write_markdown,
)


def _summary(
    *,
    downstream_beats_sv: bool = True,
    downstream_seed_count: int = 2,
    downstream_cd_improved_seed_count: int | None = None,
    downstream_f_improved_seed_count: int | None = None,
) -> dict:
    paired_delta_vs_ce = {}
    for metric in ("bin_mae", "corner_l1", "corner_l2", "center_error", "size_rel_error"):
        paired_delta_vs_ce[metric] = {"mean": -0.1, "n": 2, "improved_seed_count": 2}
    paired_delta_vs_ce["aabb_iou"] = {"mean": 0.1, "n": 2, "improved_seed_count": 2}
    if downstream_cd_improved_seed_count is None:
        downstream_cd_improved_seed_count = downstream_seed_count if downstream_beats_sv else 0
    if downstream_f_improved_seed_count is None:
        downstream_f_improved_seed_count = downstream_seed_count if downstream_beats_sv else 0
    return {
        "layout": {
            "runs": {
                "D_geometry": {
                    "paired_delta_vs_ce": paired_delta_vs_ce,
                    "seeds": {
                        "11": {
                            "visual_count": 2,
                            "exact_uid_match": True,
                            "paired_uid_count": 2,
                            "ce_uid_count": 2,
                            "run_uid_count": 2,
                        },
                        "23": {
                            "visual_count": 2,
                            "exact_uid_match": True,
                            "paired_uid_count": 2,
                            "ce_uid_count": 2,
                            "run_uid_count": 2,
                        },
                    },
                }
            }
        },
        "downstream": {
            "comparison_to_sv": {
                "E_stage2_best": {
                    "cd_beats_sv": downstream_beats_sv,
                    "f_score_beats_sv": downstream_beats_sv,
                    "object_set_equal": True,
                    "paired_count": 2,
                    "avg_cd_delta_vs_sv": -0.01 if downstream_beats_sv else 0.01,
                    "avg_f_score_delta_vs_sv": 0.02 if downstream_beats_sv else -0.02,
                    "seed_count": downstream_seed_count,
                    "cd_improved_seed_count": downstream_cd_improved_seed_count,
                    "f_score_improved_seed_count": downstream_f_improved_seed_count,
                }
            }
        },
    }


def _set_run_deltas(summary: dict, run: str, *, benefit: float) -> None:
    paired_delta_vs_ce = {}
    for metric in ("bin_mae", "corner_l1", "corner_l2", "center_error", "size_rel_error"):
        paired_delta_vs_ce[metric] = {"mean": -benefit, "n": 2, "improved_seed_count": 2}
    paired_delta_vs_ce["aabb_iou"] = {"mean": benefit, "n": 2, "improved_seed_count": 2}
    summary["layout"]["runs"][run] = {
        "paired_delta_vs_ce": paired_delta_vs_ce,
        "seeds": summary["layout"]["runs"]["D_geometry"]["seeds"],
    }


def _category_metrics(*, stable: bool = True) -> dict:
    metrics = {}
    for metric in ("bin_mae", "corner_l1", "corner_l2", "center_error", "size_rel_error"):
        metrics[metric] = {"mean": -0.1 if stable else 0.1, "n": 2, "improved_seed_count": 2 if stable else 0}
    metrics["aabb_iou"] = {"mean": 0.1 if stable else -0.1, "n": 2, "improved_seed_count": 2 if stable else 0}
    return metrics


def _summary_with_categories(*, stable: bool = True) -> dict:
    summary = _summary()
    summary["layout"]["runs"]["D_geometry"]["category_paired_delta_vs_ce"] = {
        "chair": _category_metrics(stable=True),
        "table": _category_metrics(stable=stable),
    }
    summary["layout"]["runs"]["D_geometry"]["missing_category_count"] = 0
    return summary


def _report(*, bin_mae: float, aabb_iou: float, uids: list[str] | None = None, **extra) -> dict:
    report = {
        "summary": {
            "bin_mae": {"mean": bin_mae},
            "aabb_iou": {"mean": aabb_iou},
        }
    }
    report.update(extra)
    if uids is not None:
        report.update(
            {
                "_report_path": "synthetic/report.json",
                "_per_sample_path": "synthetic/per_sample.jsonl",
                "_uids": sorted(uids),
                "_uid_count": len(uids),
            }
        )
    return report


def _write_report_bundle(root, *, bin_mae: float, aabb_iou: float, uids: list[str]):
    root.mkdir(parents=True)
    (root / "report.json").write_text(json.dumps(_report(bin_mae=bin_mae, aabb_iou=aabb_iou)))
    with (root / "per_sample.jsonl").open("w") as f:
        for uid in uids:
            f.write(json.dumps({"uid": uid}) + "\n")


def test_council_review_passes_when_all_gates_are_satisfied(tmp_path):
    result = evaluate_council(
        summary=_summary(),
        best_layout_run="D_geometry",
        downstream_run="E_stage2_best",
        best_layout_report=_report(bin_mae=2.0, aabb_iou=0.7),
        negative_controls={
            "one_view_eval": _report(bin_mae=2.5, aabb_iou=0.65),
            "no_aabb": _report(bin_mae=2.0, aabb_iou=0.5),
        },
        required_controls=["one_view_eval", "no_aabb"],
    )
    assert result["status"] == "pass"
    assert result["recommendation"] == "merge"
    assert result["issues"] == []

    out = tmp_path / "council_review.md"
    write_markdown(result, out)
    text = out.read_text()
    assert text.startswith("status: pass")
    assert "recommendation: merge" in text
    assert "Checked commands/artifacts" in text
    assert json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0])


def test_council_review_passes_with_required_category_stability():
    result = evaluate_council(
        summary=_summary_with_categories(),
        best_layout_run="D_geometry",
        downstream_run="E_stage2_best",
        best_layout_report=_report(bin_mae=2.0, aabb_iou=0.7),
        negative_controls={"one_view_eval": _report(bin_mae=2.5, aabb_iou=0.65)},
        required_controls=["one_view_eval"],
        require_category_stability=True,
    )

    assert result["status"] == "pass"
    assert result["checks"]["layout_vs_ce"]["category_stability"]["passes"]


def test_council_review_fails_when_required_category_regresses():
    result = evaluate_council(
        summary=_summary_with_categories(stable=False),
        best_layout_run="D_geometry",
        downstream_run="E_stage2_best",
        best_layout_report=_report(bin_mae=2.0, aabb_iou=0.7),
        negative_controls={"one_view_eval": _report(bin_mae=2.5, aabb_iou=0.65)},
        required_controls=["one_view_eval"],
        require_category_stability=True,
    )

    assert result["status"] == "fail"
    assert any("category 'table'" in issue for issue in result["issues"])


def test_council_review_fails_without_downstream_sv_win():
    result = evaluate_council(
        summary=_summary(downstream_beats_sv=False),
        best_layout_run="D_geometry",
        downstream_run="E_stage2_best",
        best_layout_report=_report(bin_mae=2.0, aabb_iou=0.7),
        negative_controls={"one_view_eval": _report(bin_mae=2.5, aabb_iou=0.65)},
        required_controls=["one_view_eval"],
    )

    assert result["status"] == "fail"
    assert any("does not beat SV on downstream CD" in issue for issue in result["issues"])
    assert any("does not beat SV on downstream F-score" in issue for issue in result["issues"])


def test_council_review_fails_without_grouped_downstream_seed_evidence():
    summary = _summary()
    cmp = summary["downstream"]["comparison_to_sv"]["E_stage2_best"]
    cmp.pop("seed_count")
    cmp.pop("cd_improved_seed_count")
    cmp.pop("f_score_improved_seed_count")

    result = evaluate_council(
        summary=summary,
        best_layout_run="D_geometry",
        downstream_run="E_stage2_best",
        best_layout_report=_report(bin_mae=2.0, aabb_iou=0.7),
        negative_controls={"one_view_eval": _report(bin_mae=2.5, aabb_iou=0.65)},
        required_controls=["one_view_eval"],
    )

    assert result["status"] == "fail"
    assert any("does not report grouped downstream seed evidence" in issue for issue in result["issues"])


def test_council_review_fails_when_downstream_mean_hides_seed_regression():
    result = evaluate_council(
        summary=_summary(
            downstream_beats_sv=True,
            downstream_seed_count=2,
            downstream_cd_improved_seed_count=1,
            downstream_f_improved_seed_count=2,
        ),
        best_layout_run="D_geometry",
        downstream_run="E_stage2_best",
        best_layout_report=_report(bin_mae=2.0, aabb_iou=0.7),
        negative_controls={"one_view_eval": _report(bin_mae=2.5, aabb_iou=0.65)},
        required_controls=["one_view_eval"],
    )

    assert result["status"] == "fail"
    assert any("does not beat SV on downstream CD/F for every seed" in issue for issue in result["issues"])


def test_council_review_fails_when_downstream_seed_labels_are_duplicated():
    summary = _summary()
    cmp = summary["downstream"]["comparison_to_sv"]["E_stage2_best"]
    cmp["unique_seed_count"] = 1
    cmp["seed_labels"] = ["seed11", "seed11"]
    cmp["expected_seed_labels"] = ["seed11", "seed23"]
    cmp["duplicate_seed_labels"] = ["seed11"]
    cmp["missing_expected_seed_labels"] = ["seed23"]

    result = evaluate_council(
        summary=summary,
        best_layout_run="D_geometry",
        downstream_run="E_stage2_best",
        best_layout_report=_report(bin_mae=2.0, aabb_iou=0.7),
        negative_controls={"one_view_eval": _report(bin_mae=2.5, aabb_iou=0.65)},
        required_controls=["one_view_eval"],
    )

    assert result["status"] == "fail"
    assert any("duplicate downstream seed labels" in issue for issue in result["issues"])
    assert any("missing downstream seed labels" in issue for issue in result["issues"])


def test_council_review_fails_when_negative_control_does_not_degrade():
    result = evaluate_council(
        summary=_summary(),
        best_layout_run="D_geometry",
        downstream_run="E_stage2_best",
        best_layout_report=_report(bin_mae=2.0, aabb_iou=0.7),
        negative_controls={"one_view_eval": _report(bin_mae=1.5, aabb_iou=0.8)},
        required_controls=["one_view_eval"],
    )

    assert result["status"] == "fail"
    assert any("does not degrade" in issue for issue in result["issues"])


def test_council_review_fails_when_negative_control_uids_differ():
    result = evaluate_council(
        summary=_summary(),
        best_layout_run="D_geometry",
        downstream_run="E_stage2_best",
        best_layout_report=_report(bin_mae=2.0, aabb_iou=0.7, uids=["a", "b"]),
        negative_controls={"one_view_eval": _report(bin_mae=2.5, aabb_iou=0.65, uids=["a", "c"])},
        required_controls=["one_view_eval"],
    )

    assert result["status"] == "fail"
    assert any("not UID-paired" in issue for issue in result["issues"])
    uid_pairing = result["checks"]["negative_controls"]["one_view_eval"]["per_report"][0]["uid_pairing"]
    assert uid_pairing["missing_vs_baseline"] == ["b"]
    assert uid_pairing["extra_vs_baseline"] == ["c"]


def test_council_review_fails_when_view_control_identity_is_wrong():
    result = evaluate_council(
        summary=_summary(),
        best_layout_run="D_geometry",
        downstream_run="E_stage2_best",
        best_layout_report=_report(bin_mae=2.0, aabb_iou=0.7),
        negative_controls={
            "one_view_eval": _report(
                bin_mae=2.5,
                aabb_iou=0.65,
                view_limit=2,
                reference_only=False,
                shuffle_views=False,
            )
        },
        required_controls=["one_view_eval"],
    )

    assert result["status"] == "fail"
    assert any("expected ablation settings" in issue for issue in result["issues"])
    identity = result["checks"]["negative_controls"]["one_view_eval"]["per_report"][0]["identity"]
    assert identity["observed"]["view_limit"] == 2
    assert any("view_limit" in issue for issue in identity["issues"])


def test_council_review_fails_when_training_control_override_is_missing():
    result = evaluate_council(
        summary=_summary(),
        best_layout_run="D_geometry",
        downstream_run="E_stage2_best",
        best_layout_report=_report(bin_mae=2.0, aabb_iou=0.7),
        negative_controls={"no_aabb": _report(bin_mae=2.5, aabb_iou=0.65, overrides=[])},
        required_controls=["no_aabb"],
    )

    assert result["status"] == "fail"
    identity = result["checks"]["negative_controls"]["no_aabb"]["per_report"][0]["identity"]
    assert "dataset.model.mv_obj_aabb_token=false" in identity["expected"]["overrides"]
    assert any("missing overrides" in issue for issue in identity["issues"])


def test_report_bundle_loads_sibling_per_sample_uids(tmp_path):
    _write_report_bundle(tmp_path / "run" / "seed11", bin_mae=2.0, aabb_iou=0.7, uids=["b", "a"])

    bundle = read_report_bundle(tmp_path / "run" / "seed11" / "report.json")

    assert bundle["_uids"] == ["a", "b"]
    assert bundle["_uid_count"] == 2
    assert bundle["_per_sample_path"].endswith("per_sample.jsonl")


def test_council_review_fails_when_downstream_is_not_object_paired():
    summary = _summary()
    summary["downstream"]["comparison_to_sv"]["E_stage2_best"]["object_set_equal"] = False

    result = evaluate_council(
        summary=summary,
        best_layout_run="D_geometry",
        downstream_run="E_stage2_best",
        best_layout_report=_report(bin_mae=2.0, aabb_iou=0.7),
        negative_controls={"one_view_eval": _report(bin_mae=2.5, aabb_iou=0.65)},
        required_controls=["one_view_eval"],
    )

    assert result["status"] == "fail"
    assert any("not UID/object-paired" in issue for issue in result["issues"])


def test_council_review_fails_when_layout_uids_are_not_exactly_paired():
    summary = _summary()
    summary["layout"]["runs"]["D_geometry"]["seeds"]["11"]["exact_uid_match"] = False
    summary["layout"]["runs"]["D_geometry"]["seeds"]["11"]["run_uid_count"] = 1
    summary["layout"]["runs"]["D_geometry"]["seeds"]["11"]["missing_vs_ce"] = ["dropped-bad-case"]
    summary["layout"]["missing"] = ["D_geometry/seed11 UID mismatch vs A_ce: missing=1 extra=0"]

    result = evaluate_council(
        summary=summary,
        best_layout_run="D_geometry",
        downstream_run="E_stage2_best",
        best_layout_report=_report(bin_mae=2.0, aabb_iou=0.7),
        negative_controls={"one_view_eval": _report(bin_mae=2.5, aabb_iou=0.65)},
        required_controls=["one_view_eval"],
    )

    assert result["status"] == "fail"
    assert any("not exactly UID-paired with CE" in issue for issue in result["issues"])


def test_council_review_fails_when_selected_best_run_is_dominated():
    summary = _summary()
    _set_run_deltas(summary, "C_coord", benefit=0.2)

    result = evaluate_council(
        summary=summary,
        best_layout_run="D_geometry",
        downstream_run="E_stage2_best",
        best_layout_report=_report(bin_mae=2.0, aabb_iou=0.7),
        negative_controls={"one_view_eval": _report(bin_mae=2.5, aabb_iou=0.65)},
        required_controls=["one_view_eval"],
    )

    assert result["status"] == "fail"
    assert any("dominated by C_coord" in issue for issue in result["issues"])


def test_council_review_allows_best_run_metric_tradeoff():
    summary = _summary()
    tradeoff = {}
    for metric in ("bin_mae", "corner_l1", "corner_l2", "center_error"):
        tradeoff[metric] = {"mean": -0.2, "n": 2, "improved_seed_count": 2}
    tradeoff["size_rel_error"] = {"mean": -0.05, "n": 2, "improved_seed_count": 2}
    tradeoff["aabb_iou"] = {"mean": 0.2, "n": 2, "improved_seed_count": 2}
    summary["layout"]["runs"]["C_coord"] = {
        "paired_delta_vs_ce": tradeoff,
        "seeds": summary["layout"]["runs"]["D_geometry"]["seeds"],
    }

    result = evaluate_council(
        summary=summary,
        best_layout_run="D_geometry",
        downstream_run="E_stage2_best",
        best_layout_report=_report(bin_mae=2.0, aabb_iou=0.7),
        negative_controls={"one_view_eval": _report(bin_mae=2.5, aabb_iou=0.65)},
        required_controls=["one_view_eval"],
    )

    assert result["status"] == "pass"
    assert result["checks"]["layout_vs_ce"]["best_run_selection"]["passes"]


def test_council_review_requires_control_report_for_each_best_report():
    result = evaluate_council(
        summary=_summary(),
        best_layout_run="D_geometry",
        downstream_run="E_stage2_best",
        best_layout_report=[
            _report(bin_mae=2.0, aabb_iou=0.7),
            _report(bin_mae=2.1, aabb_iou=0.72),
        ],
        negative_controls={"one_view_eval": [_report(bin_mae=2.5, aabb_iou=0.65)]},
        required_controls=["one_view_eval"],
    )

    assert result["status"] == "fail"
    assert any("has 1 reports but 2 best-layout reports" in issue for issue in result["issues"])


def test_council_recommendation_distinguishes_missing_from_reject():
    assert recommendation_for("pass", []) == "merge"
    assert recommendation_for("fail", ["summary.json is missing or unreadable"]) == "keep-experimental"
    assert recommendation_for("fail", ["D_geometry does not improve bin_mae against CE across all paired seeds"]) == "reject"
