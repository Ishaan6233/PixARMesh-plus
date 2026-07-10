import json

from scripts.eval.write_mv_layout_council_review import evaluate_council, write_markdown


def _summary(*, downstream_beats_sv: bool = True) -> dict:
    paired_delta_vs_ce = {}
    for metric in ("bin_mae", "corner_l1", "corner_l2", "center_error", "size_rel_error"):
        paired_delta_vs_ce[metric] = {"mean": -0.1, "n": 2, "improved_seed_count": 2}
    paired_delta_vs_ce["aabb_iou"] = {"mean": 0.1, "n": 2, "improved_seed_count": 2}
    return {
        "layout": {
            "runs": {
                "D_geometry": {
                    "paired_delta_vs_ce": paired_delta_vs_ce,
                    "seeds": {
                        "11": {"visual_count": 2},
                        "23": {"visual_count": 2},
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
                }
            }
        },
    }


def _report(*, bin_mae: float, aabb_iou: float) -> dict:
    return {
        "summary": {
            "bin_mae": {"mean": bin_mae},
            "aabb_iou": {"mean": aabb_iou},
        }
    }


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
    assert result["issues"] == []

    out = tmp_path / "council_review.md"
    write_markdown(result, out)
    text = out.read_text()
    assert text.startswith("status: pass")
    assert "Checked commands/artifacts" in text
    assert json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0])


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
