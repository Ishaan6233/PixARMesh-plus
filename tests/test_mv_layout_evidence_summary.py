import json
import pytest
from pathlib import Path

from scripts.eval.summarize_mv_layout_evidence import downstream_summary, layout_summary, load_uid_metadata


def _write_seed(root: Path, run: str, seed: int, records: list[dict]) -> None:
    out = root / run / f"seed{seed}"
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps({"num_records": len(records)}))
    with (out / "per_sample.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _record(uid: str, *, bin_mae: float, aabb_iou: float) -> dict:
    return {
        "uid": uid,
        "valid": True,
        "invalid_reason": None,
        "layout_token_count": 24,
        "expected_layout_tokens": 24,
        "token_accuracy": 0.5,
        "valid_token_frac": 1.0,
        "bin_mae": bin_mae,
        "corner_l1": bin_mae / 10.0,
        "corner_l2": bin_mae / 8.0,
        "center_error": bin_mae / 6.0,
        "size_rel_error": bin_mae / 12.0,
        "aabb_iou": aabb_iou,
    }


def test_layout_summary_uses_uid_paired_deltas(tmp_path):
    _write_seed(
        tmp_path,
        "A_ce",
        11,
        [_record("a", bin_mae=10.0, aabb_iou=0.2), _record("b", bin_mae=8.0, aabb_iou=0.4)],
    )
    _write_seed(
        tmp_path,
        "B_ordinal",
        11,
        [_record("b", bin_mae=6.0, aabb_iou=0.5), _record("a", bin_mae=7.0, aabb_iou=0.3)],
    )

    summary = layout_summary(tmp_path, ["A_ce", "B_ordinal"], [11], "A_ce")

    delta_bin = summary["runs"]["B_ordinal"]["paired_delta_vs_ce"]["bin_mae"]
    delta_iou = summary["runs"]["B_ordinal"]["paired_delta_vs_ce"]["aabb_iou"]
    assert delta_bin["mean"] == -2.5
    assert delta_bin["improved_seed_count"] == 1
    assert delta_iou["mean"] == pytest.approx(0.1)
    assert delta_iou["improved_seed_count"] == 1


def test_layout_summary_includes_invalid_rows_in_valid_token_fraction(tmp_path):
    ce_records = [
        _record("a", bin_mae=10.0, aabb_iou=0.2),
        _record("b", bin_mae=8.0, aabb_iou=0.4),
    ]
    run_records = [
        _record("a", bin_mae=10.0, aabb_iou=0.2),
        {
            **_record("b", bin_mae=30.0, aabb_iou=0.0),
            "valid": False,
            "invalid_reason": "short",
            "layout_token_count": 1,
            "valid_token_frac": 0.0,
        },
    ]
    _write_seed(tmp_path, "A_ce", 11, ce_records)
    _write_seed(tmp_path, "B_ordinal", 11, run_records)

    summary = layout_summary(tmp_path, ["A_ce", "B_ordinal"], [11], "A_ce")

    run = summary["runs"]["B_ordinal"]
    assert run["seeds"]["11"]["paired_uid_count"] == 2
    assert run["metrics"]["valid_token_frac"]["mean"] == 0.5
    assert run["paired_delta_vs_ce"]["valid_token_frac"]["mean"] == -0.5


def test_layout_summary_reports_partial_uid_overlap_as_missing_evidence(tmp_path):
    _write_seed(
        tmp_path,
        "A_ce",
        11,
        [_record("a", bin_mae=10.0, aabb_iou=0.2), _record("b", bin_mae=8.0, aabb_iou=0.4)],
    )
    _write_seed(
        tmp_path,
        "B_ordinal",
        11,
        [_record("b", bin_mae=6.0, aabb_iou=0.5)],
    )

    summary = layout_summary(tmp_path, ["A_ce", "B_ordinal"], [11], "A_ce")

    assert not summary["runs"]["B_ordinal"]["seeds"]["11"]["exact_uid_match"]
    assert any("UID mismatch" in item for item in summary["missing"])


def test_layout_summary_reports_category_paired_deltas(tmp_path):
    _write_seed(
        tmp_path,
        "A_ce",
        11,
        [_record("a", bin_mae=10.0, aabb_iou=0.2), _record("b", bin_mae=6.0, aabb_iou=0.5)],
    )
    _write_seed(
        tmp_path,
        "B_ordinal",
        11,
        [_record("a", bin_mae=8.0, aabb_iou=0.4), _record("b", bin_mae=7.0, aabb_iou=0.45)],
    )
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        "\n".join(
            [
                json.dumps({"uid": "a", "category": "chair"}),
                json.dumps({"uid": "b", "category": "table"}),
            ]
        )
        + "\n"
    )

    summary = layout_summary(tmp_path, ["A_ce", "B_ordinal"], [11], "A_ce", load_uid_metadata(metadata))
    category_delta = summary["runs"]["B_ordinal"]["category_paired_delta_vs_ce"]

    assert category_delta["chair"]["bin_mae"]["mean"] == pytest.approx(-2.0)
    assert category_delta["chair"]["bin_mae"]["improved_seed_count"] == 1
    assert category_delta["table"]["bin_mae"]["mean"] == pytest.approx(1.0)
    assert category_delta["table"]["bin_mae"]["improved_seed_count"] == 0


def _write_downstream(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_downstream_summary_uses_object_paired_deltas_and_cd_gate(tmp_path):
    sv = tmp_path / "sv.jsonl"
    mv = tmp_path / "mv.jsonl"
    _write_downstream(
        sv,
        [
            {"uid": "a", "obj_id": 0, "cd": 0.5, "f_score": 0.4},
            {"uid": "b", "obj_id": 1, "cd": 0.3, "f_score": 0.5},
            {"avg_cd": 0.4, "avg_f_score": 0.45, "num_evaluated": 2},
        ],
    )
    _write_downstream(
        mv,
        [
            {"uid": "b", "obj_id": 1, "cd": 0.2, "f_score": 0.7},
            {"uid": "a", "obj_id": 0, "cd": 0.4, "f_score": 0.6},
            {"avg_cd": 0.3, "avg_f_score": 0.65, "num_evaluated": 2},
        ],
    )

    summary = downstream_summary([f"E_stage2_best={mv}"], str(sv))
    cmp = summary["comparison_to_sv"]["E_stage2_best"]

    assert cmp["object_set_equal"]
    assert cmp["paired_count"] == 2
    assert cmp["avg_cd_delta_vs_sv"] == pytest.approx(-0.1)
    assert cmp["avg_f_score_delta_vs_sv"] == pytest.approx(0.2)
    assert cmp["cd_beats_sv"]
    assert cmp["f_score_beats_sv"]


def test_downstream_summary_reports_unpaired_objects(tmp_path):
    sv = tmp_path / "sv.jsonl"
    mv = tmp_path / "mv.jsonl"
    _write_downstream(
        sv,
        [
            {"uid": "a", "obj_id": 0, "cd": 0.5, "f_score": 0.4},
            {"avg_cd": 0.5, "avg_f_score": 0.4, "num_evaluated": 1},
        ],
    )
    _write_downstream(
        mv,
        [
            {"uid": "b", "obj_id": 0, "cd": 0.2, "f_score": 0.7},
            {"avg_cd": 0.2, "avg_f_score": 0.7, "num_evaluated": 1},
        ],
    )

    summary = downstream_summary([f"E_stage2_best={mv}"], str(sv))

    assert not summary["comparison_to_sv"]["E_stage2_best"]["object_set_equal"]
    assert any("downstream object set mismatch" in item for item in summary["missing"])


def test_downstream_summary_aggregates_repeated_run_name_as_seed_group(tmp_path):
    sv = tmp_path / "sv.jsonl"
    seed11 = tmp_path / "stage2" / "seed11" / "eval_obj_results.jsonl"
    seed23 = tmp_path / "stage2" / "seed23" / "eval_obj_results.jsonl"
    _write_downstream(
        sv,
        [
            {"uid": "a", "obj_id": 0, "cd": 0.5, "f_score": 0.4},
            {"uid": "b", "obj_id": 1, "cd": 0.3, "f_score": 0.5},
            {"avg_cd": 0.4, "avg_f_score": 0.45, "num_evaluated": 2},
        ],
    )
    _write_downstream(
        seed11,
        [
            {"uid": "a", "obj_id": 0, "cd": 0.4, "f_score": 0.6},
            {"uid": "b", "obj_id": 1, "cd": 0.2, "f_score": 0.7},
            {"avg_cd": 0.3, "avg_f_score": 0.65, "num_evaluated": 2},
        ],
    )
    _write_downstream(
        seed23,
        [
            {"uid": "b", "obj_id": 1, "cd": 0.25, "f_score": 0.65},
            {"uid": "a", "obj_id": 0, "cd": 0.45, "f_score": 0.55},
            {"avg_cd": 0.35, "avg_f_score": 0.60, "num_evaluated": 2},
        ],
    )

    summary = downstream_summary(
        [f"E_stage2_best={seed11}", f"E_stage2_best={seed23}"],
        str(sv),
    )
    run = summary["runs"]["E_stage2_best"]
    cmp = summary["comparison_to_sv"]["E_stage2_best"]

    assert run["seed_count"] == 2
    assert run["avg_cd"] == pytest.approx(0.325)
    assert run["avg_cd_ci95"] is not None
    assert [item["seed"] for item in run["per_seed"]] == ["seed11", "seed23"]
    assert cmp["object_set_equal"]
    assert cmp["seed_count"] == 2
    assert cmp["avg_cd_delta_vs_sv"] == pytest.approx(-0.075)
    assert cmp["avg_f_score_delta_vs_sv"] == pytest.approx(0.175)
    assert cmp["cd_improved_seed_count"] == 2
    assert cmp["f_score_improved_seed_count"] == 2
    assert cmp["cd_beats_sv"]
    assert cmp["f_score_beats_sv"]


def test_downstream_summary_reports_duplicate_or_missing_seed_labels(tmp_path):
    sv = tmp_path / "sv.jsonl"
    seed11 = tmp_path / "stage2" / "seed11" / "eval_obj_results.jsonl"
    _write_downstream(
        sv,
        [
            {"uid": "a", "obj_id": 0, "cd": 0.5, "f_score": 0.4},
            {"avg_cd": 0.5, "avg_f_score": 0.4, "num_evaluated": 1},
        ],
    )
    _write_downstream(
        seed11,
        [
            {"uid": "a", "obj_id": 0, "cd": 0.4, "f_score": 0.6},
            {"avg_cd": 0.4, "avg_f_score": 0.6, "num_evaluated": 1},
        ],
    )

    summary = downstream_summary(
        [f"E_stage2_best={seed11}", f"E_stage2_best={seed11}"],
        str(sv),
        expected_seeds=[11, 23],
    )
    run = summary["runs"]["E_stage2_best"]
    cmp = summary["comparison_to_sv"]["E_stage2_best"]

    assert run["seed_count"] == 2
    assert run["unique_seed_count"] == 1
    assert run["duplicate_seed_labels"] == ["seed11"]
    assert run["missing_expected_seed_labels"] == ["seed23"]
    assert cmp["duplicate_seed_labels"] == ["seed11"]
    assert any("duplicate downstream seed labels" in item for item in summary["missing"])
    assert any("missing expected downstream seed labels" in item for item in summary["missing"])
