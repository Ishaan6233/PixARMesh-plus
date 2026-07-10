import json
import pytest
from pathlib import Path

from scripts.eval.summarize_mv_layout_evidence import downstream_summary, layout_summary


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
