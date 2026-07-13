import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.data.trellis2_mv import (
    MV_FEATURE_CACHE_CONTRACT_KEY,
    MV_FEATURE_CACHE_FINGERPRINT_KEY,
    MV_FEATURE_CACHE_VERSION,
)

TEST_CACHE_FP = "ready-cache-fp"
TEST_PROTOCOL = {
    "num_sample_points": 10000,
    "align_sample_points": 5000,
    "alignment_protocol": "separate_alignment_sample",
    "no_align": False,
    "mask_area_thresh": 1600,
    "evaluator_seed": 12345,
    "rng_policy": "per_object_stable_seed_v1",
    "eval_protocol_fingerprint": "protocol-fp",
}


def _cache_provenance():
    contract = {
        "cache_version": MV_FEATURE_CACHE_VERSION,
        "fingerprint": TEST_CACHE_FP,
        "certifiable": True,
        "uncertified_reasons": [],
    }
    return {
        MV_FEATURE_CACHE_FINGERPRINT_KEY: np.asarray(TEST_CACHE_FP),
        MV_FEATURE_CACHE_CONTRACT_KEY: np.asarray(json.dumps(contract)),
    }


def _write_ready_fixture(root):
    mesh = root / "mesh"
    hf = root / "hf"
    cache = root / "cache"
    sv = root / "sv.jsonl"
    uid_meta = root / "uid_categories.jsonl"

    (mesh / "mesh_dumps").mkdir(parents=True)
    (mesh / "mv_cond").mkdir()
    (mesh / "metadata.csv").write_text("sha256,uid,scene_id\nsha,uid-a,scene-a\n")
    (mesh / "conditioning_filter.csv").write_text("sha256,keep\nsha,true\n")
    (mesh / "conditioning_filter.meta.json").write_text(json.dumps({"frame_correction": True}))

    (hf / "train").mkdir(parents=True)
    (hf / "dataset_dict.json").write_text("{}\n")
    (hf / "train" / "state.json").write_text("{}\n")

    cache.mkdir()
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "cache_version": MV_FEATURE_CACHE_VERSION,
                "fingerprint": TEST_CACHE_FP,
                "certifiable": True,
                "uncertified_reasons": [],
            }
        )
    )
    np.savez(
        cache / "uid-a.npz",
        cache_version=np.asarray(MV_FEATURE_CACHE_VERSION, dtype=np.int64),
        local_points=np.zeros((1, 2, 2, 3), dtype=np.float16),
        conf=np.ones((1, 2, 2, 1), dtype=np.float16),
        dino_feats=np.zeros((1, 2, 1, 1), dtype=np.float16),
        view_indices=np.asarray([0], dtype=np.int64),
        view_mask=np.asarray([True]),
        ref_view=np.asarray(0, dtype=np.int64),
        **_cache_provenance(),
    )

    with sv.open("w") as f:
        f.write(
            json.dumps(
                {
                    "uid": "uid-a",
                    "obj_id": 0,
                    "cd": 0.1,
                    "f_score": 0.2,
                    "object_eval_seed": 7,
                    **TEST_PROTOCOL,
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "avg_cd": 0.1,
                    "avg_f_score": 0.2,
                    "num_evaluated": 1,
                    "coverage": 1.0,
                    **TEST_PROTOCOL,
                }
            )
            + "\n"
        )
    uid_meta.write_text(
        json.dumps({"uid": "uid-a", "category": "chair"}) + "\n"
        + json.dumps({"uid": "uid-b", "category": "table"}) + "\n"
    )

    return mesh, hf, cache, sv, uid_meta


def _run_readiness(tmp_path, *extra):
    mesh, hf, cache, sv, uid_meta = _write_ready_fixture(tmp_path)
    return _run_readiness_for_paths(tmp_path, mesh, hf, cache, sv, uid_meta, *extra)


def _run_readiness_for_paths(tmp_path, mesh, hf, cache, sv, uid_meta, *extra):
    out = tmp_path / "readiness.json"
    cmd = [
        sys.executable,
        "scripts/experiments/check_mv_layout_loss_readiness.py",
        "--mesh-dataset",
        str(mesh),
        "--hf-dataset",
        str(hf),
        "--mv-feature-cache",
        str(cache),
        "--sv-downstream",
        str(sv),
        "--uid-metadata",
        str(uid_meta),
        "--cache-precompute-script",
        str(tmp_path / "precompute_cache.sh"),
        "--stage1-train-root",
        str(tmp_path / "stage1"),
        "--stage2-train-root",
        str(tmp_path / "stage2"),
        "--no-require-gpu",
        "--sv-baseline-glob",
        str(tmp_path / "candidate_sv" / "**" / "eval_obj_results.jsonl"),
        "--out",
        str(out),
        *extra,
    ]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return result, json.loads(out.read_text())


def test_mv_layout_loss_readiness_accepts_complete_prerequisites(tmp_path):
    result, report = _run_readiness(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert report["ok"]
    assert report["issues"] == []
    assert report["checks"]["mv_feature_cache"]["npz_count"] == 1
    assert report["checks"]["mv_feature_cache"]["coverage"]["expected_total"] == 1
    assert report["checks"]["mv_feature_cache"]["coverage"]["missing_count"] == 0
    assert report["checks"]["mv_feature_cache"]["expected_cache_version"] == MV_FEATURE_CACHE_VERSION
    assert report["checks"]["sv_downstream"]["object_rows"] == 1
    assert report["checks"]["uid_metadata"]["category_count"] == 2


def _write_empty_marker_npz(cache):
    np.savez(
        cache / "uid-a.npz",
        cache_version=np.asarray(MV_FEATURE_CACHE_VERSION, dtype=np.int64),
        local_points=np.zeros((1, 1, 1, 3), dtype=np.float16),
        conf=np.zeros((1, 1, 1, 1), dtype=np.float16),
        dino_feats=np.zeros((1, 1, 1, 1), dtype=np.float16),
        view_indices=np.asarray([0], dtype=np.int64),
        view_mask=np.asarray([False]),
        ref_view=np.asarray(0, dtype=np.int64),
        empty_reason=np.asarray("runtime rejected"),
        **_cache_provenance(),
    )


def test_mv_layout_loss_readiness_recognizes_empty_marker_cache_files_as_present(tmp_path):
    # An empty-marker file still counts as present (not missing) for coverage purposes;
    # this fixture's single expected uid is 100% empty-marker, so with the default
    # thresholds it now also blocks readiness (see the two tests below) -- this test
    # isolates the "present, not missing" bookkeeping from the threshold gate.
    mesh, hf, cache, sv, uid_meta = _write_ready_fixture(tmp_path)
    _write_empty_marker_npz(cache)

    result, report = _run_readiness_for_paths(
        tmp_path, mesh, hf, cache, sv, uid_meta, "--max-empty-marker-frac", "1.0"
    )

    assert result.returncode == 0, result.stdout + result.stderr
    coverage = report["checks"]["mv_feature_cache"]["coverage"]
    assert coverage["missing_count"] == 0
    assert coverage["empty_marker_count"] == 1
    assert coverage["feature_file_count"] == 0


def test_mv_layout_loss_readiness_blocks_on_excessive_empty_marker_fraction(tmp_path):
    mesh, hf, cache, sv, uid_meta = _write_ready_fixture(tmp_path)
    _write_empty_marker_npz(cache)

    result, report = _run_readiness_for_paths(tmp_path, mesh, hf, cache, sv, uid_meta)

    assert result.returncode == 1
    assert not report["ok"]
    coverage = report["checks"]["mv_feature_cache"]["coverage"]
    assert coverage["empty_marker_frac"] == 1.0
    assert any("empty-marker placeholders" in issue for issue in report["issues"])
    assert any(item["check"] == "mv_feature_cache" for item in report["remediations"])


def test_mv_layout_loss_readiness_warns_without_blocking_below_max_empty_marker_frac(tmp_path):
    mesh, hf, cache, sv, uid_meta = _write_ready_fixture(tmp_path)
    _write_empty_marker_npz(cache)

    result, report = _run_readiness_for_paths(
        tmp_path,
        mesh,
        hf,
        cache,
        sv,
        uid_meta,
        "--max-empty-marker-frac",
        "1.0",
        "--warn-empty-marker-frac",
        "0.05",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert report["ok"]
    assert any("empty-marker placeholders" in warning for warning in report["warnings"])


def test_mv_layout_loss_readiness_reports_filter_keep_cache_coverage_gaps(tmp_path):
    mesh, hf, cache, sv, uid_meta = _write_ready_fixture(tmp_path)
    with (mesh / "metadata.csv").open("a") as f:
        f.write("sha-b,uid-b,scene-b\n")
    with (mesh / "conditioning_filter.csv").open("a") as f:
        f.write("sha-b,true\n")

    result, report = _run_readiness_for_paths(tmp_path, mesh, hf, cache, sv, uid_meta)

    assert result.returncode == 1
    coverage = report["checks"]["mv_feature_cache"]["coverage"]
    assert coverage["expected_total"] == 2
    assert coverage["missing_count"] == 1
    assert coverage["missing_examples"] == ["uid-b"]
    assert any("missing 1 files required" in issue for issue in report["issues"])
    assert any(item["check"] == "mv_feature_cache" for item in report["remediations"])


def test_mv_layout_loss_readiness_reports_cache_precompute_remediation(tmp_path):
    mesh, hf, cache, sv, uid_meta = _write_ready_fixture(tmp_path)
    shutil.rmtree(cache)
    out = tmp_path / "readiness.json"
    precompute = tmp_path / "precompute_cache.sh"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/check_mv_layout_loss_readiness.py",
            "--mesh-dataset",
            str(mesh),
            "--hf-dataset",
            str(hf),
            "--mv-feature-cache",
            str(cache),
            "--sv-downstream",
            str(sv),
            "--uid-metadata",
            str(uid_meta),
            "--cache-precompute-script",
            str(precompute),
            "--stage1-train-root",
            str(tmp_path / "stage1"),
            "--stage2-train-root",
            str(tmp_path / "stage2"),
            "--no-require-gpu",
            "--out",
            str(out),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    report = json.loads(out.read_text())

    assert result.returncode == 1
    assert any("missing MV feature cache root" in issue for issue in report["issues"])
    assert {
        "check": "mv_feature_cache",
        "action": "Build or rebuild the frozen DA3+DINO Trellis2-MV feature cache before launching A-D training.",
        "command": f"bash {precompute}",
    } in report["remediations"]


def test_mv_layout_loss_readiness_reports_missing_uid_metadata(tmp_path):
    missing = tmp_path / "missing_uid_categories.jsonl"
    result, report = _run_readiness(tmp_path, "--uid-metadata", str(missing))

    assert result.returncode == 1
    assert not report["ok"]
    assert any("missing UID category metadata" in issue for issue in report["issues"])
    assert any(item["check"] == "uid_metadata" and "raw model_id" in item["action"] for item in report["remediations"])


def test_mv_layout_loss_readiness_requires_uid_metadata_by_default(tmp_path):
    result, report = _run_readiness(tmp_path, "--uid-metadata", "")

    assert result.returncode == 1
    assert not report["ok"]
    assert report["checks"]["uid_metadata"]["required"]
    assert any("no --uid-metadata provided" in issue for issue in report["issues"])
    assert any(item["check"] == "uid_metadata" and "--no-require-uid-metadata" in item["action"] for item in report["remediations"])


def test_mv_layout_loss_readiness_omitted_uid_metadata_uses_required_default(tmp_path):
    mesh, hf, cache, sv, _uid_meta = _write_ready_fixture(tmp_path)
    out = tmp_path / "readiness.json"
    script = Path(__file__).resolve().parents[1] / "scripts/experiments/check_mv_layout_loss_readiness.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mesh-dataset",
            str(mesh),
            "--hf-dataset",
            str(hf),
            "--mv-feature-cache",
            str(cache),
            "--sv-downstream",
            str(sv),
            "--cache-precompute-script",
            str(tmp_path / "precompute_cache.sh"),
            "--stage1-train-root",
            str(tmp_path / "stage1"),
            "--stage2-train-root",
            str(tmp_path / "stage2"),
            "--no-require-gpu",
            "--out",
            str(out),
        ],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    report = json.loads(out.read_text())

    assert result.returncode == 1
    assert not report["ok"]
    assert report["checks"]["uid_metadata"]["required"]
    assert report["checks"]["uid_metadata"]["path"] == "metadata/layout_uid_categories.jsonl"
    assert any("missing UID category metadata" in issue for issue in report["issues"])


def test_mv_layout_loss_readiness_allows_uid_metadata_opt_out_for_dry_run(tmp_path):
    result, report = _run_readiness(tmp_path, "--uid-metadata", "", "--no-require-uid-metadata")

    assert result.returncode == 0, result.stdout + result.stderr
    assert report["ok"]
    assert report["issues"] == []
    assert not report["checks"]["uid_metadata"]["required"]
    assert any("no --uid-metadata provided" in warning for warning in report["warnings"])


def test_mv_layout_loss_readiness_reports_malformed_uid_metadata(tmp_path):
    bad = tmp_path / "bad_uid_categories.jsonl"
    bad.write_text(json.dumps({"uid": "uid-a"}) + "\n" + json.dumps({"category": "chair"}) + "\n")
    result, report = _run_readiness(tmp_path, "--uid-metadata", str(bad))

    assert result.returncode == 1
    assert not report["ok"]
    assert report["checks"]["uid_metadata"]["missing_uid_rows"] == 1
    assert report["checks"]["uid_metadata"]["missing_category_rows"] == 1
    assert any("rows without a UID key" in issue for issue in report["issues"])
    assert any("rows without a category key" in issue for issue in report["issues"])
    assert any("fewer than two categories" in issue for issue in report["issues"])
    assert any(item["check"] == "uid_metadata" and "accepted UID key" in item["action"] for item in report["remediations"])


def test_mv_layout_loss_readiness_reports_existing_checkpoint_guards(tmp_path):
    existing = tmp_path / "stage1" / "A_ce" / "seed11" / "checkpoints"
    existing.mkdir(parents=True)
    result, report = _run_readiness(tmp_path)

    assert result.returncode == 1
    assert not report["ok"]
    assert str(existing) in report["checks"]["existing_checkpoints"]["paths"]
    assert any("existing checkpoint directories" in issue for issue in report["issues"])
    assert any(item["check"] == "existing_checkpoints" for item in report["remediations"])


def test_mv_layout_loss_readiness_blocks_on_disagreeing_sv_baseline(tmp_path):
    # 2026-07-13 red-team Finding 2: two on-disk "frozen SV baseline" files disagreed by
    # ~2x avg_cd on identical objects with no warning anywhere. This is the guard.
    mesh, hf, cache, sv, uid_meta = _write_ready_fixture(tmp_path)
    other_dir = tmp_path / "other_sv_baseline"
    other_dir.mkdir()
    other_sv = other_dir / "eval_obj_results.jsonl"
    with other_sv.open("w") as f:
        f.write(
            json.dumps(
                {
                    "uid": "uid-a",
                    "obj_id": 0,
                    "cd": 0.05,
                    "f_score": 0.6,
                    "object_eval_seed": 7,
                    **TEST_PROTOCOL,
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "avg_cd": 0.05,
                    "avg_f_score": 0.6,
                    "num_evaluated": 1,
                    "coverage": 1.0,
                    **TEST_PROTOCOL,
                }
            )
            + "\n"
        )

    result, report = _run_readiness_for_paths(
        tmp_path,
        mesh,
        hf,
        cache,
        sv,
        uid_meta,
        "--sv-baseline-glob",
        str(tmp_path / "**" / "eval_obj_results.jsonl"),
    )

    assert result.returncode == 1
    assert not report["ok"]
    disagreements = report["checks"]["sv_downstream"]["baseline_disagreements"]
    assert len(disagreements) == 1
    assert disagreements[0]["path"] == str(other_sv)
    assert disagreements[0]["avg_cd"] == 0.05
    assert any("disagree with" in issue for issue in report["issues"])


def test_mv_layout_loss_readiness_silent_when_candidate_baselines_agree(tmp_path):
    mesh, hf, cache, sv, uid_meta = _write_ready_fixture(tmp_path)
    other_dir = tmp_path / "other_sv_baseline"
    other_dir.mkdir()
    other_sv = other_dir / "eval_obj_results.jsonl"
    with other_sv.open("w") as f:
        f.write(json.dumps({"uid": "uid-a", "obj_id": 0, "cd": 0.101, "f_score": 0.2}) + "\n")
        f.write(json.dumps({"avg_cd": 0.101, "avg_f_score": 0.2, "num_evaluated": 1}) + "\n")

    result, report = _run_readiness_for_paths(
        tmp_path,
        mesh,
        hf,
        cache,
        sv,
        uid_meta,
        "--sv-baseline-glob",
        str(tmp_path / "**" / "eval_obj_results.jsonl"),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "baseline_disagreements" not in report["checks"]["sv_downstream"]
    assert not any("disagree with" in warning for warning in report["warnings"])
