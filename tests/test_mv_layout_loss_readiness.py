import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.data.trellis2_mv import MV_FEATURE_CACHE_VERSION


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
    np.savez(
        cache / "uid-a.npz",
        cache_version=np.asarray(MV_FEATURE_CACHE_VERSION, dtype=np.int64),
        local_points=np.zeros((1, 2, 2, 3), dtype=np.float16),
        conf=np.ones((1, 2, 2, 1), dtype=np.float16),
        dino_feats=np.zeros((1, 2, 1, 1), dtype=np.float16),
        view_indices=np.asarray([0], dtype=np.int64),
        view_mask=np.asarray([True]),
        ref_view=np.asarray(0, dtype=np.int64),
    )

    with sv.open("w") as f:
        f.write(json.dumps({"uid": "uid-a", "obj_id": 0, "cd": 0.1, "f_score": 0.2}) + "\n")
        f.write(json.dumps({"avg_cd": 0.1, "avg_f_score": 0.2, "num_evaluated": 1}) + "\n")
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


def test_mv_layout_loss_readiness_accepts_empty_marker_cache_files(tmp_path):
    mesh, hf, cache, sv, uid_meta = _write_ready_fixture(tmp_path)
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
    )

    result, report = _run_readiness_for_paths(tmp_path, mesh, hf, cache, sv, uid_meta)

    assert result.returncode == 0, result.stdout + result.stderr
    coverage = report["checks"]["mv_feature_cache"]["coverage"]
    assert coverage["empty_marker_count"] == 1
    assert coverage["feature_file_count"] == 0


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
