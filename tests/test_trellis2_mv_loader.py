import numpy as np
import pytest

import json

from src.data.collator import get_mesh_data_collator
from src.data.trellis2_mv import (
    MV_FEATURE_CACHE_CONTRACT_KEY,
    MV_FEATURE_CACHE_FINGERPRINT_KEY,
    MV_FEATURE_CACHE_VERSION,
    Trellis2MVDataset,
    _mask_sanity_for_view,
    _select_diverse_views,
    build_mv_feature_cache_contract,
    view_selection_policy_fingerprint,
)
from src.utils.config import DataConfig, ModelConfig


TEST_CACHE_FP = "test-cache-fp"


def _test_contract(fingerprint: str = TEST_CACHE_FP) -> dict:
    return {
        "schema": "trellis2_mv_feature_cache",
        "cache_version": MV_FEATURE_CACHE_VERSION,
        "fingerprint": fingerprint,
        "certifiable": True,
        "uncertified_reasons": [],
    }


def _write_cache_manifest(root, fingerprint: str = TEST_CACHE_FP) -> None:
    (root / "manifest.json").write_text(json.dumps(_test_contract(fingerprint)))


def _provenance_kwargs(fingerprint: str = TEST_CACHE_FP) -> dict:
    return {
        MV_FEATURE_CACHE_FINGERPRINT_KEY: np.asarray(fingerprint),
        MV_FEATURE_CACHE_CONTRACT_KEY: np.asarray(json.dumps(_test_contract(fingerprint))),
    }


def _model_cfg_for_cache(tmp_path, dino_dir=None) -> ModelConfig:
    dino_dir = dino_dir or (tmp_path / "dino")
    return ModelConfig(
        vocab_size=64,
        num_pos_tokens=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        pc_token_id=3,
        tokenization_method="meshxl",
        max_seq_length=32,
        pos_token_offset=6,
        layout_tokenization_method="tri",
        image_encoder=str(dino_dir),
        da3_ckpt_path=str(tmp_path / "da3"),
    )


def _write_artifact_dirs(tmp_path):
    for name in ("da3", "dino", "preproc"):
        root = tmp_path / name
        root.mkdir()
        (root / "config.json").write_text(json.dumps({"name": name}))
    return tmp_path / "da3", tmp_path / "dino", tmp_path / "preproc"


def test_select_diverse_views_requires_minimum_reference_support():
    pts = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)
    cams = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    Ks = np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0)

    assert _select_diverse_views(
        pts,
        cams,
        Ks,
        (4, 4),
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        k_max=2,
        min_support_pts=4,
    ) == []


def test_select_diverse_views_keeps_highest_support_first():
    pts = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)
    cams = np.repeat(np.eye(4, dtype=np.float32)[None], 4, axis=0)
    Ks = np.repeat(np.eye(3, dtype=np.float32)[None], 4, axis=0)

    selected = _select_diverse_views(
        pts,
        cams,
        Ks,
        (4, 4),
        np.array([10.0, 80.0, 60.0, 2.0], dtype=np.float32),
        k_max=3,
        min_support_pts=50,
    )

    assert selected[0] == 1


def test_mask_sanity_accepts_identifiable_target_seed_hits():
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.2]], dtype=np.float32)
    w2c = np.eye(4, dtype=np.float32)
    K = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.int32)
    mask[1, 1] = 1007
    row = {"objects": {"inst_ids": [7]}}

    score = _mask_sanity_for_view(
        points,
        w2c,
        K,
        mask,
        row,
        min_area_px=1,
        min_hit_pts=1,
        min_hit_frac=0.1,
    )

    assert score["enforced"]
    assert score["ok"]
    assert score["target_hit_count"] == 2


def test_feature_cache_loads_valid_cached_tensors(tmp_path):
    ds = Trellis2MVDataset.__new__(Trellis2MVDataset)
    ds.feature_cache = tmp_path
    _write_cache_manifest(tmp_path)
    np.savez_compressed(
        tmp_path / "uid-a.npz",
        cache_version=np.array(MV_FEATURE_CACHE_VERSION, dtype=np.int64),
        local_points=np.zeros((2, 4, 5, 3), dtype=np.float16),
        conf=np.ones((2, 4, 5, 1), dtype=np.float16),
        dino_feats=np.zeros((2, 8, 2, 3), dtype=np.float16),
        view_indices=np.array([3, 1], dtype=np.int64),
        view_mask=np.array([True, False]),
        ref_view=np.array(0, dtype=np.int64),
        **_provenance_kwargs(),
    )

    cache = ds._load_feature_cache("uid-a", n_avail=4)

    assert cache["view_indices"].tolist() == [3, 1]
    assert cache["view_mask"].tolist() == [True, False]
    assert cache["cached_local_points"].shape == (2, 4, 5, 3)


def test_feature_cache_loads_empty_marker_for_runtime_rejected_item(tmp_path):
    ds = Trellis2MVDataset.__new__(Trellis2MVDataset)
    ds.feature_cache = tmp_path
    _write_cache_manifest(tmp_path)
    np.savez_compressed(
        tmp_path / "uid-a.npz",
        cache_version=np.array(MV_FEATURE_CACHE_VERSION, dtype=np.int64),
        local_points=np.zeros((2, 1, 1, 3), dtype=np.float16),
        conf=np.zeros((2, 1, 1, 1), dtype=np.float16),
        dino_feats=np.zeros((2, 1, 1, 1), dtype=np.float16),
        view_indices=np.array([0, 0], dtype=np.int64),
        view_mask=np.array([False, False]),
        ref_view=np.array(0, dtype=np.int64),
        empty_reason=np.asarray("mask sanity rejected all support views"),
        **_provenance_kwargs(),
    )

    cache = ds._load_feature_cache("uid-a", n_avail=4)

    assert cache["empty_marker"]
    assert cache["empty_reason"] == "mask sanity rejected all support views"
    assert cache["view_mask"].tolist() == [False, False]


def test_dataset_maps_empty_feature_cache_marker_to_graceful_empty_item(tmp_path):
    ds = Trellis2MVDataset.__new__(Trellis2MVDataset)
    ds.instances = ["sha-a"]
    ds.data_cfg = DataConfig(
        type="3d-front-trellis2-mv",
        path="unused",
        num_points=4,
        num_views=2,
        load_images=False,
    )
    ds.image_preprocessor = None
    ds.is_train = False
    ds.norm_bound = 0.9995
    ds.feature_cache = tmp_path
    _write_cache_manifest(tmp_path)
    ds._uid_to_idx = {"uid-a": 0}
    ds._scene_id_to_idx = {"scene-a": 0}
    ds._pan_key = None
    ds._hf_has_panoptic = False
    ds._cached_img_chw = None
    ds._hf = [
        {
            "wrd2cam_rects": [np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)],
            "Ks": [np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)],
        }
    ]
    ds._load_mesh = lambda _sha: (
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.asarray([[0, 1, 2]], dtype=np.int64),
    )
    ds._load_cond = lambda _sha: {
        "uid": "uid-a",
        "scene_id": "scene-a",
        "T_norm_from_output": np.eye(4, dtype=np.float32),
        "T_output_from_norm": np.eye(4, dtype=np.float32),
        "scene_point_clouds": np.zeros((1, 3), dtype=np.float32),
        "bboxes": np.zeros((1, 8, 3), dtype=np.float32),
        "object_to_norm_transforms": np.eye(4, dtype=np.float32),
    }
    np.savez_compressed(
        tmp_path / "uid-a.npz",
        cache_version=np.array(MV_FEATURE_CACHE_VERSION, dtype=np.int64),
        local_points=np.zeros((2, 1, 1, 3), dtype=np.float16),
        conf=np.zeros((2, 1, 1, 1), dtype=np.float16),
        dino_feats=np.zeros((2, 1, 1, 1), dtype=np.float16),
        view_indices=np.array([0, 0], dtype=np.int64),
        view_mask=np.array([False, False]),
        ref_view=np.array(0, dtype=np.int64),
        empty_reason=np.asarray("runtime rejected"),
        **_provenance_kwargs(),
    )
    # A real (non-marker) cache entry must exist as the shape template for the
    # zero cached features that empty items ship in cache mode.
    np.savez_compressed(
        tmp_path / "uid-template.npz",
        cache_version=np.array(MV_FEATURE_CACHE_VERSION, dtype=np.int64),
        local_points=np.zeros((2, 4, 5, 3), dtype=np.float16),
        conf=np.zeros((2, 4, 5, 1), dtype=np.float16),
        dino_feats=np.zeros((2, 8, 2, 3), dtype=np.float16),
        view_indices=np.array([0, 1], dtype=np.int64),
        view_mask=np.array([True, False]),
        ref_view=np.array(0, dtype=np.int64),
        **_provenance_kwargs(),
    )

    with pytest.warns(UserWarning, match="runtime rejected"):
        item = ds[0]

    assert not item["point_clouds_valid"]
    assert item["view_mask"].tolist() == [False, False]
    # Cache mode: empty items must carry template-shaped zero cached features and
    # pixel_values so a batch mixing real and empty examples keeps cached_* keys
    # (the collator only forwards them when every example has them, and the model
    # has no live geo_encoder fallback in cache mode).
    assert item["cached_local_points"].shape == (2, 4, 5, 3)
    assert item["cached_conf"].shape == (2, 4, 5, 1)
    assert item["cached_dino_feats"].shape == (2, 8, 2, 3)
    assert not item["cached_local_points"].any()
    assert tuple(item["pixel_values"].shape) == (2, 3, 4, 5)


def test_feature_cache_rejects_stale_old_format(tmp_path):
    ds = Trellis2MVDataset.__new__(Trellis2MVDataset)
    ds.feature_cache = tmp_path
    np.savez_compressed(
        tmp_path / "uid-a.npz",
        local_points=np.zeros((2, 4, 5, 3), dtype=np.float16),
        conf=np.ones((2, 4, 5, 1), dtype=np.float16),
        dino_feats=np.zeros((2, 8, 2, 3), dtype=np.float16),
        view_indices=np.array([3, 1], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="missing required cache keys"):
        ds._load_feature_cache("uid-a", n_avail=4)


def test_feature_cache_rejects_wrong_cache_version(tmp_path):
    ds = Trellis2MVDataset.__new__(Trellis2MVDataset)
    ds.feature_cache = tmp_path
    _write_cache_manifest(tmp_path)
    np.savez_compressed(
        tmp_path / "uid-a.npz",
        cache_version=np.array(1, dtype=np.int64),
        local_points=np.zeros((2, 4, 5, 3), dtype=np.float16),
        conf=np.ones((2, 4, 5, 1), dtype=np.float16),
        dino_feats=np.zeros((2, 8, 2, 3), dtype=np.float16),
        view_indices=np.array([3, 1], dtype=np.int64),
        view_mask=np.array([True, False]),
        ref_view=np.array(0, dtype=np.int64),
        **_provenance_kwargs(),
    )

    with pytest.raises(ValueError, match="cache_version=1"):
        ds._load_feature_cache("uid-a", n_avail=4)


def test_feature_cache_rejects_ref_view_outside_valid_mask(tmp_path):
    ds = Trellis2MVDataset.__new__(Trellis2MVDataset)
    ds.feature_cache = tmp_path
    _write_cache_manifest(tmp_path)
    np.savez_compressed(
        tmp_path / "uid-a.npz",
        cache_version=np.array(MV_FEATURE_CACHE_VERSION, dtype=np.int64),
        local_points=np.zeros((2, 4, 5, 3), dtype=np.float16),
        conf=np.ones((2, 4, 5, 1), dtype=np.float16),
        dino_feats=np.zeros((2, 8, 2, 3), dtype=np.float16),
        view_indices=np.array([3, 1], dtype=np.int64),
        view_mask=np.array([False, True]),
        ref_view=np.array(0, dtype=np.int64),
        **_provenance_kwargs(),
    )

    with pytest.raises(ValueError, match="view_mask\\[ref_view\\] is false"):
        ds._load_feature_cache("uid-a", n_avail=4)


def test_view_selection_policy_fingerprint_changes_with_policy_fields():
    base = DataConfig(type="trellis2_mv", path="unused")
    changed = DataConfig(type="trellis2_mv", path="unused", mv_mask_min_area_px=base.mv_mask_min_area_px + 1)
    same = DataConfig(type="trellis2_mv", path="unused")

    assert view_selection_policy_fingerprint(base) != view_selection_policy_fingerprint(changed)
    assert view_selection_policy_fingerprint(base) == view_selection_policy_fingerprint(same)


def test_mv_feature_cache_contract_fingerprint_covers_policy_and_artifact_bytes(tmp_path):
    _da3, dino, preproc = _write_artifact_dirs(tmp_path)
    data_cfg = DataConfig(
        type="trellis2_mv",
        path="unused",
        num_views=8,
        image_preprocessor=str(preproc),
    )
    model_cfg = _model_cfg_for_cache(tmp_path, dino)

    base = build_mv_feature_cache_contract(data_cfg, model_cfg)
    same = build_mv_feature_cache_contract(data_cfg, model_cfg)
    changed_views = build_mv_feature_cache_contract(
        DataConfig(
            type="trellis2_mv",
            path="unused",
            num_views=4,
            image_preprocessor=str(preproc),
        ),
        model_cfg,
    )
    (dino / "config.json").write_text(json.dumps({"name": "dino", "changed": True}))
    changed_dino = build_mv_feature_cache_contract(data_cfg, model_cfg)

    assert base["certifiable"]
    assert base["fingerprint"] == same["fingerprint"]
    assert base["fingerprint"] != changed_views["fingerprint"]
    assert base["fingerprint"] != changed_dino["fingerprint"]


def _cache_kwargs(fingerprint: str | None = TEST_CACHE_FP):
    kwargs = dict(
        cache_version=np.array(MV_FEATURE_CACHE_VERSION, dtype=np.int64),
        local_points=np.zeros((2, 4, 5, 3), dtype=np.float16),
        conf=np.ones((2, 4, 5, 1), dtype=np.float16),
        dino_feats=np.zeros((2, 8, 2, 3), dtype=np.float16),
        view_indices=np.array([3, 1], dtype=np.int64),
        view_mask=np.array([True, False]),
        ref_view=np.array(0, dtype=np.int64),
    )
    if fingerprint is not None:
        kwargs.update(_provenance_kwargs(fingerprint))
    return kwargs


def test_feature_cache_rejects_row_fingerprint_mismatch(tmp_path):
    ds = Trellis2MVDataset.__new__(Trellis2MVDataset)
    ds.feature_cache = tmp_path
    _write_cache_manifest(tmp_path, fingerprint="root-fp")
    np.savez_compressed(tmp_path / "uid-a.npz", **_cache_kwargs("row-fp"))

    with pytest.raises(ValueError, match="does not match root manifest"):
        ds._load_feature_cache("uid-a", n_avail=4)


def test_feature_cache_loads_when_row_fingerprint_matches_manifest(tmp_path):
    ds = Trellis2MVDataset.__new__(Trellis2MVDataset)
    ds.feature_cache = tmp_path
    _write_cache_manifest(tmp_path, fingerprint="same-fp")
    np.savez_compressed(tmp_path / "uid-a.npz", **_cache_kwargs("same-fp"))

    cache = ds._load_feature_cache("uid-a", n_avail=4)

    assert cache["view_indices"].tolist() == [3, 1]


def test_feature_cache_rejects_missing_provenance_legacy_cache(tmp_path):
    ds = Trellis2MVDataset.__new__(Trellis2MVDataset)
    ds.feature_cache = tmp_path
    _write_cache_manifest(tmp_path)
    np.savez_compressed(tmp_path / "uid-a.npz", **_cache_kwargs(None))

    with pytest.raises(ValueError, match="missing v3 provenance keys"):
        ds._load_feature_cache("uid-a", n_avail=4)


def test_mv_collator_keeps_only_plural_scene_transforms():
    data_cfg = DataConfig(
        type="3d-front-trellis2-mv",
        path="unused",
        num_pos_tokens=32,
        num_points=2,
        use_masked_obj_pc=True,
    )
    model_cfg = ModelConfig(
        vocab_size=64,
        num_pos_tokens=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        pc_token_id=3,
        prefix_len=2,
        tokenization_method="meshxl",
        max_seq_length=32,
        pos_token_offset=6,
        layout_tokenization_method="tri",
    )
    collator = get_mesh_data_collator(data_cfg, model_cfg)

    example = {
        "bboxes": np.zeros((1, 8, 3), dtype=np.float32),
        "obj_indices": 0,
        "vertices": None,
        "faces": None,
        "point_clouds": np.zeros((2, 3), dtype=np.float32),
        "point_clouds_2d": np.zeros((2, 2), dtype=np.float32),
        "point_clouds_valid": True,
        "scene_transforms": np.eye(4, dtype=np.float32)[None].repeat(2, axis=0),
        "K_per_view": np.eye(3, dtype=np.float32)[None].repeat(2, axis=0),
        "view_mask": np.array([True, False]),
    }

    batch = collator([example, example])

    assert "scene_transforms" in batch
    assert "scene_transform" not in batch
    assert batch["scene_transforms"].shape == (2, 2, 4, 4)
