import argparse
import json

import numpy as np
import pytest
import torch

from scripts.eval.eval_layout_mv import (
    LAYOUT_COORD_TOKENS,
    apply_view_ablation,
    bbox_metrics,
    decode_layout_tokens,
    filter_batch_rows,
    row_view_usage,
    save_visuals,
    select_visual_payloads,
    summarize_view_usage,
)


def test_shuffle_views_keeps_cameras_fixed_and_shuffles_observations():
    batch = {
        "view_mask": torch.tensor([[True, True, True]]),
        "pixel_values": torch.arange(3).view(1, 3, 1, 1, 1).float(),
        "cached_local_points": torch.arange(3).view(1, 3, 1, 1, 1).float(),
        "cached_conf": torch.arange(3).view(1, 3, 1, 1, 1).float(),
        "cached_dino_feats": torch.arange(3).view(1, 3, 1, 1, 1).float(),
        "panoptic_masks": torch.arange(3).view(1, 3, 1, 1),
        "scene_transforms": torch.arange(48).view(1, 3, 4, 4).float(),
        "K_per_view": torch.arange(27).view(1, 3, 3, 3).float(),
        "ref_view": torch.tensor([0]),
    }
    original_scene = batch["scene_transforms"].clone()
    original_k = batch["K_per_view"].clone()

    meta = apply_view_ablation(
        batch,
        argparse.Namespace(reference_only=False, view_limit=0, shuffle_views=True),
    )

    assert batch["pixel_values"].flatten().tolist() == [2.0, 1.0, 0.0]
    assert batch["cached_local_points"].flatten().tolist() == [2.0, 1.0, 0.0]
    assert torch.equal(batch["scene_transforms"], original_scene)
    assert torch.equal(batch["K_per_view"], original_k)
    assert batch["ref_view"].item() == 0
    assert meta["shuffle_perm"] == [2, 1, 0]
    assert "cameras" in meta["shuffle_semantics"]
    assert meta["input_valid_view_counts"] == [3]
    assert meta["enabled_view_counts"] == [3]
    assert meta["ref_views"] == [0]


def test_view_limit_records_before_and_after_view_counts():
    batch = {
        "view_mask": torch.tensor([[True, True, True, False], [True, False, True, True]]),
        "ref_view": torch.tensor([0, 2]),
    }

    meta = apply_view_ablation(
        batch,
        argparse.Namespace(reference_only=False, view_limit=2, shuffle_views=False),
    )

    assert meta["num_view_slots"] == 4
    assert meta["input_valid_view_counts"] == [3, 3]
    assert meta["enabled_view_counts"] == [2, 2]
    assert row_view_usage(meta, 1) == {
        "input_valid_view_count": 3,
        "enabled_view_count": 2,
        "num_view_slots": 4,
        "enabled_view_fraction": 0.5,
        "ref_view": 2,
    }
    assert batch["view_mask"].tolist() == [[True, True, False, False], [True, False, True, False]]


def test_summarize_view_usage_reports_multiview_fraction():
    summary = summarize_view_usage(
        [
            {
                "view_usage": {
                    "input_valid_view_count": 3,
                    "enabled_view_count": 3,
                    "num_view_slots": 4,
                    "enabled_view_fraction": 0.75,
                }
            },
            {
                "view_usage": {
                    "input_valid_view_count": 2,
                    "enabled_view_count": 1,
                    "num_view_slots": 4,
                    "enabled_view_fraction": 0.25,
                }
            },
        ]
    )

    assert summary["num_records_with_view_usage"] == 2
    assert summary["mean_input_valid_view_count"] == pytest.approx(2.5)
    assert summary["mean_enabled_view_count"] == pytest.approx(2.0)
    assert summary["multi_view_record_frac"] == pytest.approx(0.5)
    assert summary["single_view_record_frac"] == pytest.approx(0.5)
    assert summary["mean_enabled_view_fraction"] == pytest.approx(0.5)


def test_reference_only_rejects_masked_ref_view():
    batch = {
        "view_mask": torch.tensor([[False, True]]),
        "ref_view": torch.tensor([0]),
    }

    with pytest.raises(ValueError, match="ref_view=0"):
        apply_view_ablation(
            batch,
            argparse.Namespace(reference_only=True, view_limit=0, shuffle_views=False),
        )


def test_filter_batch_rows_keeps_valid_examples_only():
    batch = {
        "uid": ["bad", "good"],
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "view_mask": torch.tensor([[False, False], [True, False]]),
        "global": torch.tensor([9]),
    }

    filtered = filter_batch_rows(batch, torch.tensor([False, True]))

    assert filtered["uid"] == ["good"]
    assert filtered["input_ids"].tolist() == [[3, 4]]
    assert filtered["view_mask"].tolist() == [[True, False]]
    assert filtered["global"].tolist() == [9]


def test_decode_layout_tokens_keeps_raw_invalid_bins_for_metrics():
    tokens = np.array([[4] * 24], dtype=np.int64)

    decoded = decode_layout_tokens(tokens, pos_token_offset=6, num_pos_tokens=32)

    assert decoded.raw_bins.min() == -2
    assert not decoded.token_valid.any()
    assert not decoded.sample_valid[0]
    assert decoded.invalid_reasons == ["invalid_position_token"]


def _layout_metrics_for(decoded):
    gt_tokens = np.full(LAYOUT_COORD_TOKENS, 6, dtype=np.int64)
    gt = decode_layout_tokens(gt_tokens[None, :], pos_token_offset=6, num_pos_tokens=32)
    return bbox_metrics(
        decoded.boxes[0],
        gt.boxes[0],
        decoded.token_valid[0],
        gt.raw_bins[0],
        decoded.raw_bins[0],
        gt.scored_tokens[0],
        decoded.scored_tokens[0],
        bool(decoded.structural_valid[0]),
    )


def test_decode_layout_tokens_marks_empty_sequence_invalid():
    decoded = decode_layout_tokens(
        np.empty((1, 0), dtype=np.int64), pos_token_offset=6, num_pos_tokens=32
    )

    assert decoded.token_counts.tolist() == [0]
    assert decoded.invalid_reasons == ["empty"]
    assert not decoded.structural_valid[0]
    assert not decoded.sample_valid[0]
    assert decoded.scored_tokens.shape == (1, LAYOUT_COORD_TOKENS)
    assert _layout_metrics_for(decoded)["valid_token_frac"] == 0.0


def test_decode_layout_tokens_marks_one_token_sequence_invalid():
    decoded = decode_layout_tokens(
        np.array([[2]], dtype=np.int64), pos_token_offset=6, num_pos_tokens=32
    )

    assert decoded.token_counts.tolist() == [1]
    assert decoded.tokens[0].tolist() == [2]
    assert decoded.invalid_reasons == ["short"]
    assert not decoded.structural_valid[0]
    assert not decoded.sample_valid[0]
    assert _layout_metrics_for(decoded)["valid_token_frac"] == 0.0


def test_decode_layout_tokens_marks_truncated_sequence_invalid():
    tokens = np.full((1, LAYOUT_COORD_TOKENS - 1), 6, dtype=np.int64)

    decoded = decode_layout_tokens(tokens, pos_token_offset=6, num_pos_tokens=32)

    assert decoded.token_counts.tolist() == [LAYOUT_COORD_TOKENS - 1]
    assert decoded.invalid_reasons == ["truncated"]
    assert not decoded.structural_valid[0]
    assert not decoded.sample_valid[0]
    assert _layout_metrics_for(decoded)["valid_token_frac"] == 0.0


def test_decode_layout_tokens_marks_overlong_sequence_invalid():
    tokens = np.full((1, LAYOUT_COORD_TOKENS + 1), 6, dtype=np.int64)

    decoded = decode_layout_tokens(tokens, pos_token_offset=6, num_pos_tokens=32)

    assert decoded.token_counts.tolist() == [LAYOUT_COORD_TOKENS + 1]
    assert decoded.invalid_reasons == ["overlong"]
    assert not decoded.structural_valid[0]
    assert not decoded.sample_valid[0]
    assert decoded.tokens[0].tolist() == [6] * (LAYOUT_COORD_TOKENS + 1)
    assert decoded.scored_tokens[0].tolist() == [6] * LAYOUT_COORD_TOKENS
    assert _layout_metrics_for(decoded)["valid_token_frac"] == 0.0


def test_decode_layout_tokens_accepts_terminal_eos_after_layout_tokens():
    tokens = np.array([[*([6] * LAYOUT_COORD_TOKENS), 2]], dtype=np.int64)

    decoded = decode_layout_tokens(tokens, pos_token_offset=6, num_pos_tokens=32)

    assert decoded.token_counts.tolist() == [LAYOUT_COORD_TOKENS + 1]
    assert decoded.invalid_reasons == [None]
    assert decoded.structural_valid[0]
    assert decoded.sample_valid[0]
    assert decoded.scored_tokens[0].tolist() == [6] * LAYOUT_COORD_TOKENS
    assert _layout_metrics_for(decoded)["valid_token_frac"] == 1.0


def test_select_visual_payloads_supports_worst_and_fixed_uids():
    payloads = [
        {"uid": "a", "record": {"aabb_iou": 0.8, "bin_mae": 1.0, "center_error": 1.0, "valid_token_frac": 1.0}},
        {"uid": "b", "record": {"aabb_iou": 0.1, "bin_mae": 9.0, "center_error": 4.0, "valid_token_frac": 1.0}},
    ]

    worst = select_visual_payloads(payloads, mode="worst", max_visuals=1, fixed_uids=set())
    fixed = select_visual_payloads(payloads, mode="first", max_visuals=1, fixed_uids={"a"})

    assert worst[0][0]["uid"] == "b"
    assert worst[0][1]["mode"] == "worst"
    assert fixed[0][0]["uid"] == "a"
    assert fixed[0][1]["mode"] == "fixed_uids"


def test_save_visuals_writes_projection_and_conditioning_metadata(tmp_path):
    batch = {
        "cond_pcs": torch.zeros(1, 2, 3),
        "scene_transforms": torch.eye(4).view(1, 1, 4, 4),
        "K_per_view": torch.eye(3).view(1, 1, 3, 3),
        "view_mask": torch.tensor([[True]]),
        "panoptic_masks": torch.zeros(1, 1, 8, 8, dtype=torch.long),
        "view_indices": torch.tensor([[0]]),
    }
    diagnostics = {
        "obj_voxels": torch.zeros(1, 2, 3),
        "ctx_voxels": torch.zeros(1, 2, 3),
        "seed_pcs": torch.zeros(1, 2, 3),
        "seed_pcs_2d": torch.zeros(1, 2, 2),
        "view_mask": torch.tensor([[True]]),
        "ref_view": torch.tensor([0]),
    }
    box = np.array(
        [
            [-0.1, -0.1, 1.0],
            [0.1, -0.1, 1.0],
            [0.1, 0.1, 1.0],
            [-0.1, 0.1, 1.0],
            [-0.1, -0.1, 1.2],
            [0.1, -0.1, 1.2],
            [0.1, 0.1, 1.2],
            [-0.1, 0.1, 1.2],
        ],
        dtype=np.float32,
    )
    record = {
        "token_accuracy": 1.0,
        "valid_token_frac": 1.0,
        "bin_mae": 0.0,
        "corner_l1": 0.0,
        "corner_l2": 0.0,
        "center_error": 0.0,
        "size_rel_error": 0.0,
        "aabb_iou": 1.0,
    }

    save_visuals(
        tmp_path,
        "uid-a",
        box,
        box,
        batch,
        0,
        diagnostics,
        record=record,
        selection_meta={"mode": "fixed_uids"},
        ablation_meta={"shuffle_perm": [0]},
    )

    meta = json.loads((tmp_path / "visuals" / "uid-a" / "conditioning.json").read_text())
    points = np.load(tmp_path / "visuals" / "uid-a" / "conditioning_points.npz")
    assert meta["projection"]["per_view"][0]["gt_positive_depth_corners"] == 8
    assert meta["selection"]["mode"] == "fixed_uids"
    assert points["seed_pcs"].shape == (2, 3)
