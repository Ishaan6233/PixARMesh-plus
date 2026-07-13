import pytest
import torch

from src.models.utils import _validate_loading_report, build_mv_architecture_contract
from src.utils.config import ModelConfig


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mv_voxel_encoder = torch.nn.Linear(2, 2)
        self.decoder = torch.nn.Linear(2, 2)


def _report(*, missing=(), unexpected=(), mismatched=()):
    return {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "mismatched_keys": [{"key": key, "checkpoint_shape": "old", "model_shape": "new"} for key in mismatched],
        "error_msgs": [],
    }


def _model_cfg(**overrides):
    values = dict(
        vocab_size=64,
        num_pos_tokens=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        pc_token_id=3,
        prefix_len=10,
        pc_latent_len=4,
        tokenization_method="meshxl",
        max_seq_length=32,
        pos_token_offset=6,
        layout_tokenization_method="tri",
        mv_voxel_encoder=True,
        mv_obj_pc_cond=True,
        mv_num_obj_queries=3,
        mv_num_scene_queries=2,
        mv_obj_aabb_token=True,
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_strict_warm_start_rejects_missing_trainable_tensor():
    model = TinyModel()

    with pytest.raises(ValueError, match="strict_warm_start rejected"):
        _validate_loading_report(
            model,
            _report(missing=["decoder.weight"]),
            "strict_warm_start",
        )


def test_base_init_allows_documented_mv_missing_tensor_only():
    model = TinyModel()

    _validate_loading_report(
        model,
        _report(missing=["mv_voxel_encoder.weight"]),
        "base_init",
    )
    with pytest.raises(ValueError, match="base_init only allows"):
        _validate_loading_report(
            model,
            _report(missing=["decoder.weight"]),
            "base_init",
        )


def test_mv_architecture_contract_changes_with_prefix_layout():
    base = build_mv_architecture_contract(_model_cfg())
    changed = build_mv_architecture_contract(_model_cfg(mv_num_obj_queries=4, prefix_len=11))

    assert base["fingerprint"] != changed["fingerprint"]
    assert base["prefix_decomposition"]["total"] == 10
