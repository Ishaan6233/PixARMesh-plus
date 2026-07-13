import os
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.data.typing import TokenType
from src.models.edgerunner import ShapeOPT, ShapeOPTConfig
from src.models.utils import get_model
from src.utils.config import ModelConfig, mv_prefix_len
from train import _build_model_config


LAYOUT_GEOMETRY_TOKENS = 24
SMALL_NUM_POS = 32
SMALL_POS_OFFSET = 6

GEOMETRY_LAYOUT_DEFAULTS = {
    "loss_layout_ordinal_sigma": 2.0,
    "loss_layout_ordinal_weight": 1.0,
    "loss_layout_coord_weight": 1.0,
}


class _DummyCondEncoder(torch.nn.Module):
    output_dim = 16

    def forward(self, *args, **kwargs):
        raise AssertionError("layout loss wiring tests should not execute cond_encoder")


def _compose_config(config_name: str, overrides: list[str] | None = None):
    os.environ.setdefault("RUN_TS", "pytest")
    OmegaConf.register_new_resolver("sub", lambda x, y: x - y, replace=True)
    repo_root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(repo_root / "configs"), version_base=None):
        return compose(config_name=config_name, overrides=overrides or [])


def _bbox_bins(num_pos_tokens: int) -> torch.Tensor:
    assert num_pos_tokens >= 16
    x0, x1 = num_pos_tokens // 8, num_pos_tokens // 3
    y0, y1 = num_pos_tokens // 7, num_pos_tokens // 2
    z0, z1 = num_pos_tokens // 6, (num_pos_tokens * 2) // 3
    return torch.tensor(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=torch.long,
    ).reshape(-1)


def _tiny_pretrained_shapeopt(tmp_path, model_cfg: ModelConfig) -> Path:
    torch.manual_seed(0)
    checkpoint = tmp_path / "tiny-shapeopt"
    config = ShapeOPTConfig(
        vocab_size=model_cfg.pos_token_offset + model_cfg.num_pos_tokens + 8,
        hidden_size=16,
        word_embed_proj_dim=16,
        ffn_dim=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        layerdrop=0.0,
        max_position_embeddings=max(model_cfg.max_position_embeddings, 64),
        bos_token_id=model_cfg.bos_token_id,
        eos_token_id=model_cfg.eos_token_id,
        pad_token_id=model_cfg.pad_token_id,
        num_pos_tokens=model_cfg.num_pos_tokens,
        pos_token_offset=model_cfg.pos_token_offset,
        with_ctx_pc=model_cfg.with_ctx_pc,
    )
    ShapeOPT(config, cond_encoder=_DummyCondEncoder()).save_pretrained(checkpoint)
    return checkpoint


def _layout_batch(model) -> dict[str, torch.Tensor]:
    bins = _bbox_bins(model.config.num_pos_tokens)
    labels = torch.full((1, LAYOUT_GEOMETRY_TOKENS + 1), -100, dtype=torch.long)
    labels[0, 1:] = model.config.pos_token_offset + bins

    input_ids = torch.full_like(labels, model.config.bos_token_id)
    input_ids[0, 1:] = labels[0, 1:]
    token_type_ids = torch.full_like(labels, TokenType.LAYOUT)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "token_type_ids": token_type_ids,
        "num_items_in_batch": torch.tensor(1.0),
    }


def _run_deterministic_layout_forward(model):
    model.eval()
    with torch.no_grad():
        model.lm_head.weight.zero_()
    return model(**_layout_batch(model))


def _assert_weighted_layout_total(model, output):
    assert output.loss is not None
    assert output.loss_layout is not None
    assert output.loss_layout_token is not None
    assert torch.allclose(output.loss_layout, output.loss_layout_token)
    assert output.loss_object is not None
    assert torch.allclose(output.loss_object, torch.zeros_like(output.loss_object))

    expected = output.loss_layout_token * LAYOUT_GEOMETRY_TOKENS
    for weight_name, component_name in (
        ("loss_layout_ordinal_weight", "loss_layout_ordinal"),
        ("loss_layout_coord_weight", "loss_layout_coord"),
    ):
        weight = float(getattr(model.config, weight_name))
        component = getattr(output, component_name)
        if weight:
            assert component is not None, f"{component_name} missing despite {weight_name}={weight}"
            assert torch.isfinite(component).all()
            assert component.item() > 0.0
            expected = expected + weight * component

    assert torch.allclose(output.loss, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("config_name", "ignore_obj_seq"),
    [
        ("edgerunner_3d_front_trellis2_mv_stage1", True),
        ("edgerunner_3d_front_trellis2_mv_stage2", False),
    ],
)
def test_mv_stage_configs_round_trip_layout_losses_through_loader_and_forward(
    tmp_path, config_name, ignore_obj_seq
):
    cfg = _compose_config(config_name)
    model_cfg = _build_model_config(cfg)

    assert cfg.dataset.src_data.ignore_obj_seq is ignore_obj_seq
    assert model_cfg.prefix_len == 2371
    assert model_cfg.loss_layout_geometry_tokens == LAYOUT_GEOMETRY_TOKENS
    for field, expected in GEOMETRY_LAYOUT_DEFAULTS.items():
        assert getattr(model_cfg, field) == expected

    checkpoint = _tiny_pretrained_shapeopt(tmp_path, model_cfg)
    model = get_model(str(checkpoint), model_cfg, cond_encoder=_DummyCondEncoder())
    for field, expected in GEOMETRY_LAYOUT_DEFAULTS.items():
        assert getattr(model.config, field) == expected

    output = _run_deterministic_layout_forward(model)
    _assert_weighted_layout_total(model, output)


@pytest.mark.parametrize(
    ("weight_name", "component_name", "weight", "sigma"),
    [
        ("loss_layout_ordinal_weight", "loss_layout_ordinal", 1.25, 2.0),
        ("loss_layout_coord_weight", "loss_layout_coord", 1.5, None),
    ],
)
def test_each_layout_loss_weight_independently_reaches_forward_total(
    tmp_path, weight_name, component_name, weight, sigma
):
    kwargs = {
        "vocab_size": SMALL_POS_OFFSET + SMALL_NUM_POS + 8,
        "num_pos_tokens": SMALL_NUM_POS,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "pos_token_offset": SMALL_POS_OFFSET,
        "ar_model_type": "edgerunner",
        "max_position_embeddings": 64,
        "loss_layout_geometry_tokens": LAYOUT_GEOMETRY_TOKENS,
        "loss_layout_ordinal_sigma": sigma,
        "loss_layout_ordinal_weight": 0.0,
        "loss_layout_coord_weight": 0.0,
    }
    kwargs[weight_name] = weight
    model_cfg = ModelConfig(**kwargs)

    checkpoint = _tiny_pretrained_shapeopt(tmp_path, model_cfg)
    model = get_model(str(checkpoint), model_cfg, cond_encoder=_DummyCondEncoder())
    output = _run_deterministic_layout_forward(model)

    component = getattr(output, component_name)
    assert component is not None
    assert torch.isfinite(component).all()
    expected = output.loss_layout_token * LAYOUT_GEOMETRY_TOKENS + weight * component
    assert torch.allclose(output.loss, expected, rtol=1e-6, atol=1e-6)


def test_mv_ce_control_round_trip_disables_auxiliary_layout_losses(tmp_path):
    cfg = _compose_config(
        "edgerunner_3d_front_trellis2_mv_stage1",
        overrides=["+experiment=mv_layout_loss_ce"],
    )
    model_cfg = _build_model_config(cfg)

    assert model_cfg.prefix_len == mv_prefix_len(model_cfg)
    assert model_cfg.loss_layout_ordinal_sigma is None
    assert model_cfg.loss_layout_ordinal_weight == 0.0
    assert model_cfg.loss_layout_coord_weight == 0.0

    checkpoint = _tiny_pretrained_shapeopt(tmp_path, model_cfg)
    model = get_model(str(checkpoint), model_cfg, cond_encoder=_DummyCondEncoder())
    output = _run_deterministic_layout_forward(model)

    assert output.loss_layout_token is not None
    assert output.loss_layout_ordinal is None
    assert output.loss_layout_coord is None
    assert torch.allclose(
        output.loss,
        output.loss_layout_token * LAYOUT_GEOMETRY_TOKENS,
        rtol=1e-6,
        atol=1e-6,
    )
