from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import torch
import torch.nn as nn


def _mv_test_inputs(feat_dim: int = 4):
    obj_voxels = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.2, 0.0, 1.2], [0.0, 0.2, 1.4]]],
        dtype=torch.float32,
    )
    ctx_voxels = torch.tensor(
        [[[0.1, 0.1, 1.1], [0.3, 0.0, 1.5], [0.0, 0.3, 1.7], [0.2, 0.2, 1.3]]],
        dtype=torch.float32,
    )
    scene_transforms = torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    K_per_view = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 2, 1, 1)
    geo_depth = torch.full((1, 2, 4, 4), 3.0, dtype=torch.float32)
    view_mask = torch.ones(1, 2, dtype=torch.bool)
    panoptic_masks = torch.ones(1, 2, 4, 4, dtype=torch.long)
    target_ids = torch.ones(1, 2, dtype=torch.long)
    conf = torch.ones(1, 2, 4, 4, dtype=torch.float32)
    dino_feats = torch.arange(1 * 2 * feat_dim * 2 * 2, dtype=torch.float32)
    dino_feats = dino_feats.view(1, 2, feat_dim, 2, 2).to(torch.bfloat16)
    return {
        "obj_voxels": obj_voxels,
        "ctx_voxels": ctx_voxels,
        "scene_transforms": scene_transforms,
        "K_per_view": K_per_view,
        "geo_depth": geo_depth,
        "view_mask": view_mask,
        "panoptic_masks": panoptic_masks,
        "target_ids": target_ids,
        "conf": conf,
        "dino_feats": dino_feats,
    }


class _FakeDepthAnything3(nn.Module):
    last_instance: "_FakeDepthAnything3 | None" = None

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 1)
        self.forward_seen: dict[str, torch.dtype | bool] = {}

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        _ = args, kwargs
        inst = cls()
        cls.last_instance = inst
        return inst

    def forward(self, images: torch.Tensor, **kwargs):
        _ = kwargs
        self.forward_seen = {
            "dtype": images.dtype,
            "autocast": torch.is_autocast_enabled(images.device.type),
        }
        bsz, num_views, _channels, height, width = images.shape
        depth = torch.ones(
            bsz,
            num_views,
            height,
            width,
            device=images.device,
            dtype=images.dtype,
        )
        intrinsics = torch.eye(3, device=images.device, dtype=images.dtype)
        intrinsics = intrinsics.expand(bsz, num_views, 3, 3).clone()
        conf = torch.ones_like(depth)
        return {"depth": depth, "intrinsics": intrinsics, "conf": conf}


def test_da3_encoder_keeps_backbone_fp32_and_disables_autocast(monkeypatch):
    fake_pkg = types.ModuleType("depth_anything_3")
    fake_api = types.ModuleType("depth_anything_3.api")
    fake_api.DepthAnything3 = _FakeDepthAnything3
    fake_pkg.api = fake_api
    monkeypatch.setitem(sys.modules, "depth_anything_3", fake_pkg)
    monkeypatch.setitem(sys.modules, "depth_anything_3.api", fake_api)

    from src.models.utils import get_da3_encoder

    encoder = get_da3_encoder(
        SimpleNamespace(geo_encoder_type="da3", da3_ckpt_path="unused")
    )
    fake_model = _FakeDepthAnything3.last_instance
    assert fake_model is not None
    assert next(fake_model.parameters()).dtype == torch.float32

    pixel_values = torch.ones(1, 2, 3, 4, 4, dtype=torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = encoder.forward_all_views_joint(pixel_values)

    assert fake_model.forward_seen == {
        "dtype": torch.float32,
        "autocast": False,
    }
    assert out["local_points"].dtype == torch.bfloat16
    assert out["conf"].dtype == torch.bfloat16


def test_obj_view_feature_fusion_accepts_bf16_features_and_fp32_geometry_under_autocast():
    from src.models.edgerunner import ShapeOPT

    data = _mv_test_inputs(feat_dim=4)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = ShapeOPT._fuse_obj_view_features(
            None,
            data["obj_voxels"],
            data["dino_feats"],
            data["scene_transforms"],
            data["K_per_view"],
            data["geo_depth"],
            data["view_mask"],
            data["panoptic_masks"],
            data["target_ids"],
            data["conf"],
        )

    assert out.shape == (1, 3, 4)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


def test_mv_voxel_encoder_accepts_fp32_geometry_and_bf16_features_under_autocast():
    from src.models.mv_voxel_encoder import MultiViewVoxelAlignedEncoder

    data = _mv_test_inputs(feat_dim=4)
    encoder = MultiViewVoxelAlignedEncoder(
        feat_dim=4,
        voxel_dim=8,
        out_dim=6,
        num_obj_queries=2,
        num_scene_queries=1,
        num_heads=2,
        use_geometry=True,
    ).to(torch.bfloat16)

    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = encoder(
            data["obj_voxels"],
            data["ctx_voxels"],
            data["dino_feats"],
            data["scene_transforms"],
            data["K_per_view"],
            data["geo_depth"],
            data["view_mask"],
            panoptic_masks=data["panoptic_masks"],
            target_ids=data["target_ids"],
            conf=data["conf"],
            obj_geom_voxels=data["obj_voxels"],
        )

    assert out["z_i"].shape == (1, 2, 6)
    assert out["z_scene"].shape == (1, 1, 6)
    assert out["z_i"].dtype == torch.bfloat16
    assert out["z_scene"].dtype == torch.bfloat16
    assert torch.isfinite(out["z_i"].float()).all()
    assert torch.isfinite(out["z_scene"].float()).all()
