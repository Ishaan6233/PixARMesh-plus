from __future__ import annotations

import logging
from pathlib import Path

import torch

from .frozen_geo_encoder import FrozenGeoEncoder, register_geo_encoder

logger = logging.getLogger(__name__)


def _resolve_hf_cache_snapshot(path: str) -> str:
    """Resolve either a flat checkpoint dir or a Hugging Face cache root."""
    root = Path(path).expanduser()
    if (root / "config.json").exists():
        return str(root)

    snapshots = root / "snapshots"
    refs_main = root / "refs" / "main"
    if snapshots.exists():
        if refs_main.exists():
            commit = refs_main.read_text().strip()
            candidate = snapshots / commit
            if (candidate / "config.json").exists():
                return str(candidate)
        candidates = sorted(p for p in snapshots.iterdir() if (p / "config.json").exists())
        if candidates:
            return str(candidates[-1])

    return path


@register_geo_encoder("da3")
class Da3FrozenEncoder(FrozenGeoEncoder):
    """Frozen Depth Anything 3 wrapper for MV geometry conditioning.

    The wrapper consumes the repo's image-normalized tensor batch directly and calls DA3
    without dataset camera poses. DA3 predicts its own intrinsics; depth is converted to
    OpenCV camera-frame XYZ so the downstream discovery/voxel code sees the same contract
    as the previous frozen geometry backbone.
    """

    def __init__(self, ckpt_path: str):
        super().__init__()
        from depth_anything_3.api import DepthAnything3

        resolved = _resolve_hf_cache_snapshot(ckpt_path)
        logger.info("Loading DA3 frozen encoder from %s", resolved)
        self.model = DepthAnything3.from_pretrained(
            resolved,
            local_files_only=True,
        )
        self.model.float()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @classmethod
    def from_model_cfg(cls, model_cfg) -> "Da3FrozenEncoder":
        return cls(getattr(model_cfg, "da3_ckpt_path", "checkpoints/da3/DA3-GIANT"))

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if pixel_values.ndim != 4:
            raise ValueError(f"expected (B,3,H,W) pixel_values, got {tuple(pixel_values.shape)}")
        out = self.forward_all_views_joint(pixel_values[:, None])
        return {k: v[:, 0] for k, v in out.items()}

    @torch.no_grad()
    def forward_all_views_joint(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if pixel_values.ndim != 5:
            raise ValueError(
                f"expected (B,N,3,H,W) pixel_values, got {tuple(pixel_values.shape)}"
            )

        da3_inputs = pixel_values.to(dtype=torch.float32)
        with torch.autocast(device_type=da3_inputs.device.type, enabled=False):
            raw = self.model.forward(
                da3_inputs,
                extrinsics=None,
                intrinsics=None,
                export_feat_layers=[],
                infer_gs=False,
                use_ray_pose=False,
            )

        depth = self._get_raw_tensor(raw, ("depth",), required=False)
        intrinsics = self._get_raw_tensor(raw, ("intrinsics",), required=False)
        conf = self._get_raw_tensor(raw, ("depth_conf", "conf"), required=False)
        if depth is None or intrinsics is None:
            pred = self.model._convert_to_prediction(raw)
            depth = torch.as_tensor(pred.depth, device=pixel_values.device)
            intrinsics = torch.as_tensor(pred.intrinsics, device=pixel_values.device)
            conf = (
                torch.as_tensor(pred.conf, device=pixel_values.device)
                if pred.conf is not None
                else None
            )

        if depth.ndim == 3:
            depth = depth[:, None]
        if intrinsics.ndim == 3:
            intrinsics = intrinsics[None]
        if conf is None:
            conf = torch.ones_like(depth)
        elif conf.ndim == 3:
            conf = conf[:, None]

        depth = depth.to(device=pixel_values.device, dtype=torch.float32)
        intrinsics = intrinsics.to(device=pixel_values.device, dtype=torch.float32)
        conf = conf.to(device=pixel_values.device, dtype=torch.float32)
        if conf.ndim == 4:
            conf = conf.unsqueeze(-1)
        elif conf.ndim != 5:
            raise ValueError(f"expected DA3 conf rank 4 or 5, got {tuple(conf.shape)}")

        local_points = self._depth_to_camera_points(depth, intrinsics)
        out_dtype = pixel_values.dtype if pixel_values.is_floating_point() else torch.float32
        return {
            "local_points": local_points.to(out_dtype),
            "conf": conf.to(out_dtype),
        }

    @staticmethod
    def _get_raw_tensor(raw, names: tuple[str, ...], required: bool = True) -> torch.Tensor | None:
        for name in names:
            value = raw.get(name, None) if hasattr(raw, "get") else getattr(raw, name, None)
            if value is not None:
                return value
        if required:
            raise KeyError(f"DA3 raw output missing any of {names}")
        return None

    @staticmethod
    def _depth_to_camera_points(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        B, N, H, W = depth.shape
        device = depth.device
        dtype = depth.dtype

        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )
        xx = xx.view(1, 1, H, W)
        yy = yy.view(1, 1, H, W)

        fx = intrinsics[..., 0, 0].clamp(min=1e-6).view(B, N, 1, 1)
        fy = intrinsics[..., 1, 1].clamp(min=1e-6).view(B, N, 1, 1)
        cx = intrinsics[..., 0, 2].view(B, N, 1, 1)
        cy = intrinsics[..., 1, 2].view(B, N, 1, 1)

        x = (xx - cx) / fx * depth
        y = (yy - cy) / fy * depth
        z = depth
        return torch.stack((x, y, z), dim=-1)
