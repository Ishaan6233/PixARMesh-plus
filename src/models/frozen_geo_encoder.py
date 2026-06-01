"""
Abstract base class and shared utilities for frozen geometry encoder backbones.

A frozen geometry encoder takes DiNOv2-normalized RGB images and returns
per-pixel 3D points in OpenCV camera frame (X-right, Y-down, Z-forward).

Adding a new backbone:
  1. Subclass FrozenGeoEncoder and implement forward() + from_model_cfg().
  2. Decorate the class with @register_geo_encoder("your_name").
  3. Add any backbone-specific fields to ModelConfig.
  4. build_geo_encoder() dispatches automatically via geo_encoder_type.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_GEO_ENCODER_REGISTRY: dict[str, type["FrozenGeoEncoder"]] = {}


def register_geo_encoder(name: str):
    """Class decorator — registers a FrozenGeoEncoder subclass under `name`."""
    def _dec(cls: type["FrozenGeoEncoder"]) -> type["FrozenGeoEncoder"]:
        _GEO_ENCODER_REGISTRY[name] = cls
        return cls
    return _dec


def build_geo_encoder(model_cfg) -> "FrozenGeoEncoder | None":
    """Factory: construct the geo encoder specified by model_cfg.geo_encoder_type.

    Returns None when geo_encoder_type is empty (baseline — no geometry backbone).
    Raises ValueError for unknown types so misconfigured runs fail loudly.
    """
    enc_type = getattr(model_cfg, "geo_encoder_type", "")
    if not enc_type:
        return None
    if enc_type not in _GEO_ENCODER_REGISTRY:
        available = ", ".join(sorted(_GEO_ENCODER_REGISTRY))
        raise ValueError(
            f"Unknown geo_encoder_type {enc_type!r}. Available: {available}"
        )
    enc = _GEO_ENCODER_REGISTRY[enc_type].from_model_cfg(model_cfg)
    return enc.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class FrozenGeoEncoder(ABC, nn.Module):
    """Frozen geometry backbone producing per-pixel camera-frame 3D points.

    Contract
    --------
    Input:
        pixel_values — (B, 3, H, W) DiNOv2-preprocessed images (ImageNet mean/std,
                        bfloat16 or float16). Pass as-is from the batch; each subclass
                        handles its own internal dtype and normalization.
    Output:
        dict with:
          'local_points': (B, H, W, 3)  camera-frame XYZ; depth = local_points[..., 2]
          'conf':         (B, H, W, 1)  per-pixel confidence in (0, ∞]

    Properties
    ----------
    - All parameters must be frozen (requires_grad=False) before use.
    - state_dict() returns {} — weights are excluded from training checkpoints
      and always re-loaded from the pretrained checkpoint at startup.
    """

    @classmethod
    @abstractmethod
    def from_model_cfg(cls, model_cfg) -> "FrozenGeoEncoder":
        """Construct from a ModelConfig object (called by build_geo_encoder)."""
        ...

    @torch.no_grad()
    @abstractmethod
    def forward(self, pixel_values: torch.Tensor) -> dict:
        """See class docstring for I/O contract."""
        ...

    # -- Checkpoint exclusion ------------------------------------------------
    def state_dict(self, *args, **kwargs) -> dict:
        return {}

    def load_state_dict(self, state_dict, strict=True):
        return


# ---------------------------------------------------------------------------
# Shared point-cloud utilities (backbone-agnostic)
# ---------------------------------------------------------------------------

def build_geo_ctx_pc(
    local_points: torch.Tensor,
    conf: torch.Tensor,
    scene_transform: torch.Tensor,
    num_ctx_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a context point cloud from geo-encoder output and transform to scene space.

    Args:
        local_points:    (B, H, W, 3)  camera-frame 3D points
        conf:            (B, H, W, 1)  confidence weights
        scene_transform: (B, 4, 4)     camera-frame → normalised scene space
        num_ctx_points:  target number of context points to sample
    Returns:
        ctx_pcs:    (B, num_ctx_points, 3)  normalised scene space, same dtype as input
        ctx_pcs_2d: (B, num_ctx_points, 2)  normalised pixel coords [-1,1] (x=col, y=row)
    """
    B, H, W, _ = local_points.shape
    device    = local_points.device
    out_dtype = local_points.dtype

    # Build normalised pixel-grid — same convention as cond_pcs_2d in the collator
    h_lin = torch.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, device=device, dtype=torch.float32)
    w_lin = torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(h_lin, w_lin, indexing="ij")
    pix_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    pts_flat  = local_points.float().reshape(B, H * W, 3)
    conf_flat = conf.float().reshape(B, H * W)
    pix_flat  = pix_grid.reshape(B, H * W, 2)

    valid = pts_flat[..., 2] > 0   # only points with positive depth

    ctx_cam_list   = []
    ctx_pcs_2d_list = []
    for b in range(B):
        v       = valid[b]
        pts_b   = pts_flat[b][v]
        conf_b  = conf_flat[b][v].clamp(min=1e-6)
        pix_b   = pix_flat[b][v]
        n_valid = pts_b.shape[0]
        if n_valid == 0:
            # No valid depth predictions for this item — fall back to zeros.
            # The caller's warning_once in edgerunner.py covers the Pi3X-unavailable case;
            # zeros here avoid a crash and let the run continue (degenerate but safe).
            ctx_cam_list.append(torch.zeros(num_ctx_points, 3, device=pts_flat.device))
            ctx_pcs_2d_list.append(torch.zeros(num_ctx_points, 2, device=pix_flat.device))
            continue
        inds = torch.multinomial(conf_b, num_ctx_points, replacement=(n_valid < num_ctx_points))
        ctx_cam_list.append(pts_b[inds])
        ctx_pcs_2d_list.append(pix_b[inds])

    ctx_cam    = torch.stack(ctx_cam_list,    dim=0)   # (B, N, 3) float32
    ctx_pcs_2d = torch.stack(ctx_pcs_2d_list, dim=0)  # (B, N, 2) float32

    ctx_pcs = _apply_scene_transform(ctx_cam, scene_transform)   # float32 internally
    return ctx_pcs.to(out_dtype), ctx_pcs_2d


def build_geo_obj_pc(
    local_points: torch.Tensor,
    cond_pcs_2d: torch.Tensor,
    scene_transform: torch.Tensor,
) -> torch.Tensor:
    """Replace data-loader object PCs with geo-encoder 3D points at the same pixels.

    Preserves the data-loader's mask-based pixel selection while substituting
    richer encoder geometry for back-projected depth.

    Args:
        local_points:    (B, H, W, 3)  camera-frame points
        cond_pcs_2d:     (B, N_pts, 2) pixel coords in [-1,1] from data loader
        scene_transform: (B, 4, 4)
    Returns:
        (B, N_pts, 3) in normalised scene space, same dtype as local_points
    """
    out_dtype = local_points.dtype
    lp   = local_points.float().permute(0, 3, 1, 2)   # (B, 3, H, W)
    grid = cond_pcs_2d.float().unsqueeze(1)             # (B, 1, N_pts, 2)
    sampled = F.grid_sample(lp, grid, mode="bilinear", align_corners=False, padding_mode="zeros")
    obj_cam = sampled.squeeze(2).permute(0, 2, 1)       # (B, N_pts, 3)
    return _apply_scene_transform(obj_cam, scene_transform).to(out_dtype)


def _apply_scene_transform(pts: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Apply a (B, 4, 4) affine transform to (B, N, 3) points.

    Always operates in float32 regardless of input dtype — bfloat16 has only 7
    mantissa bits (~1 cm error at 1 m scale), which is unacceptable for geometry.
    Convention: p_out = p_in @ R.T + t  (row-vector, same as src.data.utils.transform_3d_points).
    """
    M_f = M.to(pts.device, dtype=torch.float32)
    R   = M_f[:, :3, :3]   # (B, 3, 3)
    t   = M_f[:, :3, 3]    # (B, 3)
    return torch.bmm(pts.float(), R.permute(0, 2, 1)) + t.unsqueeze(1)
