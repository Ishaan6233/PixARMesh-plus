"""Multi-view voxel-aligned encoder for DA3/Trellis2-MV conditioning.

The encoder consumes scene-frame object/context voxels, samples frozen per-view
features at their image projections, fuses visible views with an IBRNet-style
mean/variance statistic, injects explicit voxel geometry, and pools the result
into object and scene latent tokens for the OPT decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pc_edgerunner.encoder import PointEmbed


def _project_to_views(
    voxels_scene: torch.Tensor,
    scene_transforms: torch.Tensor,
    K_per_view: torch.Tensor,
    H: int,
    W: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project scene-frame voxels into every view using align_corners=False coords.

    Args:
        voxels_scene: (B, V, 3) scene-space points.
        scene_transforms: (B, N, 4, 4) camera-to-scene transforms.
        K_per_view: (B, N, 3, 3) camera intrinsics.
        H, W: full-resolution image/depth map size in pixels.

    Returns:
        pix_coords: (B, V, N, 2) normalized grid_sample coordinates.
        voxels_cam: (B, V, N, 3) camera-frame voxels.
    """
    B, V, _ = voxels_scene.shape
    st_inv = torch.linalg.inv(scene_transforms.float())
    voxels_h = torch.cat(
        [voxels_scene.float(), torch.ones(B, V, 1, device=voxels_scene.device)],
        dim=-1,
    )
    voxels_cam_h = voxels_h.unsqueeze(1) @ st_inv.permute(0, 1, 3, 2)
    voxels_cam = voxels_cam_h[..., :3].permute(0, 2, 1, 3)

    vc_nv = voxels_cam.permute(0, 2, 1, 3)
    projected = (
        vc_nv.unsqueeze(-2) @ K_per_view.float().unsqueeze(2).transpose(-1, -2)
    ).squeeze(-2)
    z = projected[..., 2].clamp(min=1e-6)
    u = projected[..., 0] / z
    v = projected[..., 1] / z

    u_norm = (u + 0.5) / W * 2.0 - 1.0
    v_norm = (v + 0.5) / H * 2.0 - 1.0
    pix_coords = torch.stack([u_norm, v_norm], dim=-1).permute(0, 2, 1, 3)
    return pix_coords, voxels_cam


def _sample_features(
    img_feats: torch.Tensor,
    pix_coords: torch.Tensor,
) -> torch.Tensor:
    """Sample per-view feature maps at projected voxel coordinates.

    Args:
        img_feats: (B, N, C, Hf, Wf).
        pix_coords: (B, V, N, 2) normalized full-image coordinates.

    Returns:
        (B, V, N, C) sampled feature tensor.
    """
    B, N, C, Hf, Wf = img_feats.shape
    V = pix_coords.shape[1]
    feats_flat = img_feats.reshape(B * N, C, Hf, Wf)
    coords_flat = pix_coords.permute(0, 2, 1, 3).reshape(B * N, V, 1, 2)
    sampled = F.grid_sample(
        feats_flat,
        coords_flat,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    sampled = sampled.squeeze(-1).permute(0, 2, 1)
    return sampled.reshape(B, N, V, C).permute(0, 2, 1, 3)


def _compute_visibility_mask(
    voxels_cam: torch.Tensor,
    geo_depth: torch.Tensor,
    pix_coords: torch.Tensor,
    view_mask: torch.Tensor,
    depth_rtol: float = 0.05,
    panoptic_masks: torch.Tensor | None = None,
    target_ids: torch.Tensor | None = None,
    geometry_only: bool = False,
) -> torch.Tensor:
    """Compute per-voxel-per-view visibility.

    Object voxels use mask consensus when panoptic masks and target IDs are supplied.
    Context voxels use positive-depth plus in-frame geometry (`geometry_only=True`).
    The depth occlusion fallback is retained only for callers without masks.
    """
    B, V, N, _ = voxels_cam.shape
    H, W = geo_depth.shape[-2:]
    z_ok = voxels_cam[..., 2] > 0
    vm = view_mask[:, None, :].expand(B, V, N)
    coords_flat = pix_coords.permute(0, 2, 1, 3).reshape(B * N, V, 1, 2)

    if panoptic_masks is not None and target_ids is not None:
        Hp, Wp = panoptic_masks.shape[-2:]
        masks_flat = panoptic_masks.float().reshape(B * N, 1, Hp, Wp)
        ids = F.grid_sample(
            masks_flat,
            coords_flat,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        ids = ids.squeeze(-1).squeeze(1).reshape(B, N, V).permute(0, 2, 1).round().long()
        target = target_ids[:, None, :].expand(B, V, N)
        return z_ok & (ids == target) & (target > 0) & vm

    depth_flat = geo_depth.float().reshape(B * N, 1, H, W)
    sampled_depth = F.grid_sample(
        depth_flat,
        coords_flat,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    sampled_depth = sampled_depth.squeeze(-1).squeeze(1).reshape(B, N, V).permute(0, 2, 1)
    in_bounds = sampled_depth > 0
    if geometry_only:
        return z_ok & in_bounds & vm
    depth_ok = voxels_cam[..., 2] <= sampled_depth * (1.0 + depth_rtol)
    return z_ok & depth_ok & in_bounds & vm


def _ibr_fusion(
    per_view_feats: torch.Tensor,
    vis_mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse visible per-view features as weighted mean plus variance."""
    mask = vis_mask.unsqueeze(-1).float()
    if weights is not None:
        mask = mask * weights.unsqueeze(-1).clamp(min=0.0)
        denom = mask.sum(dim=2).clamp(min=1e-6)
    else:
        denom = mask.sum(dim=2).clamp(min=1.0)
    mu = (per_view_feats * mask).sum(dim=2) / denom
    mu2 = ((per_view_feats**2) * mask).sum(dim=2) / denom
    var = (mu2 - mu**2).clamp(min=0.0)
    return torch.cat([mu, var], dim=-1)


class MultiViewVoxelAlignedEncoder(nn.Module):
    """Voxel-aligned MV encoder without deformable cross-attention refinement."""

    def __init__(
        self,
        feat_dim: int,
        voxel_dim: int,
        out_dim: int,
        num_obj_queries: int,
        num_scene_queries: int,
        num_heads: int,
        use_geometry: bool = True,
    ):
        super().__init__()
        self.voxel_dim = voxel_dim
        self.feat_dim = feat_dim

        self.fusion_proj = nn.Linear(2 * feat_dim, voxel_dim)
        self.use_geometry = use_geometry
        self.point_embed = PointEmbed(dim=voxel_dim) if use_geometry else None

        self.obj_queries = nn.Parameter(torch.randn(num_obj_queries, voxel_dim) * 0.02)
        self.scene_queries = nn.Parameter(torch.randn(num_scene_queries, voxel_dim) * 0.02)

        self.obj_cross_attn = nn.MultiheadAttention(voxel_dim, num_heads, batch_first=True)
        self.scene_cross_attn = nn.MultiheadAttention(voxel_dim, num_heads, batch_first=True)
        self.scene_ctx_cross_attn = nn.MultiheadAttention(voxel_dim, num_heads, batch_first=True)
        self.scene_ctx_norm = nn.LayerNorm(voxel_dim)

        self.obj_out_proj = nn.Linear(voxel_dim, out_dim)
        self.scene_out_proj = nn.Linear(voxel_dim, out_dim)

    def _process_voxels(
        self,
        voxels: torch.Tensor,
        dino_feats: torch.Tensor,
        scene_transforms: torch.Tensor,
        K_per_view: torch.Tensor,
        geo_depth: torch.Tensor,
        view_mask: torch.Tensor,
        conf: torch.Tensor | None = None,
        panoptic_masks: torch.Tensor | None = None,
        target_ids: torch.Tensor | None = None,
        geometry_only: bool = False,
        geom_voxels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, V, _ = voxels.shape
        N = scene_transforms.shape[1]
        H_full, W_full = geo_depth.shape[-2:]

        pix_coords, voxels_cam = _project_to_views(
            voxels, scene_transforms, K_per_view, H_full, W_full
        )
        vis_mask = _compute_visibility_mask(
            voxels_cam,
            geo_depth,
            pix_coords,
            view_mask,
            panoptic_masks=panoptic_masks,
            target_ids=target_ids,
            geometry_only=geometry_only,
        )

        conf_vox = None
        if conf is not None:
            coords_flat = pix_coords.permute(0, 2, 1, 3).reshape(B * N, V, 1, 2)
            conf_map = conf.float().reshape(B * N, 1, H_full, W_full)
            conf_vox = F.grid_sample(
                conf_map,
                coords_flat,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            conf_vox = conf_vox.squeeze(-1).squeeze(1).reshape(B, N, V).permute(0, 2, 1)

        per_view_feats = _sample_features(dino_feats, pix_coords).to(voxels.dtype)

        fused = _ibr_fusion(per_view_feats.float(), vis_mask, weights=conf_vox)
        voxel_feats = self.fusion_proj(fused.to(next(self.fusion_proj.parameters()).dtype))

        if self.point_embed is not None:
            pe_in = geom_voxels if geom_voxels is not None else voxels
            voxel_feats = voxel_feats + self.point_embed(pe_in.to(voxel_feats.dtype))

        return voxel_feats

    def forward(
        self,
        obj_voxels: torch.Tensor,
        ctx_voxels: torch.Tensor,
        dino_feats: torch.Tensor,
        scene_transforms: torch.Tensor,
        K_per_view: torch.Tensor,
        geo_depth: torch.Tensor,
        view_mask: torch.Tensor,
        panoptic_masks: torch.Tensor | None = None,
        target_ids: torch.Tensor | None = None,
        conf: torch.Tensor | None = None,
        obj_geom_voxels: torch.Tensor | None = None,
        obj_view_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B = obj_voxels.shape[0]
        eff_obj_mask = obj_view_mask if obj_view_mask is not None else view_mask

        obj_voxel_feats = self._process_voxels(
            obj_voxels,
            dino_feats,
            scene_transforms,
            K_per_view,
            geo_depth,
            eff_obj_mask,
            conf=conf,
            panoptic_masks=panoptic_masks,
            target_ids=target_ids,
            geom_voxels=obj_geom_voxels,
        )
        ctx_voxel_feats = self._process_voxels(
            ctx_voxels,
            dino_feats,
            scene_transforms,
            K_per_view,
            geo_depth,
            view_mask,
            conf=conf,
            geometry_only=True,
        )

        obj_q = self.obj_queries.unsqueeze(0).expand(B, -1, -1).to(obj_voxel_feats)
        z_i, _ = self.obj_cross_attn(obj_q, obj_voxel_feats, obj_voxel_feats)

        all_voxel_feats = torch.cat([obj_voxel_feats, ctx_voxel_feats], dim=1)
        scene_q = self.scene_queries.unsqueeze(0).expand(B, -1, -1).to(all_voxel_feats)
        z_scene, _ = self.scene_cross_attn(scene_q, all_voxel_feats, all_voxel_feats)

        z_i_ctx, _ = self.scene_ctx_cross_attn(z_i, z_scene, z_scene)
        z_i = self.scene_ctx_norm(z_i + z_i_ctx)

        return {
            "z_i": self.obj_out_proj(z_i),
            "z_scene": self.scene_out_proj(z_scene),
        }
