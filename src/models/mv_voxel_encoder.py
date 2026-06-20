"""Multi-view Voxel-Aligned PC-Encoder.

Given:
  - obj_voxels  (B, V_obj, 3) scene space (from cond_pcs via Pi3X)
  - ctx_voxels  (B, V_ctx, 3) scene space (merged multi-view geometry)
  - dino_feats  (B, N, C_d, H', W') spatial DINOv2 feature maps
  - mask_feats  (B, N, C_m, Hm, Wm) binary-projection-mask conv features (ShapeR-style)
  - scene_transforms (B, N, 4, 4)  camera_n → scene
  - K_per_view       (B, N, 3, 3)  GT intrinsics
  - pi3x_depth       (B, N, H, W)  local_points[..., 2] from Pi3X (full resolution)
  - view_mask        (B, N) bool

Produces z_i (B, M, out_dim) and z_scene (B, S, out_dim) for the OPT decoder.

Feature resolution note: dino_feats and mask_feats may have different spatial sizes.
Rather than concatenating them as feature maps, we sample each separately at each
voxel's projected pixel coordinates and concatenate the per-voxel per-view vectors.
This avoids any spatial alignment requirement between the two feature sources.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from .frozen_geo_encoder import _apply_scene_transform, fps_centroid_seeded
from .pc_edgerunner.encoder import PointEmbed


# ---------------------------------------------------------------------------
# Projection and sampling helpers
# ---------------------------------------------------------------------------

def _project_to_views(
    voxels_scene: torch.Tensor,       # (B, V, 3) scene space
    scene_transforms: torch.Tensor,   # (B, N, 4, 4) cam_n → scene  (invert for reverse)
    K_per_view: torch.Tensor,         # (B, N, 3, 3)
    H: int,
    W: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project voxels from scene space into every view's image plane.

    Coordinate convention: align_corners=False.
      pixel center i ↔ grid coord (i + 0.5) / S * 2 − 1   (S = H or W)

    Returns:
        pix_coords: (B, V, N, 2)  normalised [-1, 1] for grid_sample  (x=col, y=row)
        voxels_cam: (B, V, N, 3)  voxels in each camera frame  (z = depth)
    """
    B, V, _ = voxels_scene.shape
    N = scene_transforms.shape[1]

    # Invert scene_transforms: scene → camera_n
    st_inv = torch.linalg.inv(scene_transforms.float())   # (B, N, 4, 4)

    # Homogeneous voxels: (B, V, 4)
    v_hom = torch.cat([
        voxels_scene.float(),
        torch.ones(B, V, 1, device=voxels_scene.device),
    ], dim=-1)

    # (B, 1, V, 4) @ (B, N, 4, 4).T → (B, N, V, 4)
    v_hom_exp = v_hom.unsqueeze(1)                              # (B, 1, V, 4)
    st_inv_T  = st_inv.permute(0, 1, 3, 2)                     # (B, N, 4, 4)
    voxels_cam_hom = v_hom_exp @ st_inv_T                       # (B, N, V, 4)
    voxels_cam = voxels_cam_hom[..., :3].permute(0, 2, 1, 3)   # (B, V, N, 3)

    # Perspective projection: K @ p_cam
    K = K_per_view.float()          # (B, N, 3, 3)
    vc_nv = voxels_cam.permute(0, 2, 1, 3)   # (B, N, V, 3)
    K_exp = K.unsqueeze(2)                    # (B, N, 1, 3, 3)
    projected = (vc_nv.unsqueeze(-2) @ K_exp.transpose(-1, -2)).squeeze(-2)  # (B, N, V, 3)
    z_proj = projected[..., 2].clamp(min=1e-6)
    u = projected[..., 0] / z_proj   # (B, N, V)  pixel-space x
    v = projected[..., 1] / z_proj   # (B, N, V)  pixel-space y

    # Normalise to [-1, 1] (align_corners=False convention)
    u_norm = (u + 0.5) / W * 2.0 - 1.0
    v_norm = (v + 0.5) / H * 2.0 - 1.0

    pix_coords = torch.stack([u_norm, v_norm], dim=-1)  # (B, N, V, 2)
    pix_coords = pix_coords.permute(0, 2, 1, 3)         # (B, V, N, 2)

    return pix_coords, voxels_cam


def _sample_features(
    img_feats: torch.Tensor,   # (B, N, C, H', W')
    pix_coords: torch.Tensor,  # (B, V, N, 2)  normalised [-1, 1]
) -> torch.Tensor:             # (B, V, N, C)
    """Bilinear-sample img_feats at projected voxel coordinates.

    The same normalised pix_coords work for any spatial resolution of img_feats
    because grid_sample scales internally — no need to adjust coords.
    """
    B, N, C, Hf, Wf = img_feats.shape
    V = pix_coords.shape[1]
    feats_flat  = img_feats.reshape(B * N, C, Hf, Wf)
    coords_flat = pix_coords.permute(0, 2, 1, 3).reshape(B * N, V, 1, 2)
    sampled = F.grid_sample(
        feats_flat, coords_flat,
        mode="bilinear", padding_mode="zeros", align_corners=False,
    )                                   # (B*N, C, V, 1)
    sampled = sampled.squeeze(-1)       # (B*N, C, V)
    sampled = sampled.permute(0, 2, 1)  # (B*N, V, C)
    return sampled.reshape(B, N, V, C).permute(0, 2, 1, 3)  # (B, V, N, C)


def _compute_visibility_mask(
    voxels_cam: torch.Tensor,            # (B, V, N, 3) camera-frame xyz
    pi3x_depth: torch.Tensor,            # (B, N, H, W)
    pix_coords: torch.Tensor,            # (B, V, N, 2)
    view_mask: torch.Tensor,             # (B, N) bool
    depth_rtol: float = 0.05,
    panoptic_masks: torch.Tensor | None = None,  # (B, N, Hp, Wp) long instance IDs
    target_ids: torch.Tensor | None = None,      # (B, N) long per-view target instance ID
    geometry_only: bool = False,
) -> torch.Tensor:                       # (B, V, N) bool
    """Per-voxel-per-view visibility.

    Three modes (in priority order):
      1. Mask consensus (panoptic_masks + target_ids given) — a voxel sees view n iff
         it projects into the target instance's segmentation mask there. This is the
         Experiment-3 mechanism; it replaces the Pi3X depth occlusion gate, which is
         unreliable because Pi3X depth is not metrically consistent across wide-baseline
         views (8-13% systematic error). Used for object voxels.
      2. Geometry-only (geometry_only=True) — z>0 + valid Pi3X depth at the pixel, no
         occlusion gate. Used for context voxels (no single target instance).
      3. Soft depth occlusion gate (default) — backward-compatible behaviour.
    """
    B, V, N, _ = voxels_cam.shape
    H, W = pi3x_depth.shape[-2:]

    # Test 1: positive z in camera frame (point is in front of camera)
    z_ok = voxels_cam[..., 2] > 0   # (B, V, N)
    vm   = view_mask[:, None, :].expand(B, V, N)   # (B, V, N)
    c_flat = pix_coords.permute(0, 2, 1, 3).reshape(B * N, V, 1, 2)

    if panoptic_masks is not None and target_ids is not None:
        # Mask consensus: trust the segmentation mask, not the depth check.
        Hp, Wp = panoptic_masks.shape[-2:]
        p_flat = panoptic_masks.float().reshape(B * N, 1, Hp, Wp)
        ids = F.grid_sample(p_flat, c_flat, mode="nearest",
                            padding_mode="zeros", align_corners=False)
        ids = ids.squeeze(-1).squeeze(1).reshape(B, N, V).permute(0, 2, 1).round().long()  # (B, V, N)
        tgt = target_ids[:, None, :].expand(B, V, N)
        mask_ok = (ids == tgt) & (tgt > 0)   # out-of-frame → 0 → no match
        return z_ok & mask_ok & vm

    # Sample Pi3X predicted depth at projected pixel locations
    d_flat = pi3x_depth.float().reshape(B * N, 1, H, W)
    pi3x_d = F.grid_sample(d_flat, c_flat, mode="bilinear",
                            padding_mode="zeros", align_corners=False)
    pi3x_d = pi3x_d.squeeze(-1).squeeze(1).reshape(B, N, V).permute(0, 2, 1)  # (B, V, N)

    # Pi3X returned valid depth at this pixel (depth > 0) → proxy for in-frame surface
    in_bounds = pi3x_d > 0

    if geometry_only:
        return z_ok & in_bounds & vm

    # Default: soft depth occlusion test (voxel depth ≤ Pi3X depth × (1 + rtol))
    depth_ok = voxels_cam[..., 2] <= pi3x_d * (1.0 + depth_rtol)
    return z_ok & depth_ok & in_bounds & vm


def _ibr_fusion(
    per_view_feats: torch.Tensor,        # (B, V, N, C)
    vis_mask: torch.Tensor,              # (B, V, N) bool
    weights: torch.Tensor | None = None,  # (B, V, N) per-view confidence weights in [0, ∞)
) -> torch.Tensor:                       # (B, V, 2C)
    """IBRNet-style weighted mean + variance fusion across visible views.

    When ``weights`` (e.g. Pi3X per-view confidence) is given, the fusion becomes a
    confidence-weighted mean/variance — realising the "weighted residual (μ, σ²)" of
    the architecture and down-weighting the systematically-inconsistent wide-baseline
    views (Experiment 2: 8-13% depth error).
    """
    m = vis_mask.unsqueeze(-1).float()
    if weights is not None:
        m = m * weights.unsqueeze(-1).clamp(min=0.0)
        denom = m.sum(dim=2).clamp(min=1e-6)
    else:
        denom = m.sum(dim=2).clamp(min=1.0)
    mu  = (per_view_feats * m).sum(dim=2) / denom         # (B, V, C)
    mu2 = ((per_view_feats ** 2) * m).sum(dim=2) / denom
    var = (mu2 - mu ** 2).clamp(min=0.0)
    return torch.cat([mu, var], dim=-1)                   # (B, V, 2C)


# ---------------------------------------------------------------------------
# Soft rasterization helper
# ---------------------------------------------------------------------------

def _rasterize_points(
    pix_coords: torch.Tensor,   # (B, V, N, 2) in [-1, 1]
    img_H: int,
    img_W: int,
    sigma: float = 2.0,
) -> torch.Tensor:              # (B, N, 1, img_H, img_W)
    """Scatter V 3D-object points into a soft (Gaussian-blob) binary mask per view."""
    B, V, N, _ = pix_coords.shape
    device = pix_coords.device

    # Convert normalised [-1,1] → continuous pixel coords (align_corners=False)
    u = (pix_coords[..., 0] + 1.0) * 0.5 * img_W - 0.5   # (B, V, N)
    v = (pix_coords[..., 1] + 1.0) * 0.5 * img_H - 0.5

    # Gaussian kernel
    r  = int(3 * sigma + 1)
    gy, gx = torch.meshgrid(
        torch.arange(-r, r + 1, device=device, dtype=torch.float32),
        torch.arange(-r, r + 1, device=device, dtype=torch.float32),
        indexing="ij",
    )
    kernel = torch.exp(-(gx ** 2 + gy ** 2) / (2 * sigma ** 2))  # (ks, ks)

    # Round to nearest integer pixel; reorder to (B*N, V)
    u_int = u.round().long().clamp(0, img_W - 1)   # (B, V, N)
    v_int = v.round().long().clamp(0, img_H - 1)
    u_bn  = u_int.permute(0, 2, 1).reshape(B * N, V)   # (B*N, V)
    v_bn  = v_int.permute(0, 2, 1).reshape(B * N, V)

    masks = torch.zeros(B * N, img_H * img_W, device=device)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            w  = kernel[dy + r, dx + r].item()
            yy = (v_bn + dy).clamp(0, img_H - 1)   # (B*N, V)
            xx = (u_bn + dx).clamp(0, img_W - 1)
            idx = yy * img_W + xx                  # (B*N, V) flat index
            masks.scatter_add_(1, idx, torch.full_like(idx, w, dtype=masks.dtype))

    masks = masks.clamp(max=1.0).reshape(B, N, 1, img_H, img_W)
    return masks


# ---------------------------------------------------------------------------
# BinaryMaskConvExtractor  (ShapeR-style)
# ---------------------------------------------------------------------------

class BinaryMaskConvExtractor(nn.Module):
    """Project the 3D object PC to each view → soft binary mask → conv features.

    Output spatial resolution: img_H // 4  ×  img_W // 4  (stride 4 total).
    Feature sampling is done in normalised coords, so the exact resolution does
    not need to match DINOv2 — pix_coords work at any feature resolution.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, out_channels, 1),
        )

    def forward(
        self,
        obj_voxels: torch.Tensor,           # (B, V, 3) scene space
        scene_transforms: torch.Tensor,     # (B, N, 4, 4)
        K_per_view: torch.Tensor,           # (B, N, 3, 3)
        view_mask: torch.Tensor,            # (B, N) bool
        img_H: int,
        img_W: int,
    ) -> torch.Tensor:                      # (B, N, out_channels, img_H//4, img_W//4)
        B = obj_voxels.shape[0]
        N = scene_transforms.shape[1]
        pix_coords, _ = _project_to_views(
            obj_voxels, scene_transforms, K_per_view, img_H, img_W
        )
        masks = _rasterize_points(pix_coords, img_H, img_W)      # (B, N, 1, H, W)
        out   = self.conv(masks.reshape(B * N, 1, img_H, img_W)) # (B*N, C, H', W')
        _, C, Hf, Wf = out.shape
        return out.reshape(B, N, C, Hf, Wf)


# ---------------------------------------------------------------------------
# MultiViewVoxelAlignedEncoder
# ---------------------------------------------------------------------------

class MultiViewVoxelAlignedEncoder(nn.Module):
    """Multi-view voxel-aligned encoder that replaces MICHE in the MV path.

    Produces:
        z_i     (B, num_obj_queries,   out_dim)
        z_scene (B, num_scene_queries, out_dim)
    """

    def __init__(
        self,
        feat_dim: int,           # per-view feature dim (C_dino, or C_dino+C_mask)
        voxel_dim: int,          # D
        out_dim: int,
        num_obj_queries: int,
        num_scene_queries: int,
        num_heads: int,
        use_geometry: bool = True,
    ):
        super().__init__()
        self.voxel_dim = voxel_dim
        self.feat_dim  = feat_dim

        # IBRNet fusion: 2 * feat_dim → voxel_dim
        self.fusion_proj = nn.Linear(2 * feat_dim, voxel_dim)

        # Explicit geometry stream: Fourier (PointEmbed) embedding of voxel XYZ, added as
        # a residual to the DINO-fused voxel feature. Without this the query pooling is
        # position-blind (appearance-only); with it, z_i / z_scene carry "point-cloud cues"
        # — the geometry-centric conditioning that single-view PixARMesh relies on.
        # Reuses EdgeRunner's PointEmbed so the geometry featurizer matches single-view.
        self.use_geometry = use_geometry
        self.point_embed = PointEmbed(dim=voxel_dim) if use_geometry else None

        # Variance-aware deformable offset predictor (zero-init → identity at start)
        self.offset_net = nn.Linear(voxel_dim, 2)
        nn.init.zeros_(self.offset_net.weight)
        nn.init.zeros_(self.offset_net.bias)

        # Per-view KV projection for deformable cross-attention
        self.deform_kv_proj = nn.Linear(feat_dim, voxel_dim)

        self.deform_cross_attn = nn.MultiheadAttention(
            voxel_dim, num_heads, batch_first=True
        )
        self.deform_norm = nn.LayerNorm(voxel_dim)

        # Learnable queries
        self.obj_queries   = nn.Parameter(torch.randn(num_obj_queries,   voxel_dim) * 0.02)
        self.scene_queries = nn.Parameter(torch.randn(num_scene_queries, voxel_dim) * 0.02)

        # Query aggregation cross-attention
        self.obj_cross_attn   = nn.MultiheadAttention(voxel_dim, num_heads, batch_first=True)
        self.scene_cross_attn = nn.MultiheadAttention(voxel_dim, num_heads, batch_first=True)

        # Scene-context aggregation: object latents attend to scene latents
        self.scene_ctx_cross_attn = nn.MultiheadAttention(voxel_dim, num_heads, batch_first=True)
        self.scene_ctx_norm       = nn.LayerNorm(voxel_dim)

        # Output projections
        self.obj_out_proj   = nn.Linear(voxel_dim, out_dim)
        self.scene_out_proj = nn.Linear(voxel_dim, out_dim)

    def _process_voxels(
        self,
        voxels: torch.Tensor,               # (B, V, 3) scene space
        dino_feats: torch.Tensor,           # (B, N, C_d, H', W')
        mask_feats: torch.Tensor | None,    # (B, N, C_m, Hm, Wm) or None
        scene_transforms: torch.Tensor,    # (B, N, 4, 4)
        K_per_view: torch.Tensor,          # (B, N, 3, 3)
        pi3x_depth: torch.Tensor,          # (B, N, H_full, W_full)
        view_mask: torch.Tensor,           # (B, N) bool
        conf: torch.Tensor | None = None,           # (B, N, H_full, W_full) Pi3X confidence
        panoptic_masks: torch.Tensor | None = None,  # (B, N, Hp, Wp) long
        target_ids: torch.Tensor | None = None,      # (B, N) long
        geometry_only: bool = False,
    ) -> torch.Tensor:                     # (B, V, voxel_dim)
        B, V, _ = voxels.shape
        N = scene_transforms.shape[1]
        H_full = pi3x_depth.shape[-2]
        W_full = pi3x_depth.shape[-1]

        # Project voxels to all views (using full-resolution H, W for correct K units)
        pix_coords, voxels_cam = _project_to_views(
            voxels, scene_transforms, K_per_view, H_full, W_full
        )  # pix_coords: (B, V, N, 2);  voxels_cam: (B, V, N, 3)

        # Per-point-per-view visibility mask:
        #  - object voxels (panoptic + target_ids) → mask consensus (Experiment 3)
        #  - context voxels (geometry_only)        → z>0 + valid depth, no occlusion gate
        vis_mask = _compute_visibility_mask(
            voxels_cam, pi3x_depth, pix_coords, view_mask,
            panoptic_masks=panoptic_masks, target_ids=target_ids,
            geometry_only=geometry_only,
        )  # (B, V, N) bool

        # Sample Pi3X confidence at projected coords → per-view fusion weights
        conf_vox = None
        if conf is not None:
            c_flat   = pix_coords.permute(0, 2, 1, 3).reshape(B * N, V, 1, 2)
            conf_map = conf.float().reshape(B * N, 1, H_full, W_full)
            conf_vox = F.grid_sample(conf_map, c_flat, mode="bilinear",
                                     padding_mode="zeros", align_corners=False)
            conf_vox = conf_vox.squeeze(-1).squeeze(1).reshape(B, N, V).permute(0, 2, 1)  # (B, V, N)

        # Sample dino features; optionally also mask conv features
        dino_sampled = _sample_features(dino_feats, pix_coords)    # (B, V, N, C_d)
        if mask_feats is not None:
            mask_sampled = _sample_features(mask_feats, pix_coords)
            per_view_feats = torch.cat([
                dino_sampled.to(voxels.dtype),
                mask_sampled.to(voxels.dtype),
            ], dim=-1)                                              # (B, V, N, C_d+C_m)
        else:
            per_view_feats = dino_sampled.to(voxels.dtype)         # (B, V, N, C_d)

        # IBRNet permutation-invariant confidence-weighted fusion
        fused = _ibr_fusion(per_view_feats.float(), vis_mask, weights=conf_vox)  # (B, V, 2*(C_d+C_m))
        voxel_feats = self.fusion_proj(
            fused.to(next(self.fusion_proj.parameters()).dtype)
        )  # (B, V, D)

        # Inject explicit voxel geometry (point-cloud cue) as a residual, so the
        # deformable offsets, refinement, and query pooling are all position-aware.
        if self.point_embed is not None:
            voxel_feats = voxel_feats + self.point_embed(voxels.to(voxel_feats.dtype))  # (B, V, D)

        # Variance-aware deformable offsets
        offsets = torch.tanh(self.offset_net(voxel_feats)) * 0.1   # (B, V, 2)
        pix_refined = (
            pix_coords + offsets.unsqueeze(2).expand(B, V, N, 2)
        ).clamp(-1.0, 1.0)                                          # (B, V, N, 2)

        # Resample with refined coordinates
        dino_ref = _sample_features(dino_feats, pix_refined)
        if mask_feats is not None:
            mask_ref = _sample_features(mask_feats, pix_refined)
            refined_feats = torch.cat([
                dino_ref.to(voxel_feats.dtype),
                mask_ref.to(voxel_feats.dtype),
            ], dim=-1)                                               # (B, V, N, C_total)
        else:
            refined_feats = dino_ref.to(voxel_feats.dtype)          # (B, V, N, C_d)

        # Deformable cross-attention: each voxel query attends to its N view features
        C_total = refined_feats.shape[-1]
        kv_in = self.deform_kv_proj(refined_feats.reshape(B * V, N, C_total))  # (B*V, N, D)
        q_in  = voxel_feats.reshape(B * V, 1, self.voxel_dim)
        pad   = ~vis_mask.reshape(B * V, N)

        # Guard: if a voxel is invisible in all views, unmask view 0 to avoid NaN
        all_masked    = pad.all(dim=1, keepdim=True)
        pad           = pad & ~all_masked

        attn_out, _   = self.deform_cross_attn(q_in, kv_in, kv_in, key_padding_mask=pad)
        voxel_feats   = self.deform_norm(
            voxel_feats + attn_out.squeeze(1).reshape(B, V, self.voxel_dim)
        )
        return voxel_feats   # (B, V, D)

    def forward(
        self,
        obj_voxels: torch.Tensor,           # (B, V_obj, 3) scene space
        ctx_voxels: torch.Tensor,           # (B, V_ctx, 3) scene space
        dino_feats: torch.Tensor,           # (B, N, C_d, H', W')
        mask_feats: torch.Tensor | None,    # (B, N, C_m, Hm, Wm) or None
        scene_transforms: torch.Tensor,    # (B, N, 4, 4)
        K_per_view: torch.Tensor,          # (B, N, 3, 3)
        pi3x_depth: torch.Tensor,          # (B, N, H, W)  full-resolution Pi3X depth
        view_mask: torch.Tensor,           # (B, N) bool
        panoptic_masks: torch.Tensor | None = None,  # (B, N, Hp, Wp) long — enables mask consensus
        target_ids: torch.Tensor | None = None,      # (B, N) long — per-view target instance ID
        conf: torch.Tensor | None = None,            # (B, N, H, W) Pi3X confidence — weighted fusion
    ) -> dict:
        B = obj_voxels.shape[0]

        # Object voxels: mask-consensus visibility (Experiment 3) when panoptic + target_ids
        # are supplied; otherwise the depth-gate fallback inside _process_voxels.
        obj_voxel_feats = self._process_voxels(
            obj_voxels, dino_feats, mask_feats,
            scene_transforms, K_per_view, pi3x_depth, view_mask,
            conf=conf, panoptic_masks=panoptic_masks, target_ids=target_ids,
        )   # (B, V_obj, D)

        # Context voxels: no single target instance → geometry-only visibility.
        ctx_voxel_feats = self._process_voxels(
            ctx_voxels, dino_feats, mask_feats,
            scene_transforms, K_per_view, pi3x_depth, view_mask,
            conf=conf, geometry_only=True,
        )   # (B, V_ctx, D)

        # Object latents: learnable queries cross-attend to object voxels
        obj_q  = self.obj_queries.unsqueeze(0).expand(B, -1, -1).to(obj_voxel_feats)
        z_i, _ = self.obj_cross_attn(obj_q, obj_voxel_feats, obj_voxel_feats)

        # Scene latents: learnable queries cross-attend to all voxels
        all_voxels   = torch.cat([obj_voxel_feats, ctx_voxel_feats], dim=1)
        scene_q      = self.scene_queries.unsqueeze(0).expand(B, -1, -1).to(all_voxels)
        z_scene, _   = self.scene_cross_attn(scene_q, all_voxels, all_voxels)

        # Scene-context aggregation: each object latent attends to the scene latent
        z_i_ctx, _   = self.scene_ctx_cross_attn(z_i, z_scene, z_scene)
        z_i          = self.scene_ctx_norm(z_i + z_i_ctx)

        return {
            "z_i":     self.obj_out_proj(z_i),
            "z_scene": self.scene_out_proj(z_scene),
        }
