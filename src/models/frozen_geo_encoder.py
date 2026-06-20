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


def adaptive_fps_voxelize(
    pts: torch.Tensor,
    conf: torch.Tensor | None,
    n_voxels: int,
    conf_threshold: float = 0.3,
) -> torch.Tensor:
    """AnySplat-style adaptive-size voxelization via confidence-filtered FPS.

    High-confidence (reliable) points are kept for FPS; unreliable points are
    filtered out.  Dense, well-reconstructed regions therefore receive finer
    voxel coverage (more voxels per unit area), while uncertain or missing
    geometry is not voxelized — exactly the adaptive-size property described
    in AnySplat.

    Args:
        pts:              (B, M, 3)
        conf:             (B, M)  per-point confidence, or None (fall back to plain FPS)
        n_voxels:         target number of voxels
        conf_threshold:   minimum confidence to include a point (default 0.3)
    Returns:
        (B, n_voxels, 3)
    """
    if conf is None:
        return fps_centroid_seeded(pts, n_voxels)

    B, M, _ = pts.shape
    results = []
    for b in range(B):
        keep = conf[b] >= conf_threshold
        pts_keep = pts[b][keep]
        if pts_keep.shape[0] < max(n_voxels, 4):
            pts_keep = pts[b]   # fall back to all points if too few pass threshold
        vox = fps_centroid_seeded(pts_keep.unsqueeze(0), n_voxels).squeeze(0)
        results.append(vox)
    return torch.stack(results, dim=0)


def fps_centroid_seeded(pts: torch.Tensor, n_sample: int) -> torch.Tensor:
    """FPS seeded from the point closest to the cloud centroid — order-invariant.

    Args:
        pts:      (B, M, 3)
        n_sample: number of points to keep
    Returns:
        (B, n_sample, 3)
    """
    B, M, _ = pts.shape
    if n_sample >= M:
        if n_sample == M:
            return pts
        # oversample by repeating last point
        pad = pts[:, -1:, :].expand(B, n_sample - M, 3)
        return torch.cat([pts, pad], dim=1)

    pts_f = pts.float()
    centroid = pts_f.mean(dim=1, keepdim=True)                   # (B, 1, 3)
    seed_idx = (pts_f - centroid).norm(dim=-1).argmin(dim=1)     # (B,)

    # Pure-PyTorch FPS (no pointnet2_ops dependency)
    device = pts.device
    selected = torch.zeros(B, n_sample, dtype=torch.long, device=device)
    selected[:, 0] = seed_idx
    dist = torch.full((B, M), float("inf"), device=device)

    for i in range(1, n_sample):
        prev = selected[:, i - 1]                                # (B,)
        prev_pts = pts_f[torch.arange(B, device=device), prev]  # (B, 3)
        d = (pts_f - prev_pts.unsqueeze(1)).norm(dim=-1)         # (B, M)
        dist = torch.minimum(dist, d)
        selected[:, i] = dist.argmax(dim=1)

    idx = selected.unsqueeze(-1).expand(B, n_sample, 3)
    return pts.gather(1, idx)


def fps_score_seeded(pts: torch.Tensor, scores: torch.Tensor, n_sample: int) -> torch.Tensor:
    """FPS seeded from the highest-score point rather than the centroid.

    Biases uniform coverage toward the most-agreed geometry — points seen in
    more views (higher hit_count) are preferred as the initial seed, so the
    selected set starts from the confidently-reconstructed object core.

    Args:
        pts:      (B, M, 3)
        scores:   (B, M)  per-point confidence; higher = seed first
        n_sample: number of points to keep
    Returns:
        (B, n_sample, 3)
    """
    B, M, _ = pts.shape
    if n_sample >= M:
        if n_sample == M:
            return pts
        pad = pts[:, -1:, :].expand(B, n_sample - M, 3)
        return torch.cat([pts, pad], dim=1)

    pts_f   = pts.float()
    seed_idx = scores.argmax(dim=1)   # (B,) — start from highest-confidence point

    device   = pts.device
    selected = torch.zeros(B, n_sample, dtype=torch.long, device=device)
    selected[:, 0] = seed_idx
    dist = torch.full((B, M), float("inf"), device=device)

    for i in range(1, n_sample):
        prev     = selected[:, i - 1]
        prev_pts = pts_f[torch.arange(B, device=device), prev]   # (B, 3)
        d        = (pts_f - prev_pts.unsqueeze(1)).norm(dim=-1)  # (B, M)
        dist     = torch.minimum(dist, d)
        selected[:, i] = dist.argmax(dim=1)

    idx = selected.unsqueeze(-1).expand(B, n_sample, 3)
    return pts.gather(1, idx)


def build_geo_ctx_voxels_mv(
    local_points: torch.Tensor,
    conf: torch.Tensor,
    scene_transforms: torch.Tensor,
    num_voxels: int,
    view_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Merge context geometry from all N views and downsample with centroid-seeded FPS.

    Args:
        local_points:    (B, N, H, W, 3)  per-view camera-frame points
        conf:            (B, N, H, W, 1)  per-pixel confidence (unused here, kept for API parity)
        scene_transforms:(B, N, 4, 4)     camera_n → scene space
        num_voxels:      target voxel count
        view_mask:       (B, N) bool, True = valid view; None = all valid
    Returns:
        (B, num_voxels, 3) context voxels in scene space, same dtype as local_points
    """
    B, N, H, W, _ = local_points.shape
    out_dtype = local_points.dtype
    device = local_points.device

    all_scene_pts: list[torch.Tensor] = []
    for b in range(B):
        view_pts_list = []
        for n in range(N):
            if view_mask is not None and not view_mask[b, n]:
                continue
            lp_n = local_points[b, n]           # (H, W, 3)
            valid = lp_n[..., 2] > 0            # z > 0 in camera frame
            pts_cam = lp_n[valid].float()       # (P, 3)
            if pts_cam.shape[0] == 0:
                continue
            st_n = scene_transforms[b, n].unsqueeze(0)   # (1, 4, 4)
            pts_scene = _apply_scene_transform(pts_cam.unsqueeze(0), st_n).squeeze(0)
            view_pts_list.append(pts_scene)
        if len(view_pts_list) == 0:
            all_scene_pts.append(torch.zeros(num_voxels, 3, device=device))
        else:
            merged = torch.cat(view_pts_list, dim=0)  # (P_total, 3)
            # centroid-seeded FPS on merged cloud
            sampled = fps_centroid_seeded(merged.unsqueeze(0), num_voxels).squeeze(0)
            all_scene_pts.append(sampled)

    result = torch.stack(all_scene_pts, dim=0)  # (B, num_voxels, 3)
    return result.to(out_dtype)


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


# ---------------------------------------------------------------------------
# Grounded-SAM consensus instance discovery
# ---------------------------------------------------------------------------

def _project_pts_to_views(
    pts_scene: torch.Tensor,         # (V, 3) scene space (single batch item)
    scene_transforms_n: torch.Tensor,  # (N, 4, 4)
    K_per_view_n: torch.Tensor,        # (N, 3, 3)
    H: int,
    W: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project V scene-space points to N views.

    Returns:
        pix_coords: (V, N, 2)  normalised [-1,1] for grid_sample
        pts_cam:    (V, N, 3)  camera-frame xyz
    """
    V = pts_scene.shape[0]
    N = scene_transforms_n.shape[0]

    st_inv = torch.linalg.inv(scene_transforms_n.float())  # (N, 4, 4)
    v_hom  = torch.cat([pts_scene.float(),
                         torch.ones(V, 1, device=pts_scene.device)], dim=-1)  # (V, 4)
    # v_hom @ st_inv[n].T  for each n
    pts_cam_hom = v_hom @ st_inv.permute(0, 2, 1)   # (N, V, 4)  via broadcast
    pts_cam     = pts_cam_hom[..., :3].permute(1, 0, 2)  # (V, N, 3)

    K = K_per_view_n.float()   # (N, 3, 3)
    vc_nv = pts_cam.permute(1, 0, 2)      # (N, V, 3)
    projected = (vc_nv.unsqueeze(-2) @ K.unsqueeze(1).transpose(-1, -2)).squeeze(-2)  # (N, V, 3)
    z = projected[..., 2].clamp(min=1e-6)
    u = projected[..., 0] / z   # (N, V)
    v = projected[..., 1] / z

    u_norm = (u + 0.5) / W * 2.0 - 1.0
    v_norm = (v + 0.5) / H * 2.0 - 1.0
    pix_coords = torch.stack([u_norm, v_norm], dim=-1).permute(1, 0, 2)  # (V, N, 2)

    return pix_coords, pts_cam


def _sample_mask_ids(
    pix_coords: torch.Tensor,    # (V, N, 2) normalised [-1,1]
    pts_cam: torch.Tensor,       # (V, N, 3) camera frame
    panoptic_masks: torch.Tensor,  # (N, H, W) long
    local_points_z: torch.Tensor,  # (N, H, W) Pi3X predicted depth
    view_mask: torch.Tensor,       # (N,) bool
    depth_rtol: float = 0.10,
) -> torch.Tensor:               # (V, N) long, -1 = not visible / invalid
    V, N, _ = pix_coords.shape
    H, W    = panoptic_masks.shape[1:]
    device  = pix_coords.device

    # --- Visibility test ---
    z_ok    = pts_cam[..., 2] > 0   # (V, N) in front of camera

    # Sample Pi3X depth at projected positions
    d_flat  = local_points_z.float().unsqueeze(1)      # (N, 1, H, W)
    c_flat  = pix_coords.permute(1, 0, 2).unsqueeze(2) # (N, V, 1, 2)
    pi3x_d  = F.grid_sample(d_flat, c_flat, mode="bilinear",
                              padding_mode="zeros", align_corners=False)
    pi3x_d  = pi3x_d.squeeze(1).squeeze(-1).T          # (V, N) via (N, V, 1) → squeeze

    depth_ok   = pts_cam[..., 2] <= pi3x_d * (1.0 + depth_rtol)
    in_bounds  = pi3x_d > 0
    vm         = view_mask.unsqueeze(0).expand(V, N)
    vis        = z_ok & depth_ok & in_bounds & vm       # (V, N) bool

    # --- Nearest-neighbour mask ID sampling ---
    m_flat  = panoptic_masks.float().unsqueeze(1)      # (N, 1, H, W)
    sampled = F.grid_sample(m_flat, c_flat, mode="nearest",
                             padding_mode="zeros", align_corners=False)
    ids     = sampled.squeeze(1).squeeze(-1).T.long()  # (V, N)

    ids[~vis] = -1
    return ids


def _get_per_view_target_ids(
    seed_pcs: torch.Tensor,             # (P, 3) scene space
    scene_transforms_n: torch.Tensor,  # (N, 4, 4)
    K_per_view_n: torch.Tensor,        # (N, 3, 3)
    panoptic_masks: torch.Tensor,      # (N, H, W) long
    local_points_z: torch.Tensor,      # (N, H, W) Pi3X depth
    view_mask: torch.Tensor,           # (N,) bool
    depth_rtol: float = 0.10,
    min_seed_count: int = 10,
) -> torch.Tensor:                     # (N,) long, -1 = no valid seed visible
    """Per-view target instance ID from seed projection + plurality vote.

    For each view, projects seed_pcs into the view and returns the plurality
    Grounded-SAM instance ID among visible seed points. Instance IDs may differ
    across views (separate SAM runs), so this is computed independently per view.
    """
    N = scene_transforms_n.shape[0]
    H, W = panoptic_masks.shape[1:]
    device = seed_pcs.device
    target_ids = torch.full((N,), -1, dtype=torch.long, device=device)
    if seed_pcs.shape[0] == 0:
        return target_ids
    pix_coords, pts_cam = _project_pts_to_views(
        seed_pcs, scene_transforms_n, K_per_view_n, H, W
    )   # pix_coords: (P, N, 2),  pts_cam: (P, N, 3)
    # depth_rtol=100.0: skip depth occlusion test for target ID discovery.
    # Any depth_rtol eliminates seed points at the object's surface boundary,
    # leaving a thin minority slice that votes for the wrong SAM region.
    seed_ids = _sample_mask_ids(
        pix_coords, pts_cam, panoptic_masks, local_points_z, view_mask,
        depth_rtol=100.0,
    )   # (P, N) long, -1 = not visible / invalid
    for n in range(N):
        if not view_mask[n]:
            continue
        ids_n = seed_ids[:, n]
        # Skip views where too few seed points landed in-bounds: the plurality
        # vote over a handful of points is unreliable and often picks background.
        if int((ids_n >= 0).sum().item()) < min_seed_count:
            continue
        valid = ids_n > 0   # exclude background (0) and not-visible (-1)
        if valid.any():
            ids_valid = ids_n[valid]
            max_id = int(ids_valid.max().item()) + 1
            target_ids[n] = torch.bincount(ids_valid, minlength=max_id).argmax()
    return target_ids


def _seed_biased_pool(
    all_pts: torch.Tensor,
    all_conf: torch.Tensor | None,
    seed_f: torch.Tensor,
    pool_size: int,
    conf_threshold: float,
    device: torch.device,
) -> torch.Tensor:
    """Half FPS from 2× seed-radius neighbourhood, half global FPS."""
    pool_size_b = min(pool_size, all_pts.shape[0])
    if seed_f.shape[0] > 0:
        seed_center = seed_f.mean(0)
        seed_radius = (seed_f - seed_center).norm(dim=-1).max().clamp(min=1e-4)
        near_mask = (all_pts - seed_center).norm(dim=-1) < 2.0 * seed_radius
        near_pts  = all_pts[near_mask]
        near_conf = all_conf[near_mask] if all_conf is not None else None
    else:
        near_pts  = torch.empty(0, 3, device=device)
        near_conf = None

    if near_pts.shape[0] >= 4:
        n_near   = min(pool_size_b // 2, near_pts.shape[0])
        n_global = pool_size_b - n_near
        pool_near = adaptive_fps_voxelize(
            near_pts.unsqueeze(0),
            near_conf.unsqueeze(0) if near_conf is not None else None,
            n_near, conf_threshold=conf_threshold,
        ).squeeze(0)
        pool_global = adaptive_fps_voxelize(
            all_pts.unsqueeze(0),
            all_conf.unsqueeze(0) if all_conf is not None else None,
            n_global, conf_threshold=conf_threshold,
        ).squeeze(0)
        return torch.cat([pool_near, pool_global], dim=0)
    else:
        return adaptive_fps_voxelize(
            all_pts.unsqueeze(0),
            all_conf.unsqueeze(0) if all_conf is not None else None,
            pool_size_b, conf_threshold=conf_threshold,
        ).squeeze(0)


def discover_instance_points_mv(
    local_points: torch.Tensor,        # (B, N, H, W, 3)  Pi3X per-view camera frame
    scene_transforms: torch.Tensor,    # (B, N, 4, 4)  cam_n → scene
    panoptic_masks: torch.Tensor,      # (B, N, H, W)  long — Grounded-SAM instance IDs
    K_per_view: torch.Tensor,          # (B, N, 3, 3)
    view_mask: torch.Tensor,           # (B, N) bool
    seed_pcs: torch.Tensor,            # (B, P, 3)  scene-space seed (cond_pcs)
    num_obj_voxels: int,
    num_ctx_voxels: int,
    conf: torch.Tensor | None = None,  # (B, N, H, W, 1) Pi3X per-pixel confidence
    pool_size: int = 8192,
    depth_rtol: float = 0.10,
    adaptive_fallback: bool = True,
    conf_threshold: float = 0.3,
    min_views: int = 2,
    mask_seeded_pool: bool = False,
    boundary_bias_alpha: float = 0.0,
    return_diagnostics: bool = False,
    return_pool_diagnostics: bool = False,
    return_target_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict]:
    """Grounded-SAM direct-mask instance discovery with adaptive voxelization.

    For each Pi3X 3D point (from confidence-filtered adaptive FPS merged across N
    views), project into every visible view and directly check whether it falls
    inside the target object's Grounded-SAM mask.  A point is labelled as belonging
    to the object if it matches the target mask ID in at least one visible view.

    The per-view target ID is found independently per view by projecting seed_pcs
    (data-loader object PC) and taking the plurality Grounded-SAM ID among visible
    seed points.  This handles the case where different views have different instance
    ID numbering (e.g. separate SAM runs).

    Args:
        local_points:    (B, N, H, W, 3)
        scene_transforms:(B, N, 4, 4)
        panoptic_masks:  (B, N, H, W) int  — 0 = background, >0 = instance
        K_per_view:      (B, N, 3, 3)
        view_mask:       (B, N) bool
        seed_pcs:        (B, P, 3)  scene-space object seed
        num_obj_voxels:  V_obj
        num_ctx_voxels:  V_ctx
        conf:            (B, N, H, W, 1) Pi3X per-pixel confidence, or None
        pool_size:       number of scene points before final FPS
        conf_threshold:  minimum confidence to include a point
        min_views:       a pool point must match the target mask in this many views
                         (clamped to the number of views with a valid target ID)
        mask_seeded_pool: if True, build pool by back-projecting target SAM mask pixels
                         from all N views instead of seed-biased FPS.  Directly addresses
                         the pool-coverage bottleneck identified by Gate-0 v2 (87.9% of
                         missed pool points are on-surface — they were never sampled).
        boundary_bias_alpha: weight added to boundary pixels when mask_seeded_pool=True.
                         0.0 = uniform; 0.5 = boundary pixels get 1.5× weight in FPS.
        return_diagnostics: if True, return third element dict with per-sample lists
    Returns:
        obj_voxels: (B, num_obj_voxels, 3)
        ctx_voxels: (B, num_ctx_voxels, 3)
        [diag]:     dict with "pool_hit_rate" and "n_obj_pts_raw" (only if return_diagnostics)
    """
    B, N, H, W, _ = local_points.shape
    device    = local_points.device
    out_dtype = local_points.dtype

    if return_pool_diagnostics:
        return_diagnostics = True  # pool diag implies base diag

    obj_list: list[torch.Tensor] = []
    ctx_list: list[torch.Tensor] = []
    target_ids_list: list[torch.Tensor] = []
    diag_pool_hit_rate: list[float] = []
    diag_n_obj_raw: list[int] = []
    diag_pool_pts: list[torch.Tensor] = []
    diag_pool_hit_strict: list[torch.Tensor] = []
    diag_pool_hit_2d: list[torch.Tensor] = []

    for b in range(B):
        st_b = scene_transforms[b]           # (N, 4, 4)
        K_b  = K_per_view[b]                 # (N, 3, 3)
        vm_b = view_mask[b]                  # (N,) bool
        pm_b = panoptic_masks[b]             # (N, H, W)
        lp_z = local_points[b, :, :, :, 2]  # (N, H, W) depth

        # --- 1. Merge valid Pi3X points from all views to scene space ---
        pts_scene_list = []
        conf_list      = []
        for n in range(N):
            if not vm_b[n]:
                continue
            lp_n   = local_points[b, n]
            conf_n = conf[b, n, :, :, 0] if conf is not None else None
            valid  = lp_n[..., 2] > 0
            pts_c  = lp_n[valid].float()
            if pts_c.shape[0] == 0:
                continue
            pts_s = _apply_scene_transform(pts_c.unsqueeze(0), st_b[n:n+1]).squeeze(0)
            pts_scene_list.append(pts_s)
            if conf_n is not None:
                conf_list.append(conf_n[valid].float())

        if len(pts_scene_list) == 0:
            obj_list.append(fps_centroid_seeded(seed_pcs[b:b+1].float(), num_obj_voxels).squeeze(0))
            ctx_list.append(fps_centroid_seeded(seed_pcs[b:b+1].float(), num_ctx_voxels).squeeze(0))
            target_ids_list.append(torch.full((N,), -1, dtype=torch.long, device=device))
            diag_pool_hit_rate.append(0.0)
            diag_n_obj_raw.append(0)
            if return_pool_diagnostics:
                diag_pool_pts.append(torch.empty(0, 3))
                diag_pool_hit_strict.append(torch.empty(0, dtype=torch.bool))
                diag_pool_hit_2d.append(torch.empty(0, dtype=torch.bool))
            continue

        all_pts  = torch.cat(pts_scene_list, dim=0)
        all_conf = torch.cat(conf_list, dim=0) if conf_list else None

        # --- 3 (moved up). Per-view target instance ID from seed projection ---
        # Must precede pool construction when mask_seeded_pool=True so the pool loop
        # knows which SAM ID to back-project per view.  _get_per_view_target_ids only
        # depends on seed_pcs and scene geometry — no pool dependency.
        seed_b = seed_pcs[b].float()
        target_ids_n = _get_per_view_target_ids(
            seed_b, st_b, K_b, pm_b, lp_z, vm_b, depth_rtol
        ) if seed_b.shape[0] > 0 else torch.full((N,), -1, dtype=torch.long, device=device)
        # target_ids_n: (N,) long, -1 = no valid seed visible in this view
        target_ids_list.append(target_ids_n)

        # --- 2a. Pool construction ---
        if mask_seeded_pool:
            # Direct back-projection of target SAM mask pixels from all N views.
            # Fixes the pool-coverage bottleneck: the seed-biased FPS approach only
            # samples ~5% of scene cloud slots for a small object, leaving 87.9% of
            # object-surface points unsampled (Gate-0 v2 finding).
            mask_pool_list: list[torch.Tensor] = []
            mask_conf_list: list[torch.Tensor] = []
            mask_weight_list: list[torch.Tensor] = []
            for n in range(N):
                if not vm_b[n] or target_ids_n[n] <= 0:
                    continue
                lp_n = local_points[b, n]            # (H_lp, W_lp, 3)
                H_lp, W_lp = lp_n.shape[:2]
                # Resize panoptic_masks (original res e.g. 484×648) to match
                # local_points resolution (Pi3X pads to size_divisor=28, e.g. 504×672).
                pm_n_rs = F.interpolate(
                    pm_b[n].float().unsqueeze(0).unsqueeze(0),
                    size=(H_lp, W_lp), mode="nearest",
                ).squeeze().long()
                obj_mask_n = pm_n_rs == target_ids_n[n]   # (H_lp, W_lp) bool
                valid_n    = lp_n[..., 2] > 0
                combined   = obj_mask_n & valid_n
                pts_c = lp_n[combined].float()
                if pts_c.shape[0] == 0:
                    continue
                pts_s = _apply_scene_transform(
                    pts_c.unsqueeze(0), st_b[n:n+1]
                ).squeeze(0)
                mask_pool_list.append(pts_s)
                if conf is not None:
                    mask_conf_list.append(conf[b, n, :, :, 0][combined].float())
                if boundary_bias_alpha > 0.0:
                    # Boundary = dilated mask & ~eroded mask (approximate via max_pool2d).
                    # Erosion: -max_pool2d(-mask) with kernel 3.
                    m_f = obj_mask_n.float().unsqueeze(0).unsqueeze(0)
                    eroded = (-F.max_pool2d(-m_f, kernel_size=3, stride=1, padding=1)).squeeze().bool()
                    boundary = obj_mask_n & ~eroded
                    w = torch.ones(pts_c.shape[0], device=device)
                    w[boundary[combined]] = 1.0 + boundary_bias_alpha
                    mask_weight_list.append(w)

            if mask_pool_list:
                raw_pts  = torch.cat(mask_pool_list, dim=0)
                raw_conf = torch.cat(mask_conf_list, dim=0) if mask_conf_list else None
                # Boundary bias: multiply conf by weight so boundary pixels are
                # less likely to be discarded by the conf threshold.
                if raw_conf is not None and mask_weight_list:
                    raw_w    = torch.cat(mask_weight_list, dim=0)
                    raw_conf = (raw_conf * raw_w).clamp(max=1.0)
                pool_size_b = min(pool_size, raw_pts.shape[0])
                pool_pts = adaptive_fps_voxelize(
                    raw_pts.unsqueeze(0),
                    raw_conf.unsqueeze(0) if raw_conf is not None else None,
                    pool_size_b, conf_threshold=conf_threshold,
                ).squeeze(0)
            else:
                # All target_ids_n == -1 (degenerate scene): fall back to seed-biased pool.
                pool_pts = _seed_biased_pool(
                    all_pts, all_conf, seed_b, pool_size, conf_threshold, device
                )
        else:
            pool_pts = _seed_biased_pool(
                all_pts, all_conf, seed_b, pool_size, conf_threshold, device
            )

        # --- 2b. Context voxels: directly from full cloud to num_ctx_voxels ---
        # Avoids the redundant pool→ctx re-downsampling; confidence property applies end-to-end.
        ctx_voxels_b = adaptive_fps_voxelize(
            all_pts.unsqueeze(0),
            all_conf.unsqueeze(0) if all_conf is not None else None,
            num_ctx_voxels,
            conf_threshold=conf_threshold,
        ).squeeze(0)   # (num_ctx_voxels, 3)

        # --- 4. Direct Grounded-SAM masking ---
        # A pool point is labelled as object if it matches the target mask in
        # at least min_views views (clamped so we never require more views than
        # those that have a valid target ID).
        n_valid_views = int((target_ids_n > 0).sum().item())
        if n_valid_views > 0:
            pix_coords, pts_cam = _project_pts_to_views(pool_pts, st_b, K_b, H, W)
            P = pool_pts.shape[0]
            threshold = min(min_views, n_valid_views)

            def _pool_mask(rtol: float) -> torch.Tensor:
                ids = _sample_mask_ids(
                    pix_coords, pts_cam, pm_b, lp_z, vm_b, depth_rtol=rtol
                )
                hc = torch.zeros(P, dtype=torch.long, device=device)
                for n in range(N):
                    if not vm_b[n] or target_ids_n[n] <= 0:
                        continue
                    hc += (ids[:, n] == target_ids_n[n]).long()
                return hc

            hit_count = _pool_mask(depth_rtol)
            if adaptive_fallback and hit_count.max() < threshold:
                # Fallback: Pi3X depth maps are not globally consistent across wide-
                # baseline views (same 3D point can be 0.5+ scene units apart in
                # different views).  depth_rtol=100.0 skips the depth check entirely;
                # the 2D projection still uses ground-truth camera poses.
                hit_count = _pool_mask(100.0)

            is_obj = hit_count >= threshold
            obj_pts = pool_pts[is_obj]
            obj_scores = hit_count[is_obj].float() / n_valid_views  # (n_obj,) in [0,1]
            diag_pool_hit_rate.append(is_obj.float().mean().item())
            diag_n_obj_raw.append(int(is_obj.sum().item()))
            if return_pool_diagnostics:
                # Strict = at depth_rtol; 2D-only = at rtol=100 (no depth check)
                hc_strict = _pool_mask(depth_rtol)
                hc_2d     = _pool_mask(100.0)
                diag_pool_pts.append(pool_pts.cpu())
                diag_pool_hit_strict.append((hc_strict >= threshold).cpu())
                diag_pool_hit_2d.append((hc_2d >= threshold).cpu())
        else:
            obj_pts    = torch.empty(0, 3, device=device)
            obj_scores = torch.empty(0, device=device)
            diag_pool_hit_rate.append(0.0)  # no valid target view
            diag_n_obj_raw.append(0)
            if return_pool_diagnostics:
                P_b = pool_pts.shape[0]
                diag_pool_pts.append(pool_pts.cpu())
                diag_pool_hit_strict.append(torch.zeros(P_b, dtype=torch.bool))
                diag_pool_hit_2d.append(torch.zeros(P_b, dtype=torch.bool))

        # --- 5. Fallback + FPS to fixed obj size ---
        if obj_pts.shape[0] == 0:
            obj_pts    = seed_pcs[b].float()
            obj_scores = None  # no hit-count scores available in fallback
        if obj_pts.shape[0] == 0:
            obj_pts    = pool_pts[:1]
            obj_scores = None

        if obj_scores is not None and obj_scores.shape[0] > 0:
            obj_list.append(
                fps_score_seeded(obj_pts.unsqueeze(0), obj_scores.unsqueeze(0), num_obj_voxels).squeeze(0)
            )
        else:
            obj_list.append(fps_centroid_seeded(obj_pts.unsqueeze(0), num_obj_voxels).squeeze(0))
        ctx_list.append(ctx_voxels_b)

    obj_t = torch.stack(obj_list, dim=0).to(out_dtype)
    ctx_t = torch.stack(ctx_list, dim=0).to(out_dtype)
    target_ids_t = torch.stack(target_ids_list, dim=0) if return_target_ids else None  # (B, N) long
    if return_diagnostics:
        diag: dict = {"pool_hit_rate": diag_pool_hit_rate, "n_obj_pts_raw": diag_n_obj_raw}
        if return_pool_diagnostics:
            diag["pool_pts"]        = diag_pool_pts         # list of (pool_size_b, 3) cpu tensors
            diag["pool_hit_strict"] = diag_pool_hit_strict  # list of (pool_size_b,) bool tensors
            diag["pool_hit_2d"]     = diag_pool_hit_2d      # list of (pool_size_b,) bool tensors
        if return_target_ids:
            return obj_t, ctx_t, diag, target_ids_t
        return obj_t, ctx_t, diag
    if return_target_ids:
        return obj_t, ctx_t, target_ids_t
    return obj_t, ctx_t
