from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    type: str
    path: str
    num_points: int = 4096
    norm_bound: float = 0.9995
    with_normals: bool = False
    mesh_path: str = ""
    num_pos_tokens: int = 128
    random_scale: bool = True
    random_scale_min: float = 0.75
    random_rotate_sampled_instance: bool = False
    random_rotate: bool = True
    random_rotate_min: float = 0.0
    random_rotate_max: float = 360.0
    random_jitter_point_clouds: bool = True
    random_jitter_probability: float = 0.5
    random_jitter_offset: float = 0.01
    random_jitter_depth: bool = True
    random_jitter_depth_offset: float = 0.02
    random_shift: bool = False
    random_shift_max: float = 0.2
    visualize: bool = False
    # For local condition
    local_obj_num_points: int = -1
    mask_path: str = ""
    mask_erosion_size: int = 3
    depth_path: str = ""
    use_predicted_depth: bool = False
    predicted_depth_aligned: bool = False
    local_obj_cond_drop_prob: float = 0.2
    # Object point cloud in global frame
    use_masked_obj_pc: bool = False
    # Image conditions
    load_images: bool = False
    image_preprocessor: str = "facebook/dpt-dinov2-small-nyu"
    image_size_divisor: int = 28
    # Context point clouds
    num_ctx_points: int = 0
    # Test-2 overfit: train+eval on the first N examples only (0 = full dataset).
    overfit_n: int = 0
    # Multi-view frozen-feature cache dir (geometry local_points/conf + DINOv2 feats,
    # precomputed by scripts/data/precompute_mv_features.py). "" = compute live.
    mv_feature_cache: str = ""
    # Ablations
    ignore_obj_seq: bool = False
    ignore_layout_seq: bool = False
    # Multi-view
    num_views: int = (
        1  # max slots to pad to; actual count determined by covisibility selection
    )
    # Trellis2-MV dataset: path to local HF dataset used to cross-ref images/cameras.
    # Defaults to <dataset_path>/../../3d-front-multiview-full when empty.
    trellis2_hf_path: str = ""
    # Covisibility-based view selection (Trellis2-MV / mesh_datasets).
    # Scores all available views by object pixel support, then greedily selects up to
    # mv_covis_k_max diverse views.  Views with fewer than mv_covis_min_support_pts
    # in-frame projected object points are excluded before selection.
    mv_covis_k_max: int = 8
    mv_covis_min_support_pts: int = 50
    # Reference view sanity gates applied after support-based covisibility selection.
    mv_mask_min_area_px: int = 256
    mv_mask_min_hit_pts: int = 8
    mv_mask_min_hit_frac: float = 0.05
    # Drop instances with degenerate conditioning (HF row has <2 views, or the object
    # projects in-frame in NO view). Requires the conditioning_filter.csv sidecar from
    # scripts/data/build_conditioning_filter.py next to the trellis2 metadata.csv.
    mv_filter_degenerate: bool = False
    # Exact per-scene norm→HF-world frame correction (norm_to_world_transform): aligns
    # covisibility scoring, scene_transforms, and seed 2D projection with the HF camera
    # world. Requires a uid column in the HF dataset and a sidecar built in the same
    # frame (checked against conditioning_filter.meta.json).
    mv_frame_correction: bool = False


@dataclass
class ModelConfig:
    vocab_size: int
    num_pos_tokens: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    sep_token_id: Optional[int] = None
    indicator_token_id: int = -50
    obj_pc_token_id: int = -49
    pc_token_id: int = -48
    prefix_len: int = 0
    pc_latent_len: int = 0
    cond: bool = False
    obj_cond: bool = False
    local_path: str = ""
    local_cond_path: str = ""
    cond_enc_type: str = "miche"
    freeze_cond_encoder: bool = True
    # Test-2 overfit / frozen-decoder regime: train ONLY the mv_voxel_encoder, freeze
    # everything else (OPT decoder, lm_head, MICHE cond_encoder, embeddings).
    freeze_decoder: bool = False
    ar_model_type: str = "meshxl"
    tokenization_method: str = "meshxl"
    max_seq_length: int = 8192
    max_position_embeddings: int = 8192
    pos_token_offset: int = 0
    no_layout_loss: bool = False
    layout_tokenization_method: str = "tri"
    img_cond: bool = False
    image_encoder: str = ""
    high_res_image_encoder: bool = False
    high_res_image_encoder_hidden_size: int = 256
    image_encoder_layers: List[str] = field(default_factory=lambda: [])
    with_ctx_pc: bool = False
    img_cond_drop_prob: float = 0.0
    loss_layout_scale: Optional[float] = None
    # Experimental MV stage-1 layout supervision. Defaults keep the current CE-only
    # objective unchanged unless an experiment config opts in.
    loss_layout_ordinal_sigma: Optional[float] = None
    loss_layout_ordinal_weight: float = 0.0
    loss_layout_coord_weight: float = 0.0
    loss_layout_geometry_tokens: int = 24
    # DA3: frozen any-view image→3D backbone for MV geometry.
    use_da3: bool = False
    da3_ckpt_path: str = "checkpoints/da3/DA3-GIANT"
    # Generic frozen geometry encoder registry selector. Empty keeps the SV baseline.
    geo_encoder_type: str = ""
    # Multi-view voxel encoder
    mv_voxel_encoder: bool = False
    mv_num_obj_voxels: int = 512
    mv_num_ctx_voxels: int = 1024
    mv_voxel_dim: int = 512
    mv_num_obj_queries: int = 257
    mv_num_scene_queries: int = 64
    mv_num_heads: int = 8
    mv_mask_seeded_pool: bool = False
    mv_boundary_bias_alpha: float = 0.0
    # Instance-discovery method (MV segmentation tournament). Resolved via
    # src.models.discovery.get_discovery_fn; "consensus" = current mask-consensus
    # voting baseline. New Family-A candidates register under their own name.
    mv_discovery_method: str = "consensus"
    # Route the multi-view-discovered canonical points through the SV cond_encoder
    # (native obj-PC channel the decoder exploits). When True, prefix gains pc_latent_len
    # obj-PC tokens. mv_use_voxel_encoder keeps the z_i/z_scene appearance-fusion channel.
    mv_obj_pc_cond: bool = False
    mv_use_voxel_encoder: bool = True
    # Inject confidence-weighted multi-view DINO appearance into the obj-PC channel's
    # cond_encoder via extra_feat. The SV cond_encoder was trained with extra_feat ALWAYS
    # present (img_cond_drop_prob=0), so geometry-only obj-PC is off-distribution by the
    # appearance term; this restores it and adds the multi-view texture cue. prefix_len
    # is unchanged (extra_feat is added inside cond_encoder; latent count stays pc_latent_len).
    mv_obj_pc_appearance: bool = False
    # DEBUG-ONLY oracle ceiling: replace the observed, self-normalized obj_geom_voxels with
    # FPS-sampled points from the GT canonical mesh (full-extent, leak-by-design). Used to
    # bound whether ANY conditioning fix can beat SV and to quantify the partial-observation
    # self-norm scale cost (oracle uses full extent). MUST stay false in any shippable config.
    mv_obj_pc_oracle: bool = False
    # Discovery / voxelization parameters.  These must be in ModelConfig (not just accessed
    # via getattr defaults) so that _filter_dataclass_kwargs passes them through to
    # ShapeOPTConfig, and _mv_fields propagates them when loading single-view checkpoints.
    # Defaults match the getattr fallbacks in edgerunner.py so existing checkpoints are
    # behaviour-identical; yaml overrides (e.g. mv_min_views: 2) now actually take effect.
    mv_min_views: int = 3
    mv_conf_threshold: float = 0.5
    mv_depth_rtol: float = 100.0
    mv_pool_size: int = 8192
    mv_intra_obj_register: bool = False
    mv_register_iters: int = 4
    mv_geom_norm_quantile: float = 0.0
    # If quantile trimming collapses the max-axis span below this fraction of the
    # raw span, fall back to raw min/max for that object to avoid unbounded scale.
    mv_geom_norm_trim_fallback_ratio: float = 0.2
    mv_use_geometry: bool = True
    # Sampling strategy for obj_voxels after discovery: "fps" = score-seeded FPS (default,
    # maximises spread), "grid" = fixed voxel-grid reps, "adaptive" = occupied-cell
    # reps at an adaptive grid resolution followed by score-seeded FPS.
    mv_voxel_sampling: str = "fps"
    # Per-object pixel-support gate for IBRNet fusion: views with fewer than this many
    # panoptic pixels matching the target instance are excluded from the object-voxel
    # encoder path (scene context voxels still use all valid views).
    mv_covis_min_support_pix: int = 200
    # Per-view rogue gate (0 = off): drop whole views whose mean confidence falls
    # below this before discovery/fusion. Thresholds are backbone-specific and must
    # be calibrated before use. Never drops below mv_view_gate_min_views and never
    # drops the reference view.
    mv_view_conf_gate: float = 0.0
    mv_view_gate_min_views: int = 2
    # Append ONE scene-frame obj-AABB token to the prefix. The geometry channels are
    # canonicalized by observed extent, which strips the layout head's pose/size
    # evidence — this token is the only explicit
    # scene-frame coordinate signal for the target object (2026-07-03 council).
    mv_obj_aabb_token: bool = False


def mv_prefix_len(model_cfg) -> int:
    """Number of conditioning (pc_token) slots the MV path emits, in prefix order
    [obj-PC latents] + [z_i, z_scene] + [obj-AABB] + num_face. Single source of truth
    shared by training/runtime config builders so the collator's prefix_len always
    matches what the model produces (else masked_scatter mis-sizes)."""
    n = 0
    if getattr(model_cfg, "mv_obj_pc_cond", False):
        n += model_cfg.pc_latent_len
    if getattr(model_cfg, "mv_use_voxel_encoder", True) and getattr(
        model_cfg, "mv_voxel_encoder", False
    ):
        n += model_cfg.mv_num_obj_queries + model_cfg.mv_num_scene_queries
    if getattr(model_cfg, "mv_obj_aabb_token", False):
        n += 1
    return n + 1  # num_face
