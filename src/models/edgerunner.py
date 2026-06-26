import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import OPTForCausalLM, OPTConfig, OPTModel, PreTrainedModel
from transformers.models.opt.modeling_opt import OPTDecoder
from .cond import EdgeRunnerProjector, ContextAggregator
from .embed import CoordEmbed
from .loss import causal_lm_loss_with_token_types, CustomCausalLMOutputWithTokenTypes
from .frozen_geo_encoder import (
    build_geo_ctx_pc, build_geo_obj_pc,
    fps_centroid_seeded,
)
from .discovery import get_discovery_fn
from .mv_voxel_encoder import (
    MultiViewVoxelAlignedEncoder,
    _project_to_views,
    _compute_visibility_mask,
    _sample_features,
)

logger = logging.getLogger(__name__)


class OPTLearnedPositionalEmbeddingNoOffset(nn.Embedding):
    def forward(
        self,
        attention_mask: torch.LongTensor,
        past_key_values_length: int = 0,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        if position_ids is None:
            position_ids = torch.cumsum(attention_mask, dim=1)
            position_ids = (position_ids * attention_mask - 1).long()
            position_ids = position_ids[:, past_key_values_length:]
        position_ids.clip_(0)
        return super().forward(position_ids)


class ShapeOPTConfig(OPTConfig):
    model_type = "shape-opt"

    def __init__(
        self,
        indicator_token_id=-50,
        obj_pc_token_id=-49,
        pc_token_id=-48,
        with_ctx_pc=False,
        img_cond_drop_prob=0.0,
        loss_layout_scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.indicator_token_id = indicator_token_id
        self.obj_pc_token_id = obj_pc_token_id
        self.pc_token_id = pc_token_id
        self.with_ctx_pc = with_ctx_pc
        self.tie_word_embeddings = False
        self.img_cond_drop_prob = img_cond_drop_prob
        self.loss_layout_scale = loss_layout_scale


class ShapeOPTDecoder(OPTDecoder):
    config_class = ShapeOPTConfig

    def __init__(self, config: ShapeOPTConfig):
        super().__init__(config)
        self.embed_positions = OPTLearnedPositionalEmbeddingNoOffset(
            config.max_position_embeddings, config.hidden_size
        )
        self.post_init()


class ShapeOPTModel(OPTModel):
    def __init__(self, config: ShapeOPTConfig):
        super().__init__(config)
        self.decoder = ShapeOPTDecoder(config)
        self.post_init()


class ShapeOPT(OPTForCausalLM):
    _tied_weights_keys = []
    config_class = ShapeOPTConfig

    def __init__(
        self,
        config: ShapeOPTConfig,
        cond_encoder=None,
        cond_encoder_img=None,
        pi3x_encoder=None,   # frozen image→3D backbone (Pi3X; replaces depth back-projection)
        is_scene=False,
    ):
        super().__init__(config)
        self.model = ShapeOPTModel(config)
        self.lm_head = nn.Linear(
            config.word_embed_proj_dim, config.vocab_size, bias=False
        )
        self.embed_num_face = nn.Embedding(10, config.word_embed_proj_dim)

        self.is_scene = is_scene
        if self.is_scene:
            self.indicator_embed = CoordEmbed(
                num_points=3,
                dim=config.word_embed_proj_dim,
                freq_embed_dim=48,
            )
        else:
            self.indicator_embed = None

        self.post_init()

        if cond_encoder is not None:
            self.projector = EdgeRunnerProjector(
                cond_encoder.output_dim,
                config.word_embed_proj_dim,
            )
            self.projector.apply(self._init_weights)
        self.cond_encoder = cond_encoder
        self.cond_encoder_img = cond_encoder_img
        # Frozen geometry encoder — declared here so hasattr() is always reliable.
        # Populated (or left None) by get_model() after from_pretrained().
        self.pi3x_encoder = pi3x_encoder   # image→3D (Pi3X; replaces depth back-projection)

        self.ctx_aggregator = None
        if self.config.with_ctx_pc:
            self.ctx_aggregator = ContextAggregator(
                cond_encoder.output_dim, num_heads=8
            )
            self.ctx_aggregator.apply(self._init_weights)

        # Multi-view voxel encoder (replaces MICHE when pixel_values.dim() == 5).
        # Instance discovery uses Grounded-SAM consensus voting (no binary mask conv).
        self.mv_voxel_encoder = None
        if getattr(config, "mv_voxel_encoder", False) and cond_encoder_img is not None:
            img_feat_dim = getattr(cond_encoder_img, "output_feat_dim", 384)
            self.mv_voxel_encoder = MultiViewVoxelAlignedEncoder(
                feat_dim          = img_feat_dim,
                voxel_dim         = getattr(config, "mv_voxel_dim",          512),
                out_dim           = config.word_embed_proj_dim,
                num_obj_queries   = getattr(config, "mv_num_obj_queries",   257),
                num_scene_queries = getattr(config, "mv_num_scene_queries",  64),
                num_heads         = getattr(config, "mv_num_heads",           8),
                use_geometry      = getattr(config, "mv_use_geometry",     True),
            )
            self.mv_voxel_encoder.apply(self._init_weights)

    def _init_weights(self, module):
        return PreTrainedModel._init_weights(self, module)

    def get_inputs_with_cond(
        self,
        input_ids,
        cond_pcs=None,
        cond_pcs_2d=None,
        ctx_pcs=None,
        ctx_pcs_2d=None,
        cond_num_faces=None,
        obj_indices=None,
        obj_bboxes=None,
        obj_cond_pcs=None,
        pixel_values=None,
    ):
        input_ids = input_ids.clone()
        extra_feat_mask = None
        sampled_feats = None
        ctx_img_feats = None
        if pixel_values is not None and self.cond_encoder_img is not None:
            img_feats = self.cond_encoder_img(pixel_values=pixel_values)
            sampled_feats = F.grid_sample(
                img_feats,
                cond_pcs_2d.unsqueeze(1),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled_feats = sampled_feats.squeeze(2).permute(0, 2, 1)  # (B, N, C)
            if ctx_pcs_2d is not None:
                ctx_img_feats = F.grid_sample(
                    img_feats,
                    ctx_pcs_2d.unsqueeze(1),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                ctx_img_feats = ctx_img_feats.squeeze(2).permute(0, 2, 1)  # (B, N, C)
            if self.training:
                extra_feat_mask = (
                    torch.rand(cond_pcs.size(0)) < self.config.img_cond_drop_prob
                )
                extra_feat_mask = extra_feat_mask.to(cond_pcs.device)

        cond_token_mask = input_ids == self.config.pc_token_id
        indicator_mask = input_ids == self.config.indicator_token_id
        obj_cond_token_mask = input_ids == self.config.obj_pc_token_id
        input_ids[indicator_mask | obj_cond_token_mask | cond_token_mask] = (
            self.config.pad_token_id
        )
        inputs_embeds = self.model.decoder.embed_tokens(input_ids)

        if obj_indices is not None and self.is_scene:
            valid_inds = obj_indices >= 0
            if obj_bboxes is not None:
                indicator_embeds = self.indicator_embed(obj_bboxes)
                inputs_embeds.masked_scatter_(
                    indicator_mask.unsqueeze(-1), indicator_embeds[valid_inds]
                )
            if obj_cond_pcs is not None and len(obj_cond_pcs) > 0:
                obj_conds = self.cond_encoder(obj_cond_pcs)
                obj_cond_embeds = self.projector(obj_conds)
                inputs_embeds.masked_scatter_(
                    obj_cond_token_mask.unsqueeze(-1), obj_cond_embeds.flatten(0, 1)
                )

        all_cond_embeds = []
        if cond_pcs is not None:
            conds = self.cond_encoder(
                cond_pcs, extra_feat=sampled_feats, extra_feat_mask=extra_feat_mask
            )
            if ctx_pcs is not None and self.ctx_aggregator is not None:
                ctx_conds = self.cond_encoder(
                    ctx_pcs, extra_feat=ctx_img_feats, extra_feat_mask=extra_feat_mask
                )
                conds = self.ctx_aggregator(conds, ctx_conds)
            cond_embeds = self.projector(conds)
            all_cond_embeds.append(cond_embeds)
        if cond_num_faces is None:
            cond_num_faces = torch.zeros(
                (inputs_embeds.size(0), 1),
                dtype=torch.long,
                device=inputs_embeds.device,
            )
        num_face_embeds = self.embed_num_face(cond_num_faces)
        all_cond_embeds.append(num_face_embeds)
        if len(all_cond_embeds) > 0:
            all_cond_embeds = torch.cat(
                [e.to(inputs_embeds.dtype) for e in all_cond_embeds], dim=1
            ).flatten(0, 1)
            inputs_embeds.masked_scatter_(
                cond_token_mask.unsqueeze(-1), all_cond_embeds
            )
        return inputs_embeds

    def _select_ref_view(self, local_points: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        """Return the reference-view index per batch item.

        Reference = view with most positive-depth pixels; deterministic (argmax, first-wins tie).
        Args:
            local_points: (B, N, H, W, 3) per-view camera frame
            view_mask:    (B, N) bool
        Returns:
            ref_idx: (B,) long
        """
        B, N, H, W, _ = local_points.shape
        valid_count = (local_points[..., 2] > 0).reshape(B, N, -1).sum(-1).float()  # (B, N)
        if view_mask is not None:
            valid_count = valid_count * view_mask.float()
        return valid_count.argmax(dim=1)   # (B,)

    def _fuse_obj_view_features(
        self,
        voxels,            # (B, V, 3) scene frame (for projection)
        dino_feats,        # (B, N, C_d, H', W')
        scene_transforms,  # (B, N, 4, 4)
        K_per_view,        # (B, N, 3, 3)
        pi3x_depth,        # (B, N, H, W)
        view_mask,         # (B, N) bool
        panoptic_masks,    # (B, N, Hp, Wp) long or None
        target_ids,        # (B, N) long or None
        conf,              # (B, N, H, W) Pi3X confidence or None
    ):
        """Confidence-weighted multi-view MEAN of DINO features at the object voxels, in
        the raw DINO feature space (C_d) — the appearance term the SV cond_encoder expects
        as `extra_feat`. The SV cond_encoder was trained with extra_feat ALWAYS present
        (img_cond_drop_prob=0), so the geometry-only obj-PC call is off-distribution by
        exactly this term; supplying it restores the trained distribution and adds the
        multi-view texture cue. Visibility reuses the obj voxels' mask-consensus gate
        (Experiment 3); voxels visible in no view get a zero feature (== the img-drop case).
        Returns (B, V, C_d) aligned 1:1 with the (canonical) obj_geom_voxels the encoder embeds.
        """
        B, V, _ = voxels.shape
        H_full, W_full = pi3x_depth.shape[-2:]
        pix_coords, voxels_cam = _project_to_views(
            voxels, scene_transforms, K_per_view, H_full, W_full
        )   # (B, V, N, 2), (B, V, N, 3)
        vis_mask = _compute_visibility_mask(
            voxels_cam, pi3x_depth, pix_coords, view_mask,
            panoptic_masks=panoptic_masks, target_ids=target_ids,
        )   # (B, V, N)
        dino_sampled = _sample_features(dino_feats, pix_coords)   # (B, V, N, C_d)

        m = vis_mask.unsqueeze(-1).float()   # (B, V, N, 1)
        if conf is not None:
            N = pix_coords.shape[2]
            c_flat   = pix_coords.permute(0, 2, 1, 3).reshape(B * N, V, 1, 2)
            conf_map = conf.float().reshape(B * N, 1, H_full, W_full)
            conf_vox = F.grid_sample(conf_map, c_flat, mode="bilinear",
                                     padding_mode="zeros", align_corners=False)
            conf_vox = conf_vox.squeeze(-1).squeeze(1).reshape(B, N, V).permute(0, 2, 1)  # (B, V, N)
            m = m * conf_vox.unsqueeze(-1).clamp(min=0.0)
        denom = m.sum(dim=2).clamp(min=1e-6)                      # (B, V, 1)
        mu = (dino_sampled.float() * m).sum(dim=2) / denom        # (B, V, C_d)
        return mu.to(dino_feats.dtype)

    def get_mv_inputs_with_cond(
        self,
        input_ids,
        pixel_values,           # (B, N, C, H, W)
        scene_transforms,       # (B, N, 4, 4)
        K_per_view,             # (B, N, 3, 3)
        view_mask,              # (B, N) bool
        panoptic_masks,         # (B, N, H, W) long or None
        cond_pcs,               # (B, P, 3)  data-loader seed PC in scene space
        cond_pcs_2d,            # (B, P, 2)  reference-view pixel coords
        cond_num_faces,
        cached_local_points=None,  # (B, N, H, W, 3) precomputed Pi3X geometry
        cached_conf=None,          # (B, N, H, W, 1) precomputed Pi3X confidence
        cached_dino_feats=None,    # (B, N, C_d, H', W') precomputed DINOv2 features
        obj_canon_transform=None,  # (B, 4, 4) scene -> per-object canonical (rotation used)
        gt_obj_vertices=None,      # (B, V, 3) DEBUG oracle: GT-canonical surface points
    ):
        """Build the multi-view conditioning prefix embeddings.

        Runs Pi3X → instance discovery → MV voxel encoder, then scatters
        z_i + z_scene + num_face into the pc_token slots of the embedded prefix.
        Returns inputs_embeds (B, L, D). Shared by the training forward
        (_forward_multiview) and by generation (infer.py passes the result to
        self.generate(inputs_embeds=...)).

        Pi3X and DINOv2 are frozen and depend only on the (un-augmented) images, so
        their outputs can be precomputed once and passed in via cached_local_points /
        cached_conf / cached_dino_feats — skipping the two heavy ViT forwards per step.
        """
        B, N, C, H, W = pixel_values.shape
        device = pixel_values.device

        if view_mask is None:
            view_mask = torch.ones(B, N, dtype=torch.bool, device=device)

        # --- Pi3X: all N views in one forward (or reuse cached features) ---
        if cached_local_points is not None:
            lp = cached_local_points.to(device)
            pi3x_out = {"conf": cached_conf.to(device) if cached_conf is not None else None}
        else:
            pi3x_out = self.pi3x_encoder.forward_all_views_joint(pixel_values)
            lp = pi3x_out["local_points"]       # (B, N, H, W, 3) camera frame
        pi3x_depth = lp[..., 2]                 # (B, N, H, W)
        st         = scene_transforms.to(device, dtype=torch.float32)
        K_f        = K_per_view.float().to(device)

        # --- Instance discovery via Grounded-SAM consensus voting ---
        # For every Pi3X 3D point: project to all views, collect mask IDs,
        # majority-vote to assign each point to an instance.
        # seed_pcs selects the target instance (data-loader object PC).
        mv_num_obj_voxels = getattr(self.config, "mv_num_obj_voxels", 512)
        mv_num_ctx_voxels = getattr(self.config, "mv_num_ctx_voxels", 1024)

        obj_voxels_geom = None   # registered geometry-stream cloud (set in discover path)
        if panoptic_masks is not None:
            # Enhance seed first: use Pi3X geometry at reference-view obj pixels
            ref_idx = self._select_ref_view(lp, view_mask)
            seed_list = []
            for b in range(B):
                rv   = ref_idx[b].item()
                pc_b = build_geo_obj_pc(
                    lp[b:b+1, rv], cond_pcs_2d[b:b+1], st[b:b+1, rv]
                )
                seed_list.append(pc_b)
            seed_pcs = torch.cat(seed_list, dim=0)   # (B, P, 3)

            discover_fn = get_discovery_fn(getattr(self.config, "mv_discovery_method", "consensus"))
            obj_voxels, ctx_voxels, mv_target_ids, obj_voxels_geom = discover_fn(
                local_points        = lp,
                scene_transforms    = st,
                panoptic_masks      = panoptic_masks.to(device),
                K_per_view          = K_f,
                view_mask           = view_mask,
                seed_pcs            = seed_pcs.float(),
                num_obj_voxels      = mv_num_obj_voxels,
                num_ctx_voxels      = mv_num_ctx_voxels,
                conf                = pi3x_out["conf"],
                # Proven operating point (Experiments 1-5): mask consensus with 3-view
                # agreement and no depth gate. min_views/depth_rtol must be passed
                depth_rtol          = getattr(self.config, "mv_depth_rtol", 100.0),
                min_views           = getattr(self.config, "mv_min_views", 3),
                mask_seeded_pool    = getattr(self.config, "mv_mask_seeded_pool", True),
                boundary_bias_alpha = getattr(self.config, "mv_boundary_bias_alpha", 0.0),
                pool_size           = getattr(self.config, "mv_pool_size", 8192),
                intra_obj_register  = getattr(self.config, "mv_intra_obj_register", False),
                register_iters      = getattr(self.config, "mv_register_iters", 4),
                return_target_ids   = True,
            )   # (B, V_obj, 3), (B, V_ctx, 3), (B, N)
        else:
            # Panoptic masks not available — fall back to FPS of Pi3X geometry
            logger.warning_once(
                "panoptic_masks not provided; falling back to seed-FPS for obj_voxels."
            )
            mv_target_ids = None  # no mask consensus possible without panoptic masks
            ref_idx = self._select_ref_view(lp, view_mask)
            seed_list = []
            for b in range(B):
                rv   = ref_idx[b].item()
                pc_b = build_geo_obj_pc(
                    lp[b:b+1, rv], cond_pcs_2d[b:b+1], st[b:b+1, rv]
                )
                seed_list.append(pc_b)
            seed_pcs   = torch.cat(seed_list, dim=0)
            obj_voxels = fps_centroid_seeded(seed_pcs.float(), mv_num_obj_voxels)
            # ctx: all valid Pi3X scene points merged across views
            ctx_list = []
            for b in range(B):
                pts_all = []
                for n in range(N):
                    if not view_mask[b, n]:
                        continue
                    lp_n = lp[b, n]
                    valid = lp_n[..., 2] > 0
                    if valid.any():
                        from .frozen_geo_encoder import _apply_scene_transform
                        pts_s = _apply_scene_transform(
                            lp_n[valid].float().unsqueeze(0), st[b:b+1, n]
                        ).squeeze(0)
                        pts_all.append(pts_s)
                merged = torch.cat(pts_all, dim=0) if pts_all else seed_pcs[b].float()
                ctx_list.append(fps_centroid_seeded(merged.unsqueeze(0), mv_num_ctx_voxels).squeeze(0))
            ctx_voxels = torch.stack(ctx_list, dim=0)

        obj_voxels = obj_voxels.to(lp.dtype)
        ctx_voxels = ctx_voxels.to(lp.dtype)

        # --- Canonicalize object voxels for the geometry (PointEmbed) stream ---
        # obj_voxels are scene-frame (required for view projection + feature sampling),
        # but the decoder emits vertices in the per-object CANONICAL frame. Rotate by the
        # data-derived scene->object rotation, then re-center/re-scale by the voxels' OWN
        # observed extent — this cancels Pi3X's unknown global scale and reproduces
        # normalize_vertices(bound=0.95) on the observed surface. Only the geometry stream
        # sees these; projection/feature-sampling keep the scene-frame obj_voxels.
        obj_geom_voxels = None
        if obj_canon_transform is not None:
            # Fix A: the geometry stream uses the cross-view-REGISTERED cloud when available
            # (obj_voxels_geom), while obj_voxels stays scene-frame for projection/DINO.
            src_voxels = obj_voxels_geom if obj_voxels_geom is not None else obj_voxels
            R = obj_canon_transform[:, :3, :3].to(device=src_voxels.device, dtype=src_voxels.dtype)
            v = torch.bmm(src_voxels, R.transpose(1, 2))   # (B, V, 3) rotate about origin
            # Robust per-axis extent (MoGe-ROE style, Fix B): raw min/max lets a few
            # partial-observation outliers / mask-bleed stragglers inflate the scale and
            # blow up the aspect ratio (measured: ~1.4x inflation, anisotropy 4.24). Using
            # the [q, 1-q] quantile box instead trims those tails. q=0 recovers the original
            # min/max behaviour (clean A/B). Rotation stays from obj_canon_transform.
            q = float(getattr(self.config, "mv_geom_norm_quantile", 0.0) or 0.0)
            if q > 0.0:
                qs = torch.tensor([q, 1.0 - q], device=v.device, dtype=torch.float32)
                bounds = torch.quantile(v.float(), qs, dim=1)   # (2, B, 3)
                vmin = bounds[0].unsqueeze(1).to(v.dtype)        # (B, 1, 3)
                vmax = bounds[1].unsqueeze(1).to(v.dtype)
            else:
                vmin = v.amin(dim=1, keepdim=True)
                vmax = v.amax(dim=1, keepdim=True)
            center = 0.5 * (vmin + vmax)
            scale = (2 * 0.95) / (vmax - vmin).amax(dim=-1, keepdim=True).clamp_min(1e-6)
            obj_geom_voxels = (v - center) * scale          # (B, V, 3) canonical frame

        # --- DEBUG oracle ceiling: bypass the observed self-norm and condition the
        # geometry stream on GT-canonical surface points (full-extent, leak-by-design).
        # Bounds whether perfect conditioning beats SV and isolates Finding 2's scale cost.
        # NOTE: leave mv_obj_pc_appearance OFF under the oracle — the appearance feature is
        # aligned to the observed obj_voxels, not these GT points.
        if getattr(self.config, "mv_obj_pc_oracle", False) and gt_obj_vertices is not None:
            obj_geom_voxels = gt_obj_vertices.to(
                device=obj_voxels.device, dtype=obj_voxels.dtype
            )

        # --- DINOv2 features (shared by the obj-PC appearance term + voxel encoder) ---
        # Computed once when either consumer needs it; frozen + depends only on the
        # un-augmented images, so the cache path is numerically identical.
        mv_conf = pi3x_out["conf"][..., 0] if pi3x_out.get("conf") is not None else None
        need_dino = getattr(self.config, "mv_obj_pc_appearance", False) or getattr(
            self.config, "mv_use_voxel_encoder", True
        )
        dino_feats = None
        if need_dino:
            if cached_dino_feats is not None:
                dino_feats = cached_dino_feats.to(device)
            else:
                if self.cond_encoder_img is None:
                    raise RuntimeError("cond_encoder_img is required for multi-view path")
                pv_flat    = pixel_values.reshape(B * N, C, H, W)
                dino_flat  = self.cond_encoder_img(pixel_values=pv_flat)
                _, C_d, Hf, Wf = dino_flat.shape
                dino_feats = dino_flat.reshape(B, N, C_d, Hf, Wf)

        # --- Native obj-PC GEOMETRY channel (the decoder's exploited channel) ---
        # Route the multi-view-discovered, canonical points through the SV cond_encoder
        # (->2048 latents) + projector — the exact channel the decoder was trained on and
        # provably exploits (Test-1: +17% from obj-PC completeness). "More views = a more
        # complete point cloud" then flows through a channel the decoder actually uses.
        # obj_geom_voxels is canonical / normalize_vertices(0.95), matching the output frame.
        # When mv_obj_pc_appearance is on, supply the confidence-weighted multi-view DINO
        # at the obj voxels as extra_feat — the SV cond_encoder was trained with extra_feat
        # ALWAYS present (img_cond_drop_prob=0), so geometry-only is off-distribution.
        obj_pc_embeds = None
        if getattr(self.config, "mv_obj_pc_cond", False):
            if obj_geom_voxels is None:
                raise RuntimeError("mv_obj_pc_cond=True requires obj_canon_transform")
            obj_pc_extra_feat = None
            if getattr(self.config, "mv_obj_pc_appearance", False):
                obj_pc_extra_feat = self._fuse_obj_view_features(
                    obj_voxels, dino_feats, st, K_f, pi3x_depth, view_mask,
                    panoptic_masks.to(device) if panoptic_masks is not None else None,
                    mv_target_ids, mv_conf,
                )   # (B, V_obj, C_d) aligned with obj_geom_voxels
            obj_pc_conds  = self.cond_encoder(obj_geom_voxels, extra_feat=obj_pc_extra_feat)
            obj_pc_embeds = self.projector(obj_pc_conds)   # (B, pc_latent_len, D)

        # --- Multi-view voxel encoder → z_i, z_scene (appearance fusion; optional) ---
        # Pass panoptic_masks + per-view target IDs so the encoder gates per-view
        # features by mask consensus (Experiment 3) instead of the Pi3X depth check,
        # and Pi3X confidence so the IBRNet fusion is confidence-weighted.
        z_i = z_scene = None
        if getattr(self.config, "mv_use_voxel_encoder", True):
            mv_out  = self.mv_voxel_encoder(
                obj_voxels, ctx_voxels,
                dino_feats, None,                    # mask_feats=None (discovery handles localization)
                st, K_f, pi3x_depth, view_mask,
                panoptic_masks = panoptic_masks.to(device) if panoptic_masks is not None else None,
                target_ids     = mv_target_ids,
                conf           = mv_conf,
                obj_geom_voxels = obj_geom_voxels,   # canonical-frame geometry for PointEmbed
            )
            z_i     = mv_out["z_i"]      # (B, M, out_dim)
            z_scene = mv_out["z_scene"]  # (B, S, out_dim)

        # --- Assemble input embeddings ---
        if cond_num_faces is None:
            cond_num_faces = torch.zeros(B, 1, dtype=torch.long, device=device)
        num_face_embeds = self.embed_num_face(cond_num_faces)   # (B, 1, D)

        input_ids_clone = input_ids.clone()
        cond_token_mask = input_ids_clone == self.config.pc_token_id
        indicator_mask  = input_ids_clone == self.config.indicator_token_id
        obj_pc_mask     = input_ids_clone == self.config.obj_pc_token_id
        input_ids_clone[cond_token_mask | indicator_mask | obj_pc_mask] = (
            self.config.pad_token_id
        )
        inputs_embeds = self.model.decoder.embed_tokens(input_ids_clone)

        # Fill pc_token slots, in order: [obj-PC latents] + [z_i, z_scene] + num_face.
        # Channels are included only when produced (see flags above); the total token
        # count must equal prefix_len (the collator emits that many pc_token slots).
        cond_parts = []
        if obj_pc_embeds is not None:
            cond_parts.append(obj_pc_embeds.to(inputs_embeds.dtype))   # (B, pc_latent_len, D)
        if z_i is not None:
            cond_parts.append(z_i.to(inputs_embeds.dtype))             # (B, M, D)
            cond_parts.append(z_scene.to(inputs_embeds.dtype))         # (B, S, D)
        cond_parts.append(num_face_embeds.to(inputs_embeds.dtype))     # (B, 1, D)
        all_cond = torch.cat(cond_parts, dim=1).flatten(0, 1)          # (B*prefix_cond, D)
        # Guard the prefix_len <-> produced-token invariant: masked_scatter silently
        # mis-fills when sizes differ, so assert rather than emit garbage conditioning.
        assert all_cond.shape[0] == int(cond_token_mask.sum()), (
            f"MV cond token count {all_cond.shape[0]} != pc_token slots "
            f"{int(cond_token_mask.sum())}; prefix_len must equal mv_prefix_len(config)."
        )
        # Out-of-place: when the decoder is frozen (overfit / frozen-decoder regime)
        # inputs_embeds is a frozen leaf, and in-place masked_scatter_ on it errors.
        inputs_embeds = inputs_embeds.masked_scatter(
            cond_token_mask.unsqueeze(-1), all_cond
        )
        return inputs_embeds

    def _forward_multiview(
        self,
        input_ids,
        pixel_values,           # (B, N, C, H, W)
        scene_transforms,       # (B, N, 4, 4)
        K_per_view,             # (B, N, 3, 3)
        view_mask,              # (B, N) bool
        panoptic_masks,         # (B, N, H, W) long or None
        cond_pcs,               # (B, P, 3)
        cond_pcs_2d,            # (B, P, 2)
        cond_num_faces,
        cached_local_points=None,
        cached_conf=None,
        cached_dino_feats=None,
        obj_canon_transform=None,
        gt_obj_vertices=None,
        **decoder_kwargs,
    ):
        inputs_embeds = self.get_mv_inputs_with_cond(
            input_ids=input_ids,
            pixel_values=pixel_values,
            scene_transforms=scene_transforms,
            K_per_view=K_per_view,
            view_mask=view_mask,
            panoptic_masks=panoptic_masks,
            cond_pcs=cond_pcs,
            cond_pcs_2d=cond_pcs_2d,
            cond_num_faces=cond_num_faces,
            cached_local_points=cached_local_points,
            cached_conf=cached_conf,
            cached_dino_feats=cached_dino_feats,
            obj_canon_transform=obj_canon_transform,
            gt_obj_vertices=gt_obj_vertices,
        )

        # --- OPT decoder ---
        output_attentions    = decoder_kwargs.pop("output_attentions",    self.config.output_attentions)
        output_hidden_states = decoder_kwargs.pop("output_hidden_states", self.config.output_hidden_states)
        return_dict          = decoder_kwargs.pop("return_dict",          self.config.use_return_dict)
        labels               = decoder_kwargs.pop("labels", None)
        attention_mask       = decoder_kwargs.pop("attention_mask", None)
        position_ids         = decoder_kwargs.pop("position_ids", None)
        head_mask            = decoder_kwargs.pop("head_mask", None)
        past_key_values      = decoder_kwargs.pop("past_key_values", None)
        use_cache            = decoder_kwargs.pop("use_cache", None)
        cache_position       = decoder_kwargs.pop("cache_position", None)

        outputs = self.model.decoder(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **decoder_kwargs,
        )
        logits = self.lm_head(outputs[0]).contiguous()

        loss = loss_layout = loss_object = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss, loss_layout, loss_object = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                loss_layout_scale=self.config.loss_layout_scale,
                **decoder_kwargs,
            )

        return CustomCausalLMOutputWithTokenTypes(
            loss=loss,
            loss_layout=loss_layout,
            loss_object=loss_object,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @property
    def loss_function(self):
        return causal_lm_loss_with_token_types

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        cond_pcs=None,
        cond_pcs_2d=None,
        ctx_pcs=None,
        ctx_pcs_2d=None,
        cond_num_faces=None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        obj_indices=None,
        obj_bboxes=None,
        obj_cond_pcs=None,
        pixel_values=None,
        scene_transform=None,
        # Multi-view fields
        scene_transforms=None,    # (B, N, 4, 4) — per-view; present in MV batch
        K_per_view=None,          # (B, N, 3, 3)
        view_mask=None,           # (B, N) bool
        panoptic_masks=None,      # (B, N, H, W) long — Grounded-SAM instance IDs
        cached_local_points=None, # precomputed frozen-Pi3X geometry (skips Pi3X forward)
        cached_conf=None,
        cached_dino_feats=None,   # precomputed frozen-DINOv2 features (skips DINOv2 forward)
        obj_canon_transform=None, # (B, 4, 4) scene -> per-object canonical (geometry frame)
        gt_obj_vertices=None,     # (B, V, 3) DEBUG oracle: GT-canonical surface points
        **kwargs,
    ):
        # Multi-view path: pixel_values is (B, N, C, H, W) when N > 1.
        # All N views are processed in a single Pi3X forward, producing per-view geometry;
        if (
            self.pi3x_encoder is not None
            and self.mv_voxel_encoder is not None
            and pixel_values is not None
            and pixel_values.dim() == 5
            and scene_transforms is not None
        ):
            return self._forward_multiview(
                input_ids=input_ids,
                pixel_values=pixel_values,
                scene_transforms=scene_transforms,
                K_per_view=K_per_view,
                view_mask=view_mask,
                panoptic_masks=panoptic_masks,
                cond_pcs=cond_pcs,
                cond_pcs_2d=cond_pcs_2d,
                cond_num_faces=cond_num_faces,
                cached_local_points=cached_local_points,
                cached_conf=cached_conf,
                cached_dino_feats=cached_dino_feats,
                obj_canon_transform=obj_canon_transform,
                gt_obj_vertices=gt_obj_vertices,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                position_ids=position_ids,
                cache_position=cache_position,
                obj_indices=obj_indices,
                obj_bboxes=obj_bboxes,
                **kwargs,
            )

        # Pi3X override: replace data-loader point clouds with Pi3X predictions.
        # Requires both pixel_values (for Pi3X inference) and scene_transform (to convert
        # camera-frame XYZ → normalised scene space). If scene_transform is absent
        # (e.g. legacy inference scripts that predate Pi3X), we log once and fall back
        # to the data-loader point clouds so the model still runs correctly.
        if self.pi3x_encoder is not None and pixel_values is not None:
            if scene_transform is None:
                logger.warning_once(
                    "Pi3X encoder is active but scene_transform was not provided. "
                    "Falling back to data-loader point clouds. "
                    "Populate scene_transforms in the data pipeline to enable Pi3X."
                )
            else:
                # Pi3X.encode() receives pre-normalised images; pass pixel_values directly.
                pi3x_out = self.pi3x_encoder(pixel_values)
                lp = pi3x_out["local_points"]   # already in pixel_values.dtype
                cf = pi3x_out["conf"]
                # scene_transform: (B, 4, 4).  _apply_scene_transform runs in float32
                # internally — no need to cast st to bfloat16 here.
                st = scene_transform.to(pixel_values.device)
                if ctx_pcs is not None:
                    ctx_pcs, ctx_pcs_2d = build_geo_ctx_pc(lp, cf, st, ctx_pcs.shape[1])
                if cond_pcs is not None and cond_pcs_2d is not None:
                    cond_pcs = build_geo_obj_pc(lp, cond_pcs_2d, st)

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.cond_encoder is not None and cond_pcs is not None:
            inputs_embeds = self.get_inputs_with_cond(
                input_ids,
                cond_pcs=cond_pcs,
                cond_pcs_2d=cond_pcs_2d,
                ctx_pcs=ctx_pcs,
                ctx_pcs_2d=ctx_pcs_2d,
                cond_num_faces=cond_num_faces,
                obj_indices=obj_indices,
                obj_bboxes=obj_bboxes,
                obj_cond_pcs=obj_cond_pcs,
                pixel_values=pixel_values,
            )
            input_ids = None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = loss_layout = loss_object = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss, loss_layout, loss_object = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                loss_layout_scale=self.config.loss_layout_scale,
                **kwargs,
            )

        return CustomCausalLMOutputWithTokenTypes(
            loss=loss,
            loss_layout=loss_layout,
            loss_object=loss_object,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
