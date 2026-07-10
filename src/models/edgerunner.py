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
from .frozen_geo_encoder import _apply_scene_transform, build_geo_ctx_pc, build_geo_obj_pc, fps_centroid_seeded
from .discovery import get_discovery_fn
from .mv_voxel_encoder import (
    MultiViewVoxelAlignedEncoder,
    _compute_visibility_mask,
    _project_to_views,
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
        geo_encoder=None,
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
        # Generic frozen image-to-geometry backbone (DA3 for the current MV port).
        self.geo_encoder = geo_encoder

        self.ctx_aggregator = None
        if self.config.with_ctx_pc:
            self.ctx_aggregator = ContextAggregator(
                cond_encoder.output_dim, num_heads=8
            )
            self.ctx_aggregator.apply(self._init_weights)

        self.mv_voxel_encoder = None
        if getattr(config, "mv_voxel_encoder", False) and cond_encoder_img is not None:
            img_feat_dim = getattr(cond_encoder_img, "output_dim", 384)
            self.mv_voxel_encoder = MultiViewVoxelAlignedEncoder(
                feat_dim=img_feat_dim,
                voxel_dim=getattr(config, "mv_voxel_dim", 512),
                out_dim=config.word_embed_proj_dim,
                num_obj_queries=getattr(config, "mv_num_obj_queries", 257),
                num_scene_queries=getattr(config, "mv_num_scene_queries", 64),
                num_heads=getattr(config, "mv_num_heads", 8),
                use_geometry=getattr(config, "mv_use_geometry", True),
            )
            self.mv_voxel_encoder.apply(self._init_weights)

        self.mv_aabb_embed = None
        if getattr(config, "mv_obj_aabb_token", False):
            self.mv_aabb_embed = nn.Linear(6, config.word_embed_proj_dim)
            self.mv_aabb_embed.apply(self._init_weights)

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

    def _select_ref_view(
        self,
        local_points: torch.Tensor,
        view_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Reference view fallback: view with the most positive-depth pixels."""
        bsz, num_views, _, _, _ = local_points.shape
        valid_count = (local_points[..., 2] > 0).reshape(bsz, num_views, -1).sum(-1).float()
        if view_mask is not None:
            valid_count = valid_count * view_mask.float()
        return valid_count.argmax(dim=1)

    def _fuse_obj_view_features(
        self,
        voxels,
        dino_feats,
        scene_transforms,
        K_per_view,
        geo_depth,
        view_mask,
        panoptic_masks,
        target_ids,
        conf,
    ):
        """Confidence-weighted mean DINO features at object voxels."""
        B, V, _ = voxels.shape
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
        )
        dino_sampled = _sample_features(dino_feats, pix_coords)

        weights = vis_mask.unsqueeze(-1).float()
        if conf is not None:
            N = pix_coords.shape[2]
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
            weights = weights * conf_vox.unsqueeze(-1).clamp(min=0.0)
        denom = weights.sum(dim=2).clamp(min=1e-6)
        return ((dino_sampled.float() * weights).sum(dim=2) / denom).to(dino_feats.dtype)

    def get_mv_inputs_with_cond(
        self,
        input_ids,
        pixel_values,
        scene_transforms,
        K_per_view,
        view_mask,
        panoptic_masks,
        cond_pcs,
        cond_pcs_2d,
        cond_num_faces,
        cached_local_points=None,
        cached_conf=None,
        cached_dino_feats=None,
        obj_canon_transform=None,
        gt_obj_vertices=None,
        ref_view=None,
    ):
        """Build multi-view conditioning embeddings and scatter them into pc_token slots."""
        B, N, C, H, W = pixel_values.shape
        device = pixel_values.device
        if view_mask is None:
            view_mask = torch.ones(B, N, dtype=torch.bool, device=device)

        if cached_local_points is not None:
            local_points = cached_local_points.to(device)
            geo_out = {"conf": cached_conf.to(device) if cached_conf is not None else None}
        else:
            if self.geo_encoder is None:
                raise RuntimeError("Multi-view path requires a frozen geo_encoder or cached_local_points")
            geo_out = self.geo_encoder.forward_all_views_joint(pixel_values)
            local_points = geo_out["local_points"]
        geo_depth = local_points[..., 2]
        st = scene_transforms.to(device, dtype=torch.float32)
        K_f = K_per_view.float().to(device)
        if geo_out.get("conf") is not None:
            geo_out["conf"] = geo_out["conf"].to(device)

        view_gate = float(getattr(self.config, "mv_view_conf_gate", 0.0) or 0.0)
        if view_gate > 0.0 and geo_out.get("conf") is not None:
            mean_view_conf = geo_out["conf"][..., 0].float().mean(dim=(-1, -2))
            gate_ok = mean_view_conf >= view_gate
            if ref_view is not None:
                gate_ok.scatter_(1, ref_view.to(gate_ok.device).long().view(-1, 1), True)
            min_keep = int(getattr(self.config, "mv_view_gate_min_views", 2))
            kept = (view_mask & gate_ok).sum(dim=-1)
            if bool((kept < min_keep).any()):
                ranked = mean_view_conf.masked_fill(~view_mask, -1.0)
                k = min(min_keep, ranked.shape[1])
                force = torch.zeros_like(gate_ok)
                force.scatter_(1, ranked.topk(k, dim=-1).indices, True)
                gate_ok = torch.where((kept < min_keep).view(-1, 1), gate_ok | force, gate_ok)
            view_mask = view_mask & gate_ok

        mv_num_obj_voxels = getattr(self.config, "mv_num_obj_voxels", 512)
        mv_num_ctx_voxels = getattr(self.config, "mv_num_ctx_voxels", 1024)
        obj_voxels_geom = None

        if panoptic_masks is not None:
            ref_idx = (
                ref_view.to(local_points.device).long()
                if ref_view is not None
                else self._select_ref_view(local_points, view_mask)
            )
            seed_list = []
            for b in range(B):
                rv = int(ref_idx[b].item())
                seed_list.append(
                    build_geo_obj_pc(
                        local_points[b:b + 1, rv],
                        cond_pcs_2d[b:b + 1],
                        st[b:b + 1, rv],
                    )
                )
            seed_pcs = torch.cat(seed_list, dim=0)

            discover_fn = get_discovery_fn(getattr(self.config, "mv_discovery_method", "consensus"))
            obj_voxels, ctx_voxels, mv_target_ids, obj_voxels_geom = discover_fn(
                local_points=local_points,
                scene_transforms=st,
                panoptic_masks=panoptic_masks.to(device),
                K_per_view=K_f,
                view_mask=view_mask,
                seed_pcs=seed_pcs.float(),
                num_obj_voxels=mv_num_obj_voxels,
                num_ctx_voxels=mv_num_ctx_voxels,
                conf=geo_out["conf"],
                conf_threshold=getattr(self.config, "mv_conf_threshold", 0.5),
                depth_rtol=getattr(self.config, "mv_depth_rtol", 100.0),
                min_views=getattr(self.config, "mv_min_views", 3),
                mask_seeded_pool=getattr(self.config, "mv_mask_seeded_pool", True),
                boundary_bias_alpha=getattr(self.config, "mv_boundary_bias_alpha", 0.0),
                pool_size=getattr(self.config, "mv_pool_size", 8192),
                intra_obj_register=getattr(self.config, "mv_intra_obj_register", False),
                register_iters=getattr(self.config, "mv_register_iters", 4),
                voxel_sampling=getattr(self.config, "mv_voxel_sampling", "fps"),
                return_target_ids=True,
            )
        else:
            logger.warning("panoptic_masks not provided; falling back to seed-FPS MV voxels.")
            mv_target_ids = None
            ref_idx = (
                ref_view.to(local_points.device).long()
                if ref_view is not None
                else self._select_ref_view(local_points, view_mask)
            )
            seed_list = []
            for b in range(B):
                rv = int(ref_idx[b].item())
                seed_list.append(
                    build_geo_obj_pc(
                        local_points[b:b + 1, rv],
                        cond_pcs_2d[b:b + 1],
                        st[b:b + 1, rv],
                    )
                )
            seed_pcs = torch.cat(seed_list, dim=0)
            obj_voxels = fps_centroid_seeded(seed_pcs.float(), mv_num_obj_voxels)
            ctx_list = []
            for b in range(B):
                pts_all = []
                for n in range(N):
                    if not view_mask[b, n]:
                        continue
                    lp_n = local_points[b, n]
                    valid = lp_n[..., 2] > 0
                    if valid.any():
                        pts_s = _apply_scene_transform(
                            lp_n[valid].float().unsqueeze(0), st[b:b + 1, n]
                        ).squeeze(0)
                        pts_all.append(pts_s.float())
                merged = torch.cat(pts_all, dim=0) if pts_all else seed_pcs[b].float()
                ctx_list.append(
                    fps_centroid_seeded(merged.unsqueeze(0), mv_num_ctx_voxels).squeeze(0)
                )
            ctx_voxels = torch.stack(ctx_list, dim=0)

        obj_voxels = obj_voxels.to(local_points.dtype)
        ctx_voxels = ctx_voxels.to(local_points.dtype)

        obj_geom_voxels = None
        if obj_canon_transform is not None:
            src_voxels = obj_voxels_geom if obj_voxels_geom is not None else obj_voxels
            R = obj_canon_transform[:, :3, :3].to(device=src_voxels.device, dtype=src_voxels.dtype)
            v = torch.bmm(src_voxels, R.transpose(1, 2))
            q = float(getattr(self.config, "mv_geom_norm_quantile", 0.0) or 0.0)
            if q > 0.0:
                qs = torch.tensor([q, 1.0 - q], device=v.device, dtype=torch.float32)
                bounds = torch.quantile(v.float(), qs, dim=1)
                vmin = bounds[0].unsqueeze(1).to(v.dtype)
                vmax = bounds[1].unsqueeze(1).to(v.dtype)
            else:
                vmin = v.amin(dim=1, keepdim=True)
                vmax = v.amax(dim=1, keepdim=True)
            center = 0.5 * (vmin + vmax)
            scale = (2 * 0.95) / (vmax - vmin).amax(dim=-1, keepdim=True).clamp_min(1e-6)
            obj_geom_voxels = (v - center) * scale

        if getattr(self.config, "mv_obj_pc_oracle", False) and gt_obj_vertices is not None:
            obj_geom_voxels = gt_obj_vertices.to(device=obj_voxels.device, dtype=obj_voxels.dtype)

        mv_conf = geo_out["conf"][..., 0] if geo_out.get("conf") is not None else None
        need_dino = getattr(self.config, "mv_obj_pc_appearance", False) or (
            getattr(self.config, "mv_use_voxel_encoder", True) and self.mv_voxel_encoder is not None
        )
        dino_feats = None
        if need_dino:
            if cached_dino_feats is not None:
                dino_feats = cached_dino_feats.to(device)
            else:
                if self.cond_encoder_img is None:
                    raise RuntimeError("cond_encoder_img is required for multi-view DINO features")
                pv_flat = pixel_values.reshape(B * N, C, H, W)
                dino_flat = self.cond_encoder_img(pixel_values=pv_flat)
                _, C_d, Hf, Wf = dino_flat.shape
                dino_feats = dino_flat.reshape(B, N, C_d, Hf, Wf)

        obj_pc_embeds = None
        if getattr(self.config, "mv_obj_pc_cond", False):
            if obj_geom_voxels is None:
                raise RuntimeError("mv_obj_pc_cond=True requires obj_canon_transform")
            obj_pc_extra_feat = None
            if getattr(self.config, "mv_obj_pc_appearance", False):
                obj_pc_extra_feat = self._fuse_obj_view_features(
                    obj_voxels,
                    dino_feats,
                    st,
                    K_f,
                    geo_depth,
                    view_mask,
                    panoptic_masks.to(device) if panoptic_masks is not None else None,
                    mv_target_ids,
                    mv_conf,
                )
            obj_pc_conds = self.cond_encoder(obj_geom_voxels, extra_feat=obj_pc_extra_feat)
            obj_pc_embeds = self.projector(obj_pc_conds)

        obj_view_mask = None
        if mv_target_ids is not None and panoptic_masks is not None:
            pm_dev = panoptic_masks.to(device)
            pixel_support_n = (pm_dev == mv_target_ids[:, :, None, None]).sum((-1, -2)).float()
            min_sup_pix = float(getattr(self.config, "mv_covis_min_support_pix", 200))
            obj_view_mask = view_mask & (pixel_support_n >= min_sup_pix)

        z_i = z_scene = None
        if getattr(self.config, "mv_use_voxel_encoder", True):
            if self.mv_voxel_encoder is None:
                raise RuntimeError("mv_use_voxel_encoder=True but mv_voxel_encoder was not built")
            mv_out = self.mv_voxel_encoder(
                obj_voxels,
                ctx_voxels,
                dino_feats,
                st,
                K_f,
                geo_depth,
                view_mask,
                panoptic_masks=panoptic_masks.to(device) if panoptic_masks is not None else None,
                target_ids=mv_target_ids,
                conf=mv_conf,
                obj_geom_voxels=obj_geom_voxels,
                obj_view_mask=obj_view_mask,
            )
            z_i = mv_out["z_i"]
            z_scene = mv_out["z_scene"]

        if cond_num_faces is None:
            cond_num_faces = torch.zeros(B, 1, dtype=torch.long, device=device)
        num_face_embeds = self.embed_num_face(cond_num_faces)

        input_ids_clone = input_ids.clone()
        cond_token_mask = input_ids_clone == self.config.pc_token_id
        indicator_mask = input_ids_clone == self.config.indicator_token_id
        obj_pc_mask = input_ids_clone == self.config.obj_pc_token_id
        input_ids_clone[cond_token_mask | indicator_mask | obj_pc_mask] = self.config.pad_token_id
        inputs_embeds = self.model.decoder.embed_tokens(input_ids_clone)

        cond_parts = []
        if obj_pc_embeds is not None:
            cond_parts.append(obj_pc_embeds.to(inputs_embeds.dtype))
        if z_i is not None:
            cond_parts.append(z_i.to(inputs_embeds.dtype))
            cond_parts.append(z_scene.to(inputs_embeds.dtype))
        if self.mv_aabb_embed is not None:
            aabb = torch.cat([obj_voxels.amin(dim=1), obj_voxels.amax(dim=1)], dim=-1)
            cond_parts.append(self.mv_aabb_embed(aabb.to(inputs_embeds.dtype)).unsqueeze(1))
        cond_parts.append(num_face_embeds.to(inputs_embeds.dtype))
        all_cond = torch.cat(cond_parts, dim=1).flatten(0, 1)
        assert all_cond.shape[0] == int(cond_token_mask.sum()), (
            f"MV cond token count {all_cond.shape[0]} != pc_token slots "
            f"{int(cond_token_mask.sum())}; prefix_len must equal mv_prefix_len(config)."
        )
        return inputs_embeds.masked_scatter(cond_token_mask.unsqueeze(-1), all_cond)

    def _forward_multiview(
        self,
        input_ids,
        pixel_values,
        scene_transforms,
        K_per_view,
        view_mask,
        panoptic_masks,
        cond_pcs,
        cond_pcs_2d,
        cond_num_faces,
        cached_local_points=None,
        cached_conf=None,
        cached_dino_feats=None,
        obj_canon_transform=None,
        gt_obj_vertices=None,
        ref_view=None,
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
            ref_view=ref_view,
        )

        output_attentions = decoder_kwargs.pop("output_attentions", self.config.output_attentions)
        output_hidden_states = decoder_kwargs.pop(
            "output_hidden_states", self.config.output_hidden_states
        )
        decoder_kwargs.pop("return_dict", None)
        labels = decoder_kwargs.pop("labels", None)
        attention_mask = decoder_kwargs.pop("attention_mask", None)
        position_ids = decoder_kwargs.pop("position_ids", None)
        head_mask = decoder_kwargs.pop("head_mask", None)
        past_key_values = decoder_kwargs.pop("past_key_values", None)
        use_cache = decoder_kwargs.pop("use_cache", None)
        cache_position = decoder_kwargs.pop("cache_position", None)
        token_type_ids = decoder_kwargs.pop("token_type_ids", None)
        num_items_in_batch = decoder_kwargs.pop("num_items_in_batch", None)
        decoder_kwargs.pop("obj_indices", None)
        decoder_kwargs.pop("obj_bboxes", None)

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
                token_type_ids=token_type_ids,
                num_items_in_batch=num_items_in_batch,
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
        scene_transforms=None,
        K_per_view=None,
        view_mask=None,
        panoptic_masks=None,
        cached_local_points=None,
        cached_conf=None,
        cached_dino_feats=None,
        obj_canon_transform=None,
        gt_obj_vertices=None,
        ref_view=None,
        **kwargs,
    ):
        if (
            pixel_values is not None
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
                ref_view=ref_view,
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

        if self.geo_encoder is not None and pixel_values is not None:
            if scene_transform is None:
                logger.warning(
                    "geo_encoder is active but scene_transform was not provided; "
                    "falling back to data-loader point clouds."
                )
            else:
                geo_out = self.geo_encoder(pixel_values)
                lp = geo_out["local_points"]
                conf = geo_out.get("conf")
                st = scene_transform.to(pixel_values.device)
                if ctx_pcs is not None:
                    ctx_pcs, ctx_pcs_2d = build_geo_ctx_pc(lp, conf, st, ctx_pcs.shape[1])
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
