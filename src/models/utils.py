import logging
import math
import torch
import torch.nn as nn
from src.utils.config import ModelConfig
from .meshxl import MeshOPT, MeshOPTConfig
from .edgerunner import ShapeOPT, ShapeOPTConfig
from .bpt import BPTModel, BPTConfig
from .pc_miche.encoder import PointCloudEncoder
from .pc_edgerunner.encoder import EdgeRunnerPointEncoder
from .cond import ConditionEncoder
from .img_cond import ImageConditionEncoder, HighResImageConditionEncoder
from .frozen_geo_encoder import FrozenGeoEncoder  # noqa: F401 (re-exported for external use)

logger = logging.getLogger(__name__)


def _fix_uninit_params(model):
    """Reinitialize any NaN/Inf parameters left by from_pretrained's no_init_weights context.

    from_pretrained runs __init__ under no_init_weights(), which patches kaiming_uniform_/normal_
    to no-ops. Modules whose keys are absent from the checkpoint are never overwritten, leaving
    them as uninitialized GPU memory. After casting to bfloat16, garbage float32 values can
    become bfloat16 NaN and corrupt the entire forward pass.
    """
    init_std = getattr(getattr(model, "config", None), "init_std", 0.02)
    for module in model.modules():
        has_bad = any(
            p.is_floating_point() and (
                torch.isnan(p.data).any() or torch.isinf(p.data).any()
                # Uninitialized GPU memory is finite-but-huge (e.g. ~1e31) and does NOT
                # become NaN/inf — the mv_voxel_encoder (absent from single-view
                # checkpoints) hits exactly this. Scope the magnitude test to trainable
                # params so frozen Pi3X/DINOv2 real weights are never touched.
                or (p.requires_grad and p.data.abs().max() > 1e4)
            )
            for p in module.parameters(recurse=False)
        )
        if not has_bad:
            continue
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, mean=0.0, std=init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight.data, mean=0.0, std=init_std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight.data)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)
        else:
            for p in module.parameters(recurse=False):
                if p.is_floating_point():
                    nn.init.normal_(p.data, mean=0.0, std=init_std)


def _fix_pointembed_basis(model):
    """Recompute every PointEmbed Fourier `basis` buffer after from_pretrained.

    `basis` is a deterministic constant, but a buffer absent from the checkpoint is
    materialized as GARBAGE memory under low_cpu_mem_usage/meta init. That garbage is NOT
    reliably NaN/Inf — it is often finite-but-huge (observed ~2.8e38), which a finiteness
    check passes; `x @ huge` then overflows bf16 to inf and `sin(inf)` = NaN, corrupting the
    conditioning forward data-dependently. Since the basis is a constant, recompute it
    UNCONDITIONALLY (targets BUFFERS, which _fix_uninit_params does not). Only the
    from-scratch path hits this; a warm checkpoint carries a valid saved basis but
    recomputing it is identical and harmless."""
    n = 0
    for module in model.modules():
        if hasattr(module, "reset_basis") and hasattr(module, "basis"):
            module.reset_basis()
            n += 1
    if n:
        logger.info(f"_fix_pointembed_basis: recomputed {n} PointEmbed basis buffer(s)")


def get_pi3x_encoder(model_cfg: ModelConfig) -> "FrozenGeoEncoder":
    """Build the Pi3X frozen geometry encoder from ModelConfig."""
    from .pi3x_cond import Pi3XFrozenEncoder
    enc = Pi3XFrozenEncoder.from_model_cfg(model_cfg)
    # Keep Pi3X in fp32: its forward_head deliberately upcasts features to fp32 and
    # runs the point/conf conv heads under autocast(enabled=False) for numerical
    # precision. Casting the whole module to bf16 makes those fp32 features meet
    # bf16 conv weights -> dtype-mismatch crash. The enclosing autocast still runs
    # the heavy transformer encode/decode in bf16, so fp32 here costs little.
    return enc


def get_model(
    local_model_path,
    model_cfg: ModelConfig,
    cond_encoder=None,
    cond_encoder_img=None,
    pi3x_encoder=None,
):
    extra_args = {}
    if model_cfg is not None:
        extra_args["bos_token_id"] = model_cfg.bos_token_id
        extra_args["eos_token_id"] = model_cfg.eos_token_id
        extra_args["pad_token_id"] = model_cfg.pad_token_id
        extra_args["max_position_embeddings"] = model_cfg.max_position_embeddings
        extra_args["indicator_token_id"] = model_cfg.indicator_token_id
        extra_args["obj_pc_token_id"] = model_cfg.obj_pc_token_id
        extra_args["pc_token_id"] = model_cfg.pc_token_id
        extra_args["with_ctx_pc"] = model_cfg.with_ctx_pc
        extra_args["img_cond_drop_prob"] = model_cfg.img_cond_drop_prob
        extra_args["loss_layout_scale"] = model_cfg.loss_layout_scale
        extra_args["loss_layout_ordinal_sigma"] = model_cfg.loss_layout_ordinal_sigma
        if model_cfg.sep_token_id is not None:
            extra_args["sep_token_id"] = model_cfg.sep_token_id
        # MV voxel encoder fields — override stale values in single-view checkpoints
        # so that from_pretrained() creates mv_voxel_encoder when mv_voxel_encoder=True.
        _mv_fields = [
            "mv_voxel_encoder", "mv_num_obj_voxels", "mv_num_ctx_voxels",
            "mv_voxel_dim", "mv_num_obj_queries", "mv_num_scene_queries",
            "mv_num_heads", "mv_mask_seeded_pool", "mv_boundary_bias_alpha",
            "mv_obj_pc_cond", "mv_use_voxel_encoder", "mv_obj_pc_appearance",
            "mv_obj_pc_oracle", "mv_discovery_method",
            # Discovery / voxelization params — must be here so yaml overrides reach
            # ShapeOPTConfig; without this, getattr fallbacks in edgerunner.py fire
            # instead of the configured values (e.g. mv_min_views: 2 → was using 3).
            "mv_min_views", "mv_conf_threshold", "mv_depth_rtol", "mv_pool_size",
            "mv_intra_obj_register", "mv_register_iters",
            "mv_geom_norm_quantile", "mv_use_geometry", "mv_voxel_sampling",
            "mv_covis_min_support_pix", "mv_view_conf_gate",
            "mv_view_gate_min_views", "mv_obj_aabb_token",
        ]
        for _f in _mv_fields:
            if hasattr(model_cfg, _f):
                extra_args[_f] = getattr(model_cfg, _f)
        model_type = model_cfg.ar_model_type
    else:
        model_type = "meshxl"

    is_scene = "-scene" in model_type

    match model_type:
        case "meshxl":
            config_class = MeshOPTConfig
            model_class = MeshOPT
        case "edgerunner" | "edgerunner-scene":
            config_class = ShapeOPTConfig
            model_class = ShapeOPT
        case "bpt":
            config_class = BPTConfig
            model_class = BPTModel
        case _:
            raise ValueError(f"Unknown model type: {model_type}")

    config = config_class.from_pretrained(local_model_path, **extra_args)
    # PretrainedConfig.from_pretrained discards kwargs that are not declared config
    # attributes (e.g. the mv_* fields on ShapeOPTConfig), so force them onto the
    # config here. Without this, config.mv_voxel_encoder is absent and ShapeOPT
    # never builds the multi-view encoder.
    if model_cfg is not None:
        if "loss_layout_ordinal_sigma" in extra_args:
            setattr(
                config,
                "loss_layout_ordinal_sigma",
                extra_args["loss_layout_ordinal_sigma"],
            )
        for _f in _mv_fields:
            if _f in extra_args:
                setattr(config, _f, extra_args[_f])
    # Snapshot the pre-loaded cond_encoder state before from_pretrained, because
    # from_pretrained detects cond_encoder.* as "MISSING" from the main checkpoint
    # and re-initializes them with random weights, discarding the pretrained values.
    cond_enc_state = (
        {k: v.clone() for k, v in cond_encoder.state_dict().items()}
        if cond_encoder is not None
        else None
    )
    model = model_class.from_pretrained(
        local_model_path,
        config=config,
        cond_encoder=cond_encoder,
        cond_encoder_img=cond_encoder_img,
        is_scene=is_scene,
        ignore_mismatched_sizes=True,
    )
    # Restore pretrained cond_encoder keys that the checkpoint did NOT supply.
    # We detect which keys the checkpoint provided by comparing the post-load model state
    # against the pre-load snapshot: keys that now differ were loaded from the checkpoint
    # and must be kept; keys still matching the snapshot were absent from the checkpoint
    # and need the pretrained values restored (to undo any no_init_weights corruption).
    # This works for both local paths and HuggingFace model IDs.
    if cond_enc_state is not None:
        loaded_state = model.cond_encoder.state_dict()
        restore = {
            k: v for k, v in cond_enc_state.items()
            if k not in loaded_state
            or torch.equal(loaded_state[k].float().cpu(), v.float().cpu())
        }
        model.cond_encoder.load_state_dict(restore, strict=False)
    # If extra_feat_proj is still all-zeros or has garbage values after loading
    # (shape mismatch caused ignore_mismatched_sizes to skip it, leaving uninitialized
    # GPU memory that can overflow float32 norm to inf), re-init with Normal(0, 0.02).
    if cond_encoder_img is not None and cond_encoder is not None:
        enc = model.cond_encoder.encoder
        if hasattr(enc, "extra_feat_proj"):
            w = enc.extra_feat_proj.weight
            w_f32 = w.detach().float()
            w_norm = w_f32.norm().item()
            needs_reinit = not math.isfinite(w_norm) or not w_f32.any()
            if needs_reinit:
                logger.info(
                    f"extra_feat_proj has unusable values (norm={w_norm:.4g}); "
                    "re-initializing with Normal(0, 0.02)"
                )
                w_init = torch.empty(w.shape, dtype=torch.float32).normal_(std=0.02)
                enc.extra_feat_proj.weight.data.copy_(w_init.to(w.dtype))
                enc.extra_feat_proj.bias.data.zero_()
    model = model.to(torch.bfloat16)
    # Attach frozen Pi3X AFTER the bf16 cast so it stays fp32: Pi3X.forward_head
    # upcasts features to fp32 and runs the point/conf conv heads under
    # autocast(enabled=False); if those conv weights are bf16 they crash on the
    # fp32 features. Pi3X manages its own internal autocast for the heavy
    # transformer, so keeping it fp32 here is correct (and was the latent bug that
    # prevented the multi-view path from ever running). Its state_dict() returns {}
    # so from_pretrained never saw it as missing keys.
    if pi3x_encoder is not None and hasattr(model, "pi3x_encoder"):
        model.pi3x_encoder = pi3x_encoder
    # Catch NaN/Inf params left by no_init_weights (absent checkpoint keys → garbage memory
    # → bfloat16 NaN).  This covers ctx_aggregator when loading from the original edgerunner
    # checkpoint (no ctx_aggregator keys) AND avoids overwriting trained ctx_aggregator values
    # when loading from a stage-2 checkpoint that already contains them.
    _fix_uninit_params(model)
    _fix_pointembed_basis(model)
    model.config._attn_implementation = "flash_attention_2"
    if config.vocab_size != model_cfg.vocab_size:
        model.resize_token_embeddings(model_cfg.vocab_size, pad_to_multiple_of=64)
    return model


def get_condition_encoder(
    local_model_path, model_cfg: ModelConfig, cond_encoder_img=None
):
    cond_enc_type = model_cfg.cond_enc_type
    match cond_enc_type:
        case "miche":
            model_class = PointCloudEncoder
        case "edgerunner":
            model_class = EdgeRunnerPointEncoder
        case _:
            raise ValueError(f"Unknown cond enc type: {cond_enc_type}")
    extra_args = {}
    if cond_encoder_img is not None:
        extra_args["with_extra_feat"] = True
        extra_args["extra_feat_dim"] = cond_encoder_img.output_dim
    model = model_class.from_pretrained(local_model_path, **extra_args)
    model = model.to(torch.bfloat16)
    _fix_uninit_params(model)
    _fix_pointembed_basis(model)
    # extra_feat_proj is zero-initialized in encoder.__init__ (to survive no_init_weights).
    # Re-initialize with Normal(0, 0.02) here — outside no_init_weights — so image features
    # contribute from training step 1 instead of gradually turning on from zero.
    if hasattr(model, "extra_feat_proj"):
        w = model.extra_feat_proj.weight
        w_init = torch.empty(w.shape, dtype=torch.float32).normal_(std=0.02)
        w.data.copy_(w_init.to(w.dtype))
        nn.init.zeros_(model.extra_feat_proj.bias.data)
    return ConditionEncoder(model, freeze=model_cfg.freeze_cond_encoder)


def get_image_condition_encoder(model_cfg: ModelConfig):
    if model_cfg.high_res_image_encoder:
        out_features = model_cfg.image_encoder_layers
        model = HighResImageConditionEncoder(
            model_name=model_cfg.image_encoder,
            out_features=out_features,
            hidden_size=model_cfg.high_res_image_encoder_hidden_size,
        )
    else:
        model = ImageConditionEncoder(model_name=model_cfg.image_encoder)
    return model.to(torch.bfloat16)
