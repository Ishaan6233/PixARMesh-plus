import os

os.environ["OMP_NUM_THREADS"] = "1"
import warnings

warnings.filterwarnings("ignore")

import argparse
import torch
import numpy as np
import trimesh
from tqdm import tqdm
from accelerate import PartialState
from pathlib import Path
from transformers import set_seed, AutoImageProcessor
from src.utils.inference import (
    prepare_model_for_inference,
    prepare_mv_model_for_inference,
    prepare_mv_test_set,
    get_prefix_allowed_tokens_fn_edgerunner,
    decode_mesh_edgerunner,
    decode_bpt,
    prepare_test_set,
    joint_filter,
)
from src.data.collator import get_mesh_data_collator
from src.data import utils as data_utils, tokenize_bpt


EDGERUNNER_DEFAULT_MAX_FACES = 4096
EDGERUNNER_DEFAULT_MIN_FACES = 8


def _max_tokens_for_edgerunner_faces(max_faces: int) -> int:
    # Worst case: every face starts a new patch, BOM + 9 coordinate tokens.
    return max(1, int(max_faces) * 10)


def _min_tokens_for_edgerunner_faces(min_faces: int) -> int:
    if min_faces <= 0:
        return 0
    # Best case: one BOM face, then linked faces as L/R + 3 coordinate tokens.
    return 10 + max(0, int(min_faces) - 1) * 4


def _edgerunner_generation_kwargs(args, prompt_len, collator, model, batch_size):
    position_budget = collator.max_seq_length - int(prompt_len)
    if position_budget <= 0:
        raise RuntimeError(
            f"Prompt length {prompt_len} leaves no room under max_seq_length="
            f"{collator.max_seq_length}"
        )

    face_cap_tokens = _max_tokens_for_edgerunner_faces(args.max_faces)
    requested_cap = (
        int(args.max_new_tokens)
        if args.max_new_tokens is not None
        else face_cap_tokens
    )
    max_new_tokens = max(1, min(position_budget, requested_cap, face_cap_tokens))
    min_new_tokens = min(
        max_new_tokens,
        _min_tokens_for_edgerunner_faces(args.min_faces),
    )

    kwargs = {
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "do_sample": False,
        "prefix_allowed_tokens_fn": get_prefix_allowed_tokens_fn_edgerunner(
            model, batch_size=batch_size
        ),
    }
    if min_new_tokens > 0:
        kwargs["min_new_tokens"] = min_new_tokens
    if args.num_beams > 1:
        kwargs.update(
            num_beams=args.num_beams,
            early_stopping=True,
            length_penalty=args.length_penalty,
        )
    if args.do_sample:
        kwargs.update(do_sample=True, top_k=10)
    return kwargs


def _strip_padding_and_eos(tokens, pad_token_id, eos_token_id):
    tokens = tokens[tokens != pad_token_id]
    eos_idx = (tokens == eos_token_id).nonzero()[0]
    if len(eos_idx) > 0:
        tokens = tokens[: eos_idx[0]]
    return tokens


def _write_decode_failure_placeholder(out_path):
    # Write a ZERO-FACE mesh so a decode failure is counted as a coverage MISS, not a
    # scored prediction. A 1-triangle placeholder would have len(triangles)==1, slipping
    # past eval_obj's degenerate guard (len(triangles)==0) -> it would inflate coverage to
    # ~100% (hiding decode failures) AND inject a bad CD into the mean.
    trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [1e-3, 0, 0], [0, 1e-3, 0]], dtype=np.float32),
        faces=np.zeros((0, 3), dtype=np.int64),
    ).export(out_path)


def _export_edgerunner_mesh(tokens, collator, model, out_path, uid):
    tokens = _strip_padding_and_eos(
        tokens, collator.pad_token_id, model.config.eos_token_id
    )
    try:
        mesh = decode_mesh_edgerunner(
            tokens, collator.tokenizer, clean=True, verbose=False
        )
        mesh.export(out_path)
    except Exception as e:
        print(f"[WARN] decode failed for {uid} ({len(tokens)} tokens): {e}")
        # Keep eval coverage honest: a decode failure should score badly, not disappear.
        _write_decode_failure_placeholder(out_path)


def run_multiview_inference(args):
    """Multi-view EdgeRunner inference: build the conditioning prefix via the model's
    get_mv_inputs_with_cond (Pi3X -> instance discovery -> MV voxel encoder), then
    autoregressively decode the mesh. Writes <uid>.ply to the output dir."""
    state = PartialState()
    device = state.device
    set_seed(args.seed)

    # Plain overrides (no '+'): these keys exist in canonical_3d_front_multiview.yaml,
    # so '+' (append) would raise ConfigCompositionException.
    extra_overrides = []
    if getattr(args, "obj_pc_cond", False):
        extra_overrides.append("dataset.model.mv_obj_pc_cond=true")
    if getattr(args, "no_voxel_encoder", False):
        extra_overrides.append("dataset.model.mv_use_voxel_encoder=false")
    if getattr(args, "obj_pc_appearance", False):
        extra_overrides.append("dataset.model.mv_obj_pc_appearance=true")
    model, model_cfg, data_cfg = prepare_mv_model_for_inference(
        checkpoint=args.checkpoint, config_name=args.mv_config,
        extra_overrides=extra_overrides or None,
    )
    # The production stage-2 MV checkpoint is trained WITH the obj-PC geometry channel
    # (mv_obj_pc_cond=true, prefix_len=2370). The default --mv-config disables it
    # (prefix_len=322); since the cond_token_mask assert still passes either way, a
    # mismatched eval silently drops the geometry the decoder relies on and produces
    # meaningless CD/F. Warn loudly (not raise — ablations deliberately vary channels).
    if state.is_main_process and not getattr(model_cfg, "mv_obj_pc_cond", False):
        warnings.warn(
            f"MV eval built WITHOUT the obj-PC channel (mv_obj_pc_cond=False, "
            f"prefix_len={model_cfg.prefix_len}). Stage-2 checkpoints are trained WITH "
            f"it (prefix_len=2370); evaluating without it yields meaningless CD/F. Pass "
            f"--obj-pc-cond (or --mv-config edgerunner_3d_front_multiview_stage2) unless "
            f"this is a deliberate ablation.",
            stacklevel=2,
        )
    model.to(device)
    model.eval()

    out_dir = Path(args.output_dir) / "obj" / "edgerunner" / "mv"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_set = prepare_mv_test_set(data_cfg)
    n_total = len(test_set)
    if args.limit is not None:
        n_total = min(n_total, args.limit)
    indices = list(range(n_total))
    shard = indices[state.process_index :: state.num_processes]

    collator = get_mesh_data_collator(data_cfg, model_cfg)
    prefix_len = collator.prefix_len
    pc_token_id = collator.pc_token_id
    bos_token_id = collator.bos_token_id
    indicator_token_id = collator.indicator_token_id

    bs = args.batch_size
    n_iters = (len(shard) + bs - 1) // bs

    # View-count parity guard (Finding 2): the obj-PC geometry stream self-normalizes
    # to the OBSERVED extent, so evaluating with fewer views than training shrinks that
    # extent and inflates the conditioning scale (off-distribution). Warn loudly once
    # rather than silently regress. Default (args.num_views=None) keeps all training views.
    train_num_views = getattr(data_cfg, "num_views", None)
    if args.num_views is not None and train_num_views and args.num_views < train_num_views:
        warnings.warn(
            f"--num-views={args.num_views} < training num_views={train_num_views}: the "
            "obj-PC self-normalization was trained on the full-view observed extent; fewer "
            "views inflate the conditioning scale and push it off-distribution. Use all "
            "training views for a faithful eval.",
            stacklevel=2,
        )

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for it in tqdm(range(n_iters), position=state.process_index, leave=False):
                batch_idx = shard[it * bs : (it + 1) * bs]
                examples = []
                uids = []
                for i in batch_idx:
                    ex = test_set[i]
                    uid = ex["uid"] if not isinstance(ex["uid"], list) else ex["uid"][0]
                    if (out_dir / f"{uid}.ply").exists():
                        continue
                    examples.append(ex)
                    uids.append(uid)
                if not examples:
                    continue

                batch = collator(examples)

                # Test-3 N-views ablation: restrict to the first K views.
                if args.num_views is not None and "view_mask" in batch:
                    vm = batch["view_mask"].clone()
                    vm[:, args.num_views:] = False
                    batch["view_mask"] = vm

                # Build prefix-only input_ids: [pc]*prefix_len + [bos] + layout + [indicator]
                prefix_ids = []
                for ex in examples:
                    bboxes = np.array(ex["bboxes"], dtype=np.float32)
                    obj_index = int(ex["obj_indices"])
                    layout_seq = collator._tokenize_bbox(bboxes[[obj_index]])
                    seq = (
                        [pc_token_id] * prefix_len
                        + [bos_token_id]
                        + layout_seq
                        + [indicator_token_id]
                    )
                    prefix_ids.append(seq)
                input_ids = torch.as_tensor(prefix_ids, dtype=torch.long, device=device)

                def _to(x):
                    return x.to(device) if torch.is_tensor(x) else x

                # Outer guard: catastrophic batch-level failures (conditioning or
                # generation crash the whole rank). On failure every object in the
                # batch is an honest coverage MISS — we genuinely have no tokens.
                try:
                    inputs_embeds = model.get_mv_inputs_with_cond(
                        input_ids=input_ids,
                        pixel_values=_to(batch["pixel_values"]),
                        scene_transforms=_to(batch["scene_transforms"]),
                        K_per_view=_to(batch["K_per_view"]),
                        view_mask=_to(batch["view_mask"]),
                        panoptic_masks=_to(batch.get("panoptic_masks")),
                        cond_pcs=_to(batch["cond_pcs"]),
                        cond_pcs_2d=_to(batch["cond_pcs_2d"]),
                        cond_num_faces=None,
                        obj_canon_transform=_to(batch.get("obj_canon_transform")),
                        gt_obj_vertices=_to(batch.get("gt_obj_vertices")),
                        ref_view=_to(batch.get("ref_view")),
                    )

                    results = model.generate(
                        inputs_embeds=inputs_embeds,
                        **_edgerunner_generation_kwargs(
                            args,
                            prompt_len=inputs_embeds.shape[1],
                            collator=collator,
                            model=model,
                            batch_size=len(examples),
                        ),
                    )
                    results = results.cpu().numpy()
                    # Per-object guard: a degenerate token sequence for one object
                    # must not forfeit the rest of the batch.
                    for uid, tokens in zip(uids, results):
                        try:
                            _export_edgerunner_mesh(
                                tokens, collator, model, out_dir / f"{uid}.ply", uid
                            )
                        except Exception as e_obj:
                            warnings.warn(
                                f"MV mesh export failed for {uid} "
                                f"({type(e_obj).__name__}: {e_obj}); "
                                f"writing decode-failure placeholder."
                            )
                            _write_decode_failure_placeholder(out_dir / f"{uid}.ply")
                except Exception as e:
                    warnings.warn(
                        f"MV conditioning/generation failed for batch {uids} "
                        f"({type(e).__name__}: {e}); writing decode-failure placeholders."
                    )
                    for uid in uids:
                        _write_decode_failure_placeholder(out_dir / f"{uid}.ply")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (per GPU) for inference",
    )
    parser.add_argument(
        "--run-type",
        type=str,
        choices=["obj", "scene"],
        required=True,
        help="Whether to run inference on object level or scene level",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["edgerunner", "bpt"],
        required=True,
        help="Type of the model to use for inference",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to Huggingface checkpoint",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Path to output directory",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--gt-layout",
        action="store_true",
        help="Whether to use ground-truth layout for inference",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Whether to use sampling during inference",
    )
    parser.add_argument(
        "--gt-depth",
        action="store_true",
        help="Whether to use ground-truth depth maps",
    )
    parser.add_argument(
        "--gt-mask",
        action="store_true",
        help="Whether to use ground-truth masks",
    )
    parser.add_argument(
        "--image-encoder",
        type=str,
        default=None,
        help="DINOv2 image encoder (e.g. facebook/dinov2-with-registers-small). "
             "Defaults to base; use small for paper/stage-1-initialized checkpoints.",
    )
    parser.add_argument(
        "--image-preprocessor",
        type=str,
        default=None,
        help="Image preprocessor matching the encoder "
             "(e.g. facebook/dpt-dinov2-small-nyu for small).",
    )
    parser.add_argument(
        "--mv",
        action="store_true",
        help="Multi-view EdgeRunner inference (datasets/3d-front-multiview).",
    )
    parser.add_argument(
        "--mv-config",
        type=str,
        default="edgerunner_3d_front_multiview",
        help="Hydra config name for the multi-view model/data.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of objects (smoke tests).",
    )
    parser.add_argument(
        "--gt-cond",
        action="store_true",
        help="Test-1 headroom: condition on a dense GT-mesh object PC (placed into "
             "the conditioning frame via the object's transform) instead of the depth "
             "point cloud. Isolates whether the frozen decoder can exploit complete "
             "geometry.",
    )
    parser.add_argument(
        "--drop-image",
        action="store_true",
        help="Geometry-only conditioning: pass pixel_values=None (no image features).",
    )
    parser.add_argument(
        "--num-dense",
        type=int,
        default=8192,
        help="Points sampled from the GT mesh for --gt-cond.",
    )
    parser.add_argument(
        "--gt-cond-partial",
        action="store_true",
        help="De-confounded Test-1: keep only the GT-mesh points within --partial-eps "
             "of the depth-visible surface (the reference-view front), so partial-vs-"
             "complete share the GT distribution and differ ONLY in coverage.",
    )
    parser.add_argument(
        "--partial-eps",
        type=float,
        default=0.05,
        help="Distance (normalized cond frame) for --gt-cond-partial visibility.",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=None,
        help="Test-3 N-views ablation (MV path): restrict each object to the first K "
             "views (view_mask[:, K:]=False) so discovery+conditioning use K views.",
    )
    parser.add_argument(
        "--obj-pc-cond",
        action="store_true",
        help="MV path: route the multi-view-discovered canonical points through the SV "
             "obj-PC channel (cond_encoder). Sets mv_obj_pc_cond=true; prefix_len auto-adjusts.",
    )
    parser.add_argument(
        "--no-voxel-encoder",
        action="store_true",
        help="MV path: drop the mv_voxel_encoder z_i/z_scene (Stage-0 obj-PC-only test). "
             "Sets mv_use_voxel_encoder=false.",
    )
    parser.add_argument(
        "--obj-pc-appearance",
        action="store_true",
        help="MV path: feed confidence-weighted multi-view DINO at the obj voxels as the "
             "cond_encoder extra_feat (restores SV's always-on appearance term). "
             "Sets mv_obj_pc_appearance=true; prefix_len unchanged.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=EDGERUNNER_DEFAULT_MAX_FACES,
        help="EdgeRunner generation cap in training face-count units. The token cap is "
             "10 * max_faces, matching the worst-case BOM+coords encoding.",
    )
    parser.add_argument(
        "--min-faces",
        type=int,
        default=EDGERUNNER_DEFAULT_MIN_FACES,
        help="EdgeRunner minimum decoded mesh size before EOS is allowed, expressed as "
             "linked-face token length. Helps beam search avoid tiny early-EOS meshes.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional stricter EdgeRunner token cap; still clipped by --max-faces and "
             "the model's max_seq_length.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="EdgeRunner beam count. Set to 1 to recover greedy decoding.",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=1.0,
        help="Length penalty passed to EdgeRunner beam search.",
    )
    args = parser.parse_args()

    if args.mv:
        run_multiview_inference(args)
        return

    is_bpt = args.model_type == "bpt"

    is_obj_level = args.run_type == "obj"
    metadata_name = "test_obj_sub_100" if is_obj_level else "test_scene"
    metadata_file = f"metadata/{metadata_name}.jsonl"

    if is_obj_level:
        use_gt_layout = True
        use_gt_mask = True
    else:
        use_gt_layout = args.gt_layout
        use_gt_mask = args.gt_mask

    use_gt_depth = args.gt_depth

    if not use_gt_mask:
        # We don't have GT layout if using predicted masks
        use_gt_layout = False

    use_gt_layout_str = "gt_layout" if use_gt_layout else "pred_layout"
    use_gt_mask_str = "gt_mask" if use_gt_mask else "pred_mask"
    use_gt_depth_str = "gt_depth" if use_gt_depth else "pred_depth"
    name = f"{use_gt_layout_str}_{use_gt_mask_str}_{use_gt_depth_str}"
    out_dir = Path(args.output_dir) / args.run_type / args.model_type / name
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    state = PartialState()

    device = state.device
    model, model_cfg, data_cfg = prepare_model_for_inference(
        is_bpt, args.checkpoint,
        image_encoder=args.image_encoder,
        image_preprocessor=args.image_preprocessor,
    )
    model.to(device)

    data_cfg.use_predicted_depth = not use_gt_depth
    if not use_gt_depth:
        data_cfg.depth_path = "datasets/depth_pro_aligned_npy"
        data_cfg.predicted_depth_aligned = True

    image_preprocessor = AutoImageProcessor.from_pretrained(
        data_cfg.image_preprocessor, size_divisor=data_cfg.image_size_divisor
    )

    with state.local_main_process_first():
        test_set = prepare_test_set(
            data_cfg, metadata_file, use_predicted_mask=not use_gt_mask
        )

    sharded_data = test_set.shard(state.num_processes, state.process_index)
    if args.limit is not None:
        sharded_data = sharded_data.select(range(min(args.limit, len(sharded_data))))

    collator = get_mesh_data_collator(data_cfg, model_cfg)
    cond_prefix = [collator.pc_token_id] * collator.prefix_len
    bos_prefix = [collator.bos_token_id] if not is_bpt else []

    n_iters = (len(sharded_data) + args.batch_size - 1) // args.batch_size

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for item in tqdm(
                sharded_data.iter(batch_size=args.batch_size),
                total=n_iters,
                position=state.process_index,
                dynamic_ncols=True,
                leave=False,
            ):
                all_pcds = np.array(item["point_clouds"], dtype=np.float32)
                all_pcds_2d = np.array(item["point_clouds_2d"], dtype=np.float32)
                sampled_pcds = []
                sampled_pcds_2d = []
                sampled_ctx_pcds = []
                sampled_ctx_pcds_2d = []
                all_input_ids = []
                all_uids = []
                images = []

                gt_pose_seqs = []

                for uid, pcds, pcds_2d, bboxes, mask, transform, image, model_id in zip(
                    item["uid"],
                    all_pcds,
                    all_pcds_2d,
                    item["bboxes"],
                    item["mask"],
                    item["transform"],
                    item["image"],
                    item["model_id"],
                ):
                    if (out_dir / f"{uid}.ply").exists():
                        continue

                    mask = np.array(mask)

                    all_uids.append(uid)
                    obj_index = int(uid.split("_")[-1])
                    bboxes = np.array(bboxes, dtype=np.float32)
                    obj_pcd_in_global = pcds[mask]
                    obj_pcd_2d = pcds_2d[mask]

                    if args.gt_cond:
                        # Test-1 headroom: replace the depth object PC with a dense
                        # GT-mesh sample placed into the conditioning frame via the
                        # object's transform (canonical -> global/cond frame).
                        import open3d as _o3d

                        _depth_pts = obj_pcd_in_global  # the reference-view visible surface
                        _gt = _o3d.io.read_triangle_mesh(
                            f"datasets/3D-FUTURE-model-ply/{model_id}.ply"
                        )
                        _gp = np.asarray(
                            _gt.sample_points_uniformly(args.num_dense).points,
                            dtype=np.float32,
                        )
                        _gph = np.concatenate(
                            [_gp, np.ones((len(_gp), 1), dtype=np.float32)], axis=1
                        )
                        _gt_global = (_gph @ np.asarray(transform, np.float32).T)[:, :3]
                        _full_n = len(_gt_global)
                        if args.gt_cond_partial and len(_depth_pts) > 0:
                            # De-confound: keep only GT points near the depth-visible
                            # surface → partial vs complete share the GT distribution and
                            # differ only in COVERAGE (front-only vs all sides).
                            from scipy.spatial import cKDTree

                            _d, _ = cKDTree(_depth_pts).query(_gt_global)
                            _keep = _d < args.partial_eps
                            if _keep.sum() >= 64:
                                _gt_global = _gt_global[_keep]
                        if state.is_main_process and len(sampled_pcds) == 0:
                            print(
                                f"[gt-cond frame check {uid}] "
                                f"depth bbox {_depth_pts.min(0)}..{_depth_pts.max(0)} | "
                                f"gt bbox {_gt_global.min(0)}..{_gt_global.max(0)} | "
                                f"kept {len(_gt_global)}/{_full_n} "
                                f"({'partial' if args.gt_cond_partial else 'complete'})",
                                flush=True,
                            )
                        obj_pcd_in_global = _gt_global.astype(np.float32)
                        obj_pcd_2d = np.zeros((len(_gt_global), 2), dtype=np.float32)
                    sampled_pcd, sample_inds = data_utils.random_sample_point_clouds(
                        obj_pcd_in_global,
                        data_cfg.num_points,
                        return_inds=True,
                    )
                    if data_cfg.with_normals:
                        sampled_pcd = data_utils.estimate_point_cloud_normals(
                            sampled_pcd
                        )
                    sampled_ctx_pcd, sampled_ctx_inds = (
                        data_utils.random_sample_point_clouds(
                            pcds.reshape(-1, 3),
                            data_cfg.num_ctx_points,
                            return_inds=True,
                        )
                    )
                    if data_cfg.with_normals:
                        sampled_ctx_pcd = data_utils.estimate_point_cloud_normals(
                            sampled_ctx_pcd
                        )
                    sampled_ctx_pcd_2d = pcds_2d.reshape(-1, 2)[sampled_ctx_inds]
                    sampled_ctx_pcds.append(sampled_ctx_pcd)
                    sampled_ctx_pcds_2d.append(sampled_ctx_pcd_2d)
                    sampled_pcds.append(sampled_pcd)
                    sampled_pcds_2d.append(obj_pcd_2d[sample_inds])
                    gt_pose_seq = collator._tokenize_bbox(bboxes[[obj_index]])
                    gt_pose_seqs.append(gt_pose_seq)
                    if use_gt_layout:
                        layout_seq = gt_pose_seq + [collator.indicator_token_id]
                    else:
                        layout_seq = []

                    prefix_seq = cond_prefix + bos_prefix + layout_seq
                    all_input_ids.append(prefix_seq)

                    images.append(np.array(image, dtype=np.uint8))

                if len(all_input_ids) == 0:
                    continue

                sampled_pcds = np.stack(sampled_pcds, axis=0)
                sampled_pcds_2d = np.stack(sampled_pcds_2d, axis=0)
                all_input_ids = np.stack(all_input_ids, axis=0)
                all_input_ids = torch.as_tensor(all_input_ids, dtype=torch.long).to(
                    device
                )
                cond_pcs = torch.as_tensor(sampled_pcds, dtype=torch.float32).to(device)
                cond_pcs_2d = torch.as_tensor(sampled_pcds_2d, dtype=torch.float32).to(
                    device
                )
                ctx_pcs = torch.as_tensor(
                    np.stack(sampled_ctx_pcds, axis=0), dtype=torch.float32
                ).to(device)
                ctx_pcs_2d = torch.as_tensor(
                    np.stack(sampled_ctx_pcds_2d, axis=0), dtype=torch.float32
                ).to(device)

                gt_pose_seqs = np.array(gt_pose_seqs)

                if args.drop_image:
                    cond_images = None
                else:
                    cond_images = image_preprocessor(images, return_tensors="pt")[
                        "pixel_values"
                    ].to(device)
                bs = all_input_ids.shape[0]

                if not use_gt_layout:
                    # Decode layout first
                    inputs_embeds = model.get_inputs_with_cond(
                        all_input_ids,
                        cond_pcs=cond_pcs,
                        cond_pcs_2d=cond_pcs_2d,
                        ctx_pcs=ctx_pcs,
                        ctx_pcs_2d=ctx_pcs_2d,
                        pixel_values=cond_images,
                    )
                    if is_bpt:
                        results = model.generate(
                            cond_embeds=inputs_embeds,
                            batch_size=bs,
                            max_new_tokens=17,
                            do_sample=False,
                        )
                        results_cpu = results.cpu().numpy()
                        poses = tokenize_bpt.detokenize_layout(
                            results_cpu[:, :-1]
                        ).reshape(-1, 8, 3)
                    else:
                        results = model.generate(
                            inputs_embeds=inputs_embeds,
                            max_new_tokens=25,
                            use_cache=True,
                            do_sample=False,
                        )
                        results_cpu = results.cpu().numpy()
                        poses = (
                            results_cpu[:, :-1] - model_cfg.pos_token_offset
                        ).reshape(-1, 8, 3)
                    poses = data_utils.dequantize_points(
                        poses, model_cfg.num_pos_tokens
                    )
                    all_input_ids = torch.cat([all_input_ids, results], dim=1)
                    for uid, tokens, pose in zip(all_uids, results_cpu, poses):
                        np.savez_compressed(
                            out_dir / f"{uid}_pred_layout.npz", tokens=tokens
                        )
                        np.savez_compressed(out_dir / f"{uid}_pose.npz", pose=pose)

                inputs_embeds = model.get_inputs_with_cond(
                    all_input_ids,
                    cond_pcs=cond_pcs,
                    cond_pcs_2d=cond_pcs_2d,
                    ctx_pcs=ctx_pcs,
                    ctx_pcs_2d=ctx_pcs_2d,
                    pixel_values=cond_images,
                )

                seq_len = all_input_ids.shape[1]
                if is_bpt:
                    max_new_tokens = min(collator.max_seq_length - seq_len, 40960)
                    results = model.generate(
                        inputs=all_input_ids,
                        cond_embeds=inputs_embeds,
                        max_new_tokens=max_new_tokens,
                        temperature=0.5,
                        filter_logits_fn=joint_filter,
                        filter_kwargs=dict(k=50, p=0.95),
                        do_sample=args.do_sample,
                        tqdm_position=state.process_index,
                    )
                    results = results[:, seq_len:]
                else:
                    results = model.generate(
                        inputs_embeds=inputs_embeds,
                        **_edgerunner_generation_kwargs(
                            args,
                            prompt_len=seq_len,
                            collator=collator,
                            model=model,
                            batch_size=bs,
                        ),
                    )
                results = results.cpu().numpy()
                for uid, tokens in zip(all_uids, results):
                    if is_bpt:
                        tokens = _strip_padding_and_eos(
                            tokens, collator.pad_token_id, model.config.eos_token_id
                        )
                        mesh = decode_bpt(tokens)
                        mesh.export(out_dir / f"{uid}.ply")
                    else:
                        _export_edgerunner_mesh(
                            tokens, collator, model, out_dir / f"{uid}.ply", uid
                        )


if __name__ == "__main__":
    main()
