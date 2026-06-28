from functools import partial
import json
import jsonlines
import datasets
import trimesh
import torch
import numpy as np
from x_transformers.autoregressive_wrapper import top_p, top_k
from src.data import tokenize_bpt
from src.data.mesh import get_mesh_dataset, transform_3d_front
from src.models.utils import (
    get_image_condition_encoder,
    get_condition_encoder,
    get_model,
)
from src.utils.config import ModelConfig, DataConfig, mv_prefix_len


def _flatten_3d_front_for_inference(examples, data_cfg, use_predicted_mask):
    examples = transform_3d_front(examples, is_train=False, data_cfg=data_cfg)
    uids = examples["uid"]
    images = examples["images"]
    depths = examples["depths"]
    all_bboxes = examples["bboxes"]
    all_obj_bounds = examples["all_obj_bounds"]
    all_obj_to_cam_transforms = examples["all_obj_to_cam_transforms"]
    all_obj_model_ids = examples["all_obj_model_ids"]
    all_obj_masks = examples["all_obj_masks"]
    all_point_clouds = examples["all_point_clouds"]
    all_point_clouds_2d = examples["all_point_clouds_2d"]

    result_uids = []
    result_images = []
    result_depths = []
    result_bboxes = []
    result_pcds = []
    result_pcds_2d = []
    result_bounds = []
    result_transforms = []
    result_model_ids = []
    result_masks = []

    if use_predicted_mask:
        placeholder_bound = np.zeros((2, 3), dtype=np.float32)
        placeholder_transform = np.eye(4, dtype=np.float32)
        model_id = ""
        for uid, image, depth, pcds, pcds_2d in zip(
            uids, images, depths, all_point_clouds, all_point_clouds_2d
        ):
            mask_info = f"datasets/grounded_sam/{uid}/mask_annotations.json"
            with open(mask_info) as fp:
                mask_data = json.load(fp)
            mask_path = f"datasets/grounded_sam/{uid}/masks.npz"
            masks = np.load(mask_path)["masks"]
            n = masks.shape[0]
            for ind, info in enumerate(mask_data):
                if info["class_name"] == "lamp":
                    continue
                result_uids.append(f"{uid}_{ind}")
                result_images.append(image)
                result_depths.append(depth)
                result_bboxes.append(np.zeros((n, 8, 3), dtype=np.float32))
                result_pcds.append(pcds)
                result_pcds_2d.append(pcds_2d)
                result_bounds.append(placeholder_bound)
                result_transforms.append(placeholder_transform)
                result_model_ids.append(model_id)
                result_masks.append(masks[ind])
    else:
        for (
            uid,
            image,
            depth,
            bboxes,
            bounds,
            transforms,
            model_ids,
            masks,
            pcds,
            pcds_2d,
        ) in zip(
            uids,
            images,
            depths,
            all_bboxes,
            all_obj_bounds,
            all_obj_to_cam_transforms,
            all_obj_model_ids,
            all_obj_masks,
            all_point_clouds,
            all_point_clouds_2d,
        ):
            for ind, (bound, transform, model_id, mask) in enumerate(
                zip(bounds, transforms, model_ids, masks)
            ):
                result_uids.append(f"{uid}_{ind}")
                result_images.append(image)
                result_depths.append(depth)
                result_bboxes.append(bboxes)
                result_pcds.append(pcds)
                result_pcds_2d.append(pcds_2d)
                result_bounds.append(bound)
                result_transforms.append(transform)
                result_model_ids.append(model_id)
                result_masks.append(mask)

    return {
        "uid": result_uids,
        "image": result_images,
        "depth": result_depths,
        "bboxes": result_bboxes,
        "point_clouds": result_pcds,
        "point_clouds_2d": result_pcds_2d,
        "bound": result_bounds,
        "transform": result_transforms,
        "model_id": result_model_ids,
        "mask": result_masks,
    }


def prepare_test_set(data_cfg, metadata, use_predicted_mask):
    test_set = get_mesh_dataset(data_cfg)[2]

    with jsonlines.open(metadata, "r") as reader:
        valid_ids = [obj["image_id"] for obj in reader]

    valid_indices = []
    all_uids = test_set["uid"]
    for i, uid in enumerate(all_uids):
        if str(uid) in valid_ids:
            valid_indices.append(i)

    subset_test_set = test_set.select(valid_indices)
    image_size = (484, 648)
    subset_test_set = subset_test_set.map(
        partial(
            _flatten_3d_front_for_inference,
            data_cfg=data_cfg,
            use_predicted_mask=use_predicted_mask,
        ),
        batched=True,
        batch_size=4,
        num_proc=4,
        remove_columns=subset_test_set.column_names,
        features=datasets.Features(
            {
                "uid": datasets.Value("string"),
                "image": datasets.Array3D(dtype="uint8", shape=(*image_size, 3)),
                "depth": datasets.Array2D(dtype="float32", shape=image_size),
                "bboxes": datasets.Sequence(
                    datasets.Array2D(dtype="float32", shape=(8, 3))
                ),
                "point_clouds": datasets.Array3D(
                    dtype="float32", shape=(*image_size, 3)
                ),
                "point_clouds_2d": datasets.Array3D(
                    dtype="float32", shape=(*image_size, 2)
                ),
                "bound": datasets.Array2D(dtype="float32", shape=(2, 3)),
                "transform": datasets.Array2D(dtype="float32", shape=(4, 4)),
                "model_id": datasets.Value("string"),
                "mask": datasets.Array2D(dtype="bool", shape=image_size),
            }
        ),
    )
    return subset_test_set


def joint_filter(logits, k=50, p=0.95):
    logits = top_k(logits, k=k)
    logits = top_p(logits, thres=p)
    return logits


def get_prefix_allowed_tokens_fn_edgerunner(model, batch_size=1):
    """Return a stateless EdgeRunner grammar mask usable by greedy or beam search.

    Token grammar:
    - 5 (BOM) starts a face patch and must be followed by 9 coordinate tokens.
    - 3/4 (L/R) extend the patch and must be followed by 3 coordinate tokens.
    - After a complete patch/extension, the next token can be L/R/BOM/EOS.

    The old implementation kept one mutable counter per batch item. Beam search calls this
    function independently for each live beam, so shared mutable state corrupts divergent
    beams. Recomputing the state from `input_ids` keeps the grammar beam-safe.
    """
    del batch_size  # Kept for API compatibility with existing call sites.
    structure_tokens = {3, 4, 5}
    coord_start = 6

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        del batch_id
        tokens = input_ids.tolist()
        if len(tokens) == 0:
            return [5]
        if tokens[-1] == model.config.eos_token_id:
            return [model.config.eos_token_id]

        last_struct_idx = None
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] in structure_tokens:
                last_struct_idx = i
                break

        if last_struct_idx is None:
            return [5]

        last_struct = tokens[last_struct_idx]
        required_coords = 9 if last_struct == 5 else 3
        consumed_coords = len(tokens) - last_struct_idx - 1
        if consumed_coords < required_coords:
            # Coordinate tokens occupy exactly [coord_start, coord_start+num_pos_tokens).
            # The old upper bound (vocab_size) also admitted the structure/sep ids and
            # unused vocab as "coordinates", letting the model emit out-of-bin vertices.
            coord_end = coord_start + model.config.num_pos_tokens
            return list(range(coord_start, coord_end))
        return [3, 4, 5, model.config.eos_token_id]

    return prefix_allowed_tokens_fn


def _remove_small_components(mesh, area_ratio=0.01, verbose=False):
    """Drop disconnected components whose surface area is below area_ratio × total.

    Removes the spurious floaters the autoregressive decoder occasionally emits. A
    single stray component would otherwise inflate the mesh bounding box and distort
    the downstream bbox-normalization (and CD/F) far beyond its own few points.

    Conservative by design: keeps *every* substantial component (not just the largest),
    so genuinely disconnected object parts (e.g. chair legs) survive. area_ratio <= 0
    disables it; if all components fall below the threshold (degenerate), the single
    largest component is kept.
    """
    if area_ratio <= 0.0 or len(mesh.faces) == 0:
        return mesh
    try:
        components = mesh.split(only_watertight=False)
    except Exception:
        return mesh
    if len(components) <= 1:
        return mesh
    areas = np.array([float(c.area) for c in components])
    total = float(areas.sum())
    if total <= 0.0:
        return mesh
    keep = [c for c, a in zip(components, areas) if a >= area_ratio * total]
    if not keep:
        keep = [components[int(areas.argmax())]]
    cleaned = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]
    if verbose:
        print(
            f"[INFO] components: {len(components)} → kept {len(keep)} "
            f"(removed {len(components) - len(keep)} floaters)"
        )
    return cleaned


def decode_mesh_edgerunner(tokens, tokenizer, clean=True, verbose=False, floater_area_ratio=0.0):
    tokens = tokens - 3
    vertices, faces, face_type = tokenizer.decode(tokens)

    if verbose:
        print(f"[INFO] vertices: {vertices.shape[0]}, faces: {faces.shape[0]}")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # fix flipped faces and merge close vertices
    if clean:
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.fix_normals()

        # Drop tiny disconnected floaters (opt-in; preserves multi-part objects).
        mesh = _remove_small_components(mesh, area_ratio=floater_area_ratio, verbose=verbose)

        if verbose:
            print(
                f"[INFO] cleaned vertices: {mesh.vertices.shape[0]}, faces: {mesh.faces.shape[0]}"
            )
    return mesh


def decode_bpt(tokens):
    vertices = tokenize_bpt.BPT_deserialize(tokens)
    num_vertices = len(vertices) // 3 * 3
    faces = np.arange(1, num_vertices + 1).reshape(-1, 3)
    mesh = tokenize_bpt.to_mesh(vertices, faces, post_process=True)
    return mesh


def recover_box_transform(P_local, Q_world):
    """
    Recover 7-DoF transform (yaw, per-axis scale, translation)
    given corresponding ordered corners.

    Args:
        P_local : (8,3) np.array — canonical box corners in object local frame
        Q_world : (8,3) np.array — observed corners in world frame (same order)

    Returns:
        yaw   : float (radians)  — rotation about Y
        scale : (3,) np.array    — [sx, sy, sz], positive
        trans : (3,) np.array    — translation vector
        T     : (4,4) np.array   — homogeneous transform (world <- local)
    """
    P = np.asarray(P_local, dtype=float)
    Q = np.asarray(Q_world, dtype=float)

    # --- remove translation ---
    muP = P.mean(0)
    muQ = Q.mean(0)
    P0, Q0 = P - muP, Q - muQ

    # --- solve in XZ plane for yaw + sx, sz ---
    P_xz, Q_xz = P0[:, [0, 2]], Q0[:, [0, 2]]
    X, *_ = np.linalg.lstsq(P_xz, Q_xz, rcond=None)
    A = X.T  # 2×2 affine part

    # Decompose A ≈ R(θ)·diag(sx, sz)
    a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    sx = np.hypot(a, c)
    sz = np.hypot(b, d)
    yaw1 = np.arctan2(-c, a)
    yaw2 = np.arctan2(b, d)
    # robust average of angles
    yaw = np.arctan2(np.sin(yaw1) + np.sin(yaw2), np.cos(yaw1) + np.cos(yaw2))

    # --- y-scale (since yaw leaves Y axis unchanged) ---
    Py, Qy = P0[:, 1], Q0[:, 1]
    sy = float(np.dot(Py, Qy) / np.dot(Py, Py))
    sy = abs(sy)

    scale = np.array([abs(sx), sy, abs(sz)])

    # --- translation ---
    R = np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
    )
    trans = muQ - (R @ (scale * muP))

    T = np.eye(4)
    T[:3, :3] = R @ np.diag(scale)
    T[:3, 3] = trans

    return yaw, scale, trans, T


def run_perspectivefields(image):
    from perspective2d import PerspectiveFields
    from perspective2d.utils.utils import general_vfov_to_focal

    img_bgr = image[..., ::-1]
    H, W, _ = img_bgr.shape
    pf_model = PerspectiveFields("Paramnet-360Cities-edina-uncentered").eval().cuda()
    pred = pf_model.inference(img_bgr=img_bgr)

    roll = np.radians(pred["pred_roll"].cpu().item())
    pitch = np.radians(pred["pred_pitch"].cpu().item())
    vfov = np.radians(pred["pred_general_vfov"].cpu().item())
    cx_rel = pred["pred_rel_cx"].cpu().item()
    cy_rel = pred["pred_rel_cy"].cpu().item()
    focal_rel = general_vfov_to_focal(cx_rel, cy_rel, 1, vfov, degree=False)
    f = focal_rel * H
    cx = (cx_rel + 0.5) * W
    cy = (cy_rel + 0.5) * H
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    return K, roll, pitch


def initialize_depth_pro(ckpt_path="checkpoint/depth_pro.pt"):
    import depth_pro

    config = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
    config.checkpoint_uri = ckpt_path
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()
    model.cuda()

    def get_depth_from_depth_pro(
        image: np.ndarray, f_px: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = transform(image).cuda()
        prediction = model.infer(image, f_px=f_px)
        depth = prediction["depth"]
        f_px = prediction["focallength_px"]
        return depth.detach().cpu(), f_px.detach().cpu()

    return get_depth_from_depth_pro


def prepare_model_for_inference(
    is_bpt,
    checkpoint,
    image_encoder=None,
    image_preprocessor=None,
):
    if is_bpt:
        cond_encoder_name = "miche-encoder-bpt"
        params = {
            "vocab_size": 5184,
            "num_pos_tokens": 128,
            "bos_token_id": -100,
            "eos_token_id": 5120,
            "pad_token_id": -1,
            "prefix_len": 0,
            "cond_enc_type": "miche",
            "ar_model_type": "bpt",
            "tokenization_method": "bpt",
            "max_seq_length": 10000,
            "max_position_embeddings": 10000,
            "pos_token_offset": 0,
            "sep_token_id": 5121,
            "indicator_token_id": 5121,
            "pc_latent_len": 0,
        }
    else:
        cond_encoder_name = "edgerunner-pc-encoder"
        params = {
            "vocab_size": 576,
            "num_pos_tokens": 512,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "prefix_len": 2049,
            "cond_enc_type": "edgerunner",
            "ar_model_type": "edgerunner",
            "tokenization_method": "edgerunner",
            "max_seq_length": 43019,
            "max_position_embeddings": 43019,
            "pos_token_offset": 6,
            "sep_token_id": 518,
            "indicator_token_id": 518,
            "pc_latent_len": 2048,
        }
    _image_encoder = image_encoder or "facebook/dinov2-with-registers-base"
    _image_preprocessor = image_preprocessor or "facebook/dinov2-with-registers-base"
    model_cfg = ModelConfig(
        cond=True,
        layout_tokenization_method="full",
        loss_layout_scale=None,
        img_cond=True,
        image_encoder=_image_encoder,
        local_cond_path=f"zx1239856/{cond_encoder_name}",
        local_path=checkpoint,
        high_res_image_encoder=False,
        freeze_cond_encoder=False,
        with_ctx_pc=True,
        **params,
    )
    data_cfg = DataConfig(
        type="3d-front-layout",
        path="datasets/3d-front-ar-packed",
        num_pos_tokens=model_cfg.num_pos_tokens,
        num_points=4096 if is_bpt else 8192,
        norm_bound=0.95,
        mesh_path="datasets/3d-front-meshes",
        random_rotate_min=-45.0,
        random_rotate_max=45.0,
        random_shift=True,
        random_shift_max=0.2,
        visualize=False,
        mask_path="datasets/3d-front-panoptic",
        mask_erosion_size=3,
        use_masked_obj_pc=True,
        with_normals=is_bpt,
        random_jitter_point_clouds=False,
        load_images=True,
        image_preprocessor=_image_preprocessor,
        image_size_divisor=28,
        num_ctx_points=16384,
    )
    cond_encoder_img = get_image_condition_encoder(model_cfg)
    cond_encoder = get_condition_encoder(
        model_cfg.local_cond_path, model_cfg, cond_encoder_img=cond_encoder_img
    )
    model = get_model(
        model_cfg.local_path,
        model_cfg,
        cond_encoder=cond_encoder,
        cond_encoder_img=cond_encoder_img,
    )
    return model, model_cfg, data_cfg


def _filter_dataclass_kwargs(dataclass_type, values):
    from dataclasses import fields

    allowed = {f.name for f in fields(dataclass_type)}
    return {k: v for k, v in values.items() if k in allowed}


def prepare_mv_model_for_inference(
    checkpoint=None, config_name="edgerunner_3d_front_multiview", extra_overrides=None
):
    """Build the multi-view EdgeRunner model + configs for inference.

    Mirrors train.py's construction (Hydra-composed config → get_model with a
    pi3x_encoder + cond encoders) so the architecture exactly matches training.
    `checkpoint` overrides model.local_path (e.g. a trained MV checkpoint); when
    None the config's local_path is used (the single-view init checkpoint).
    """
    import os
    from pathlib import Path
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from src.models.utils import get_pi3x_encoder

    # train_full.sh injects RUN_TS into output-dir fields we don't use at inference;
    # set a default so resolving model/dataset interpolations doesn't KeyError.
    os.environ.setdefault("RUN_TS", "inference")

    overrides = []
    if checkpoint is not None:
        overrides.append(f"model.local_path={checkpoint}")
    if extra_overrides:
        overrides.extend(extra_overrides)

    config_dir = Path("configs").absolute().as_posix()
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=overrides)

    data_values = OmegaConf.to_container(cfg.dataset.src_data, resolve=True)
    data_cfg = DataConfig(**_filter_dataclass_kwargs(DataConfig, data_values))

    # The multi-view dataset config (dataset/canonical_3d_front_multiview.yaml) carries
    # the MV model overrides (prefix_len=322, mv_voxel_encoder=True, mv_num_*_queries,
    # ...) under cfg.dataset.model. Merge them over cfg.model so the MV encoder is
    # actually constructed (cfg.model alone holds only single-view defaults).
    model_values = OmegaConf.to_container(cfg.model, resolve=True)
    ds_model = OmegaConf.select(cfg, "dataset.model")
    if ds_model is not None:
        model_values.update(OmegaConf.to_container(ds_model, resolve=True))
    model_cfg = ModelConfig(**_filter_dataclass_kwargs(ModelConfig, model_values))

    # Derive prefix_len from the active conditioning channels so the collator emits
    # exactly as many pc_token slots as the model produces (single source of truth in
    # mv_prefix_len). Covers obj-PC-only / augment / voxel-only, including the degenerate
    # both-off case. Only for the MV path; SV configs keep their own prefix_len.
    if getattr(model_cfg, "mv_voxel_encoder", False) or model_cfg.mv_obj_pc_cond:
        model_cfg.prefix_len = mv_prefix_len(model_cfg)

    cond_encoder_img = (
        get_image_condition_encoder(model_cfg) if model_cfg.img_cond else None
    )
    cond_encoder = (
        get_condition_encoder(
            model_cfg.local_cond_path, model_cfg, cond_encoder_img=cond_encoder_img
        )
        if model_cfg.cond
        else None
    )
    pi3x_enc = get_pi3x_encoder(model_cfg) if model_cfg.use_pi3x else None
    model = get_model(
        model_cfg.local_path,
        model_cfg,
        cond_encoder=cond_encoder,
        cond_encoder_img=cond_encoder_img,
        pi3x_encoder=pi3x_enc,
    )
    return model, model_cfg, data_cfg


def prepare_mv_test_set(data_cfg):
    """Load the multi-view dataset's evaluation split with the MV transform applied.

    The 3d-front-multiview dataset is a save_to_disk DatasetDict whose only split
    is `validation`; get_mesh_dataset's train/val/test logic does not handle it, so
    we load and wrap the split directly here.
    """
    from pathlib import Path
    from transformers import AutoImageProcessor
    from src.data.mesh import transform_3d_front_multiview

    path = Path(data_cfg.path).absolute().as_posix()
    try:
        data = datasets.load_from_disk(path)
    except Exception:
        data = datasets.load_dataset(path)
    if isinstance(data, datasets.Dataset):
        split_ds = data
    else:
        split = next(
            s for s in ("validation", "val", "test") if s in data
        )
        split_ds = data[split]

    image_preprocessor = AutoImageProcessor.from_pretrained(
        data_cfg.image_preprocessor, size_divisor=data_cfg.image_size_divisor
    )
    split_ds = split_ds.with_transform(
        partial(
            transform_3d_front_multiview,
            is_train=False,
            data_cfg=data_cfg,
            image_preprocessor=image_preprocessor,
        )
    )
    return split_ds


def build_mv_uid_to_model_id(data_cfg):
    """Map MV row uid -> (3D-FUTURE model_id, ref-view mask area).

    model_id (objects.model_ids[0]) is the GT-mesh lookup key. The ref-view mask area
    (target instance's pixel count in view 0, the seed/bbox view) lets the eval apply the
    SAME small-object `mask_area_thresh` filter the single-view protocol uses — otherwise
    MV is scored on a strict superset of objects (incl. tiny/occluded ones SV skips),
    biasing the MV-vs-SV comparison. Falls back to a large area if masks are unavailable.
    """
    from pathlib import Path

    import numpy as np

    from src.data.utils import get_masks_by_ids

    path = Path(data_cfg.path).absolute().as_posix()
    try:
        data = datasets.load_from_disk(path)
    except Exception:
        data = datasets.load_dataset(path)
    if not isinstance(data, datasets.Dataset):
        split = next(s for s in ("validation", "val", "test") if s in data)
        data = data[split]
    keep = ("uid", "objects", "panoptic_masks", "panoptic_mask")
    cols = data.remove_columns([c for c in data.column_names if c not in keep])
    mapping = {}
    for row in cols:
        objs = row.get("objects")
        mids = objs.get("model_ids") if objs else None
        if not mids:
            continue
        area = 10 ** 9
        pan = row.get("panoptic_masks")
        if pan is None:
            pan = row.get("panoptic_mask")
        inst_ids = objs.get("inst_ids") if objs else None
        if pan is not None and inst_ids:
            ref_mask = pan[0] if isinstance(pan, list) else pan  # ref view = view 0
            try:
                masks = get_masks_by_ids(ref_mask, [inst_ids[0]])
                area = int(np.asarray(masks[0]).sum())
            except Exception:
                area = 10 ** 9
        mapping[row["uid"]] = (mids[0], area)
    return mapping
